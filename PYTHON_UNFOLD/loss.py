#we need to import fasteigenpy BEFORE torch
#otherwise some fuckery happens when torch imports mkl
#which in turn breaks fasteigenpy
#I think this is related to openmp somehow??
try:
    import fasteigenpy as eigen
except ImportError:
    print("WARNING: NO FASTEIGENPY")

import numpy as np
import torch
import ioutil

class FullLoss:
    '''
    This is the full Loss model we use. The forward model works as follows:
        1. Gen-level background = G * gen
        2. Detector model applies to background-free gen component to obtain background-free reco:
            pureReco = T * (gen - gen background)
        3. Reco-level background = R * pureReco
        4. Return reco = pureReco + reco background

    We have three qualitatively different kinds of systematics:
        1. Variations in the transfer matrix. These are parameterized as 
            T = transfer0 + sum_i theta_i transferVariation_i
        2. Shape variations in the gen-level background template
            G = gamma0 + sum_i gamma_i gammaVariation_i
        3. Shape variations in the reco-level background template
            R = rho0 + sum_i rho_i rhoVariations_i

    All three have gaussian constraints,

    The loss a simple chi-squared loss function:

    L = 1/2 (reco_pred - reco_true)^2 / reco_uncertainty^2 + 1/2 sum_i (nuisance_i)^2
    TODO: is the 1/2 correct?
    '''
    def __init__(self):
        self.device = 'numpy'
        pass

    def setup(self, transfer0, transferVariations, 
              transferVarIndices,
              gamma0, gammaVariations,
              rho0, rhoVariations,
              namedNuisances=None):

        self.transfer0 = transfer0
        self.gamma0 = gamma0
        self.rho0 = rho0

        self.transferVariations = transferVariations
        self.transferVarIndices = transferVarIndices

        self.gammaVariations = gammaVariations
        self.rhoVariations = rhoVariations

        self.arrays = ['transfer0', 'gamma0', 'rho0', 
                       'transferVariations',
                       'transferVarIndices',
                       'gammaVariations',
                       'rhoVariations']

        self.nTheta = gammaVariations.shape[0]
        self.nTransfer = transferVarIndices.shape[0]
        self.nBeta = transfer0.shape[1]

        if transferVariations.shape[0] != transferVarIndices.shape[0]:
            raise ValueError("TransferVariations needs to align with its indices")

        if gammaVariations.shape[0] != rhoVariations.shape[0]:
            raise ValueError("G, R to have same number of variations")

        self.namedNuisances = namedNuisances

        print("nBeta:", self.nBeta)
        print("nTheta:", self.nTheta)

    def __str__(self):
        result = ''
        result += "FullLoss:\n"
        for name in self.arrays:
            result += '\t' + name + str(getattr(self, name).shape) + '\n'
            if name == 'transferVarIndices':
                result += '\t\t' + str(getattr(self, name)) + '\n'
        result += '\n'
        result += '\tnBeta: ' + str(self.nBeta) + '\n'
        result += '\tnTheta: ' + str(self.nTheta) + '\n'
        result += '\tnTransfer: ' + str(self.nTransfer) + '\n'

        result += '\n\tnamedNuisances:\n'
        for key in self.namedNuisances:
            result += '\t\t' + self.namedNuisances[key] + ':' + str(key) + '\n'

        return result
        

    def write_to_disk(self, path):
        import os
        print("Writing loss to", path)
        os.makedirs(path, exist_ok=True)

        self.cpu().detach().numpy()

        for name in self.arrays:
            ioutil.wrapped_write_np(os.path.join(path, f"{name}.npy"),
                                    getattr(self, name))

        ioutil.wrapped_write_json(os.path.join(path, "features.json"),
            {
                'arrays' : self.arrays,
                'nBeta': self.nBeta,
                'nTheta': self.nTheta,
                'nTransfer': self.nTransfer,
                'namedNuisances': self.namedNuisances
            })

    def read_from_disk(self, path):
        import os.path

        features = ioutil.wrapped_read_json(os.path.join(path, "features.json"))

        self.arrays = features['arrays']
        self.nBeta = features['nBeta']
        self.nTheta = features['nTheta']
        self.nTransfer = features['nTransfer']
        self.namedNuisances = features['namedNuisances']

        for name in self.arrays:
            setattr(self, name, ioutil.wrapped_read_np(os.path.join(path, f"{name}.npy")))

    def set_1d(self):
        self.covmatrix = False

    def set_2d(self):
        self.covmatrix = True

    def getGoodX0(self, reco):
        print("Building good x0 guess by inverting transfer matrix...")
        print("reco shape:", reco.shape)
        print("\tsum:", reco.sum())
        T = self.transfer0
        print("T shape:", T.shape)
        print("\tsum:", T.sum())

        # rho = recoBkg / (reco - recoBkg) 
        # -> recoBkg = rho * (reco - recoBkg)
        # -> recoBkg = rho * reco - rho * recoBkg
        # -> recoBkg * (1 + rho) = rho * reco
        # -> recoBkg = rho * reco / (1 + rho)

        recoBkgGuess = self.rho0 * reco / (1 + self.rho0)
        Rpure = reco - recoBkgGuess
        print("Rpure shape:", Rpure.shape)
        print("\tsum:", Rpure.sum())
        
        if type(T) is torch.Tensor:
            T = T.cpu().numpy()
        if type(Rpure) is torch.Tensor:
            Rpure = Rpure.cpu().numpy()
        if type(reco) is torch.Tensor:
            reco = reco.cpu().numpy()

        codT = eigen.CompleteOrthogonalDecomposition(T)
        Gpure = codT.solve(Rpure).squeeze()
        print("Gpure shape:", Gpure.shape)
        print("\tsum:", Gpure.sum())
        
        # gamma = genBkg / gen
        # -> genBkg = gamma * gen
        # -> (gen - genBkg) = gen - gamma * gen
        # -> (gen - genBkg) = gen * (1 - gamma)
        # -> gen = (gen - genBkg) / (1 - gamma)
        beta0 = Gpure / (1 - self.gamma0)
        print("beta0 shape:", beta0.shape)
        print("\tsum:", beta0.sum())

        denom = np.where(reco == 0, 1, reco)
        x0 = beta0 / denom
        x0[reco == 0] = 1
        print("x0 shape:", x0.shape)
        print("\tsum:", x0.sum())

        return x0

    def getG(self, theta):
        return self.gamma0 + torch.tensordot(theta, self.gammaVariations, 1)

    def getR(self, theta):
        return self.rho0 + torch.tensordot(theta, self.rhoVariations, 1)

    def getT(self, theta):
        return self.transfer0 + torch.tensordot(theta[self.transferVarIndices], self.transferVariations, 1) 

    def genBkg(self, beta, theta):
        return self.getG(theta) * beta

    def recoBkg(self, p, theta):
        return self.getR(theta) * p

    def forward(self, beta, theta):
        genpure = beta - self.genBkg(beta, theta)

        p = torch.matmul(self.getT(theta), genpure)

        return p + self.recoBkg(p, theta)

    def forward_1arg(self, x):
        return self.forward(self.get_beta(x), self.get_theta(x))

    def loss(self, x, reco, recoErr):
        beta = x[:self.nBeta]
        theta = x[self.nBeta:]

        negB = torch.where(beta < 0, beta, 0)
        negBTerm = 1000*torch.sum(torch.square(negB))
        beta = torch.where(beta<0, 0, beta)

        fwd = self.forward(beta*reco, theta)
        #print("FWD: ", fwd)
        diff = fwd-reco
        #print("DIFF: ", diff)

        if self.covmatrix:
            errTerm = torch.linalg.multi_dot((diff, recoErr, diff))
        else:
            errTerm = torch.sum(torch.square(diff/recoErr))
        #print("ERR: ", errTerm)

        cstrTerm = torch.sum(torch.square(theta))
        #print("CSTR: ", cstrTerm)
        
        return 0.5 * (errTerm + cstrTerm) + negBTerm

    def loss_with_frozen(self, x, reco, recoErr, 
                         frozen_mask=None, 
                         frozen_vals=None):
        if frozen_mask is None or torch.sum(frozen_mask)==0:
            return self.loss(x, reco, recoErr)
        else:
            newx = torch.zeros(self.nBeta + self.nTheta,
                               dtype=x.dtype,
                               device=x.device)
            newx[frozen_mask] = frozen_vals
            newx[~frozen_mask] = x
            return self.loss(newx, reco, recoErr)

    def one_parameter_loss(self, reco, recoErr,
                           frozen_mask=None,
                           frozen_vals=None):

        if type(reco) is not torch.Tensor:
            reco = torch.from_numpy(reco)
        if type(recoErr) is not torch.Tensor:
            recoErr = torch.from_numpy(reco)

        reco = reco.to(self.transfer0.device)
        recoErr = recoErr.to(self.transfer0.device)

        return lambda x: self.loss_with_frozen(x, reco, recoErr,
                                               frozen_mask,
                                               frozen_vals)

    def get_beta(self, x):
        return x[:self.nBeta]

    def get_theta(self, x):
        return x[self.nBeta:]

    def numpy(self, *args, **kwargs):
        for name in self.arrays:
            if type(getattr(self, name)) is not torch.Tensor:
                continue
            setattr(self, name, getattr(self, name).numpy(*args, **kwargs))

        self.device = 'numpy'

        return self

    def torch(self):
        for name in self.arrays:
            if type(getattr(self, name)) is not torch.Tensor:
                setattr(self, name, torch.from_numpy(getattr(self, name)))

        self.device = 'cpu'

        return self

    def cpu(self):
        for name in self.arrays:
            if type(getattr(self, name)) is not torch.Tensor:
                continue
            setattr(self, name, getattr(self, name).cpu())

        self.device = 'cpu'

        return self

    def cuda(self):
        for name in self.arrays:
            setattr(self, name, getattr(self, name).cuda())

        self.device = 'cuda'

        return self

    def to(self, device):
        for name in self.arrays:
            setattr(self, name, getattr(self, name).to(device))

        self.device = device

        return self

    def detach(self):
        for name in self.arrays:
            if type(getattr(self, name)) is not torch.Tensor:
                continue
            setattr(self, name, getattr(self, name).detach())

        return self
