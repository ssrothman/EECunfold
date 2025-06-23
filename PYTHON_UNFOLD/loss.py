import numpy as np
import torch

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
        pass

    def setup(self, transfer0, transferVariations, 
              transferVarIndices,
              gamma0, gammaVariations,
              rho0, rhoVariations,
              namedNuisances=None,
              covmatrix = False):

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

        self.covmatrix = covmatrix
        self.namedNuisances = namedNuisances

        print("nBeta:", self.nBeta)
        print("nTheta:", self.nTheta)
        print("Expecting inverse covariance matrix?", self.covmatrix)

    def write_to_disk(self, path):
        import os.path
        self.cpu().detach().numpy()

        for name in self.arrays:
            with open(os.path.join(path, f"{name}.npy"), 'wb') as f:
                print("Writing", name, "to", f.name)
                np.save(f, getattr(self, name))

        with open(os.path.join(path, "features.pkl"), 'wb') as f:
            print("Writing features to", f.name)
            pickle.dump({
                'arrays' : self.arrays,
                'nBeta': self.nBeta,
                'nTheta': self.nTheta,
                'nTransfer': self.nTransfer,
                'covmatrix': self.covmatrix,
                'namedNuisances': self.namedNuisances
            }, f)

    def read_from_disk(self, path):
        import os.path
        import pickle

        with open(os.path.join(path, "features.pkl"), 'rb') as f:
            print("Reading features from", f.name)
            features = pickle.load(f)

        self.arrays = features['arrays']
        self.nBeta = features['nBeta']
        self.nTheta = features['nTheta']
        self.nTransfer = features['nTransfer']
        self.covmatrix = features['covmatrix']
        self.namedNuisances = features['namedNuisances']

        for name in self.arrays:
            print("Reading", name, "from", os.path.join(path, f"{name}.npy"))
            with open(os.path.join(path, f"{name}.npy"), 'rb') as f:
                setattr(self, name, np.load(f))

    def getGoodX0(self, reco):
        print("Building good x0 guess by inverting transfer matrix...")
        T = self.transfer0

        # rho = recoBkg / (reco - recoBkg) 
        # -> recoBkg = rho * (reco - recoBkg)
        # -> recoBkg = rho * reco - rho * recoBkg
        # -> recoBkg * (1 + rho) = rho * reco
        # -> recoBkg = rho * reco / (1 + rho)

        recoBkgGuess = self.rho0 * reco / (1 + self.rho0)
        Rpure = reco - recoBkgGuess
        
        import eigenpy as eigen
        if type(T) is torch.Tensor:
            T = T.cpu().numpy()
        if type(Rpure) is torch.Tensor:
            Rpure = Rpure.cpu().numpy()
        if type(reco) is torch.Tensor:
            reco = reco.cpu().numpy()

        codT = eigen.CompleteOrthogonalDecomposition(T)
        Gpure = codT.solve(Rpure)
        
        # gamma = genBkg / gen
        # -> genBkg = gamma * gen
        # -> (gen - genBkg) = gen - gamma * gen
        # -> (gen - genBkg) = gen * (1 - gamma)
        # -> gen = (gen - genBkg) / (1 - gamma)
        beta0 = Gpure / (1 - self.gamma0)

        denom = np.where(reco == 0, 1, reco)
        x0 = beta0 / denom
        x0[reco == 0] = 1

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

        negBTerm = 1000*torch.sum(torch.square(beta)[beta<=0])
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

        cstrTerm = torch.sum(torch.square(x[self.nBeta:]))
        #print("CSTR: ", cstrTerm)
        
        return 0.5 * (errTerm + cstrTerm) + negBTerm

    def one_parameter_loss(self, reco, recoErr):
        if type(reco) is not torch.Tensor:
            reco = torch.from_numpy(reco)
        if type(recoErr) is not torch.Tensor:
            recoErr = torch.from_numpy(reco)

        reco = reco.to(self.transfer0.device)
        recoErr = recoErr.to(self.transfer0.device)

        return lambda x: self.loss(x, reco, recoErr)

    def nNuisances(self):
        return self.nTheta

    def get_beta(self, x):
        return x[:self.nBeta]

    def get_theta(self, x):
        return x[self.nBeta:]

    def numpy(self, *args, **kwargs):
        for name in self.arrays:
            setattr(self, name, getattr(self, name).numpy(*args, **kwargs))

        return self

    def torch(self):
        for name in self.arrays:
            if type(getattr(self, name)) is not torch.Tensor:
                setattr(self, name, torch.from_numpy(getattr(self, name)))

        return self

    def cpu(self):
        for name in self.arrays:
            setattr(self, name, getattr(self, name).cpu())

        return self

    def cuda(self):
        for name in self.arrays:
            setattr(self, name, getattr(self, name).cuda())

        return self

    def to(self, device):
        for name in self.arrays:
            setattr(self, name, getattr(self, name).to(device))

        return self

    def detach(self):
        for name in self.arrays:
            setattr(self, name, getattr(self, name).detach())
