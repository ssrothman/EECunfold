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
import statutil
import indexing

def map_to_indices(start, offset, maxval):
    result = start + offset
    result = torch.where(result < 0, -1 - result, result)
    result = torch.where(result >= maxval, 2 * maxval - result - 1, result)
    #for i in range(len(offset)):
    #    print("%d + %d -> %d" % (start, offset[i], result[i]))
    return result

def smooth_Tmat_score_2d(T2d, model, monotonic_weight=1):
    #model is a 1D array, representing transfered mass to bin indices -Nm to +Nm
    #T2d is the 2D transfer matrix with shape (Nreco, Ngen) [not necessarily square]
    #indexed according to T2d(iReco, iGen) = flow from iGen -> iReco

    if len(model) % 2 == 0:
        raise ValueError("Model must have odd length (ie from -N to +N)")
    
    if len(T2d.shape) != 2:
        raise ValueError("T2d must be a 2D array")

    Nm = len(model) // 2
    Nreco = T2d.shape[0]
    Ngen = T2d.shape[1]

    starts = torch.arange(Ngen, dtype=torch.int32)[None, :]  # shape (1, Ngen)
    offsets = torch.arange(-Nm, Nm+1, dtype=torch.int32)[:, None]  # shape (2*Nm+1, 1)
    Gindices = map_to_indices(starts, offsets, Ngen)

    T2d_pred = torch.zeros((Nreco, Ngen), dtype=T2d.dtype, device=T2d.device)
    
    #this is the equivalent to something like np.add.at
    T2d_pred.index_put_((Gindices, starts), model[:,None], accumulate=True)

    #now we define a loss function...
    loss = torch.sum(torch.square(T2d_pred - T2d))

    if monotonic_weight > 0:
        # Add a monotonicity penalty
        # We want the model to be monotonic, so we penalize negative slopes
        slopes_right = model[Nm:-1] - model[Nm+1:]
        slopes_left = model[1:Nm] - model[0:Nm-1]

        penalty = torch.sum(torch.relu(-slopes_right)) + torch.sum(torch.relu(-slopes_left))
        loss += monotonic_weight * penalty

    return loss

def optimize_T2d_model(T2d, Nm=5):
    theloss = lambda model: smooth_Tmat_score_2d(T2d, model)
    import torchmin
    x0 = torch.zeros(2 * Nm+1, dtype=T2d.dtype, device=T2d.device)
    x0[Nm] = 1.0
    res = torchmin.minimize(
        theloss, x0=x0,
        method='l-bfgs',
        callback = lambda x: print("Loss: %g" % theloss(x)),
    )
    return res

def t3d_pred(Np, Nm, Nreco, Ngen, model):
    starts = torch.arange(Ngen, dtype=torch.int32)[None, None, :]  # shape (1, 1, Ngen)
    offsets = torch.arange(-Nm, Nm+1, dtype=torch.int32)[None,:,None]  # shape (2*Nm+1, 1, 1)
    Gindices = map_to_indices(starts, offsets, Ngen)

    T3d_pred = torch.zeros((Np, Nreco, Ngen), dtype=model.dtype, device=model.device)

    #this is the equivalent to something like np.add.at
    T3d_pred.index_put_((torch.arange(Np)[:, None, None], Gindices, starts),
                        model[:, :, None], accumulate=True)

    return T3d_pred

def smooth_Tmat_score_3d(T3d, model, monotonic_weight=0):
    '''
    Basically solve Np 2d problems in parallel

    model is a 2D array of shape (Np, 2*Nm+1)
        representing Np x (transfered mass to bin indices -Nm to +Nm) arrays
    T3d is a 3D array of shape (Np, Nreco, Ngen)
    '''
    if T3d.shape[0] != model.shape[0]:
        raise ValueError("T3d and model must have the same first dimension")
    if len(model.shape) != 2:
        raise ValueError("Model must be a 2D array")
    if len(T3d.shape) != 3:
        raise ValueError("T3d must be a 3D array")

    Np = T3d.shape[0]
    Nm = model.shape[1] // 2
    Nreco = T3d.shape[1]
    Ngen = T3d.shape[2]

    T3d_pred = t3d_pred(Np, Nm, Nreco, Ngen, model)

    loss = torch.sum(torch.square(T3d_pred - T3d))

    if monotonic_weight > 0:
        # Add a monotonicity penalty
        # We want the model to be monotonic, so we penalize negative slopes
        slopes_right = model[:, Nm:-1] - model[:, Nm+1:]
        slopes_left = model[:, 1:Nm] - model[:, 0:Nm-1]

        penalty = torch.sum(torch.relu(-slopes_right)) + torch.sum(torch.relu(-slopes_left))
        loss += monotonic_weight * penalty

    return loss

def setup_Tdiagfit(T0, Nm=5, binning=None, device='cuda'):
    #T = T0.reshape((7, 5, 15, 15, 7, 5, 15, 15))
    #Tdiag = np.einsum('abciabcj->abcij', T)
    #Tdiag = Tdiag.reshape((-1, 15, 15))
    Nc = binning.blocks[0].ax_details['c']['extent']
    Tdiag = np.zeros((0, Nc, Nc))
    for block in binning.blocks:
        if block.ax_details['c']['extent'] != Nc:
            raise ValueError("All blocks must have the same c-axis extent")
        for ipt in range(block.ax_details['pt']['extent']-1):
            for iR in range(block.ax_details['R']['extent']-1):
                for ir in range(block.ax_details['r']['extent']-1):
                    nextT = block.get_slice_from_indices(
                            T0.T,
                            pt=(ipt, ipt+1),
                            R=(iR, iR+1),
                            r=(ir, ir+1),
                    )
                    nextT = block.get_slice_from_indices(
                            nextT.T,
                            pt=(ipt, ipt+1),
                            R=(iR, iR+1),
                            r=(ir, ir+1),
                    )
                    print(nextT.shape)
                    Tdiag = np.append(Tdiag, nextT.reshape((1, Nc, Nc)), axis=0)

    print(Tdiag.shape)
    Tdiag = torch.from_numpy(Tdiag).to(device)
    x0 = torch.zeros((Tdiag.shape[0], 2 * Nm + 1), dtype=Tdiag.dtype, device=Tdiag.device)
    x0[:, Nm] = 1.0  # Set the central value to 1.0
    theloss = lambda model: smooth_Tmat_score_3d(Tdiag, model, monotonic_weight=0)
    import torchmin
    res = torchmin.minimize(
        theloss, x0=x0,
        method='l-bfgs',
        callback=lambda x: print("Loss: %g" % theloss(x.reshape(-1, 2 * Nm + 1))),
    )
    # Reshape the result back to the original shape
    xpred = t3d_pred(
        Tdiag.shape[0], Nm, Tdiag.shape[1], Tdiag.shape[2], res.x
    ).cpu().detach().numpy()
    final_pred = T0.copy()
    i = 0
    for block in binning.blocks:
        for ipt in range(block.ax_details['pt']['extent']-1):
            for iR in range(block.ax_details['R']['extent']-1):
                for ir in range(block.ax_details['r']['extent']-1):
                    block.assign_to_indices_2d(
                        final_pred,
                        xpred[i],
                        pt=(ipt, ipt+1),
                        R=(iR, iR+1),
                        r=(ir, ir+1),
                    )
                    i += 1

    print("is it different?")
    print("T0", T0.sum())
    print("final_pred", final_pred.sum())

    return res, final_pred

def smooth_the_loss(LOSS, binning):
    if LOSS.device != 'numpy':
        LOSS = LOSS.cpu().detach().numpy()

    res, TP = setup_Tdiagfit(LOSS.transfer0, Nm=5, binning=binning)
    LOSS.transfer0 = TP
    print("Transfer0 optimized:", res.success, res.message, res.fun.item())
    for i in range(LOSS.transferVariations.shape[0]):
        print(f"Optimizing TransferVariation {i}...")
        res, TP = setup_Tdiagfit(LOSS.transferVariations[i], Nm=5, binning=binning)
        LOSS.transferVariations[i] = TP
        print(f"TransferVariation {i} optimized:", res.success, res.message, res.fun.item())
    return LOSS

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
              genBaseline, baselineRecoFlux,
              namedNuisances, binning):

        self.transfer0 = transfer0
        self.gamma0 = gamma0
        self.rho0 = rho0

        self.transferVariations = transferVariations
        self.transferVarIndices = transferVarIndices

        self.gammaVariations = gammaVariations
        self.rhoVariations = rhoVariations

        self.genBaseline = np.where(genBaseline == 0, 1e-8, genBaseline)
        self.baselineRecoFlux = baselineRecoFlux

        self.arrays = ['transfer0', 'gamma0', 'rho0', 
                       'transferVariations',
                       'transferVarIndices',
                       'gammaVariations',
                       'genBaseline',
                       'baselineRecoFlux',
                       'rhoVariations']

        self.nTheta = gammaVariations.shape[0]
        self.nTransfer = transferVarIndices.shape[0]
        self.nBeta = transfer0.shape[1]

        if transferVariations.shape[0] != transferVarIndices.shape[0]:
            raise ValueError("TransferVariations needs to align with its indices")

        if gammaVariations.shape[0] != rhoVariations.shape[0]:
            raise ValueError("G, R to have same number of variations")

        self.namedNuisances = namedNuisances
        self.binning = binning

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

        self.binning.dump_to_file(os.path.join(path, 'binning.json'))

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

        self.binning = indexing.GenRecoBinning()
        self.binning.load_from_file(os.path.join(path, 'binning.json'))

    def set_1d(self):
        self.covmatrix = False

    def set_2d(self):
        self.covmatrix = True

    def set_rescaled(self, value):
        self.rescaled = value

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

        x0 = beta0 / (self.genBaseline * (reco.sum() / self.baselineRecoFlux))
        print("x0 shape:", x0.shape)
        print("\tsum:", x0.sum())

        return x0

    def getG(self, theta):
        return self.gamma0 + torch.tensordot(theta, self.gammaVariations, 1)

    def getR(self, theta):
        return self.rho0 + torch.tensordot(theta, self.rhoVariations, 1)

    def getT(self, theta):
        if len(self.transferVarIndices) > 0:
            return self.transfer0 + torch.tensordot(theta[self.transferVarIndices], self.transferVariations, 1) 
        else:
            return self.transfer0

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
        #print("CALLING LOSS")
        beta = x[:self.nBeta]
        theta = x[self.nBeta:]

        negB = torch.where(beta < 0, beta, 0)
        negBTerm = 1000*torch.sum(torch.square(negB))
        beta = torch.where(beta<0, 0, beta)

        if self.rescaled:
            #print("SUM(beta) =", beta.sum())
            fwd = self.forward(beta, theta)
        else:
            #print("SUM(beta) =", (beta*self.genBaseline*reco.sum()/self.baselineRecoFlux).sum())

            fwd = self.forward(beta*self.genBaseline*reco.sum()/self.baselineRecoFlux, theta)

        diff = fwd-reco

        if self.covmatrix:
            errTerm = torch.linalg.multi_dot((diff, recoErr, diff))
        else:
            errTerm = torch.sum(torch.square(diff/recoErr))

        cstrTerm = torch.sum(torch.square(theta))
        
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

        return lambda x: self.loss_with_frozen(x, reco, recoErr,
                                               frozen_mask,
                                               frozen_vals)

    def loss_from_fluxes_shapes(self, fluxes, shapes, fluxbinning, 
                                theta,
                                reco, recoErr,
                                frozen_mask=None,
                                frozen_vals=None):
        raise ValueError("Don't use this, it doesn't work! :(")

        if not self.rescaled:
            raise ValueError("need to set rescaled flag = True")

        fluxes2 = fluxes.clone()
        shapes2 = shapes.clone()

        axisblocks = self.binning.genbinning.get_blocks(
            fluxbinning.axis_names
        )

        starts = []
        lengths = []
        for i, block in enumerate(axisblocks):
            idx = block['slice']
            if type(idx) is not slice:
                raise ValueError("Expected slice, got %s" % type(idx))
            starts.append(idx.start)
            lengths.append(idx.stop - idx.start)

        starts = torch.tensor(starts, dtype=torch.int32, device=fluxes2.device)
        lengths = torch.tensor(lengths, dtype=torch.int32, device=fluxes2.device)
        order = torch.argsort(starts)
        
        indices = torch.repeat_interleave(
            order, lengths[order],
            dim=0, output_size=shapes2.shape[0]
        )
        sumshapes = torch.zeros_like(fluxes)
        sumshapes.scatter_add_(0, indices, shapes2)

        shapes2 /= sumshapes[indices]
        fluxes2 *= sumshapes

        beta = self.binning.genbinning.merge_fluxes_shapes(
            fluxes2, shapes2, fluxbinning
        )
        x = torch.concatenate((beta, theta))
        return self.loss_with_frozen(x, reco, recoErr,
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
