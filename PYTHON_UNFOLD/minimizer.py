import numpy as np
import unc
import eigenpy as eigen
from scipy.optimize import minimize
from tqdm import tqdm
import torchmin
import torch
import loss

#cut = {'pt' : slice(None,None,sum)}
#tcut = {'pt_reco' : slice(None,None,sum), 'pt_gen' : slice(None,None,sum)}

cut = {}
tcut = {}

def setup_loss(histdict, covmatrix=False):
    #nominal
    Htransfer = histdict['transfer']['nominal']
    Hgen = histdict['gen']['nominal']
    HgenBkg = histdict['unmatchedGen']['nominal'] + histdict['untransferedGen']['nominal']
    Hreco = histdict['reco']['nominal']
    HrecoBkg = histdict['unmatchedReco']['nominal'] + histdict['untransferedReco']['nominal']

    Htransfer = Htransfer[tcut]
    Hgen = Hgen[cut]
    HgenBkg = HgenBkg[cut]
    Hreco = Hreco[cut]
    HrecoBkg = HrecoBkg[cut]

    reco0 = Hreco[{'bootstrap' : 0}].values(flow=True).ravel()
    gen0 = Hgen[{'bootstrap' : 0}].values(flow=True).ravel()
    recoBkg0 = HrecoBkg[{'bootstrap' : 0}].values(flow=True).ravel()
    genBkg0 = HgenBkg[{'bootstrap' : 0}].values(flow=True).ravel()
    transfer0 = Htransfer.values(flow=True).reshape((*reco0.shape, *gen0.shape))

    denom = gen0 - genBkg0
    denom = np.where(denom == 0, 1, denom)
    transfer0 = transfer0 / denom[None, :]

    transferVariations = []
    for key in histdict['transfer']:
        if key == 'nominal':
            continue
        up = histdict['transfer'][key][0][tcut].values(flow=True).reshape((*reco0.shape, *gen0.shape))
        dn = histdict['transfer'][key][1][tcut].values(flow=True).reshape((*reco0.shape, *gen0.shape))

        genUp = histdict['gen'][key][0][{'bootstrap' : 0}][cut].values(flow=True).ravel()
        genDn = histdict['gen'][key][1][{'bootstrap' : 0}][cut].values(flow=True).ravel()

        genBkgUp = (histdict['unmatchedGen'][key][0] + histdict['untransferedGen'][key][0])[{'bootstrap' : 0}][cut].values(flow=True).ravel()
        genBkgDn = (histdict['unmatchedGen'][key][1] + histdict['untransferedGen'][key][1])[{'bootstrap' : 0}][cut].values(flow=True).ravel()

        denomUp = genUp - genBkgUp
        denomDn = genDn - genBkgDn
        denomUp = np.where(denomUp==0, 1, denomUp)
        denomDn = np.where(denomDn==0, 1, denomDn)
        up = up / denomUp[None, :]
        dn = dn / denomDn[None, :]

        transferVariations.append((up - dn)/2)

    denomG = np.where(gen0==0, 1, gen0)
    denomR = np.where(reco0-recoBkg0==0, 1, reco0-recoBkg0)
    gamma0 = genBkg0 / denomG
    rho0 = recoBkg0 / denomR

    gammaVariations = []
    rhoVariations = []

    nboot = Hreco.axes['bootstrap'].size - 1
    for iboot in range(nboot):
        reco_i = Hreco[{'bootstrap' : iboot+1}].values(flow=True).ravel()
        recoBkg_i = HrecoBkg[{'bootstrap' : iboot+1}].values(flow=True).ravel()
        gen_i = Hgen[{'bootstrap' : iboot+1}].values(flow=True).ravel()
        genBkg_i = HgenBkg[{'bootstrap' : iboot+1}].values(flow=True).ravel()

        denomGi = np.where(gen_i==0, 1, gen_i)
        denomRi = np.where(reco_i - recoBkg_i == 0, 1, reco_i - recoBkg_i)

        gamma_i = genBkg_i / denomGi
        rho_i = recoBkg_i / denomRi

        gammaVariations.append((gamma_i - gamma0)/nboot)
        rhoVariations.append((rho_i - rho0)/nboot)

    transferVariations = np.asarray(transferVariations)
    gammaVariations = np.asarray(gammaVariations)
    rhoVariations = np.asarray(rhoVariations)

    print("reco0: ", reco0.shape)
    print("gen0: ", gen0.shape)

    print("gamma0: ", gamma0.shape)
    print("rho0: ", rho0.shape)

    print("transfer0: ", transfer0.shape)
    print()
    print("gammaVariations", gammaVariations.shape)
    print("rhoVariations", rhoVariations.shape)
    print("transferVariations", transferVariations.shape)

    LOSS = loss.FullLoss(transfer0, transferVariations, 
                         gamma0, gammaVariations, 
                         rho0, rhoVariations,
                         covmatrix = covmatrix)

    torch.set_default_dtype(torch.float64)

    return LOSS

def run_minimization(Hreco, LOSS, iboot=0, 
                     method='scan', x0=None,
                     compute_hessian=False,
                     recoErr = None,
                     **kwargs):

    reco = Hreco[cut][{'bootstrap' : iboot}].values(flow=True).ravel()

    if recoErr is None:
        if LOSS.covmatrix:
            print("computing cov...")
            cov = unc.cov(Hreco[cut])
            import eigenpy as eigen
            print("inverting cov...")
            cod = eigen.CompleteOrthogonalDecomposition(cov)
            recoErr = cod.pseudoInverse()

            np.fill_diagonal(recoErr, np.where(np.diagonal(recoErr)==0, 1, np.diagonal(recoErr)))

        else:
            recoErr = unc.unc(Hreco[cut])
            recoErr = np.where(recoErr==0, 1, recoErr)

        recoErr = torch.from_numpy(recoErr)
    else:
        if LOSS.covmatrix and len(recoErr.shape) != 2:
            print("ERROR: need to pass 2d inverse covariance matrix as recoErr")
            return
        elif not LOSS.covmatrix and len(recoErr.shape) != 1:
            print("ERROR: need to pass 1d standard deviation vector as recoErr")
            return

    reco = torch.from_numpy(reco)

    if x0 is None:
        x0 = torch.ones(LOSS.nBeta)
        t0 = torch.zeros(LOSS.nNuisances())
        x0 = torch.cat((x0, t0), dim=0)

    if method == 'scan':
        methodlist = ['bfgs', 'l-bfgs', 'cg', 'newton-cg', 'newton-exact', 
                      'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']
    else:
        methodlist = [method]
        
    LOSS = LOSS.cuda()
    reco = reco.cuda()
    recoErr = recoErr.cuda()
    theloss = LOSS.one_parameter_loss(reco, recoErr)
    x0 = x0.cuda()

    for method in methodlist:
        try:
            from time import time
            print(method)
            print("starting minimization")
            print("initial loss = %g"%theloss(x0).item())
            t0 = time()
            res = torchmin.minimize(
                    theloss, x0 = x0,
                    method = method,
                    callback = lambda x : print("LOSS:", theloss(x).item()),
                    options = kwargs,
            )
            print("\tt =", time()-t0)
            print("\tSuccess: ", res.success)
            print("\tStatus: ", res.status)
            print("\tMessage: ", res.message)
            print("\tL = %g"%res.fun.cpu().detach().item())
        except Exception as e:
            print(f"Method {method} failed with error: {e}")
            #print("Stack trace:")
            #import traceback
            #traceback.print_exc()
            continue

    if compute_hessian:
        print("Computing Hessian...")
        res.hess = torch.autograd.functional.hessian(theloss, res.x, vectorize=False)

    return res, reco, recoErr

def dump_result(x, Htemplate, destination):
    import hist
    import pickle

    if type(x) is torch.Tensor:
        x = x.numpy(force=True)

    axes = []
    for axis in Htemplate.axes:
        if axis.name == 'bootstrap':
            continue
        else:
                axes.append(axis)

    Hres = hist.Hist(
        hist.axis.Integer(0, 1, name='bootstrap', label='bootstrap', overflow=False, underflow=False),
        *axes,
        storage=hist.storage.Double()
    )

    shape = list(Htemplate.values(flow=True).shape)
    shape[0] = 1
    x = x.reshape(shape)

    Hres += x

    with open(destination, 'wb') as f:
        pickle.dump(Hres, f)

    return Hres

