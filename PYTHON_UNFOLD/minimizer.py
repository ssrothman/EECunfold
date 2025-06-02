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

def get_arrs(histdict, syst, iboot):
    the_tcut = tcut
    the_tcut['bootstrap'] = iboot

    the_cut = cut
    the_cut['bootstrap'] = iboot

    transfer = histdict['transfer'][syst][tcut].values(flow=True)

    gen = histdict['gen'][syst][cut].values(flow=True).ravel()
    reco = histdict['reco'][syst][cut].values(flow=True).ravel()

    genBkg = (histdict['unmatchedGen'][syst] + histdict['untransferedGen'][syst])[cut].values(flow=True).ravel()
    recoBkg = (histdict['unmatchedReco'][syst] + histdict['untransferedReco'][syst])[cut].values(flow=True).ravel()
    
    transfer = transfer.reshape((reco.shape[0], gen.shape[0]))

    tdenom = gen - genBkg
    tdenom = np.where(tdenom==0, 1, tedenom)
    transfer = transfer/tdenom[None, :]

    Gdenom = np.where(gen==0, 1, gen)
    gamma = genBkg / Gdenom

    Rdenom = np.where(reco-recoBkg==0, 1, reco-recoBkg)
    rho = recoBkg / Rdenom

    return reco, gen, rho, gamma, transfer

def setup_loss(histdict, covmatrix=False, 
               Nboot=-1,
               two_sided_systs=[],
               one_sided_systs=[]):
    #nominal
    reco0, gen0, rho0, gamma0, transfer0 = get_arrs(histdict, 'nominal', 0)

    rhoVariations = []
    gammaVariations = []
    transferVariations = []

    #stat variations
    if Nboot <= 0:
        Nboot = histdict['reco']['nominal'].axes['bootstrap'].size - 1

    for iboot in range(1, Nboot+1):
        _, _, rho_i, gamma_i, transfer_i = get_arrs(histdict, 'nominal', iboot)
        rhoVariations.append((rho_i - rho0)/Nboot)
        gammaVariations.append((gamma_i - gamma0)/Nboot)
        transferVariations.append((transfer_i - transfer0)/Nboot)

    #syst variations
    for syst in two_sided_systs:
        _, _, rho_up, gamma_up, transfer_up = get_arrs(histdict, '%sUp'%syst, 0)
        _, _, rho_dn, gamma_dn, transfer_dn = get_arrs(histdict, '%sDown'%syst, 0)
        rhoVariations.append(0.5*(rho_up - rho_dn))
        gammaVariations.append(0.5*(gamma_up - gamma_dn))
        transferVariations.append(0.5*(transfer_up - transfer_dn))

    for syst in one_sided_systs:
        _, _, rho_up, gamma_up, transfer_up = get_arrs(histdict, syst, 0)
        rhoVariations.append(rho_up - rho0)
        gammaVariations.append(gamma_up - gamma0)
        transferVariations.append(transfer_up - transfer0)

    rhoVariations = np.asarray(rhoVariations)
    gammaVariations = np.asarray(gammaVaraitions)
    transferVariations = np.asarray(transferVariations)

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
                     compute_inv_hess=False,
                     recoErr = None,
                     device='cuda',
                     **kwargs):

    if compute_inv_hess:
        compute_hessian = True

    if type(device) is str:
        device = torch.device(device)

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
        
    LOSS = LOSS.to(device)
    if type(reco) is not torch.Tensor:
        reco = torch.from_numpy(reco)
    reco = reco.to(device)
    if type(recoErr) is not torch.Tensor:
        recoErr = torch.from_numpy(recoErr)
    recoErr = recoErr.to(device)
    theloss = LOSS.one_parameter_loss(reco, recoErr)
    if type(x0) is not torch.Tensor:
        x0 = torch.from_numpy(x0)
    x0 = x0.to(device)

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

        if compute_inv_hess:
            import eigenpy as eigen
            print("Computing inverse Hessian...")
            codhess = eigen.CompleteOrthogonalDecomposition(res.hess.numpy(force=True))
            res.invhess = codhess.pseudoInverse()

    res_to_npy(res)
    reco = reco.numpy(force=True)
    recoErr = recoErr.numpy(force=True)

    return res, reco, recoErr

def res_to_npy(res):
    for key in res.keys():
        if type(res[key]) is torch.Tensor:
            res[key] = res[key].numpy(force=True)

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

