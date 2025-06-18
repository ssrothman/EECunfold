import numpy as np
import unc
import eigenpy as eigen
from scipy.optimize import minimize
from tqdm import tqdm
import torchmin
import torch
import loss
from scipy.stats import multivariate_normal

#cut = {'pt' : slice(None,None,sum)}
#tcut = {'pt_reco' : slice(None,None,sum), 'pt_gen' : slice(None,None,sum)}

cut = {}
tcut = {}

def get_arrs(histdict, syst, iboot):
    the_tcut = tcut.copy()
    the_cut = cut.copy()
    if iboot is not None:
        the_tcut['bootstrap'] = iboot
        the_cut['bootstrap'] = iboot

    transfer = histdict['transfer'][syst][the_tcut].values(flow=True)

    gen = histdict['gen'][syst][the_cut].values(flow=True)
    reco = histdict['reco'][syst][the_cut].values(flow=True)

    genBkg = (histdict['unmatchedGen'][syst] + histdict['untransferedGen'][syst])[the_cut].values(flow=True)
    recoBkg = (histdict['unmatchedReco'][syst] + histdict['untransferedReco'][syst])[the_cut].values(flow=True)

    if iboot is None:
        reco = reco.reshape((reco.shape[0], -1))
        gen = gen.reshape((gen.shape[0], -1))
        genBkg = genBkg.reshape((genBkg.shape[0], -1))
        recoBkg = recoBkg.reshape((recoBkg.shape[0], -1))

        recoshape = reco.shape[1]
        genshape = gen.shape[1]
    else:
        reco = reco.ravel()
        gen = gen.ravel()
        genBkg = genBkg.ravel()
        recoBkg = recoBkg.ravel()

        recoshape = reco.shape[0] 
        genshape = gen.shape[0]

    if transfer is not None:
        transfer = transfer.reshape((recoshape, genshape))

        tdenom = gen - genBkg
        tdenom = np.where(tdenom==0, 1, tdenom)
        if iboot is None:
            tdenom = tdenom[0]
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
    transferVarIndices = []

    #stat variations
    if Nboot <= 0:
        Nboot = histdict['reco']['nominal'].axes['bootstrap'].size - 1

    print("Building stat templates...")
    _, _, rhoboot, gammaboot, _ = get_arrs(histdict, 'nominal', None)
    for iboot in tqdm(range(1, Nboot+1)):
        rhoVariations.append((rhoboot[iboot] - rho0)/Nboot)
        gammaVariations.append((gammaboot[iboot] - gamma0)/Nboot)

    #syst variations
    print("Buiding two-sided systs...")
    for syst in tqdm(two_sided_systs):
        _, _, rho_up, gamma_up, transfer_up = get_arrs(histdict, '%sUp'%syst, 0)
        _, _, rho_dn, gamma_dn, transfer_dn = get_arrs(histdict, '%sDown'%syst, 0)
        rhoVariations.append(0.5*(rho_up - rho_dn))
        gammaVariations.append(0.5*(gamma_up - gamma_dn))
        transferVariations.append(0.5*(transfer_up - transfer_dn))
        transferVarIndices.append(len(rhoVariations)-1)

    print("Building one-sided systs...")
    for syst in tqdm(one_sided_systs):
        _, _, rho_up, gamma_up, transfer_up = get_arrs(histdict, syst, 0)
        rhoVariations.append(rho_up - rho0)
        gammaVariations.append(gamma_up - gamma0)
        transferVariations.append(transfer_up - transfer0)
        transferVarIndices.append(len(rhoVariations)-1)

    rhoVariations = np.asarray(rhoVariations)
    gammaVariations = np.asarray(gammaVariations)
    transferVariations = np.asarray(transferVariations)
    transferVarIndices = np.asarray(transferVarIndices)

    print("reco0: ", reco0.shape)
    print("gen0: ", gen0.shape)
    print("gamma0: ", gamma0.shape)
    print("rho0: ", rho0.shape)
    print("transfer0: ", transfer0.shape)
    print()
    print("gammaVariations", gammaVariations.shape)
    print("rhoVariations", rhoVariations.shape)
    print("transferVariations", transferVariations.shape)
    print("transferVarIndices: ", transferVarIndices.shape)
    print("\t", transferVarIndices)

    LOSS = loss.FullLoss(transfer0, transferVariations, 
                         transferVarIndices,
                         gamma0, gammaVariations, 
                         rho0, rhoVariations,
                         covmatrix = covmatrix)

    torch.set_default_dtype(torch.float64)

    return LOSS

def run_minimization(LOSS, reco, recoErr, 
                     method='scan', x0=None,
                     compute_hessian=False,
                     compute_inv_hess=False,
                     device='cuda',
                     **kwargs):

    if compute_inv_hess:
        compute_hessian = True

    if type(device) is str:
        device = torch.device(device)

    recoErr = np.where(recoErr==0, 1, recoErr)

    if x0 is None:
        x0 = torch.from_numpy(np.ones(LOSS.nBeta, dtype=reco.dtype))
        t0 = torch.from_numpy(np.zeros(LOSS.nNuisances(), dtype=reco.dtype))
        x0 = torch.cat((x0, t0), dim=0)

    if type(x0) is not torch.Tensor:
        x0 = torch.from_numpy(x0)

    if x0.shape[0] == LOSS.nBeta:
        t0 = torch.zeros(LOSS.nNuisances())
        x0 = torch.cat((x0, t0), dim=0)

    if method == 'scan':
        methodlist = ['bfgs', 'l-bfgs', 'cg', 'newton-cg', 'newton-exact', 
                      'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']
    else:
        methodlist = [method]
        
    LOSS = LOSS.torch()

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
            print("Stack trace:")
            import traceback
            traceback.print_exc()
            continue

    if compute_hessian:
        print("Computing Hessian...")
        res.hess = torch.autograd.functional.hessian(theloss, res.x, vectorize=False)

        if compute_inv_hess:
            import eigenpy as eigen
            print("Computing inverse Hessian...")
            codhess = eigen.CompleteOrthogonalDecomposition(res.hess.cpu().detach().numpy())
            res.invhess = codhess.pseudoInverse()

    res_to_npy(res)
    reco = reco.cpu().detach().numpy()
    recoErr = recoErr.cpu().detach().numpy()

    return res, reco, recoErr

def res_to_npy(res):
    for key in res.keys():
        if type(res[key]) is torch.Tensor:
            res[key] = res[key].cpu().detach().numpy()

def dump_result(x, invhess, reco, Htemplate, destination):
    import hist
    import pickle

    if type(x) is torch.Tensor:
        x = x.cpu().detach().numpy()
    if type(reco) is torch.Tensor:
        reco = reco.cpu().detach().numpy()

    Hres = Htemplate.copy().reset()

    shape = list(Htemplate.values(flow=True).shape[1:])

    Hres.view(flow=True)[0] += (x[:reco.shape[0]] * reco).reshape(shape)

    distr = multivariate_normal(x, invhess, allow_singular=True)
    samples = distr.rvs(size=(Hres.axes['bootstrap'].size-1,))

    Hres.view(flow=True)[1:] += (samples[:,:reco.shape[0]] * reco[None,:]).reshape((Hres.axes['bootstrap'].size-1, *shape))

    with open(destination, 'wb') as f:
        pickle.dump(Hres, f)

    return Hres

