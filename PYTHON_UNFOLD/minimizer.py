import numpy as np
import unc
import eigenpy as eigen
from scipy.optimize import minimize
from tqdm import tqdm
import torchmin
import torch
import loss

def demo(Htransfer, 
         Hgen, HunmatchedGen, HuntransferedGen, 
         Hreco, HunmatchedReco, HuntransferedReco, 
         iboot=0, LossClass='Simplest',
         method='scan', x0=None):

    Hgenpure = Hgen - HunmatchedGen - HuntransferedGen
    Hrecopure = Hreco - HunmatchedReco - HuntransferedReco

    #Htransfer = Htransfer[{'pt_gen' : slice(None,None,sum),
    #                     'pt_reco' : slice(None,None,sum)}]
    #Hgenpure = Hgenpure[{'pt' : slice(None,None,sum)}]
    #Hrecopure = Hrecopure[{'pt' : slice(None,None,sum)}]
    #Hgen = Hgen[{'pt' : slice(None,None,sum)}]
    #Hreco = Hreco[{'pt' : slice(None,None,sum)}]

    transfer = torch.from_numpy(Htransfer.values(flow=True))

    genpure = torch.from_numpy(Hgenpure[{'bootstrap' : iboot}].values(flow=True).ravel())
    recopure = torch.from_numpy(Hrecopure[{'bootstrap' : iboot}].values(flow=True).ravel())

    gen = torch.from_numpy(Hgen[{'bootstrap' : iboot}].values(flow=True).ravel())
    reco = torch.from_numpy(Hreco[{'bootstrap' : iboot}].values(flow=True).ravel())

    gamma0 = (gen-genpure) / gen
    rho0 = (reco - recopure) / recopure
    gammaErr = gamma0/20
    rhoErr = rho0/20

    #bootstrap dimension has under/overflow that we don't want
    genshape = np.prod(gen.shape)
    recoshape = np.prod(reco.shape)

    gen = torch.reshape(gen, (-1,))
    reco = torch.reshape(reco, (-1,))
    transfer = torch.reshape(transfer, (recoshape, genshape))

    recoerr = torch.from_numpy(unc.unc(Hreco).ravel())

    transfer /= gen[None,:]
    if 'FullModel' in LossClass:
        t2 = torch.zeros_like(transfer)
        t2 = torch.diagonal_scatter(t2, torch.ones(transfer.shape[0])*2e-3)
        t2 = torch.diagonal_scatter(t2, torch.ones(transfer.shape[0]-1)*(1e-3), offset=1)
        t2 = torch.diagonal_scatter(t2, torch.ones(transfer.shape[0]-1)*(1e-3), offset=-1)
        t3 = torch.zeros_like(transfer)
        t3 = torch.diagonal_scatter(t3, torch.ones(transfer.shape[0]-1)*1e-3, offset=1)
        t3 = torch.diagonal_scatter(t3, torch.ones(transfer.shape[0]-1)*(-1e-3), offset=-1)
        transfer = torch.cat((transfer[None,:,:], t2[None,:,:], t3[None,:,:]), dim=0)

    #run on GPU
    reco = reco.cuda()
    recoerr = recoerr.cuda()
    transfer = transfer.cuda()
    gamma0 = gamma0.cuda()
    gammaErr = gammaErr.cuda()
    rho0 = rho0.cuda()
    rhoErr = rhoErr.cuda()

    if isinstance(LossClass, str):
        if LossClass != "SimpleFullModelFullTemplate":
            LossClass = loss.losses[LossClass]()
            theloss = lambda x : LossClass.loss(x, transfer, reco, recoerr)
        else:
            LossClass = loss.SimpleFullModelFullTemplateLoss(transfer, 
                                                             gamma0, 
                                                             gammaErr, 
                                                             rho0, rhoErr).cuda()
            theloss = LossClass.one_parameter_loss(reco, recoerr)

    if x0 is None:
        x0 = torch.ones_like(gen).cuda()
        t0 = torch.zeros(LossClass.nNuisances(), dtype=gen.dtype).cuda()
        x0 = torch.cat((x0, t0), dim=0)

    x0 = x0.cuda()

    if method == 'scan':
        methodlist = ['bfgs', 'l-bfgs', 'cg', 'newton-cg', 'newton-exact', 
                      'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']
    else:
        methodlist = [method]
        
    for method in methodlist:
        try:
            from time import time
            t0 = time()
            res = torchmin.minimize(theloss, x0 = x0,
                                    method = method)
            print(method)
            print("\tt =", time()-t0)
            print("\tSuccess: ", res.success)
            print("\tStatus: ", res.status)
            print("\tMessage: ", res.message)
            print("\tunfolded == genpure?", np.allclose(res.x.cpu()[:-LossClass.nNuisances()]*reco.cpu(), gen.cpu()))
            print("\tL = %g"%res.fun.cpu().detach().item())
        except Exception as e:
            print(f"Method {method} failed with error: {e}")
            #print("Stack trace:")
            #import traceback
            #traceback.print_exc()
            continue

    return res, gen, reco, transfer
