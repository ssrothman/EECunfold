import numpy as np
import scipy.linalg

import torch

@torch.jit.script
def lsq_torch(truth, chi, kappas, gamma, Nbeta):
    beta = truth[:Nbeta]
    theta = truth[Nbeta:]

    K = torch.einsum('i,ijk->jk', theta, kappas)
    
    T = K + chi

    residuals = T @ beta - gamma

    return 0.5 * torch.concatenate((residuals, theta))

import numba

@numba.jit(nopython=True)
def lsq_np(truth, chi, kappas, gamma, Nbeta):
    beta = truth[:Nbeta]
    theta = truth[Nbeta:]

    K = np.empty_like(chi)
    for i in range(len(theta)):
        K += theta[i] * kappas[i]
    #K = np.einsum('i,ijk->jk', theta, kappas)
    
    T = K + chi

    residuals = T @ beta - gamma

    loss = 0.5 * (np.sum(np.square(residuals)) + np.sum(np.square(theta)))

    grad = np.empty_like(truth)
    grad[:Nbeta] = T.T @ residuals
    for i in range(len(theta)):
        grad[Nbeta+i] = theta[i] + (T @ beta - gamma).T @ (kappas[i] @ beta)

    return (loss, grad)

@numba.jit(nopython=True)
def lsq_hess(truth, chi, kappas, gamma, Nbeta):
    beta = truth[:Nbeta]
    theta = truth[Nbeta:]

    '''
    grad_beta = (chi + theta_i kappa_i).T @ [(chi + theta_i kappa_i) @ beta - gamma]
    grad_theta_i = theta_i + [(chi + theta_i kappa_i) @ beta - gamma].T @ (kappa_i @ beta)

    '''

    hess = np.empty((len(truth), len(truth)))

    hess[:Nbeta, :Nbeta] = chi.T @ chi
    hess[Nbeta:, Nbeta:] = np.eye(len(theta))
    for i in range(len(theta)):
        for j in range(len(theta)):
            hess[Nbeta+i, Nbeta+j] = (kappas[i] @ beta).T @ (kappas[j] @ beta)

    K = np.empty_like(chi)
    for i in range(len(theta)):
        K += theta[i] * kappas[i]

    T = K + chi
    for i in range(len(theta)):
        cross_hess = kappas[i].T @ (T @ beta - gamma) + T.T @ (kappas[i] @ beta)
        hess[Nbeta+i, :Nbeta] = cross_hess
        hess[:Nbeta, Nbeta+i] = cross_hess

    #hess += np.eye(len(truth)) * 0
    return hess

def build_likelihood(nom, systs, data,
                     EPS=1.0e-4,
                     force_nonsingular=False,
                     dochecks=True,
                     usetorch=True):
    '''
    nom: dict of nominal data
    systs: list of pairs of dicts [up, down]
    data: dict of 'data'
    '''
    C = data['covreco']
    if force_nonsingular:
        baddiag = np.diag(C) == 0
        diag_x0, diag_x1 = np.diag_indices_from(C)
        C[diag_x0[baddiag], diag_x1[baddiag]] = 1
    C += EPS * np.eye(C.shape[0])

    #C = np.eye(C.shape[0])

    y = data['reco']

    X = nom['transfer']

    Ks = []
    for syst in systs:
        up = syst[0]
        down = syst[1]
        Ku = up['transfer']
        Kd = down['transfer']
        Ks.append(0.5 * (Ku-Kd))

    print(C.shape)
    print(C)

    L = scipy.linalg.cholesky(C, lower=True)
    gamma = scipy.linalg.solve(L, y)
    chi = scipy.linalg.solve(L, X)

    if len(Ks) > 0:
        kappas = []
        for K in Ks:
            kappa = scipy.linalg.solve(L, K)
            kappas.append(kappa[None,:,:])
        kappas = np.concatenate(kappas, axis=0)
    else:
        kappas = np.zeros((0, *X.shape))

    if dochecks:
        #test linalg solve...
        Xtest = L @ chi
        ytest = L @ gamma
        if len(kappas) > 0:
            Ktest = L @ kappas[0]
        print("linalg solve closes? (x3)")
        print(np.allclose(Xtest, X))
        print(np.allclose(ytest, y))
        if len(kappas) > 0:
            print(np.allclose(Ktest, Ks[0]))

    Nbeta = nom['gen'].shape[0]

    if usetorch:
        def thefunc(truth):
            return lsq_torch(truth, 
                            torch.Tensor(chi.astype(np.float32)).cuda(), 
                            torch.Tensor(kappas.astype(np.float32)).cuda(), 
                            torch.Tensor(gamma.astype(np.float32)).cuda(), 
                            Nbeta)
    else:
        @numba.jit(nopython=True)
        def thefunc(truth):
            return lsq_np(truth, chi, kappas, gamma, Nbeta)
        
        @numba.jit(nopython=True)
        def thehess(truth):
            return lsq_hess(truth, chi, kappas, gamma, Nbeta)

    initialguess, _, _, _ = scipy.linalg.lstsq(chi, gamma)
    print(initialguess.shape)

    if dochecks:
        print("norm of initialguess - truth",
              scipy.linalg.norm(initialguess - data['gen']))
    
    initialtheta = np.zeros(kappas.shape[0])
    initialguess = np.concatenate([initialguess, initialtheta])

    #initialguess += np.random.random(initialguess.shape) * 0.1

    if usetorch:
        initialguess = torch.Tensor(initialguess.astype(np.float32)).cuda()

    return thefunc, initialguess, thehess
