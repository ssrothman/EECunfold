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

    K = np.zeros_like(chi)
    for i in range(len(theta)):
        K += theta[i] * kappas[i]
    #K = np.einsum('i,ijk->jk', theta, kappas)
    
    T = K + chi

    residuals = T @ beta - gamma

    loss = 0.5 * np.sum(np.square(residuals))
    loss += 0.5 * np.sum(np.square(theta))

    grad = np.zeros_like(truth)
    grad[:Nbeta] = T.T @ residuals
    for i in range(len(theta)):
        grad[Nbeta+i] = theta[i] + (T @ beta - gamma).T @ (kappas[i] @ beta)

    #print("any nan?")
    #print("\ttruth", np.any(np.isnan(truth)))
    #print("\tchi", np.any(np.isnan(chi)))
    #print("\tkappas", np.any(np.isnan(kappas)))
    #print("\tgamma", np.any(np.isnan(gamma)))
    #print("\tNbeta", np.any(np.isnan(Nbeta)))
    #print("\tK", np.any(np.isnan(K)))
    #print("\tT", np.any(np.isnan(T)))
    #print("\tresiduals", np.any(np.isnan(residuals)))
    #print("\tloss", np.any(np.isnan(loss)))

    return (loss, grad)

@numba.jit(nopython=True)
def lsq_hess(truth, chi, kappas, gamma, Nbeta):
    beta = truth[:Nbeta]
    theta = truth[Nbeta:]

    '''
    grad_beta = (chi + theta_i kappa_i).T @ [(chi + theta_i kappa_i) @ beta - gamma]
    grad_theta_i = theta_i + [(chi + theta_i kappa_i) @ beta - gamma].T @ (kappa_i @ beta)

    '''

    hess = np.zeros((len(truth), len(truth)))

    hess[:Nbeta, :Nbeta] = chi.T @ chi
    hess[Nbeta:, Nbeta:] = np.eye(len(theta))
    for i in range(len(theta)):
        for j in range(len(theta)):
            hess[Nbeta+i, Nbeta+j] = (kappas[i] @ beta).T @ (kappas[j] @ beta)

    K = np.zeros_like(chi)
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
                     EPS=1.0e+2,
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

    import eigenpy as eigen
    llt = eigen.LLT(C)
    if (llt.info() != eigen.eigenpy_pywrap.ComputationInfo.Success):
        print("WARNING: Cholesky decomposition failed")
        print("INFO: ", llt.info())
        raise ValueError("Cholesky decomposition failed with info {}".format(llt.info()))

    L = llt.matrixL()

    orthoL = eigen.CompleteOrthogonalDecomposition(L)
    if (orthoL.info() != eigen.eigenpy_pywrap.ComputationInfo.Success):
        print("WARNING: Orthogonal decomposition failed")
        print("INFO: ", orthoL.info())
        raise ValueError("Orthogonal decomposition failed with info {}".format(orthoL.info()))

    gamma = orthoL.solve(y)
    chi = orthoL.solve(X)
    

    #L = scipy.linalg.cholesky(C, lower=True)

    #gamma = scipy.linalg.solve(L, y)
    #chi = scipy.linalg.solve(L, X)
    
    #print(np.allclose(L, L1))
    #print(np.allclose(gamma, gamma1))
    #print(np.allclose(chi, chi1))

    if len(Ks) > 0:
        kappas = []
        for K in Ks:
            kappa = orthoL.solve(K)
            kappas.append(kappa[None,:,:])
        kappas = np.concatenate(kappas, axis=0)
    else:
        kappas = np.zeros((0, *X.shape))

    if dochecks:
        #test linalg solve...
        Xtest = L @ chi
        ytest = L @ gamma
        print("inversion closes? (x3)")
        print(np.allclose(Xtest, X))
        print(np.allclose(ytest, y))
        if len(kappas) > 0:
            Ktest = L @ kappas[0]
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
        #@numba.jit(nopython=True)
        def thefunc(truth):
            return lsq_np(truth, chi, kappas, gamma, Nbeta)
        
        @numba.jit(nopython=True)
        def thehess(truth):
            return lsq_hess(truth, chi, kappas, gamma, Nbeta)

    
    orthoChi = eigen.CompleteOrthogonalDecomposition(chi)
    print(orthoChi.info())
    initialguess = orthoChi.solve(gamma)

    #initialguess, _, _, _ = scipy.linalg.lstsq(chi, gamma)
    print(initialguess.shape)

    if dochecks:
        print("norm of initialguess - truth",
              scipy.linalg.norm(initialguess - data['gen']))
    
    initialtheta = np.zeros(kappas.shape[0])
    initialguess = np.concatenate([initialguess, initialtheta])

    #initialguess += np.random.random(initialguess.shape) * 0.1

    print("C")
    print("\tmax",np.max(C))
    print("\tmin",np.min(C))
    print("chi")
    print("\tmax",np.max(chi))
    print("\tmin",np.min(chi))
    print("gamma")
    print("\tmax",np.max(gamma))
    print("\tmin",np.min(gamma))
    print("intialguess")
    print("\tmax",np.max(initialguess))
    print("\tmin",np.min(initialguess))

    if usetorch:
        initialguess = torch.Tensor(initialguess.astype(np.float32)).cuda()

    return thefunc, initialguess, thehess
