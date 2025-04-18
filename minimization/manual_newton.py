import numpy as np
import eigenpy as eigen

def manual_newton(thefunc, initialguess, thehess):
    initialguess = np.zeros_like(initialguess)
    fval, grad = thefunc(initialguess)
    hess = thehess(initialguess)

    print("initial:", fval)

    for i in range(10):
        #compute condition number of hessian
        norm_hess = np.linalg.norm(hess, ord=2)
        
        codHess = eigen.CompleteOrthogonalDecomposition(hess)
        pinv = codHess.pseudoInverse()
        norm_pinv = np.linalg.norm(pinv, ord=2)
        print("Condition number of hessian", norm_hess * norm_pinv)

        step = codHess.solve(-grad)
        print(codHess.info())
        initialguess += step
        fval, grad = thefunc(initialguess)
        hess = thehess(initialguess)
        print("step", i, ":", fval)

    return fval, initialguess

