import numpy as np
import eigenpy as eigen
import scipy

def marginalize(x, invhess, slice_start, slice_end):
    '''
    It's trivial to marginalize a multivariate Gaussian
    Just slice out the dimensions you want to marginalize over

    NB "marginalize" = "profile" = "let float"
    '''
    before_dim0 = invhess[:slice_start, :]
    after_dim0 = invhess[slice_end:, :]
    tmp0 = np.concatenate((before_dim0, after_dim0), axis=0)

    before_dim1 = tmp0[:, :slice_start]
    after_dim1 = tmp0[:, slice_end:]

    newhess = np.concatenate((before_dim1, after_dim1), axis=1)

    newx = np.concatenate((x[:slice_start], x[slice_end:]), axis=0)

    return newx, newhess


def condition(x, invhess, slice_start, slice_end, values):
    '''
    This requires a bit more math. 
    I'm copying the math from https://www.wikiwand.com/en/articles/Multivariate_normal_distribution#Marginal_distributions

    NB "condition" means "set to a given value".
    If we conditionally set a nuisance to zero that's 
    equivalent to if we never had it, up to the gaussian assumption
    '''
    xkeep = np.concatenate((x[:slice_start], x[slice_end:]), axis=0)
    xkill = x[slice_start:slice_end]

    Hbefore_dim0 = invhess[:slice_start, :]
    Hafter_dim0 = invhess[slice_end:, :]
    Htmp0 = np.concatenate((Hbefore_dim0, Hafter_dim0), axis=0)

    Hbefore_dim1 = Htmp0[:, :slice_start]
    Hafter_dim1 = Htmp0[:, slice_end:]

    H11 = np.concatenate((Hbefore_dim1, Hafter_dim1), axis=1)

    H12 = Htmp0[:, slice_start:slice_end]
    H22 = invhess[slice_start:slice_end, slice_start:slice_end]

    print("H11", H11.shape)
    print("H12", H12.shape)
    print("H22", H22.shape)

    codH22 = eigen.CompleteOrthogonalDecomposition(H22)

    newx = xkeep + H12 @ codH22.solve(values - xkill)

    newhess = H11 - H12 @ codH22.solve(H12.T)
    return newx, newhess
