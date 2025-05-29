import numpy as np
import hist

def unc(H):
    nominal = H[{'bootstrap' : 0}]
    err2 = np.zeros_like(nominal.values(flow=True))
    for i in range(1, H.axes['bootstrap'].size):
        err2 += np.square((H[{'bootstrap' : i}] - nominal).values(flow=True))
    
    return (np.sqrt(err2 / (H.axes['bootstrap'].size - 1))).ravel()

def cov(H, clampboot = -1):
    nominal = H[{'bootstrap' : 0}].values(flow=True)

    shape = nominal.shape
    Nval = np.prod(shape)

    nominal = nominal.ravel()

    cov = np.zeros((Nval, Nval))
    #sumdiff = np.zeros((Nval,))
    #sumdiff2 = np.zeros((Nval,))

    if clampboot > 0:
        looplen = np.min((H.axes['bootstrap'].size, clampboot))
    else:
        looplen = H.axes['bootstrap'].size

    for i in range(1, looplen):
        var = H[{'bootstrap' : i}].values(flow=True).ravel()
        diff = var - nominal
        cov += np.outer(diff, diff)
        #sumdiff += diff
        #sumdiff2 += np.square(diff)
    
    Nboot = looplen-1

    return cov/Nboot
