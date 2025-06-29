import fasteigenpy as eigen
import numpy as np
import hist

def err(H, clamp=True):
    nominal = H[{'bootstrap' : 0}]
    err2 = np.zeros_like(nominal.values(flow=True))
    for i in range(1, H.axes['bootstrap'].size):
        err2 += np.square((H[{'bootstrap' : i}] - nominal).values(flow=True))
    
    result = (np.sqrt(err2 / (H.axes['bootstrap'].size - 1))).ravel()
    if clamp:
        result[result==0] = 1

    return result

def maybe_normalized(H, norm=True):
    values = H.values(flow=True).ravel()
    N = H.axes['bootstrap'].size 
    values = values.reshape((N, -1))
    nomsum = values[0].sum(axis=0, keepdims=True)
    if norm:
        values = nomsum * values / np.sum(values, axis=1, keepdims=True)

    return values

def dymat(H, norm=True):
    values = maybe_normalized(H, norm=norm)

    nominal = values[0][None,:]
    boots = values[1:]

    Nboot = boots.shape[0]

    return boots - nominal

def cov(H, clamp=True):
    DY = dymat(H, norm=False)
    Nboot = DY.shape[0]

    result = DY.T @ DY / Nboot

    return result

def invcov(H, clamp=True):
    covmat = cov(H, clamp=clamp)

    decomp = eigen.CompleteOrthogonalDecomposition(covmat)
    invcov = decomp.solve(np.eye(covmat.shape[0]))

    return invcov

def test_cov_boots(DY, stepsize=100, show=True):
    accumulator = np.zeros((DY.shape[1], DY.shape[1]), dtype=DY.dtype)

    sums = []
    nboots = []
    from tqdm import tqdm
    for i in tqdm(range(0, DY.shape[0]//stepsize)):
        Nboot = stepsize * (i + 1)
        Nboot_prev = stepsize * i

        nextDY = DY[Nboot_prev:Nboot, :]
        accumulator += nextDY.T @ nextDY

        sums += [np.trace(accumulator/ Nboot)]
        nboots += [Nboot]

    import matplotlib.pyplot as plt
    plt.errorbar(nboots, sums, fmt='o')
    plt.ylabel("Covariance matrix trace")
    plt.xlabel("Number of bootstrap samples")
    if show:
        plt.show()
