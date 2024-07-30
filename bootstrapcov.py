import numpy as np

def bootstrapcov(x, statsplit=None, returndiffs=False):
    '''
    shape of x is (statsplit, bootstrap, btag, pt, order, dR)
    '''

    if statsplit is None:
        x = np.sum(x, axis=0)
    else:
        x = x[statsplit]

    #now shape of x is (bootstrap, btag, pt, order, dR)

    nominal = x[0]
    bootstraps = x[1:]

    Nb = bootstraps.shape[0]

    diffs = bootstraps - nominal
    if returndiffs:
        return diffs
    cov = (1/Nb) * np.einsum('iabcd,iefgh->abcdefgh', diffs, diffs,
                             optimize=True)

    return nominal, cov
