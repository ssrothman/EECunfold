import numpy as np

def reshapetransfer(x):
    #transfer shape is (order, btag, pt, btag, pt, dR, dR)

    Norder = x.shape[0]
    diagorder = np.eye(Norder)

    transfer = np.einsum('abcjkld,ai->bcadjkil', x, diagorder, 
                         optimize=True)
    
    return transfer
