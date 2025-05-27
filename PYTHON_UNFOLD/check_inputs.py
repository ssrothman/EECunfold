import numpy as np
import hist 

def check_valid(Htransfer, Hreco, HunmatchedReco, HuntransferedReco, 
                Hgen, HunmatchedGen, HuntransferedGen):

    transfervals = Htransfer.values(flow=True)
    recovals = Hreco - HunmatchedReco - HuntransferedReco
    genvals = Hgen - HunmatchedGen - HuntransferedGen

    recovals = recovals[{'bootstrap' : 0}].values(flow=True)
    genvals = genvals[{'bootstrap' : 0}].values(flow=True)

    print("transfervals.sum() == receovals?", 
          np.allclose(transfervals.sum(axis=(4,5,6,7)), recovals))

    T = transfervals / genvals[None,None,None,None,:,:,:,:]

    fwd = np.einsum('abcdefgh,efgh->abcd',T, genvals)
    print("fwd == recovals?",
          np.allclose(fwd, recovals))
