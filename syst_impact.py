import numpy as np
from forward import doforward

def get_K_scale(matrices):
    reco = matrices['reco']
    sumDR = np.sum(reco, axis=-1, keepdims=True)
    sumDR[sumDR == 0] = 1
    sumDR, _ = np.broadcast_arrays(sumDR, reco)
    return sumDR

def syst_impact(matrices_nom, matrices_up, matrices_dn):
    reco_nom = matrices_nom['reco']
    covreco_nom = matrices_nom['covreco']   

    gen_nom = matrices_nom['gen']
    covgen_nom = matrices_nom['covgen']

    transfer_up = matrices_up['transfer']
    transfer_dn = matrices_dn['transfer']

    forward_up, covforward_up = doforward(gen_nom, 
                                        covgen_nom,
                                        transfer_up)
    forward_dn, covforward_dn = doforward(gen_nom,
                                        covgen_nom,
                                        transfer_dn)

    K = (forward_up - forward_dn) / 2
    dK = np.abs(covforward_up + covforward_dn) / 2

    denom = np.sum(reco_nom, axis=-1).copy() #sum over dR bins
    denom[denom == 0] = 1
    K = K / denom[:,:,:,None]
    dK = dK / (denom[:,:,:,None] * denom[None,None,None,None,:,:,:,None])

    return K, dK
