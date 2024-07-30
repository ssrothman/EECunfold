import pickle
import numpy as np

from get_matrices import get_matrices, dump_matrices, handle_updn
from syst_impact import syst_impact, get_K_scale
from forward import doforward

import json

with open("config.json") as f:
    config = json.load(f)

for systscan in config['systlist']:
    with open(config['systlist'][systscan]['source'], 'rb') as f:
        Hdict = pickle.load(f)

    matrices_nom = get_matrices(
        Hdict, 'nominal',
        rebinGen=True,
        smaller=False,
        doForward=True,
        nominalForForward=None,
        doKscale=True,
        destination='/data/submit/srothman/EECunfold/output/nominal'
    )

    Ks = []
    for syst in config['systlist'][systscan]['systs']:
        Ks.append(handle_updn(matrices_nom, syst))

    #FOR UNFOLDING
    shape_reco = matrices_nom['reco'].shape
    shape_gen = matrices_nom['gen'].shape

    Nreco = np.prod(shape_reco)
    Ngen = np.prod(shape_gen)

    reco = np.reshape(matrices_nom['reco'], (Nreco))
    covreco = np.reshape(matrices_nom['covreco'], (Nreco, Nreco))
    gen = np.reshape(matrices_nom['gen'], (Ngen))
    covgen = np.reshape(matrices_nom['covgen'], (Ngen, Ngen))
    transfer = np.reshape(matrices_nom['transfer'], (Nreco, Ngen))
    Kscale = get_K_scale(matrices_nom) 
    Kscale = np.reshape(Kscale, (Nreco))

    Ks = [np.reshape(K, (Nreco)) for K in Ks]
    Ks = np.stack(Ks, axis=-1)
    print("K shape", Ks.shape)
    print("Kscale shape", Kscale.shape)
    print("reco shape", reco.shape)
    print("covreco shape", covreco.shape)
    print("gen shape", gen.shape)
    print("covgen shape", covgen.shape)
    print("transfer shape", transfer.shape)

    import os
    os.makedirs('/data/submit/srothman/EECunfold/output/UNFOLD', exist_ok=True)
    np.save('/data/submit/srothman/EECunfold/output/UNFOLD/Kscale.npy', Kscale)
    np.save('/data/submit/srothman/EECunfold/output/UNFOLD/kappa.npy', Ks)
    np.save('/data/submit/srothman/EECunfold/output/UNFOLD/transfer.npy', transfer)
    np.save('/data/submit/srothman/EECunfold/output/UNFOLD/reco.npy', reco)
    np.save('/data/submit/srothman/EECunfold/output/UNFOLD/covreco.npy', covreco)
    np.save('/data/submit/srothman/EECunfold/output/UNFOLD/gen.npy', gen)
    np.save('/data/submit/srothman/EECunfold/output/UNFOLD/covgen.npy', covgen)
