import numpy as np
from bootstrapcov import bootstrapcov
from reshapetransfer import reshapetransfer

def handle_updn(matrices_nom, syst, destination_base):
    matrices_up = handle_syst(matrices_nom, syst, 'Up',
                              destination_base)
    matrices_dn = handle_syst(matrices_nom, syst, 'Down',
                              destination_base)

    K, dK = syst_impact(matrices_nom, matrices_up, matrices_dn)

    impactpath = os.path.joins(destination_base, syst + '_impact')
    np.save(impactpath + '.npy', K)
    np.save(impactpath + '_unc.npy', dK)

    return K

def handle_syst(matrices_nom, syst, updn, destination_base):
    matrices = get_matrices(
        muonSyst, syst + updn,
        rebinGen=True, 
        smaller=False,
        doForward=True,
        nominalForForward=matrices_nom,
        doKscale=True,
        destination=os.path.join(destination_base, syst + updn)
    )

    return matrices

def get_matrices(Hdict, syst, rebinGen=True, smaller=False,
                 doForward = True,
                 nominalForForward = None,
                 doKscale = True,
                 destination = None):

    reco = Hdict[syst]['recopure']
    reco_nom, covreco_nom = bootstrapcov(reco)

    gen = Hdict[syst]['genpure']
    if rebinGen:
        ptatzero = gen[:,:,:,:,:,(0,)]
        others_1 = gen[:,:,:,:,:,1::2]
        others_2 = gen[:,:,:,:,:,2::2]
        otherssum = others_1 + others_2
        gen = np.concatenate((ptatzero, otherssum), axis=-1)

    gen_nom, covgen_nom = bootstrapcov(gen)

    transfer = Hdict[syst]['transfer']
    transfer = reshapetransfer(transfer)
    if rebinGen:
        transfer0 = transfer[:,:,:,:,:,:,:,(0,)]
        transfer1 = transfer[:,:,:,:,:,:,:,1::2]
        transfer2 = transfer[:,:,:,:,:,:,:,2::2]
        transfersum = transfer1 + transfer2
        transfer = np.concatenate((transfer0, transfersum), axis=-1)

    if smaller:
        gen_nom = gen_nom[:, :, slice(4,None), :]
        covgen_nom = covgen_nom[:, :, slice(4,None), :, 
                                :, :, slice(4,None), :]
        reco_nom = reco_nom[:, :, slice(4,None), :]
        covreco_nom = covreco_nom[:, :, slice(4,None), :, 
                                  :, :, slice(4,None), :]
        transfer = transfer[:, :, slice(4,None), :, 
                            :, :, slice(4,None), :]

    shape_reco = reco_nom.shape
    shape_gen = gen_nom.shape

    Nreco = np.prod(shape_reco)
    Ngen = np.prod(shape_gen)

    reco_nom = np.reshape(reco_nom, (Nreco))
    covreco_nom = np.reshape(covreco_nom, (Nreco, Nreco))
    gen_nom = np.reshape(gen_nom, (Ngen))
    covgen_nom = np.reshape(covgen_nom, (Ngen, Ngen))
    transfer = np.reshape(transfer, (Nreco, Ngen))

    denom = gen_nom[None,:].copy()
    denom[denom == 0] = 1
    transfer /= denom

    reco_nom = reco_nom.reshape(shape_reco)
    covreco_nom = covreco_nom.reshape((*shape_reco, *shape_reco))
    gen_nom = gen_nom.reshape(shape_gen)
    covgen_nom = covgen_nom.reshape((*shape_gen, *shape_gen))
    transfer = transfer.reshape((*shape_reco, *shape_gen))

    result = {
        'reco' : reco_nom, 
        'covreco' : covreco_nom, 
        'gen' : gen_nom, 
        'covgen' : covgen_nom, 
        'transfer' : transfer
    }

    if doForward:
        if nominalForForward is None:
            genForForward = gen_nom
            covgenForForward = covgen_nom
        else:
            genForForward = nominalForForward['gen']
            covgenForForward = nominalForForward['covgen']

        forward_nom, covforward_nom = doforward(genForForward,
                                                covgenForForward,
                                                transfer)
        result['forward'] = forward_nom
        result['covforward'] = covforward_nom

    if doKscale:
        result['Kscale'] = get_K_scale(result)

    if destination is not None:
        dump_matrices(destination, result)

    return result

def dump_matrices(folder, matrices):
    import os
    import os.path

    os.makedirs(folder, exist_ok=True)

    for key in matrices:
        np.save(os.path.join(folder,key), matrices[key])
