import numpy as np

def bootstrapcov(x):
    nominal = x[0]
    toys = x[1:]

    diffs = toys - nominal

    Ntoys = diffs.shape[0]
    cov = np.einsum('iabcd,iefgh->abcdefgh', diffs, diffs, optimize=True)
    cov /= Ntoys

    return nominal, cov

def reshapetransfer(x):
    #transfer shape is (order, pt, pt, dR, dR)

    Norder = x.shape[0]
    diagorder = np.eye(Norder)

    transfer = np.einsum('opqtw,ou->powqut', x, diagorder,
                         optimize=True)
    
    #print("AFTER RESHAPE, TRANSFER SHAPE IS")
    #print(transfer.shape)
    return transfer

def rebingen(gen, covgen, transfer):
    gen0 = gen[:,:,(0,)]
    gen1 = gen[:,:,1::2]
    gen2 = gen[:,:,2::2]
    gensum = gen1 + gen2
    gen = np.concatenate((gen0, gensum), axis=-1)

    covgen0 = covgen[:,:,0,:,:,0]
    covgen1 = covgen[:,:,1::2,:,:,1::2]
    covgen2 = covgen[:,:,2::2,:,:,2::2]
    covgensum = covgen1 + covgen2
    new_covshape = list(covgen.shape)
    new_covshape[2] = covgensum.shape[2] + 1
    new_covshape[5] = covgensum.shape[5] + 1
    covgen = np.zeros(new_covshape)
    covgen[:,:,0,:,:,0] = covgen0
    covgen[:,:,1:,:,:,1:] = covgensum

    transfer0 = transfer[:,:,:,:,:,(0,)]
    transfer1 = transfer[:,:,:,:,:,1::2]
    transfer2 = transfer[:,:,:,:,:,2::2]
    transfersum = transfer1 + transfer2
    transfer = np.concatenate((transfer0, transfersum), axis=-1)
    
    #print("AFTER REBIN GEN, TRANSFER SHAPE IS")
    #print(transfer.shape)

    return gen, covgen, transfer

def rebinreco(reco, covreco, transfer):
    reco0 = reco[:,:,(0,)]
    reco1 = reco[:,:,1::2]
    reco2 = reco[:,:,2::2]
    recosum = reco1 + reco2
    reco = np.concatenate((reco0, recosum), axis=-1)

    covreco0 = covreco[:,:,0,:,:,0]
    covreco1 = covreco[:,:,1::2,:,:,1::2]
    covreco2 = covreco[:,:,2::2,:,:,2::2]
    covrecosum = covreco1 + covreco2
    new_covshape = list(covreco.shape)
    new_covshape[2] = covrecosum.shape[2] + 1
    new_covshape[5] = covrecosum.shape[5] + 1
    covreco = np.zeros(new_covshape)
    covreco[:,:,0,:,:,0] = covreco0
    covreco[:,:,1:,:,:,1:] = covrecosum

    transfer0 = transfer[:,:,(0,),:,:,:]
    transfer1 = transfer[:,:,1::2,:,:,:]
    transfer2 = transfer[:,:,2::2,:,:,:]
    transfersum = transfer1 + transfer2
    transfer = np.concatenate((transfer0, transfersum), axis=2)

    #print("AFTER REBIN RECO, TRANSFER SHAPE IS")
    #print(transfer.shape)

    return reco, covreco, transfer

def get_arrays(hists, syst, statsplit, 
               checkclosure=True,
               restrictorder=None,
               restrictpt=None,
               restrictbtag=None,):
    data = hists[syst]

    transfer = data['transfer']
    recopure = data['recopure']
    genpure = data['genpure']

    if statsplit is None:
        transfer = np.sum(transfer, axis=1)
        recopure = np.sum(recopure, axis=0)
        genpure = np.sum(genpure, axis=0)
    else:
        transfer = transfer[:,statsplit]
        recopure = recopure[statsplit]
        genpure = genpure[statsplit]

    if 'covreco' in data:
        covreco = data['covreco'][0]
        covgen = data['covgen'][0]
        gen = genpure[0]
        reco = recopure[0]
    else:
        print("DOING BOOTSTRAP COV")
        reco, covreco = bootstrapcov(recopure)
        gen, covgen = bootstrapcov(genpure)
   
    gen = np.sum(gen, axis=0)
    reco = np.sum(reco, axis=0)
    covgen = np.sum(covgen, axis=(0,4))
    covreco = np.sum(covreco, axis=(0,4))

    #reshape transfer
    transfer = reshapetransfer(transfer)

    #rebin gen, covgen, transfer
    gen, covgen, transfer = rebingen(gen, covgen, transfer)
    #reco, covreco, transfer = rebinreco(reco, covreco, transfer)

    if restrictorder is None:
        restrictorder = tuple(range(gen.shape[1]))
    elif type(restrictorder) is int:
        restrictorder = (restrictorder,)
    elif type(restrictorder) is not str:
        restrictorder = tuple(restrictorder)

    if restrictpt is None:
        restrictpt = tuple(range(gen.shape[0]))
    elif type(restrictpt) is int:
        restrictpt = (restrictpt,)
    elif type(restrictpt) is not str:
        restrictpt = tuple(restrictpt)

    #if restrictbtag is None:
    #    restrictbtag = tuple(range(gen.shape[0]))
    #elif type(restrictbtag) is int:
    #    restrictbtag = (restrictbtag,)
    #elif type(restrictbtag) is not str:
    #    restrictbtag = tuple(restrictbtag)

    if restrictorder == 'sum':
        gen = np.sum(gen, axis=1)
        covgen = np.sum(covgen, axis=(1,4))
        transfer = np.sum(transfer, axis=(1,4))
        reco = np.sum(reco, axis=1)
        covreco = np.sum(covreco, axis=(1,4))
    else:
        gen = gen[:,restrictorder,:]
        covgen = covgen[:,restrictorder,:,:,:,:]
        covgen = covgen[:,:,:,:,restrictorder,:]
        transfer = transfer[:,restrictorder,:,:,:,:]
        transfer = transfer[:,:,:,:,restrictorder,:]
        reco = reco[:,restrictorder,:]
        covreco = covreco[:,restrictorder,:,:,:,:]
        covreco = covreco[:,:,:,:,restrictorder,:]

    #print("TRANSFER SHAPE IS")
    #print(transfer.shape)

    if restrictpt == 'sum':
        gen = np.sum(gen, axis=0)
        covgen = np.sum(covgen, axis=(0,3))
        transfer = np.sum(transfer, axis=(0,3))
        reco = np.sum(reco, axis=0)
        covreco = np.sum(covreco, axis=(0,3))
    else:
        gen = gen[restrictpt,:,:]
        covgen = covgen[restrictpt,:,:,:,:,:]
        covgen = covgen[:,:,:,restrictpt,:,:]
        transfer = transfer[restrictpt,:,:,:,:,:]
        transfer = transfer[:,:,:,restrictpt,:,:]
        reco = reco[restrictpt,:,:]
        covreco = covreco[restrictpt,:,:,:,:,:]
        covreco = covreco[:,:,:,restrictpt,:,:]

    #if restrictbtag == 'sum':
    #    gen = np.sum(gen, axis=0)
    #    covgen = np.sum(covgen, axis=(0,4))
    #    transfer = np.sum(transfer, axis=(0,4))
    #    reco = np.sum(reco, axis=0)
    #    covreco = np.sum(covreco, axis=(0,4))
    #else:
    #    gen = gen[restrictbtag,:,:,:]
    #    covgen = covgen[restrictbtag,:,:,:,:,:,:,:]
    #    covgen = covgen[:,:,:,:,restrictbtag,:,:,:]
    #    transfer = transfer[restrictbtag,:,:,:,:,:,:,:]
    #    transfer = transfer[:,:,:,:,restrictbtag,:,:,:]
    #    reco = reco[restrictbtag,:,:,:]
    #    covreco = covreco[restrictbtag,:,:,:,:,:,:,:]
    #    covreco = covreco[:,:,:,:,restrictbtag,:,:,:]
    
    print("in the end the shapes are")
    print("gen", gen.shape)
    print("covgen", covgen.shape)
    print("transfer", transfer.shape)
    print("reco", reco.shape)
    print("covreco", covreco.shape)

    genshape = gen.shape
    recoshape = reco.shape

    Ngen = np.prod(genshape)
    Nreco = np.prod(recoshape)

    reco = reco.reshape(Nreco)
    gen = gen.reshape(Ngen)
    transfer = transfer.reshape(Nreco, Ngen)
    covreco = covreco.reshape(Nreco, Nreco)
    covgen = covgen.reshape(Ngen, Ngen)

    if checkclosure:
        sumtransfer = np.sum(transfer, axis=1)
        print("transfer (sum) closes?", np.allclose(sumtransfer, reco))

    denom = gen[None,:].copy()
    denom[denom == 0] = 1
    transfer /= denom

    if checkclosure:
        forward = transfer @ gen
        print("transfer (matmul) closes?", np.allclose(forward, reco))

    return {
        'reco': reco,
        'gen': gen,
        'transfer': transfer,
        'covreco': covreco,
        'covgen': covgen,
        'recoshape': recoshape,
        'genshape': genshape
    }
