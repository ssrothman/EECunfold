import numpy as np

def doforward(gen, covgen, transfer):
    shape_gen = gen.shape
    shape_reco = transfer.shape[:len(shape_gen)]

    Ngen = np.prod(shape_gen)

    gen = np.reshape(gen, (Ngen))
    covgen = np.reshape(covgen, (Ngen, Ngen))
    transfer = np.reshape(transfer, (-1, Ngen))

    forward = transfer @ gen
    covforward = transfer @ covgen @ transfer.T

    forward = forward.reshape(shape_reco)
    covforward = covforward.reshape((*shape_reco, *shape_reco))

    return forward, covforward

