import numpy as np
import hist

def unc(H):
    nominal = H[{'bootstrap' : 0}]
    err2 = np.zeros_like(nominal.values(flow=True))
    for i in range(1, H.axes['bootstrap'].size):
        err2 += np.square((H[{'bootstrap' : i}] - nominal).values(flow=True))
    
    return np.sqrt(err2 / (H.axes['bootstrap'].size - 1))

