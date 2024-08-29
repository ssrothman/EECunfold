import pickle
import matplotlib.pyplot as plt

with open("test6/hists_file0to3018_tight_manualcov_noSyst.pkl", 'rb') as f:
    hists = pickle.load(f)

print(hists.keys())

from preprocess import get_arrays

restrictorder = None
restrictpt = 'sum' #tuple(range(10))
restrictbtag = None

nom = get_arrays(hists, 'nominal', None,
                 restrictorder=restrictorder,
                 restrictpt=restrictpt,
                 restrictbtag=restrictbtag)

#iso_up = get_arrays(hists, 'wt_isosfUp', None,
#                 restrictorder=restrictorder,
#                    restrictpt=restrictpt,
#                    restrictbtag=restrictbtag)
#iso_down = get_arrays(hists, 'wt_isosfDown', None,
#                 restrictorder=restrictorder,
#                      restrictpt=restrictpt,
#                      restrictbtag=restrictbtag)
#
#id_up = get_arrays(hists, 'wt_idsfUp', None,
#                 restrictorder=restrictorder,
#                   restrictpt=restrictpt,
#                   restrictbtag=restrictbtag)
#id_down = get_arrays(hists, 'wt_idsfDown', None,
#                 restrictorder=restrictorder,
#                     restrictpt=restrictpt,
#                     restrictbtag=restrictbtag)

from likelihood import build_likelihood

ELL, p0, HESS = build_likelihood(nom, 
                                 [],
                                 #[(iso_up, iso_down)], 
                                  #(id_up, id_down)], 
                                 nom,
                                 usetorch=False)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
#sum1 = np.sum(nom['transfer'], axis=1)
#sum0 = np.sum(nom['transfer'], axis=0)
#diag = np.diag(nom['transfer'])
#plt.plot(diag/sum0)
#plt.plot(diag/sum1)
#plt.show()

from manual_newton import manual_newton

H0 = HESS(p0)
print("Condition number of H0", np.linalg.cond(H0))
print("Largers eigenvalue of H0", np.max(np.linalg.eigvals(H0)))
print("Smallest eigenvalue of H0", np.min(np.linalg.eigvals(H0)))

fval, ans = manual_newton(ELL, p0, HESS)
print(ans)
#``
#``
#``plt.pcolormesh(H0, norm=LogNorm())
#``plt.colorbar()
#``plt.show()
#``#H0 = HESS(p0)
#``#print("Condition number of H0", np.linalg.cond(H0))
#``#from torchimize.functions.single.gda_fun_single import gradient_descent
#``#print("running gradient descent")
#``#popt = gradient_descent(p0, ELL)
#``#print("DONE")
#``#popt = popt.detach().numpy()
#``#print(scipy.linalg.norm(popt - p0.detach().numpy()))
#``
#``import scipy.optimize
#``print("running minimization with scipy")
#``res = scipy.optimize.minimize(ELL, p0,
#``                             jac=True,
#``                             options = {
#``                                 'disp' : True
#``                                 },
#``                             hess=HESS, 
#``                             method="Newton-CG")
#``print("DONE")
#``
#``import scipy.linalg
#``print(res)
#``print("result distance", scipy.linalg.norm(res.x - p0))
#``
#``
#``print('predicted nuisances:\n', res.x[-2:])
#``hess = HESS(res.x)
#``cov = scipy.linalg.pinv(hess)
#``print('predicted cov\n', cov[-2:,-2:])
