import numpy as np
import os
import os.path
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('reco_from', type=str)
parser.add_argument('nominal_from', type=str)
parser.add_argument('destination', type=str)

parser.add_argument('--basepath', type=str, required=False,
                    default='/data/submit/srothman/EECunfold/output')
parser.add_argument('--systs', type=str, nargs='+', required=False,
                    default=['wt_idsf', 'wt_triggersf',
                             'wt_prefire', 'wt_isosf'])


args = parser.parse_args()

basepath = args.basepath
reco_from = args.reco_from
nominal_from = args.nominal_from
destination = args.destination
systs = args.systs

y = np.load(os.path.join(basepath, reco_from, 'reco.npy'))
Vy = np.load(os.path.join(basepath, reco_from, 'covreco.npy'))
Kscale = np.load(os.path.join(basepath, reco_from, 'Kscale.npy'))

beta = np.load(os.path.join(basepath, nominal_from, 'gen.npy'))
Vbeta = np.load(os.path.join(basepath, nominal_from, 'covgen.npy'))
X = np.load(os.path.join(basepath, nominal_from, 'transfer.npy'))

Ny = np.prod(y.shape)
Nbeta = np.prod(beta.shape)
Ntheta = len(systs)

y = y.reshape((Ny,))
Vy = Vy.reshape((Ny, Ny))
Kscale = Kscale.reshape((Ny,))

beta = beta.reshape((Nbeta,))
Vbeta = Vbeta.reshape((Nbeta, Nbeta))
X = X.reshape((Ny, Nbeta))

syst_impacts = []
for syst in systs:
    impact = np.load(os.path.join(basepath, syst + '_impact.npy'))
    syst_impacts.append(impact.reshape((Ny,)))

kappa = np.stack(syst_impacts, axis=1)
K = kappa * Kscale[:,None]

print("Ny", Ny)
print("Ntheta", Ntheta)
print("Nbeta", Nbeta)

print("y.shape", y.shape)
print("Vy.shape", Vy.shape)
print("beta.shape", beta.shape)
print("Vbeta.shape", Vbeta.shape)
print("X.shape", X.shape)
print("K.shape", K.shape)

#test
forward = X @ beta
print("forward.shape", forward.shape)
#print(forward.sum())
print("y.shape", y.shape)
#print(y.sum())
print("forward closes?", np.allclose(forward, y))
print()

A = np.zeros((Ny+Ntheta, Nbeta+Ntheta))
A[:Ny, :Nbeta] = X
A[:Ny, Nbeta:] = K
A[Ny:, Nbeta:] = np.eye(Ntheta)

C = np.zeros((Ny+Ntheta, Ny+Ntheta))
C[:Ny, :Ny] = Vy
C[Ny:, Ny:] = np.eye(Ntheta)

g = np.zeros(Nbeta+Ntheta)
g[:Nbeta] = beta

r = np.zeros(Ny+Ntheta)
r[:Ny] = y

#test again
forward_again = A @ g;
print("forward_again.shape", forward_again.shape)
#print(forward_again.sum())
print("r.shape", r.shape)
#print(r.sum())
print("forward again closes?", np.allclose(forward_again, r))

#likelihood is (1/2) * (A*g - r)^T * C^-1 * (A*g - r)

print("SETUP INPUTS")
print()
print()
print("A.shape", A.shape)
print("C.shape", C.shape)
print("g.shape", g.shape)
print("r.shape", r.shape)
print()
os.makedirs(destination, exist_ok=True)
np.save(os.path.join(destination, "A.npy"), A)
np.save(os.path.join(destination, "C.npy"), C)
np.save(os.path.join(destination, "g.npy"), g)
np.save(os.path.join(destination, "r.npy"), r)
print("SAVED INPUTS")
