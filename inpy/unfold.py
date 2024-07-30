import pickle
import numpy as np
import numpy.linalg as la
import scipy

with open("/data/submit/srothman/EEC/Jul14_2024/DYJetsToLL/EECproj/hists_file0to2052_tight_manualcov.pkl", 'rb') as f:
    x = pickle.load(f)['nominal']

X = x['transfer']
y = x['recopure']
beta = x['genpure']
Vy = x['covreco']

print("INPUT")
#shape (order, btag_reco, pt_reco, btag_gen, pt_gen, dR_gen, dR_reco)
print(X.shape)
#shape(statindex, toyindex, btag, pt, order, dR)
print(y.shape)
#shape(statindex, toyindex, btag, pt, order, dR)
print(beta.shape)
#shape(statindex, btag1, pt1, order1, dR1, btag2, pt2, order2, dR2)
print(Vy.shape)
print()

#STEP 1: get everything into the right shape
print("RESHAPED")
diag_order = np.eye(X.shape[0])
X = np.einsum('abcjkld,ai->bcadjkil', X, diag_order, optimize=True)
y = y[0,0]
beta = beta[0,0]
Vy = Vy[0]

X=np.sum(X, axis=(1,5))
y = np.sum(y, axis=(1))
beta = np.sum(beta, axis=(1))
Vy = np.sum(Vy, axis=(1, 5))

y = y.ravel()
beta = beta.ravel()
Ny = y.shape[0]
Nbeta = beta.shape[0]
X = np.reshape(X,(Ny, Nbeta))
Vy = np.reshape(Vy,(Ny, Ny))

print(X.shape)
print(y.shape)
print(beta.shape)
print(Vy.shape)
print()

#STEP 2: check that I've got gen/reco the right way around
print("CHECK")
print("sum y", np.sum(y))
print("sum beta", np.sum(beta))
print("sum X", np.sum(X))
sumX_2 = np.sum(X, axis=(1))
print("sumX close to y?", np.allclose(sumX_2, y))
print()

#STEP 3: divide transfer by gen
print("DIVIDE BY GEN")
denom = beta[None,:]
denom[denom == 0] = 1

X = X / denom
print()

#STEP 4: forward
print("FORWARD")
forward = X @ beta
covforward = X @ Vy @ X.T
print("forward closes?", np.allclose(forward, y))
print()

#STEP 5: unfold
print("UNFOLD")
import eigenpy as eigen
LLT = eigen.LLT(Vy)
L = LLT.matrixL()
print("got L")

codL = eigen.CompleteOrthogonalDecomposition(L)
gamma = codL.solve(y)
chi = codL.solve(X)
print("got gamma, chi")

codChi = eigen.CompleteOrthogonalDecomposition(chi)
unfolded = codChi.solve(gamma)
print("got unfolded")

chiTchi = chi.T @ chi
codChiTchi = eigen.CompleteOrthogonalDecomposition(chiTchi)
covunfolded = codChiTchi.pseudoInverse()

print("unfolding closes?", np.allclose(unfolded, beta))

#STEP 6: alternative unfolding strategy: newton method
print("NEWTON")

