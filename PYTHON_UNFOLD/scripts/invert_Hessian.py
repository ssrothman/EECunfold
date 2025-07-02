import fasteigenpy as eigen
import argparse

parser = argparse.ArgumentParser(description='Run the minimizer for EEC reconstruction')
parser.add_argument('Rundir', type=str)

parser.add_argument('--force', action='store_true')
parser.add_argument('--help_condition', type=float, default=0.001)

args = parser.parse_args()

import os

hessianpath = os.path.join(args.Rundir, 'minimization_result', 'HESSIAN.npy')

if not os.path.exists(hessianpath):
    print(f"File {hessianpath} does not exist. Cannot compute inverse Hessian.")
    import sys
    sys.exit(1)

invhess_path = os.path.join(args.Rundir, 'minimization_result', 'INVHESS.npy')
invhessL_path = os.path.join(args.Rundir, 'minimization_result', 'INVHESS_L.npy')

if os.path.exists(invhess_path) and not args.force:
    print(f"File {invhess_path} already exists. Use --force to overwrite.")
    import sys
    sys.exit(0)

import ioutil
import numpy as np

H = ioutil.wrapped_read_np(hessianpath)
H = 0.5 * (H + H.T)  # Ensure symmetry

if args.help_condition > 0:
    print("Applying help condition to Hessian...")
    np.fill_diagonal(H, np.diagonal(H) * (1 + args.help_condition))

print("Computing LDLT(H)")
ldlt = eigen.LDLT(H)
if ldlt.info() != eigen.ComputationInfo.Success:
    print("LDLT decomposition failed")
    print(ldlt.info())
    import sys
    sys.exit(1)

invhess = ldlt.solve(np.eye(H.shape[0]))
ioutil.wrapped_write_np(invhess_path, invhess)

print("Computing LDLT(Hinv)")
ldlt_inv = eigen.LDLT(invhess)

PL = ldlt_inv.matrixPL()
D = ldlt_inv.vectorD()
D[D<0] = 0
Dsq = np.sqrt(D)
L = PL @ np.diag(Dsq)

#check
reconstructed = L @ L.T
if not np.allclose(reconstructed, invhess):
    print("WARNING: Reconstructed matrix does not match the inverse Hessian.")

ioutil.wrapped_write_np(invhessL_path, L)
