import argparse
import fasteigenpy as eigen

parser = argparse.ArgumentParser(description='Build EEC reco histograms')
parser.add_argument('Tag', type=str)
parser.add_argument('Sample', type=str)
parser.add_argument('--nboot', type=int, default=-1)
parser.add_argument('--statN', type=int, default=-1)
parser.add_argument('--statK', type=int, default=-1)
parser.add_argument('--firstN', type=int, default=-1)
parser.add_argument('--wtsyst', type=str, default='nominal')
parser.add_argument('--objsyst', type=str, default='nominal')

parser.add_argument('--force', action='store_true')

parser.add_argument('--projectAxes', type=str, nargs='*', default=None)

parser.add_argument('--clip_to_zero', type=float, default=1e-20)

args = parser.parse_args()

import filenames
import os

recofolder = filenames.reco_folder(
        args.Tag, args.Sample, args.nboot,
        args.statN, args.statK, args.firstN,
        args.objsyst, args.wtsyst, args.projectAxes
)

eigvals_path = os.path.join(recofolder, 'COV_EIGVALS.npy')
eigvecs_path = os.path.join(recofolder, 'COV_EIGVECS.npy')

if os.path.exists(eigvals_path) and os.path.exists(eigvecs_path) and not args.force:
    print(f"Eigenvalues and eigenvectors already exist in {recofolder}. Use --force to overwrite.")
    import sys
    sys.exit(0)

import ioutil

cov = ioutil.wrapped_read_np(os.path.join(recofolder, 'COV.npy'))

print("Computing eigendecomposition...")
solver = eigen.SelfAdjointEigenSolver(cov)
if solver.info() != eigen.ComputationInfo.Success:
    raise RuntimeError(f"Eigen decomposition failed with info: {solver.info()}")

eigvals = solver.eigenvalues()
eigvecs = solver.eigenvectors()

#check
import numpy as np
reconstructed = eigvecs @ np.diag(eigvals) @ eigvecs.T
if not np.allclose(reconstructed, cov):
    print("Reconstructed covariance does not match original covariance.")
    import sys
    sys.exit(1)

ioutil.wrapped_write_np(eigvals_path, eigvals)
ioutil.wrapped_write_np(eigvecs_path, eigvecs)

#alternative inverse
eigvals_denom = np.where(eigvals < args.clip_to_zero, 1, eigvals)
eigvals_inv = 1 / eigvals_denom

eigvals_inv2 = np.where(eigvals < args.clip_to_zero, 0, eigvals_inv)

invcov_eig = eigvecs @ np.diag(eigvals_inv) @ eigvecs.T
invcov_eig2 = eigvecs @ np.diag(eigvals_inv2) @ eigvecs.T

ioutil.wrapped_write_np(os.path.join(recofolder, 'COV_EIGINV.npy'), invcov_eig)
ioutil.wrapped_write_np(os.path.join(recofolder, 'COV_EIGINV2.npy'), invcov_eig2)

