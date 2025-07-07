import fasteigenpy as eigen
import argparse

parser = argparse.ArgumentParser(description='Run the minimizer for EEC reconstruction')
parser.add_argument('Rundir', type=str)

parser.add_argument('--force', action='store_true')

parser.add_argument('--clipLowestN', type=int, default=0)
parser.add_argument('--forcePositive', action='store_true')

parser.add_argument('--clip_wrt_corr', action='store_true')

args = parser.parse_args()

import os

hessianpath = os.path.join(args.Rundir, 'minimization_result', 'HESSIAN.npy')

if args.clip_wrt_corr:
    eigvals_path = os.path.join(args.Rundir, 'minimization_result', 'HESS_CORR_EIGVALS.npy')
    eigvecs_path = os.path.join(args.Rundir, 'minimization_result', 'HESS_CORR_EIGVECS.npy')
else:
    eigvals_path = os.path.join(args.Rundir, 'minimization_result', 'HESS_EIGVALS.npy')
    eigvecs_path = os.path.join(args.Rundir, 'minimization_result', 'HESS_EIGVECS.npy')

clipped_name = 'clip%d' % args.clipLowestN
if args.forcePositive:
    clipped_name += '_forcePos'
if args.clip_wrt_corr:
    clipped_name += '_clipCorr'

inverse_path = os.path.join(args.Rundir, 'minimization_result', 'HESS_EIGINV_%s.npy' % clipped_name)
reconstructed_path = os.path.join(args.Rundir, 'minimization_result', 'HESS_EIG_%s.npy' % clipped_name)

Linv_path = os.path.join(args.Rundir, 'minimization_result', 'HESS_EIGINV_L_%s.npy' % clipped_name)
Lreco_path = os.path.join(args.Rundir, 'minimization_result', 'HESS_EIG_L_%s.npy' % clipped_name)

if os.path.exists(eigvals_path) and os.path.exists(eigvecs_path) and os.path.exists(inverse_path) and os.path.exists(reconstructed_path) and not args.force:
    print(f"Destination {inverse_path} already exists. Use --force to overwrite.")
    import sys
    sys.exit(0)

import ioutil
import numpy as np

H = ioutil.wrapped_read_np(hessianpath)
H = 0.5 * (H + H.T)  # Ensure symmetry

if args.clip_wrt_corr:
    err = np.sqrt(np.diag(H))
    inverr = 1 / err
    corr = np.diag(inverr) @ H @ np.diag(inverr)

    print("Computing eigendecomposition of normalized Hessian...")
    solver = eigen.SelfAdjointEigenSolver(corr)
else:
    print("Computing eigendecomposition of Hessian...")
    solver = eigen.SelfAdjointEigenSolver(H)

if solver.info() != eigen.ComputationInfo.Success:
    print("Eigen decomposition failed")
    print(solver.info())
    import sys
    sys.exit(1)

ioutil.wrapped_write_np(eigvals_path, solver.eigenvalues())
ioutil.wrapped_write_np(eigvecs_path, solver.eigenvectors())

print("Inverting...")
import statutil
inverse, reconstructed, Linv, Lreco = statutil.inverse_from_eigenspectrum(
    solver, clip_lowest_N=args.clipLowestN, force_positive=args.forcePositive,
    return_sqrt=True
)

if args.clip_wrt_corr:
    inverse = np.diag(inverr) @ inverse @ np.diag(inverr)
    reconstructed = np.diag(err) @ reconstructed @ np.diag(err)

    Linv = np.diag(inverr) @ Linv
    Lreco = np.diag(err) @ Lreco

ioutil.wrapped_write_np(inverse_path, inverse)
ioutil.wrapped_write_np(reconstructed_path, reconstructed)

ioutil.wrapped_write_np(Linv_path, Linv)
ioutil.wrapped_write_np(Lreco_path, Lreco)
