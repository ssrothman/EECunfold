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
parser.add_argument('--rebin_r', type=int, default=1)
parser.add_argument('--rebin_c', type=int, default=1)

parser.add_argument('--ptoverflow', type=str, default=None)

parser.add_argument('--clipLowestN', type=int, default=0)
parser.add_argument('--forcePositive', action='store_true')

parser.add_argument('--clip_wrt_corr', action='store_true')

parser.add_argument('--oldbinning', action='store_true')

whichcov_group = parser.add_mutually_exclusive_group(required=False)
whichcov_group.add_argument('--normed', action='store_true')
whichcov_group.add_argument('--direct', action='store_true')

args = parser.parse_args()

import filenames
import os

recofolder = filenames.reco_folder(
        args.Tag, args.Sample, args.nboot,
        args.statN, args.statK, args.firstN,
        args.objsyst, args.wtsyst, 
        args.projectAxes, args.rebin_r, args.rebin_c,
        args.ptoverflow,
)
if args.oldbinning:
    recofolder += '_oldbinning'

suffix = ''
if args.normed:
    suffix = '_NORMED'
elif args.direct:
    suffix = '_DIRECT'

if args.clip_wrt_corr:
    eigvals_path = os.path.join(recofolder, 'CORR%s_EIGVALS.npy' % suffix)
    eigvecs_path = os.path.join(recofolder, 'CORR%s_EIGVECS.npy' % suffix)
else:
    eigvals_path = os.path.join(recofolder, 'COV%s_EIGVALS.npy' % suffix)
    eigvecs_path = os.path.join(recofolder, 'COV%s_EIGVECS.npy' % suffix)

clipped_name = 'clip%d' % args.clipLowestN
if args.forcePositive:
    clipped_name += '_forcePos'
if args.clip_wrt_corr:
    clipped_name += '_clipCorr'

inverse_path = os.path.join(recofolder, 'COV%s_EIGINV_%s.npy' % (suffix, clipped_name))
reconstructed_path = os.path.join(recofolder, 'COV%s_EIG_%s.npy' % (suffix, clipped_name))

if os.path.exists(eigvals_path) and os.path.exists(eigvecs_path) and os.path.exists(inverse_path) and os.path.exists(reconstructed_path) and not args.force:
    print(f"Eigenvalues and eigenvectors already exist in {recofolder}. Use --force to overwrite.")
    import sys
    sys.exit(0)

import ioutil
import numpy as np

cov = ioutil.wrapped_read_np(os.path.join(recofolder, 'COV%s.npy' % suffix))

if args.clip_wrt_corr:
    err = np.sqrt(np.diag(cov))
    err[err==0] = 1
    inverr = 1/err
    corr = np.diag(inverr) @ cov @ np.diag(inverr)

    print("Computing eigendecomposition for corr...")
    solver = eigen.SelfAdjointEigenSolver(corr)
else:
    print("Computing eigendecomposition for covariance...")
    solver = eigen.SelfAdjointEigenSolver(cov)

if solver.info() != eigen.ComputationInfo.Success:
    print("Eigen decomposition failed")
    print(solver.info())
    import sys
    sys.exit(1)

ioutil.wrapped_write_np(eigvals_path, solver.eigenvalues())
ioutil.wrapped_write_np(eigvecs_path, solver.eigenvectors())

print("Inverting...")
import statutil
inverse, reconstructed = statutil.inverse_from_eigenspectrum(
    solver, clip_lowest_N=args.clipLowestN, force_positive=args.forcePositive
)

if args.clip_wrt_corr:
    inverse = np.diag(inverr) @ inverse @ np.diag(inverr)
    reconstructed = np.diag(err) @ reconstructed @ np.diag(err)

ioutil.wrapped_write_np(inverse_path, inverse)
ioutil.wrapped_write_np(reconstructed_path, reconstructed)
