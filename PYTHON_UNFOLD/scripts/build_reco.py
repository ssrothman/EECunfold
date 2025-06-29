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

parser.add_argument('--boot_per_file', type=int, nargs='+', default=[-1])
parser.add_argument('--reweight', type=str, default=None)

parser.add_argument('--force', action='store_true')

parser.add_argument('--testcut', action='store_true')

args = parser.parse_args()

import filenames
import datasets
import numpy as np
import os
import ioutil

Hreco = datasets.get_pickled_histogram(
        args.Tag, args.Sample, 'EECres4tee', 
        args.objsyst, args.wtsyst, 'reco',
        statN=args.statN, statK=args.statK,
        boot_per_file=args.boot_per_file,
        firstN=args.firstN, 
        reweight=args.reweight,
        max_nboot = args.nboot)

if args.nboot >= 0:
    Hreco = Hreco[{'bootstrap' : slice(None, args.nboot+1)}]
if args.testcut:
    Hreco = Hreco[{'pt' : slice(None,None,sum)}]

actual_nboot = Hreco.axes['bootstrap'].size - 1

recofolder = filenames.reco_folder(
        args.Tag, args.Sample, actual_nboot,
        args.statN, args.statK, args.firstN,
        args.objsyst, args.wtsyst, args.testcut
)
if os.path.exists(recofolder) and not args.force:
    print(f"Folder {recofolder} already exists. Use --force to overwrite.")
    import sys
    sys.exit(0)

os.makedirs(recofolder, exist_ok=True)

reco = Hreco[{'bootstrap' : 0}].values(flow=True).ravel()
ioutil.wrapped_write_np(os.path.join(recofolder, 'RECO.npy'), reco)

import unc
print("building cov")
DY = unc.dymat(Hreco, norm=True)
cov = DY.T @ DY / DY.shape[0]
ioutil.wrapped_write_np(os.path.join(recofolder, 'COV.npy'), cov)

err1D = np.sqrt(np.diag(cov))
err1D[np.diag(cov) <= 0] = 1
ioutil.wrapped_write_np(os.path.join(recofolder, 'ERR1D.npy'), err1D)

print("inverting cov")
codcov = eigen.CompleteOrthogonalDecomposition(cov)
invcov = codcov.pseudoInverse()
ioutil.wrapped_write_np(os.path.join(recofolder, 'INVCOV.npy'), invcov)

err2D = 1/np.sqrt(np.diag(invcov))
err2D[np.diag(invcov) <= 0] = 1
ioutil.wrapped_write_np(os.path.join(recofolder, 'ERR2D.npy'), err2D)
