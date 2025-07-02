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

parser.add_argument('--boot_per_file', type=int, default=-1)
parser.add_argument('--reweight', type=str, default=None)
parser.add_argument('--r123type', type=str, default=None)

parser.add_argument('--force', action='store_true')

parser.add_argument('--projectAxes', type=str, nargs='*', default=None)

args = parser.parse_args()

import filenames
import datasets
import numpy as np
import os
import ioutil

Hreco = filenames.get_full_hist(
    args.Tag, args.Sample, args.boot_per_file,
    args.statN, args.statK, args.firstN,
    args.objsyst, args.wtsyst, 'reco',
    args.reweight, args.r123type,
    max_nboot=args.nboot,
    from_bkp=args.Sample!='Pythia_HTsum'
) 

if args.projectAxes is not None:
    Hreco = Hreco.project('bootstrap', *args.projectAxes)

if args.nboot >= 0:
    Hreco = Hreco[{'bootstrap' : slice(None, args.nboot+1)}]

actual_nboot = Hreco.axes['bootstrap'].size - 1

recofolder = filenames.reco_folder(
        args.Tag, args.Sample, actual_nboot,
        args.statN, args.statK, args.firstN,
        args.objsyst, args.wtsyst, args.projectAxes,
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
vals = Hreco.values(flow=True).reshape((Hreco.axes['bootstrap'].size, -1))

sums = vals.sum(axis=1)
vals = vals * sums[0] / sums[:,None]

boots = vals[1:]
nom = vals[0][None,:]

DY = boots - nom

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
