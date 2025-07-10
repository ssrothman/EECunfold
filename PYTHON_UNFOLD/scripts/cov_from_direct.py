import argparse

parser = argparse.ArgumentParser()
parser.add_argument('Tag', type=str)
parser.add_argument('Sample', type=str)
parser.add_argument('--statN', type=int, default=-1)
parser.add_argument('--statK', type=int, default=-1)
parser.add_argument('--firstN', type=int, default=-1)
parser.add_argument('--wtsyst', type=str, default='nominal')
parser.add_argument('--objsyst', type=str, default='nominal')

parser.add_argument('--reweight', type=str, default=None)
parser.add_argument('--r123type', type=str, default=None)

parser.add_argument('--force', action='store_true')

parser.add_argument('--projectAxes', type=str, nargs='*', default=None)
parser.add_argument('--rebin_r', type=int, default=1)
parser.add_argument('--rebin_c', type=int, default=1)
parser.add_argument('--ptoverflow', type=str, default=None)

parser.add_argument('--theaxes', type=str, nargs='*',
                    default=['pt', 'R', 'r', 'c'],
                    help='Axis names in covariance matrix (needed for --projectAxes). Should probably never be changed from default value')

args = parser.parse_args()

import filenames
import numpy as np
import os
import ioutil

recofolder = filenames.reco_folder(
    args.Tag, args.Sample, -1,
    args.statN, args.statK, args.firstN, 
    args.objsyst, args.wtsyst, 
    args.projectAxes, args.rebin_r, args.rebin_c,
    args.ptoverflow
)

output_path = os.path.join(recofolder, 'COV_DIRECT.npy')

if os.path.exists(output_path) and not args.force:
    print(f"Folder {recofolder} already exists. Use --force to overwrite.")
    import sys
    sys.exit(0)

cov = filenames.get_full_hist(
    args.Tag, args.Sample, 0, 
    args.statN, args.statK, args.firstN,
    args.objsyst, args.wtsyst, 'directcov_reco',
    args.reweight, args.r123type, 
    max_nboot=0, from_bkp=False
)

if args.rebin_r != 1:
    which_r_reco = args.theaxes.index('r')
    Nr_reco = cov.shape[which_r_reco] 
    cov = np.add.reduceat(cov, range(0, Nr_reco, args.rebin_r), axis=which_r_reco)

    which_r_gen = which_r_reco + len(args.theaxes)
    Nr_gen = cov.shape[which_r_gen]
    cov = np.add.reduceat(cov, range(0, Nr_gen, args.rebin_r), axis=which_r_gen)

if args.rebin_c != 1:
    which_c_reco = args.theaxes.index('c')
    Nc_reco = cov.shape[which_c_reco]
    cov = np.add.reduceat(cov, range(0, Nc_reco, args.rebin_c), axis=which_c_reco)

    which_c_gen = which_c_reco + len(args.theaxes)
    Nc_gen = cov.shape[which_c_gen]
    cov = np.add.reduceat(cov, range(0, Nc_gen, args.rebin_c), axis=which_c_gen)

if args.ptoverflow is not None:
    if args.ptoverflow == 'merge':
        reduction = list(range(cov.shape[0]))
        reduction.pop(-1)
        cov = np.add.reduceat(cov, reduction, axis=0)
        cov = np.add.reduceat(cov, reduction, axis=len(cov.shape)//2)
    elif args.ptoverflow == 'drop':
        cov = cov[:-1]
        oneshape = cov.shape[:len(cov.shape)//2]
        onesize = np.prod(oneshape)
        cov = cov.reshape((onesize, cov.shape[0]+1, -1))
        cov = cov[:, :-1]
        cov = cov.reshape((*oneshape, *oneshape))
    else:
        raise ValueError("Invalid ptoverflow option: %s. Use 'merge' or 'drop'." % args.ptoverflow)

if args.projectAxes:
    whichaxes = []
    for axis in args.projectAxes:
        if axis not in args.theaxes:
            raise ValueError(f"Axis '{axis}' not found in theaxes: {args.theaxes}")
        axisidx = args.theaxes.index(axis)
        whichaxes.append(axisidx)

    whichaxes += [i+len(args.theaxes) for i in whichaxes]
    axismask = np.ones(len(args.theaxes) * 2, dtype=bool)
    axismask[whichaxes] = False
    sumaxes = np.where(axismask)[0].tolist()

    cov = np.sum(cov, axis=tuple(sumaxes))

halfshape = cov.shape[:len(cov.shape)//2]
halfsize = np.prod(halfshape)
cov = cov.reshape(halfsize, halfsize)

ioutil.wrapped_write_np(output_path, cov)
