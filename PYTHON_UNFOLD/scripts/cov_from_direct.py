import argparse

parser = argparse.ArgumentParser()
parser.add_argument('Tag', type=str)
parser.add_argument('Sample', type=str)
parser.add_argument('--statN', type=int, default=-1)
parser.add_argument('--statK', type=int, default=-1)
parser.add_argument('--firstN', type=int, default=-1)
parser.add_argument('--nboot', type=int, default=-1,)
parser.add_argument('--wtsyst', type=str, default='nominal')
parser.add_argument('--objsyst', type=str, default='nominal')

parser.add_argument('--reweight', type=str, default=None)
parser.add_argument('--r123type', type=str, default=None)

parser.add_argument('--force', action='store_true')

parser.add_argument('--projectAxes', type=str, nargs='*', default=None)
parser.add_argument('--rebinning', type=str, default=None,)
parser.add_argument('--oldbinning', action='store_true',)

parser.add_argument('--theaxes', type=str, nargs='*',
                    default=['pt', 'R', 'r', 'c'],
                    help='Axis names in covariance matrix (needed for --projectAxes). Should probably never be changed from default value')

parser.add_argument('--what', type=str, default='reco')

args = parser.parse_args()

import filenames
import numpy as np
import os
import ioutil

recofolder = filenames.reco_folder(
    args.Tag, args.Sample, args.nboot,
    args.statN, args.statK, args.firstN, 
    args.objsyst, args.wtsyst, 
    args.projectAxes, args.rebinning,
    args.what
)
if args.oldbinning:
    recofolder += '_oldbinning'

output_path = os.path.join(recofolder, 'COV_DIRECT.npy')

if os.path.exists(output_path) and not args.force:
    print(f"File {output_path} already exists. Use --force to overwrite.")
    import sys
    sys.exit(0)

cov = filenames.get_full_hist(
    args.Tag, args.Sample, args.nboot, 
    args.statN, args.statK, args.firstN,
    args.objsyst, args.wtsyst, 'directcov_%s'%args.what,
    args.reweight, args.r123type, 
    max_nboot=0, from_bkp=args.oldbinning
)

halfshape = cov.shape[:len(cov.shape)//2]
halfsize = np.prod(halfshape)
cov = cov.reshape(halfsize, halfsize)

if args.rebinning is not None or args.projectAxes is not None:
    Hreco = filenames.get_full_hist(
        args.Tag, args.Sample, -1, 
        args.statN, args.statK, args.firstN,
        args.objsyst, args.wtsyst, args.what,
        args.reweight, args.r123type, 
        max_nboot=0, from_bkp=args.oldbinning
    )
    import indexing
    binning = indexing.Binning()
    binning.setup_from_histogram(Hreco[{'bootstrap': 0}])
    if args.rebinning is not None:
        print("Rebinning cov")
        print("\tstarting", cov.shape)
        cov, _ = binning.rebin(cov.T, os.path.join('rebinnings', args.rebinning + '.json'))
        print("\thalfway", cov.shape)
        cov, binning = binning.rebin(cov.T, os.path.join('rebinnings', args.rebinning + '.json'))
        print("\tending", cov.shape)
    if args.projectAxes is not None:
        axes_to_project = [ax for ax in binning.axis_names if ax not in args.projectAxes]
        for ax in axes_to_project:
            print("projecting out", ax)
            cov, _ = binning.project_out(cov.T, ax)
            cov, binning = binning.project_out(cov.T, ax)
            print("\tprojected shape:", cov.shape)

ioutil.wrapped_write_np(output_path, cov)

stdev = np.sqrt(np.diag(cov))
stdev_path = os.path.join(recofolder, 'STDEV_DIRECT.npy')
ioutil.wrapped_write_np(stdev_path, stdev)
