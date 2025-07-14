import argparse

parser = argparse.ArgumentParser()
parser.add_argument('Tag', type=str)
parser.add_argument('Sample', type=str)
parser.add_argument('--statN', type=int, default=-1)
parser.add_argument('--statK', type=int, default=-1)
parser.add_argument('--firstN', type=int, default=-1)
parser.add_argument('--wtsyst', type=str, default='nominal')
parser.add_argument('--objsyst', type=str, default='nominal')
parser.add_argument('--nboot', type=int, default=-1)

parser.add_argument('--reweight', type=str, default=None)
parser.add_argument('--r123type', type=str, default=None)

parser.add_argument('--force', action='store_true')

parser.add_argument('--projectAxes', type=str, nargs='*', default=None)
parser.add_argument('--rebinning', type=str, default=None)

parser.add_argument('--what', type=str, default='reco')

parser.add_argument('--out_nboot', type=int, default=15000)

args = parser.parse_args()

import filenames 
import numpy as np
import statutil
import os
import ioutil

capswhat = args.what.upper()

recofolder = filenames.reco_folder(
    args.Tag, args.Sample, args.nboot,
    args.statN, args.statK, args.firstN,
    args.objsyst, args.wtsyst, 
    args.projectAxes, args.rebinning,
    args.what
)

output_path = os.path.join(recofolder, '%s_DIRECTBOOT%d.npy' % (capswhat,args.out_nboot))
if os.path.exists(output_path) and not args.force:
    print(f"File {output_path} already exists. Use --force to overwrite.")
    import sys
    sys.exit(0)

reco = ioutil.wrapped_read_np(
    os.path.join(recofolder, '%s.npy' % capswhat)
)
cov = ioutil.wrapped_read_np(
    os.path.join(recofolder, 'COV_DIRECT.npy')
)

print("Getting matrix square root")
_, _, _, _, L = statutil.inverse_and_eigenspectrum(
    cov, clip_lowest_N=0, force_positive=True, return_sqrt=True, wrt_corr=True
)

print("Generating multivariate Gaussian samples")
samples = statutil.multivariate_gaussian_rvs(
    reco, L, args.out_nboot
)

result = np.concatenate((reco[None,:], samples), axis=0)
print(result.shape)

ioutil.wrapped_write_np(output_path, result)
