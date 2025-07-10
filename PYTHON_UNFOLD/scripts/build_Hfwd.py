import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('Rundir', type=str)

parser.add_argument("--out_nboot", type=int, default=15000)

parser.add_argument('--device', type=str, default=None)

freeze_group = parser.add_mutually_exclusive_group(required=False)
freeze_group.add_argument('--freezeAllNuisances', action='store_true')

parser.add_argument('--force', action='store_true')

parser.add_argument('--clipLowestN', type=int, default=0)
parser.add_argument('--forcePositive', action='store_true')

parser.add_argument('--clip_wrt_corr', action='store_true')

args = parser.parse_args()

if args.device is None:
    import torch
    if torch.cuda.is_available():
        args.device = 'cuda'
    else:
        args.device = 'cpu'

if args.Rundir[-1] == '/':
    args.Rundir = args.Rundir[:-1]

import os

outname = 'Hfwd_boot%d' % args.out_nboot
if args.freezeAllNuisances:
    raise NotImplementedError
    outname += '_freezeAll'
    freezeMode = 'freezeAll'
else:
    freezeMode = 'no'

clipname = 'clip%d' % args.clipLowestN
if args.forcePositive:
    clipname += '_forcePos'
if args.clip_wrt_corr:
    clipname += '_clipCorr'

outname += '_' + clipname + '.npy'
resultpath = os.path.join(args.Rundir, 'minimization_result', outname)

if os.path.exists(resultpath) and not args.force:
    print(f"File {resultpath} already exists. Use --force to overwrite.")
    import sys
    sys.exit(0)

import filenames
import datasets
import minimizer
import ioutil
import hist

res = minimizer.read_minimization_result(
    os.path.join(args.Rundir, 'minimization_result'),
)

x = res[0].x
reco = res[1]

L = ioutil.wrapped_read_np(
    os.path.join(args.Rundir, 'minimization_result', 'HESS_EIGINV_L_%s.npy' % clipname)
)

_, LOSS, configdict, _ = minimizer.setup_minimizer_from_run(args.Rundir)

import numpy as np
xfull = np.zeros(LOSS.nBeta + LOSS.nTheta, dtype=x.dtype)
if 'frozen_mask' in configdict:
    frozen_mask = np.asarray(configdict['frozen_mask'], dtype=bool)
    frozen_vals = np.asarray(configdict['frozen_vals'], dtype=x.dtype)
    xfull[frozen_mask] = frozen_vals
    xfull[~frozen_mask] = x
else:
    xfull = x

print("xfull.shape", xfull.shape)
print("L.shape", L.shape)

import minimizer
minimizer.dump_Hfwd(LOSS, xfull, L, reco, args.out_nboot, resultpath, 
                    device=args.device)

