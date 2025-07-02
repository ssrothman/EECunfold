import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('Rundir', type=str)

parser.add_argument("--out_nboot", type=int, default=5000)

parser.add_argument('--device', type=str, default='cuda')

freeze_group = parser.add_mutually_exclusive_group(required=False)
freeze_group.add_argument('--freezeAllNuisances', action='store_true')

parser.add_argument('--force', action='store_true')

args = parser.parse_args()

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

outname += '.pkl'
resultpath = os.path.join(args.Rundir, 'minimization_result', outname)

if os.path.exists(resultpath) and not args.force:
    print(f"File {resultpath} already exists. Use --force to overwrite.")
    import sys
    sys.exit(0)

import filenames
import datasets
import minimizer
import ioutil

reconame = os.path.basename(os.path.dirname(os.path.dirname(args.Rundir)))
tag, sample, _, statN, statK, firstN, objsyst, wtsyst, projectAxes = filenames.parse_reco_name(reconame)

Htemplate = filenames.get_full_hist(
        tag, sample, -1, statN, statK, firstN, 
        objsyst, wtsyst, 'reco', max_nboot=0,
        from_bkp=sample != 'Pythia_HTsum',
)
if projectAxes is not None:
    Htemplate = Htemplate.project('bootstrap', *projectAxes)

res = minimizer.read_minimization_result(
    os.path.join(args.Rundir, 'minimization_result'),
)

x = res[0].x
reco = res[1]

Hinv = ioutil.wrapped_read_np(
    os.path.join(args.Rundir, 'minimization_result', 'INVHESS.npy'),
)
L = ioutil.wrapped_read_np(
    os.path.join(args.Rundir, 'minimization_result', 'INVHESS_L.npy'),
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
minimizer.dump_Hfwd(LOSS, xfull, L, reco, Htemplate, args.out_nboot, resultpath, 
                    device=args.device)

