import argparse

parser = argparse.ArgumentParser(description='Run the minimizer for EEC reconstruction')
parser.add_argument('Rundir', type=str)

parser.add_argument('--force', action='store_true')
parser.add_argument('--device', type=str, default=None)

parser.add_argument("--clipLowestN", type=int, default=0)
parser.add_argument("--dontForcePositive", action='store_true')
parser.add_argument("--dont_clip_wrt_corr", action='store_true')

args = parser.parse_args()

if args.device is None:
    import torch
    if torch.cuda.is_available():
        args.device = 'cuda'
    else:
        args.device = 'cpu'

import os

unfpath = os.path.join(args.Rundir, 'minimization_result', 'UNFOLDED.npy')
unfsystpath = os.path.join(args.Rundir, 'minimization_result', 'UNFOLDED_SYST.npy')
covunfpath = os.path.join(args.Rundir,'minimization_result', 'COV_UNFOLDED.npy')
covunfsystpath = os.path.join(args.Rundir, 'minimization_result', 'COV_UNFOLDED_SYST.npy')

fwdpath = os.path.join(args.Rundir,'minimization_result', 'FORWARD.npy')

if os.path.exists(unfpath) and os.path.exists(covunfpath) and \
        os.path.exists(fwdpath) and os.path.exists(covunfsystpath) and \
        os.path.exists(unfsystpath) and not args.force:
    print(f"Files {unfpath}, {covunfpath}, {fwdpath}, and {covunfsystpath} already exist. Use --force to overwrite.")
    import sys
    sys.exit(0)

import loss
import minimizer
import numpy as np
import ioutil

completed, LOSS, configdict, x = minimizer.setup_minimizer_from_run(args.Rundir)
if not completed:
    print("Minimization not completed. Cannot compute unfolded distribution. Exiting.")
    import sys
    sys.exit(1)

res, reco, recoerr, x0 = x
if 'frozen_mask' not in configdict:
    print("Warning: no frozen_mask in configdict. Using default (no frozen mask).")
    configdict['frozen_mask'] = None
    configdict['frozen_vals'] = None
    xfull = res.x
else:
    configdict['frozen_mask'] = np.asarray(configdict['frozen_mask'])
    configdict['frozen_vals'] = np.asarray(configdict['frozen_vals'])
    xfull = np.empty((LOSS.nBeta+LOSS.nTheta))
    xfull[configdict['frozen_mask']] = configdict['frozen_vals']
    xfull[~configdict['frozen_mask']] = res.x

eigstr = 'clip%d' % args.clipLowestN
if not args.dontForcePositive:
    eigstr += '_forcePos'
if not args.dont_clip_wrt_corr:
    eigstr += '_clipCorr'

invhess = ioutil.wrapped_read_np(os.path.join(args.Rundir, 'minimization_result', 'HESS_EIGINV_%s.npy' % eigstr))
beta = xfull[:LOSS.nBeta]
covbeta = invhess[:LOSS.nBeta, :LOSS.nBeta]

factor = LOSS.genBaseline * reco.sum() / LOSS.baselineRecoFlux

unf = beta * factor
covunf = np.diag(factor) @ covbeta @ np.diag(factor)

scalefactor = np.ones(LOSS.nBeta + LOSS.nTheta)
scalefactor[:LOSS.nBeta] = factor
covunfsyst = np.diag(scalefactor) @ invhess @ np.diag(scalefactor)

ioutil.wrapped_write_np(unfpath, unf)
ioutil.wrapped_write_np(covunfpath, covunf)
ioutil.wrapped_write_np(covunfsystpath, covunfsyst)

theta = xfull[LOSS.nBeta:]

LOSS.torch()
LOSS.to(args.device)
unf = torch.from_numpy(unf).to(args.device)
theta = torch.from_numpy(theta).to(args.device)

fwd = LOSS.forward(unf, theta)

fwd = fwd.cpu().numpy()

ioutil.wrapped_write_np(fwdpath, fwd)

ioutil.wrapped_write_np(unfsystpath, theta.cpu().detach().numpy())
