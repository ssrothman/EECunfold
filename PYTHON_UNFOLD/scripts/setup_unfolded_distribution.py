import argparse

parser = argparse.ArgumentParser(description='Run the minimizer for EEC reconstruction')
parser.add_argument('Rundir', type=str)

parser.add_argument('--force', action='store_true')
parser.add_argument('--device', type=str, default=None)

args = parser.parse_args()

if args.device is None:
    import torch
    if torch.cuda.is_available():
        args.device = 'cuda'
    else:
        args.device = 'cpu'

import os

unfpath = os.path.join(args.Rundir, 'minimization_result', 'UNFOLDED.npy')
covunfpath = os.path.join(args.Rundir,'minimization_result', 'COV_UNFOLDED.npy')

fwdpath = os.path.join(args.Rundir,'minimization_result', 'FORWARD.npy')

if os.path.exists(unfpath) and os.path.exists(covunfpath) and os.path.exists(fwdpath) and not args.force:
    print(f"Files {unfpath}, {covunfpath}, {fwdpath} already exist. Use --force to overwrite.")
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
else:
    configdict['frozen_mask'] = np.asarray(configdict['frozen_mask'])
    configdict['frozen_vals'] = np.asarray(configdict['frozen_vals'])

hess = ioutil.wrapped_read_np(os.path.join(args.Rundir, 'minimization_result', 'HESSIAN.npy'))
beta = res.x[:LOSS.nBeta]
covbeta = hess[:LOSS.nBeta, :LOSS.nBeta]

unf = reco * beta
covunf = np.diag(reco) @ covbeta @ np.diag(reco)

ioutil.wrapped_write_np(unfpath, unf)
ioutil.wrapped_write_np(covunfpath, covunf)

theta = res.x[LOSS.nBeta:]

LOSS.torch()
LOSS.to(args.device)
unf = torch.from_numpy(unf).to(args.device)
theta = torch.from_numpy(theta).to(args.device)

fwd = LOSS.forward(unf, theta)

fwd = fwd.cpu().numpy()

ioutil.wrapped_write_np(fwdpath, fwd)
