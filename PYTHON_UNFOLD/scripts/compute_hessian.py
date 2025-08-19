import argparse
import fasteigenpy as eigen

parser = argparse.ArgumentParser()
parser.add_argument('Rundir', type=str)

parser.add_argument('--fluxes_and_shapes', type=str, nargs='+', default=None)
parser.add_argument('--device', type=str, default=None)
parser.add_argument('--force', action='store_true')

args = parser.parse_args()

if args.device is None:
    import torch
    if torch.cuda.is_available():
        args.device = 'cuda'
    else:
        args.device = 'cpu'

import os
import loss
import datasets
import minimizer
import numpy as np
import ioutil
import torch
torch.autograd.set_detect_anomaly(True)

if args.fluxes_and_shapes:
    hessianpath = os.path.join(args.Rundir, 'minimization_result', 'HESSIAN_FLUXES_SHAPES-%s.npy') % '-'.join(args.fluxes_and_shapes)
else:
    hessianpath = os.path.join(args.Rundir, 'minimization_result', 'HESSIAN.npy')

if os.path.exists(hessianpath) and not args.force:
    print(f"File {hessianpath} already exists. Use --force to overwrite.")
    import sys
    sys.exit(0)

completed, LOSS, configdict, x = minimizer.setup_minimizer_from_run(args.Rundir)
if not completed:
    print("Minimization not completed. Cannot compute Hessian. Exiting.")
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

print("computing Hessian...")
hess = minimizer.compute_hessian(LOSS, reco, recoerr, 
                                 run2d=configdict['run2d'],
                                 x=res.x,
                                 device=args.device,
                                 frozen_mask = configdict['frozen_mask'],
                                 frozen_vals = configdict['frozen_vals'],
                                 fluxes_and_shapes = args.fluxes_and_shapes)

ioutil.wrapped_write_np(hessianpath, hess)
