import argparse

parser = argparse.ArgumentParser()
parser.add_argument('Rundir', type=str)
parser.add_argument('--device', type=str, default='cuda')

args = parser.parse_args()

if args.Rundir.endswith('/'):
    args.Rundir = args.Rundir[:-1]

import minimizer

completed, LOSS, configdict, x = minimizer.setup_minimizer_from_run(args.Rundir)
if completed:
    print("Minimization already completed. No need to resume. Exiting.")
    import sys
    sys.exit(1)

import os
import ioutil
import numpy as np

x0, cptid = x

recofolder = os.path.dirname(os.path.dirname(args.Rundir))
reco = ioutil.wrapped_read_np(os.path.join(recofolder, 'RECO.npy'))
if configdict['recoerr_mode'] == 'invcov':
    recoerr_name = 'INVCOV.npy'
elif configdict['recoerr_mode'] == 'stdev_type1':
    recoerr_name = 'ERR1D.npy'
elif configdict['recoerr_mode'] == 'stdev_type2':
    recoerr_name = 'ERR2D.npy'
else:
    raise ValueError("Invalid recoerr_mode in configdict. Must be one of 'invcov', 'stdev_type1', or 'stdev_type2'.")

recoerr = ioutil.wrapped_read_np(os.path.join(recofolder, recoerr_name))

if 'frozen_mask' in configdict.keys():
    frozen_mask = np.asarray(configdict['frozen_mask'])
    frozen_vals = np.asarray(configdict['frozen_vals'])
else:
    print("Warning: no frozen_mask or frozen_vals in configdict. Using defaults (no forzen mask).")
    frozen_mask = None
    frozen_vals = None

if 'cpt_interval' in configdict.keys():
    cpt_interval = configdict['cpt_interval']
else:
    print("Warning: no cpt_interval in configdict. Using default value of 50.")
    cpt_interval = 50

res = minimizer.run_minimization(
        LOSS, reco, recoerr, 
        run2d=configdict['run2d'],
        method=configdict['method'],
        x0 = x0,
        device=args.device,
        frozen_mask = frozen_mask,
        frozen_vals = frozen_vals,
        logpath = args.Rundir,
        cpt_interval = cpt_interval,
        cpt_start = cptid + 1,
        **configdict['method_kwargs'])

minimizer.write_minimization_result(*res, destination=os.path.join(args.Rundir, 'minimization_result'))

