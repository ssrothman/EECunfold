import argparse

parser = argparse.ArgumentParser(description='Run the minimizer for EEC reconstruction')
parser.add_argument('RecoTag', type=str)
parser.add_argument('RecoSample', type=str)
parser.add_argument('--reco_nboot', type=int, default=-1)
parser.add_argument('--reco_statN', type=int, default=-1)
parser.add_argument('--reco_statK', type=int, default=-1)
parser.add_argument('--reco_firstN', type=int, default=-1)
parser.add_argument('--reco_wtsyst', type=str, default='nominal')
parser.add_argument('--reco_objsyst', type=str, default='nominal')

parser.add_argument('GenTag', type=str)
parser.add_argument('GenSample', type=str)
parser.add_argument('--gen_nboot', type=int, default=-1)
parser.add_argument('--gen_statN', type=int, default=-1)
parser.add_argument('--gen_statK', type=int, default=-1)
parser.add_argument('--gen_firstN', type=int, default=-1)

parser.add_argument('--systlist', type=str, nargs='*',
                    default=['scale', 'isosf', 'idsf', 'triggersf',
                             'PU', 'PDF', 'aS', 'PDFaS',
                             'ISR', 'FSR',
                             'CH', 'JES', 'JER', 'UNCLUSTERED',
                             'TRK_EFF'])

recoerr_group = parser.add_mutually_exclusive_group(required=True)
recoerr_group.add_argument('--invcov_normed', type=str, default=None)
recoerr_group.add_argument('--invcov_direct', type=str, default=None)
recoerr_group.add_argument('--invcov_boot', type=str, default=None)

recoerr_group.add_argument('--stdev_type1', action='store_true')
recoerr_group.add_argument('--stdev_type2', action='store_true')

x0group = parser.add_mutually_exclusive_group(required=True)
x0group.add_argument('--x0fromfile', type=str, default=None)
x0group.add_argument('--nullx0', action='store_true')
x0group.add_argument('--goodGuessX0', action='store_true')
x0group.add_argument('--MCx0', action='store_true')

parser.add_argument('--device', type=str, default=None)
parser.add_argument('--method', type=str, default='l-bfgs')
parser.add_argument('--method_kwargs', type=str, nargs='*', default=[])

parser.add_argument("--checkpoint_interval", type=int, default=50)

parser.add_argument('--projectAxes', type=str, nargs='*', default=None,)
parser.add_argument('--rebinning', type=str, default=None)

parser.add_argument('--smoothed', action='store_true')

parser.add_argument('--rescale', action='store_true',)

parser.add_argument('--oldbinning', action='store_true',)

freezegroup = parser.add_mutually_exclusive_group(required=False)
freezegroup.add_argument('--freezeAllNuisances', action='store_true')

args = parser.parse_args()

if args.device is None:
    import torch
    if torch.cuda.is_available():
        args.device = 'cuda'
    else:
        args.device = 'cpu'

#parse method_kwargs
method_kwargs = {}
for kw in args.method_kwargs:
    if '=' not in kw:
        raise ValueError(f"Invalid method_kwargs format: {kw}. Expected key=value.")
    key, value = kw.split('=', 1)
    try:
        value = int(value)
    except ValueError:
        try:
            value = float(value)
        except ValueError:
            if value.lower() in ['true', 'false']:
                value = value.lower() == 'true'
            else:
                #leave as string
                pass
    method_kwargs[key] = value

args.run2d = (args.invcov_normed is not None) or \
        (args.invcov_direct is not None) or \
        (args.invcov_boot is not None)

import os
import filenames
import loss
import datasets
import minimizer
import numpy as np
import ioutil

reco_folder = filenames.reco_folder(
    args.RecoTag, args.RecoSample, args.reco_nboot,
    args.reco_statN, args.reco_statK, args.reco_firstN,
    args.reco_objsyst, args.reco_wtsyst, 
    args.projectAxes, args.rebinning
)
loss_folder = filenames.loss_folder(
    args.GenTag, args.GenSample, args.gen_nboot,
    args.gen_statN, args.gen_statK, args.gen_firstN,
    args.systlist, 
    args.projectAxes, args.rebinning,
    args.smoothed
)
if args.oldbinning:
    reco_folder += '_oldbinning'
    loss_folder += '_oldbinning'

loss_name = os.path.basename(loss_folder)

LOSS = loss.FullLoss()
LOSS.read_from_disk(loss_folder)

reco = ioutil.wrapped_read_np(os.path.join(reco_folder, 'RECO.npy'))

if args.invcov_normed is not None:
    if args.invcov_normed == 'naive':
        recoerrpath = os.path.join(reco_folder, 'INVCOV_NORMED.npy')
    else:
        recoerrpath = os.path.join(reco_folder, 'COV_NORMED_EIGINV_%s.npy' % args.invcov_normed)
    recoerr_mode_str = 'invcov_normed_%s' % args.invcov_normed
elif args.invcov_direct is not None:
    if args.invcov_direct == 'naive':
        recoerrpath = os.path.join(reco_folder, 'INVCOV_DIRECT.npy')
    else:
        recoerrpath = os.path.join(reco_folder, 'COV_DIRECT_EIGINV_%s.npy'%args.invcov_direct)
    recoerr_mode_str = 'invcov_direct_%s' % args.invcov_direct
elif args.invcov_boot is not None:
    if args.invcov_boot == 'naive':
        recoerrpath = os.path.join(reco_folder, 'INVCOV.npy')
    else:
        recoerrpath = os.path.join(reco_folder, 'COV_EIGINV_%s.npy' % args.invcov_boot)
    recoerr_mode_str = 'invcov_boot_%s' % args.invcov_boot
elif args.stdev_type1:
    recoerrpath = os.path.join(reco_folder, 'ERR1D.npy')
    recoerr_mode_str = 'stdev_type1'
elif args.stdev_type2:
    recoerrpath = os.path.join(reco_folder, 'ERR2D.npy')
    recoerr_mode_str = 'stdev_type2'
else:
    raise ValueError("Couldn't determine recoerr type. Use --invcov_boot --invcov_normed, --invcov_direct, --stdev_type1, or --stdev_type2.")
recoerr = ioutil.wrapped_read_np(recoerrpath)

if args.x0fromfile is not None:
    x0 = ioutil.wrapped_read_np(args.x0fromfile)
    x0mode_str = "file:"+args.x0fromfile
elif args.nullx0:
    print("Using null x0")
    x0 = np.ones(LOSS.nBeta)
    x0mode_str = 'nullx0'
elif args.goodGuessX0:
    goodx0path = os.path.join(reco_folder, loss_name, 'GOODx0.npy')
    x0 = ioutil.wrapped_read_np(goodx0path)
    x0mode_str = 'goodGuessX0'
elif args.MCx0:
    MCx0path = os.path.join(loss_folder, 'MCx0.npy')
    x0 = ioutil.wrapped_read_np(MCx0path)
    x0mode_str = 'MCx0'
else:
    raise ValueError("No valid x0 source specified. Use --x0fromfile, --nullx0, --goodGuessX0, or --MCx0.")

if args.freezeAllNuisances:
    frozen_mask = np.zeros(LOSS.nTheta + LOSS.nBeta, dtype=bool)
    frozen_mask[LOSS.nBeta:] = True
    frozen_vals = np.zeros(LOSS.nTheta)
    freezemode_str = 'all'
else:
    frozen_mask = None
    frozen_vals = None
    freezemode_str = 'no'

#make unique tag based on system time
import time
unique_tag = time.strftime("%Y_%m_%d-%H_%M_%S")
resultfolder = os.path.join(reco_folder, loss_name, 'RUN_%s' % unique_tag)
os.makedirs(resultfolder, exist_ok=False)

configdict = {
    'method' : args.method,
    'method_kwargs' : method_kwargs,
    'run2d' : args.run2d,
    'recoerr_mode' : recoerr_mode_str,
    'x0mode' : x0mode_str,
    'freezeMode' : freezemode_str,
    'cpt_interval' : args.checkpoint_interval,
    'rescale' : args.rescale,
}

if frozen_mask is not None:
    configdict['frozen_mask'] = frozen_mask.tolist()
if frozen_vals is not None:
    configdict['frozen_vals'] = frozen_vals.tolist()

ioutil.wrapped_write_json(os.path.join(resultfolder, 'config.json'), configdict)

if args.rescale:
    if args.run2d:
        if "INVCOV" in recoerrpath:
            covpath = recoerrpath.replace("INVCOV", 'COV')
        elif "EIGINV" in recoerrpath:
            covpath = recoerrpath.replace("EIGINV", "EIG")
        else:
            raise ValueError("Cannot determine covariance path from recoerr path: %s" % recoerrpath)
        cov = ioutil.wrapped_read_np(covpath)
        sigma = np.sqrt(np.diagonal(cov))
    else:
        sigma = recoerr

    sigma[sigma==0] = 1
    sigma[~np.isfinite(sigma)] = 1

    reco = reco/sigma

    if args.run2d:
        recoerr = np.diag(sigma) @ recoerr @ np.diag(sigma)
    else:
        recoerr = recoerr / sigma
        
    A = np.einsum('i,j->ij', 1/sigma, reco*sigma)
    LOSS.transfer0 *= A
    for i in range(LOSS.transferVariations.shape[0]):
        LOSS.transferVariations[i] *= A

    puregen = (1 - LOSS.gamma0) * x0
    purereco = LOSS.transfer0 @ puregen
    pred = (1 + LOSS.rho0) * purereco


res = minimizer.run_minimization(LOSS, reco, recoerr, 
                                 run2d = args.run2d,
                                 method=args.method,
                                 device=args.device,
                                 x0=x0,
                                 frozen_mask = frozen_mask,
                                 frozen_vals = frozen_vals,
                                 cpt_interval=args.checkpoint_interval,
                                 logpath=resultfolder,
                                 rescaled=args.rescale,
                                 **method_kwargs)


if args.rescale:
    res, reco, recoerr, x0 = res
    reco = reco * sigma
    if args.run2d:
        recoerr = np.diag(1/sigma) @ recoerr @ np.diag(1/sigma)
    else:
        recoerr = recoerr * sigma

    res = (res, reco, recoerr, x0)

minimizer.write_minimization_result(*res, destination=os.path.join(resultfolder, 'minimization_result'))
