import argparse

parser = argparse.ArgumentParser(description='Run the minimizer for EEC reconstruction')
parser.add_argument('RecoTag', type=str)
parser.add_argument('RecoSample', type=str)
parser.add_argument('--reco_nboot', type=int, default=2000)
parser.add_argument('--reco_statN', type=int, default=-1)
parser.add_argument('--reco_statK', type=int, default=-1)
parser.add_argument('--reco_wtsyst', type=str, default='nominal')
parser.add_argument('--reco_objsyst', type=str, default='nominal')

parser.add_argument('GenTag', type=str)
parser.add_argument('GenSample', type=str)
parser.add_argument('--gen_nboot', type=int, default=2000)
parser.add_argument('--gen_statN', type=int, default=-1)
parser.add_argument('--gen_statK', type=int, default=-1)

parser.add_argument('--run2d', action='store_true')
parser.add_argument('--x0', type=str, default=None)
parser.add_argument('--nullx0', action='store_true')

parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--gtol', type=float, default=1.0)

parser.add_argument('--compute_hess', action='store_true')
parser.add_argument('--compute_inv_hess', action='store_true')

args = parser.parse_args()

import os
import datasets
import pickle
import numpy as np

reco_suffix = ''
reco_suffix += '_boot%d'%args.reco_nboot
if args.reco_statN > 0:
    reco_suffix += '_%dstat%d'%(args.reco_statN, args.reco_statK)
reco_suffix += '.pkl'

gen_suffix = ''
gen_suffix += '_boot%d'%args.gen_nboot
if args.gen_statN > 0:
    gen_suffix += '_%dstat%d'%(args.gen_statN, args.gen_statK)
gen_suffix += '.pkl'

recopath = os.path.join(datasets.basedir, args.RecoTag, args.RecoSample, 
                        'EECres4tee', 'CONSTRUCTED_RECO', 
                        'reco' + reco_suffix)
print("Reading reco from", recopath)
with open(recopath, 'rb') as f:
    reco = pickle.load(f)

if not args.run2d:
    covpath = os.path.join(datasets.basedir, args.RecoTag, args.RecoSample, 
                            'EECres4tee', 'CONSTRUCTED_RECO', 
                            'cov' + reco_suffix)
    print("Reading covariance from", covpath)
    with open(covpath, 'rb') as f:
        cov = pickle.load(f)

    recoerr = np.sqrt(np.diag(cov))
else:
    invcovpath = os.path.join(datasets.basedir, args.RecoTag, args.RecoSample,
                              'EECres4tee', 'CONSTRUCTED_RECO',
                              'invcov' + reco_suffix)
    print("Reading inverse covariance from", invcovpath)
    with open(invcovpath, 'rb') as f:
        recoerr = pickle.load(f)

if args.nullx0:
    x0 = None
else:
    if args.x0 is None:
        x0path = os.path.join(datasets.basedir, args.GenTag, args.GenSample,
                              'EECres4tee', 'CONSTRUCTED_LOSSES',
                              'MCx0' + gen_suffix)
    else:
        x0path = args.x0

    print("Reading x0 from", x0path)
    with open(x0path, 'rb') as f:
        x0 = pickle.load(f)

loss_base = 'LOSS_2d' if args.run2d else 'LOSS_1d'
losspath = os.path.join(datasets.basedir, args.GenTag, args.GenSample,
                        'EECres4tee', 'CONSTRUCTED_LOSSES',
                        loss_base + gen_suffix)
print("Reading loss from", losspath)
with open(losspath, 'rb') as f:
    LOSS = pickle.load(f)

import minimizer
print(reco.shape, reco.dtype)
print(recoerr.shape, recoerr.dtype)
res = minimizer.run_minimization(LOSS, reco, recoerr, 
                                 method='trust-ncg',
                                 gtol=args.gtol,
                                 device=args.device,
                                 compute_hessian=args.compute_hess,
                                 compute_inv_hess=args.compute_inv_hess,
                                 x0=x0)

lossname = 'LOSS_%s_%s_%s' % ('2d' if args.run2d else '1d',
                              args.GenTag, args.GenSample)
lossname += '_boot%d' % args.gen_nboot
if args.reco_statN > 0:
    lossname += '_%dstat%d' % (args.gen_statN, args.gen_statK)

reconame = 'RECO' 
reconame += '_boot%d' % args.reco_nboot
if args.reco_statN > 0:
    reconame += '_%dstat%d' % (args.reco_statN, args.reco_statK)

outpath = os.path.join(datasets.basedir, args.RecoTag, args.RecoSample,
                       'EECres4tee', 'UNFOLDED',
                       reconame, lossname,
                       'minimization_result.pkl')
os.makedirs(os.path.dirname(outpath), exist_ok=True)
print("Writing result to", outpath)
with open(outpath, 'wb') as f:
    pickle.dump(res, f)
