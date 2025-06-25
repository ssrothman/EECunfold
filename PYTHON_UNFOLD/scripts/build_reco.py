import argparse
import datasets
import numpy as np

parser = argparse.ArgumentParser(description='Build EEC reco histograms')
parser.add_argument('Tag', type=str)
parser.add_argument('Sample', type=str)
parser.add_argument('--max_nboot', type=int, default=2000)
parser.add_argument('--statN', type=int, default=-1)
parser.add_argument('--statK', type=int, default=-1)
parser.add_argument('--wtsyst', type=str, default='nominal')
parser.add_argument('--objsyst', type=str, default='nominal')

parser.add_argument('--boot_per_file', type=int, default=-1)
parser.add_argument('--reweight', type=str, default=None)

args = parser.parse_args()

Hreco = datasets.get_pickled_histogram(args.Tag, args.Sample, 'EECres4tee', 
                              args.objsyst, args.wtsyst, 'reco',
                              statN=args.statN, statK=args.statK,
                              max_nboot=args.max_nboot,
                              boot_per_file=args.boot_per_file,
                              reweight=args.reweight,
                              shuffle_boots=False,
                              verbose=False)

reco = Hreco[{'bootstrap' : 0}].values(flow=True).ravel()

import unc
DY = unc.dymat(Hreco, norm=True)
if args.max_nboot > DY.shape[0]:
    raise RuntimeError(f"Not enough bootstrap samples: {DY.shape[0]} < {args.max_nboot}")

print("building cov")
DY = DY[:args.max_nboot, :]
cov = DY.T @ DY / DY.shape[0]

print("inverting cov")
import fasteigenpy as eigen
codcov = eigen.CompleteOrthogonalDecomposition(cov)
invcov = codcov.pseudoInverse()

import pickle
import os

outfile = ''
outfile += '_boot%d'%args.max_nboot
if args.statN > 0:
    outfile += '_%dstat%d'%(args.statN, args.statK)
outfile += '.pkl'
covpath = os.path.join(datasets.basedir, args.Tag, args.Sample,
                       'EECres4tee', 'CONSTRUCTED_RECO',
                       'cov' + outfile)
invcovpath = os.path.join(datasets.basedir, args.Tag, args.Sample,
                          'EECres4tee', 'CONSTRUCTED_RECO',
                          'invcov' + outfile)
recopath = os.path.join(datasets.basedir, args.Tag, args.Sample,
                        'EECres4tee', 'CONSTRUCTED_RECO',
                        'reco' + outfile)

if not os.path.exists(os.path.dirname(covpath)):
    os.makedirs(os.path.dirname(covpath))

print("Writing ", covpath)
with open(covpath, 'wb') as f:
    pickle.dump(cov, f)

print("Writing ", invcovpath)
with open(invcovpath, 'wb') as f:
    pickle.dump(invcov, f)

print("Writing ", recopath)
with open(recopath, 'wb') as f:
    pickle.dump(reco, f)
