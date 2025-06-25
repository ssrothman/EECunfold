import argparse
from datasets import get_pickled_histogram, get_pickled_histogram_sum
import json

parser = argparse.ArgumentParser(description='Build EEC loss functions')
parser.add_argument('Tag', type=str)
parser.add_argument('Sample', type=str)

parser.add_argument('--max_nboot', type=int, default=2000)

parser.add_argument('--statN', type=int, default=-1)
parser.add_argument('--statK', type=int, default=-1)

parser.add_argument('--boot_per_file', type=int, default=-1)
parser.add_argument('--reweight', type=str, default=None)

args = parser.parse_args()

import os
outfile = ''
outfile += '_boot%d'%args.max_nboot
if args.statN > 0:
    outfile += '_%dstat%d'%(args.statN, args.statK)
outfile += '.pkl'
import datasets
outpath_1D = os.path.join(datasets.basedir, args.Tag,
                          args.Sample, 'EECres4tee', 
                          'CONSTRUCTED_LOSSES', 'LOSS_1d' + outfile)
outpath_2D = os.path.join(datasets.basedir, args.Tag,
                          args.Sample, 'EECres4tee',
                          'CONSTRUCTED_LOSSES', 'LOSS_2d' + outfile)

if not os.path.exists(os.path.dirname(outpath_1D)):
    os.makedirs(os.path.dirname(outpath_1D))

if os.path.exists(outpath_1D):
    print(f"File {outpath_1D} already exists. Exiting to avoid overwriting.")
    exit(0)

hists = {
    "reco" : {},
    "unmatchedReco" : {},
    "untransferedReco" : {},
    "gen" : {},
    "unmatchedGen" : {},
    "untransferedGen" : {},
    "transfer" : {}
}

for key in hists.keys():
    for wtsyst in ['nominal', 
                   'scaleUp', 'scaleDown', 
                   'isosfUp', 'isosfDown', 
                   'idsfUp', 'idsfDown',
                   'triggersfUp', 'triggersfDown',
                   'PUUp', 'PUDown',
                   'PDFUp', 'PDFDown',
                   'aSUp', 'aSDown',
                   'prefireUp', 'prefireDown',
                   'PDFaSUp', 'PDFaSDown',
                   'ISRUp', 'ISRDown',
                   'FSRUp', 'FSRDown']:
        print(key, wtsyst)
        hists[key][wtsyst] = get_pickled_histogram(args.Tag,
                                                   args.Sample,
                                                   "EECres4tee", 
                                                   "nominal", 
                                                   wtsyst, 
                                                   key, 
                                                   statN = args.statN, 
                                                   statK = args.statK, 
                                                   max_nboot=args.max_nboot,
                                                   reweight=args.reweight,
                                                   shuffle_boots=False,
                                                   verbose=False)

    for objsyst in ['CH_UP', 'CH_DN', 
                    'JES_UP', 'JES_DN', 
                    "JER_UP", "JER_DN", 
                    "UNCLUSTERED_UP", 'UNCLUSTERED_DN',
                    'TRK_EFF']:
        if objsyst.endswith('_UP'):
            name = objsyst[:-3]+'Up'
        elif objsyst.endswith('_DN'):
            name = objsyst[:-3]+'Down'
        else:
            name = objsyst

        print(key, objsyst)
        hists[key][name] = get_pickled_histogram(args.Tag,
                                                 args.Sample, 
                                                 "EECres4tee", 
                                                 objsyst, 
                                                 "nominal", 
                                                 key, 
                                                 statN = args.statN, 
                                                 statK = args.statK, 
                                                 max_nboot=args.max_nboot,
                                                 reweight=args.reweight,
                                                 shuffle_boots=False,
                                                 verbose=False)


two_sided = ['scale', 'isosf', 'idsf', 'triggersf', 'PU', 'PDF', 'aS', 'PDFaS', 
             'ISR', 'FSR',
             'CH', 'JES', 'JER', 'UNCLUSTERED']
one_sided = ['TRK_EFF']

import minimizer
import numpy as np
import hist
from importlib import reload

print("SETUP LOSS")
LOSS_1d = minimizer.setup_loss(hists, False,
                               two_sided_systs=two_sided,
                               one_sided_systs=one_sided)
LOSS_2d = minimizer.setup_loss(hists, True,
                               two_sided_systs=two_sided,
                               one_sided_systs=one_sided)

print("Writing ", outpath_1D)
with open(outpath_1D, 'wb') as f:
    import pickle
    pickle.dump(LOSS_1d, f)

print("Writing ", outpath_2D)
with open(outpath_2D, 'wb') as f:
    import pickle
    pickle.dump(LOSS_2d, f)

#initial guess can be perfect from MC?
MCreco = hists['reco']['nominal'][{'bootstrap' : 0}].values(flow=True).ravel()
MCgen =   hists['gen']['nominal'][{'bootstrap' : 0}].values(flow=True).ravel()
MCdenom = np.where(MCreco==0, 1, MCreco)
MCx0 = MCgen / MCdenom
MCx0[MCreco == 0] = 1

outpath_x0 = os.path.join(datasets.basedir, args.Tag,
                          args.Sample, 'EECres4tee',
                          'CONSTRUCTED_LOSSES', 'MCx0' + outfile)
print("Writing ", outpath_x0)
with open(outpath_x0, 'wb') as f:
    pickle.dump(MCx0, f)

#Hreco = hists1['reco']['nominal']
#TWO STEP MINIMIZATION PROCEDURE
#res_1d, reco_1d, recoerr_1d = minimizer.run_minimization(
#        Hreco, LOSS_1d,
#        method='trust-ncg',
#        gtol = 1.0,
#        x0 = MCx0,
#        compute_hessian=True,
#        compute_inv_hess=True,
#        device='cuda:0'
#)
#res_2d, reco_2d, recoerr_2d = minimizer.run_minimization(
#        Hreco, LOSS_2d,
#        method='trust-ncg',
#        x0 = res_1d.x,
#        compute_hessian=True,
#        compute_inv_hess=True,
#        device='cuda:1'
#)
#
#import pickle
#with open("results/res1d.pkl", 'wb') as f:
#    pickle.dump(res_1d, f)
#
#with open("results/res2d.pkl", 'wb') as f:
#    pickle.dump(res_2d, f)
