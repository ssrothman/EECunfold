import argparse

parser = argparse.ArgumentParser(description='Build EEC loss functions')
parser.add_argument('Tag', type=str)
parser.add_argument('Sample', type=str)

parser.add_argument('--nboot', type=int, default=-1)

parser.add_argument('--statN', type=int, default=-1)
parser.add_argument('--statK', type=int, default=-1)
parser.add_argument('--firstN', type=int, default=-1)

parser.add_argument('--two_sided', type=str, nargs='*',
                    default=['scale', 'isosf', 'idsf', 'triggersf',
                             'PU', 'PDF', 'aS', 'PDFaS', 
                             'ISR', 'FSR',
                             'CH', 'JES', 'JER', 'UNCLUSTERED'])
parser.add_argument('--one_sided', type=str, nargs='*',
                    default=['TRK_EFF'])

parser.add_argument('--boot_per_file', type=int, default=-1, nargs='+')
parser.add_argument('--reweight', type=str, default=None)

parser.add_argument('--force', action='store_true')

parser.add_argument('--testcut', action='store_true')

parser.add_argument('--which_objsysts', type=str, nargs='*',
                    default=['JER', 'JES', 'UNCLUSTERED', 'CH', 'TRK_EFF'],
                    help='List of which systematic names are objsysts.'
                    'Should probably never be actually passed a non-default value')

args = parser.parse_args()

import datasets
import os
import numpy as np
import ioutil

hists = {
    "reco" : {},
    "unmatchedReco" : {},
    "untransferedReco" : {},
    "gen" : {},
    "unmatchedGen" : {},
    "untransferedGen" : {},
    "transfer" : {}
}

for what in hists.keys():
    print("Loading %s %s"%(what, 'nominal'))
    hists[what]['nominal'] = datasets.get_pickled_histogram(
            args.Tag, args.Sample, 'EECres4tee', 
            'nominal', 'nominal', what, 
            statN = args.statN, statK = args.statK,
            firstN = args.firstN,
            boot_per_file=args.boot_per_file,
            max_nboot=args.nboot,
            reweight=args.reweight)

    if args.nboot >= 0 and what != 'transfer':
        if hists[what]['nominal'].axes['bootstrap'].size < args.nboot + 1:
            raise ValueError(
                f"Requested {args.nboot} bootstraps, but only "
                f"{hists[what]['nominal'].axes['bootstrap'].size - 1} available for {what}. "
                "Check your dataset and the number of bootstraps available.")

        hists[what]['nominal'] = hists[what]['nominal'][{'bootstrap' : slice(None, args.nboot+1)}]

lowest_nboot = np.inf
if args.nboot <= 0:
    for what in hists.keys():
        if what == 'transfer':
            continue
        if hists[what]['nominal'].axes['bootstrap'].size - 1 < lowest_nboot:
            lowest_nboot = hists[what]['nominal'].axes['bootstrap'].size - 1

    for what in hists.keys():
        if what == 'transfer':
            continue
        hists[what]['nominal'] = hists[what]['nominal'][{'bootstrap' : slice(None, lowest_nboot + 1)}]
    actual_nboot = lowest_nboot
else:
    actual_nboot = args.nboot

import filenames
outpath = filenames.loss_folder(
    args.Tag, args.Sample, actual_nboot,
    args.statN, args.statK, args.firstN,
    args.two_sided + args.one_sided, args.testcut)

if os.path.exists(outpath) and not args.force:
    print(f"Folder {outpath} already exists. Use --force to overwrite.")
    import sys
    sys.exit(0)
os.makedirs(outpath, exist_ok=True)

wtsysts_to_load = []
objsysts_to_load = []
for syst in args.two_sided + args.one_sided:
    if syst in args.which_objsysts:
        objsysts_to_load.append(syst)
    else:
        wtsysts_to_load.append(syst)

for what in hists.keys():
    for wtsyst in wtsysts_to_load:
        UPDN = [''] if wtsyst in args.one_sided else ['Up', 'Down']
        UPDNnames = [''] if wtsyst in args.one_sided else ['Up', 'Down']
        for updn, updnname in zip(UPDN, UPDNnames):
            print("Loading %s %s"%(what, wtsyst+updn))
            hists[what][wtsyst + updnname] = datasets.get_pickled_histogram(
                args.Tag, args.Sample, 'EECres4tee',
                'nominal', wtsyst + updn, what,
                statN = args.statN, statK = args.statK,
                boot_per_file=args.boot_per_file,
                max_nboot=0,
                firstN=args.firstN,
                reweight=args.reweight)

    for objsyst in objsysts_to_load:
        UPDN = [''] if objsyst in args.one_sided else ['_UP', '_DN']
        UPDNnames = [''] if objsyst in args.one_sided else ['Up', 'Down']
        for updn, updnname in zip(UPDN, UPDNnames):
            print("Loading %s %s"%(what, objsyst+updn))
            hists[what][objsyst + updnname] = datasets.get_pickled_histogram(
                args.Tag, args.Sample, 'EECres4tee',
                objsyst + updn, 'nominal', what,
                statN = args.statN, statK = args.statK,
                boot_per_file=args.boot_per_file,
                max_nboot=0,
                firstN=args.firstN,
                reweight=args.reweight)

import minimizer
import numpy as np
import hist
from importlib import reload
import pickle

thecut = {}
thetcut = {}
if args.testcut:
    thecut = {'pt' : slice(None,None,sum)}
    thetcut = {'pt_reco' : slice(None,None,sum), 
                'pt_gen' : slice(None,None,sum)}

print("SETUP LOSS")
LOSS = minimizer.setup_loss(hists, 
                            two_sided_systs=args.two_sided,
                            one_sided_systs=args.one_sided,
                            cut=thecut,
                            tcut=thetcut)

LOSS.write_to_disk(outpath)

#initial guess can be perfect from MC?
MCreco = hists['reco']['nominal'][{'bootstrap' : 0}][thecut].values(flow=True).ravel()
MCgen =   hists['gen']['nominal'][{'bootstrap' : 0}][thecut].values(flow=True).ravel()
MCdenom = np.where(MCreco==0, 1, MCreco)
MCx0 = MCgen / MCdenom
MCx0[MCreco == 0] = 1

ioutil.wrapped_write_np(os.path.join(outpath, 'MCx0.npy'), MCx0)
