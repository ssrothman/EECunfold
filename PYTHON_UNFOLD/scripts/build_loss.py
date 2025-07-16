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

parser.add_argument('--projectAxes', type=str, nargs='*', default=None)

parser.add_argument('--boot_per_file', type=int, default=-1, nargs='+')
parser.add_argument('--reweight', type=str, default=None)
parser.add_argument('--r123type', type=str, default=None)

parser.add_argument('--force', action='store_true')

parser.add_argument('--oldbinning', action='store_true',)

parser.add_argument('--rebinning', type=str, default=None)

parser.add_argument('--which_objsysts', type=str, nargs='*',
                    default=['JER', 'JES', 'UNCLUSTERED', 'CH', 'TRK_EFF'],
                    help='List of which systematic names are objsysts.'
                    'Should probably never be actually passed a non-default value')

args = parser.parse_args()

import os
import numpy as np
import filenames
import ioutil
import hist
import indexing

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
    hists[what]['nominal'] = filenames.get_full_hist(
            args.Tag, args.Sample, args.boot_per_file,
            args.statN, args.statK, args.firstN, 
            'nominal', 'nominal', what,
            args.reweight, args.r123type,
            max_nboot=args.nboot,
            from_bkp=args.oldbinning,
            silent=True
    )
            
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
    args.two_sided + args.one_sided, 
    args.projectAxes, args.rebinning,
    False)

if args.oldbinning:
    outpath += '_oldbinning'

if os.path.exists(outpath) and not args.force:
    print(f"Folder {outpath} already exists. Use --force to overwrite.")
    import sys
    sys.exit(0)
os.makedirs(outpath, exist_ok=True)

binning = indexing.Binning()
binning.setup_from_histogram(hists['reco']['nominal'][{'bootstrap' : 0}])

MCreco = hists['reco']['nominal'][{'bootstrap' : 0}].values(flow=True).ravel()
MCgen = hists['gen']['nominal'][{'bootstrap' : 0}].values(flow=True).ravel()
MCdenom = np.where(MCreco==0, 1, MCreco)
MCx0 = MCgen / MCdenom
MCx0[MCreco == 0] = 1

if args.rebinning is not None:
    print("Rebinning reco")
    rebinning_path = os.path.join('rebinnings', args.rebinning + '.json')
    MCx0, rebinning = binning.rebin(
            MCx0, rebinning_path
    )

    thebinning = rebinning
else:
    rebinning_path = None
    thebinning = binning

if args.projectAxes is not None:
    axes_to_project = [ax for ax in binning.axis_names if ax not in args.projectAxes]
    for ax in axes_to_project:
        print("projecting out", ax)
        MCx0, thebinning = thebinning.project_out(
            MCx0.T, ax,
        )
        MCx0 = MCx0.T
        print("\tprojected shape:", MCx0.shape)
else:
    axes_to_project = None

thebinning.dump_to_file(os.path.join(outpath, 'Binning.json'))
ioutil.wrapped_write_np(os.path.join(outpath, 'MCx0.npy'), MCx0.ravel())

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
            hists[what][wtsyst + updnname] = filenames.get_full_hist(
                args.Tag, args.Sample, args.boot_per_file,
                args.statN, args.statK, args.firstN,
                'nominal', wtsyst + updn, what,
                args.reweight, args.r123type,
                max_nboot=args.nboot,
                from_bkp=args.oldbinning,
                silent=True
            )

    for objsyst in objsysts_to_load:
        UPDN = [''] if objsyst in args.one_sided else ['_UP', '_DN']
        UPDNnames = [''] if objsyst in args.one_sided else ['Up', 'Down']
        for updn, updnname in zip(UPDN, UPDNnames):
            print("Loading %s %s"%(what, objsyst+updn))
            hists[what][objsyst + updnname] = filenames.get_full_hist(
                args.Tag, args.Sample, args.boot_per_file,
                args.statN, args.statK, args.firstN,
                objsyst + updn, 'nominal', what,
                args.reweight, args.r123type,
                max_nboot=args.nboot,
                from_bkp=args.oldbinning,
                silent=True
            )

import minimizer
import numpy as np
import hist
from importlib import reload
import pickle

print("SETUP LOSS")
LOSS = minimizer.setup_loss(hists, 
                            two_sided_systs=args.two_sided,
                            one_sided_systs=args.one_sided,
                            basebinning = binning,
                            rebinning_path = rebinning_path,
                            axes_to_project=axes_to_project)

LOSS.write_to_disk(outpath)

