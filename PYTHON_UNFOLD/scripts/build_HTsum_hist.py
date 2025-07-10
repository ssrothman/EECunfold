import argparse

parser = argparse.ArgumentParser()

parser.add_argument("Runtag")
parser.add_argument("Skimmer")
parser.add_argument("Objsyst")
parser.add_argument("Wtsyst")
parser.add_argument("what")

parser.add_argument('--statN', type=int, default=-1)
parser.add_argument('--statK', type=int, default=-1)
parser.add_argument('--firstN', type=int, default=-1)

parser.add_argument('--boot_per_file', type=int, default=-1)
parser.add_argument('--max_nboot', type=int, default=-1)

parser.add_argument('--reweight', type=str, default=None)
parser.add_argument('--r123type', type=str, default=None)

parser.add_argument('--outputtag', type=str, default='Pythia_HTsum')

parser.add_argument('--oldbinning', action='store_true')

parser.add_argument('--mute', action='store_true',
                    help='Mute stdout')

args = parser.parse_args()

if args.mute:
    import sys
    import os
    sys.stdout = open(os.devnull, 'w')

import json

with open("config/datasets.json", 'rb') as f:
    datasets_config = json.load(f)

dsets = datasets_config['StacksMC']['HT']['dsets']
xsecs = [datasets_config['DatasetsMC'][dset]['xsec'] for dset in dsets]

import filenames
import datasets
import hist

H = None
total_nboot = args.max_nboot
for dset, xsec in zip(dsets, xsecs):
    numevt = datasets.get_counts(args.Runtag, dset)
    samplewt = xsec * 1000 / numevt 
    if 'directcov' in args.what:
        samplewt = samplewt * samplewt

    Hnext = filenames.get_full_hist(
        args.Runtag, dset, args.boot_per_file,
        args.statN, args.statK, args.firstN,
        args.Objsyst, args.Wtsyst, args.what,
        args.reweight, args.r123type,
        max_nboot = total_nboot,
        from_bkp=args.oldbinning,
        silent=args.mute
    ) * samplewt
    if type(Hnext) is hist.Hist:
        print(dset, Hnext[{'bootstrap' : 0}].sum(flow=True))
    else:
        print(dset, Hnext.sum())

    if H is None:
        H = Hnext 
    else:
        if type(Hnext) is hist.Hist:
            if H.axes['bootstrap'].size < Hnext.axes['bootstrap'].size:
                Hnext = Hnext[{'bootstrap' : slice(None, H.axes['bootstrap'].size)}]
            elif H.axes['bootstrap'].size > Hnext.axes['bootstrap'].size:
                H = H[{'bootstrap' : slice(None, Hnext.axes['bootstrap'].size)}]

        H = H + Hnext

    if type(Hnext) is hist.Hist:
        if total_nboot > 0:
            total_nboot = min(total_nboot, Hnext.axes['bootstrap'].size - 1)
        else:
            total_nboot = Hnext.axes['bootstrap'].size - 1

        print("\tsum so far:", H[{'bootstrap' : 0}].sum(flow=True))
    else:
        print('\t sum so far:', H.sum())

import os
outfile = '%s_%s_%s'%(args.what, args.Objsyst, args.Wtsyst)

if type(H) is hist.Hist:
    nboot = H.axes['bootstrap'].size - 1
else:
    nboot = 0

if nboot > 0:
    outfile += '_boot%d' % nboot
if args.statN > 0:
    outfile += '_%dstat%d' % (args.statN, args.statK)
if args.firstN > 0:
    outfile += '_first%d' % args.firstN
if args.reweight is not None:
    outfile += '_%s' % args.reweight
if args.r123type is not None:
    outfile += '_%s' % args.r123type

outfile += "_HTSUM.pkl"

if type(H) is hist.Hist:
    print("FINAL SUM", H[{'bootstrap' : 0}].sum(flow=True))
else:
    print("FINAL SUM", H.sum())

if args.oldbinning:
    output_path = os.path.join(datasets.basedir, args.Runtag, 
                               args.outputtag, args.Skimmer, 
                               'hists_0to0', 'hists_bkp',
                               outfile)
else:
    output_path = os.path.join(datasets.basedir, args.Runtag, 
                               args.outputtag, args.Skimmer, 
                               'hists_0to0', args.Objsyst,
                               outfile)
os.makedirs(os.path.dirname(output_path), exist_ok=True)
import ioutil
ioutil.wrapped_write_pickle(output_path, H)
