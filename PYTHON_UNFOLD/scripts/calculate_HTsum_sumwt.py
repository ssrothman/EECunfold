import argparse

parser = argparse.ArgumentParser()

parser.add_argument('Runtag', type=str)
parser.add_argument("Skimmer", type=str)

parser.add_argument('--force', action='store_true')

parser.add_argument('--outtag', type=str, default='Pythia_HTsum')

args = parser.parse_args()

import json
import datasets
import ioutil
import os

outpath = os.path.join(
        datasets.basedir, args.Runtag, args.outtag,
        args.Skimmer, 'SUMWT.pickle'
)
if os.path.exists(outpath) and not args.force:
    print(f"File {outpath} already exists. Use --force to overwrite.")
    import sys
    sys.exit(0)

with open("config/datasets.json", 'rb') as f:
    datasets_config = json.load(f)

dsets = datasets_config['StacksMC']['HT']['dsets']
xsecs = [datasets_config['DatasetsMC'][dset]['xsec'] for dset in dsets]

sumwt = 0
sumwt2 = 0

for dset, xsec in zip(dsets, xsecs):
    numevt = datasets.get_counts(args.Runtag, dset)
    samplewt = xsec * 1000 / numevt 

    this_wts_path = os.path.join(
            datasets.basedir, args.Runtag, dset, 
            args.Skimmer, 'SUMWT.pickle'
    )
    this_wts = ioutil.wrapped_read_pickle(this_wts_path)

    sumwt += this_wts['sumwt'] * samplewt
    sumwt2 += this_wts['sumwt2'] * (samplewt ** 2)

ioutil.wrapped_write_pickle(outpath, {'sumwt': sumwt, 'sumwt2': sumwt2})
