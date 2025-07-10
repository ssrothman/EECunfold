import argparse

parser = argparse.ArgumentParser()

parser.add_argument('Runtag', type=str)
parser.add_argument('Tag', type=str)
parser.add_argument("Skimmer", type=str)

parser.add_argument('--force', action='store_true')

args = parser.parse_args()

import datasets
import os
import numpy as np

outpath = os.path.join(
        datasets.basedir, args.Runtag, args.Tag,
        args.Skimmer, 'SUMWT.pickle'
)
if os.path.exists(outpath) and not args.force:
    print(f"File {outpath} already exists. Use --force to overwrite.")
    import sys
    sys.exit(0)

dset = datasets.get_dataset(
        args.Runtag, args.Tag, args.Skimmer, 
        'nominal', 'reco'
)
print("Reading table...")
sumwt = 0 
sumwt2 = 0
from tqdm import tqdm
for batch in tqdm(dset.to_batches(columns=['wt', 'evtwt_nominal'])):
    wt = batch['wt'].to_numpy() * batch['evtwt_nominal'].to_numpy()
    sumwt += np.sum(wt)
    sumwt2 += np.sum(np.square(wt))

import ioutil
ioutil.wrapped_write_pickle(outpath, {'sumwt' : sumwt, 'sumwt2' : sumwt2})

