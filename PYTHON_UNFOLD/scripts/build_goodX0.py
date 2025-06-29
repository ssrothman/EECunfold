import argparse

parser = argparse.ArgumentParser()

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

parser.add_argument('--force', action='store_true')

parser.add_argument('--testcut', action='store_true',)

args = parser.parse_args()

import os
import filenames
import datasets

reco_folder = filenames.reco_folder(
        args.RecoTag, args.RecoSample, args.reco_nboot,
        args.reco_statN, args.reco_statK, args.reco_firstN,
        args.reco_objsyst, args.reco_wtsyst, args.testcut
)

loss_folder = filenames.loss_folder(
        args.GenTag, args.GenSample, args.gen_nboot,
        args.gen_statN, args.gen_statK, args.gen_firstN,
        args.systlist, args.testcut
)
loss_name = os.path.basename(loss_folder)

destination = os.path.join(reco_folder, loss_name, 'GOODx0.npy')

if os.path.exists(destination) and not args.force:
    print(f"File {destination} already exists. Use --force to overwrite.")
    import sys
    sys.exit(0)

import loss
import ioutil

print("Reading loss...")
LOSS = loss.FullLoss()
LOSS.read_from_disk(loss_folder)

print("Reading reco...")
reco = ioutil.wrapped_read_np(os.path.join(reco_folder, 'RECO.npy'))

print("Computing good x0...")
good_x0 = LOSS.getGoodX0(reco)

os.makedirs(os.path.dirname(destination), exist_ok=True)

ioutil.wrapped_write_np(destination, good_x0)
