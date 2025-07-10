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

parser.add_argument('--projectAxes', type=str, nargs='*', default=None,)
parser.add_argument('--rebin_r', type=int, default=1)
parser.add_argument('--rebin_c', type=int, default=1)
parser.add_argument('--ptoverflow', type=str, default=None)

mutually_exclusive = parser.add_mutually_exclusive_group(required=False)
mutually_exclusive.add_argument('--statonly', action='store_true')
mutually_exclusive.add_argument('--conditionOne', type=int, default=None)
mutually_exclusive.add_argument('--conditionRange', nargs=2, type=int, default=None)

parser.add_argument('--smoothed', action='store_true',)

parser.add_argument('--out_nboot', type=int, default=15000)
parser.add_argument('--force', action='store_true')

parser.add_argument('--clipLowestN', type=int, default=0)
parser.add_argument('--forcePositive', action='store_true')

parser.add_argument('--clip_wrt_corr', action='store_true')

args = parser.parse_args()

import filenames
import os

reco_folder = filenames.reco_folder(
    args.RecoTag, args.RecoSample, args.reco_nboot,
    args.reco_statN, args.reco_statK, args.reco_firstN,
    args.reco_objsyst, args.reco_wtsyst, 
    args.projectAxes, args.rebin_r, args.rebin_c,
    args.ptoverflow,
)
loss_folder = filenames.loss_folder(
    args.GenTag, args.GenSample, args.gen_nboot,
    args.gen_statN, args.gen_statK, args.gen_firstN,
    args.systlist, 
    args.projectAxes, args.rebin_r, args.rebin_c,
    args.ptoverflow,
    args.smoothed
)
loss_name = os.path.basename(loss_folder)

base_folder = os.path.join(reco_folder, loss_name)
runs = os.listdir(base_folder)
runs = filter(lambda x: x.startswith('RUN'), runs)

options = ['--out_nboot', str(args.out_nboot)]
if args.statonly:
    options.append('--statonly')
if args.conditionOne is not None:
    options.append('--conditionOne')
    options.append(str(args.conditionOne))
if args.conditionRange is not None:
    options.append('--conditionRange')
    options.append(str(args.conditionRange[0]))
    options.append(str(args.conditionRange[1]))

if args.force:
    options.append('--force')

options.append('--clipLowestN')
options.append(str(args.clipLowestN))
if args.forcePositive:
    options.append('--forcePositive')
if args.clip_wrt_corr:
    options.append('--clip_wrt_corr')

import subprocess
for run in runs:
    subprocess.run(['python', 'scripts/build_Hunf.py', 
                    os.path.join(base_folder, run),
                    *options])
