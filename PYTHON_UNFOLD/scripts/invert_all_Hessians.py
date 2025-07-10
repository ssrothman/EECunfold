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

parser.add_argument('--projectAxes', type=str, nargs='*', default=None)
parser.add_argument('--rebinning', type=str, default=None)

parser.add_argument('--clipLowestN', type=int, default=0)
parser.add_argument('--forcePositive', action='store_true')

parser.add_argument('--clip_wrt_corr', action='store_true')

parser.add_argument('--oldbinning', action='store_true',)

parser.add_argument('--smoothed', action='store_true',)
parser.add_argument('--force', action='store_true')

args = parser.parse_args()

import filenames
import os

reco_folder = filenames.reco_folder(
    args.RecoTag, args.RecoSample, args.reco_nboot,
    args.reco_statN, args.reco_statK, args.reco_firstN,
    args.reco_objsyst, args.reco_wtsyst, 
    args.projectAxes, args.rebinning,
)
loss_folder = filenames.loss_folder(
    args.GenTag, args.GenSample, args.gen_nboot,
    args.gen_statN, args.gen_statK, args.gen_firstN,
    args.systlist, 
    args.projectAxes, args.rebinning,
    args.smoothed
)
if args.oldbinning:
    loss_folder += '_oldbinning'
    reco_folder += '_oldbinning'

loss_name = os.path.basename(loss_folder)

base_folder = os.path.join(reco_folder, loss_name)
runs = os.listdir(base_folder)
runs = filter(lambda x: x.startswith('RUN'), runs)

def get_command(run):
    command = ['python', 'scripts/invert_Hessian.py', 
               os.path.join(base_folder, run),
               '--clipLowestN', str(args.clipLowestN)]
    if args.forcePositive:
        command.append('--forcePositive')
    if args.clip_wrt_corr:
        command.append('--clip_wrt_corr')
    if args.force:
        command.append('--force')
    return command


import subprocess
for run in runs:
    subprocess.run(get_command(run))
