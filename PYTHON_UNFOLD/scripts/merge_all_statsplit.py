import argparse

parser = argparse.ArgumentParser()

parser.add_argument("Runtag", type=str)
parser.add_argument("Skimmer", type=str)
parser.add_argument('original_statN', type=int)

parser.add_argument("--objsyst", type=str, nargs='+',
                    default=['nominal'])
parser.add_argument("--wtsyst", type=str, nargs='+',
                    default=['nominal'])

parser.add_argument("--what", type=str, nargs='+',
                    default=['directcov_reco', 'directcov_gen'])

parser.add_argument('--sample', type=str, nargs='+',
                    default=[
                        'Pythia_HT-0to70',
                        'Pythia_HT-70to100',
                        'Pythia_HT-100to200',
                        'Pythia_HT-200to400',
                        'Pythia_HT-400to600',
                        'Pythia_HT-600to800',
                        'Pythia_HT-800to1200',
                        'Pythia_HT-1200to2500',
                        'Pythia_HT-2500toInf',
                    ],)

parser.add_argument('--statN', type=int, nargs='+', required=True)
parser.add_argument('--statK', type=int, nargs='+', required=True)

parser.add_argument('--firstN', type=int, default=-1)
parser.add_argument('--nboot', type=int, default=0)

parser.add_argument('--boot_per_file', type=int, default=-1)

parser.add_argument('--reweight', type=str, default=None)
parser.add_argument('--r123type', type=str, default=None)

parser.add_argument('--oldbinning', action='store_true')

parser.add_argument('--force', action='store_true',
                    help='Force overwrite of existing output file')

parser.add_argument('-j', '--jobs', type=int, default=1,)

args = parser.parse_args()

max1 = max(len(args.statN), len(args.statK))
if args.statN is not None and len(args.statN) != max1:
    if len(args.statN) == 1:
        args.statN = args.statN * max1
    else:
        raise ValueError("statN and statK must match in length")
if args.statK is not None and len(args.statK) != max1:
    if len(args.statK) == 1:
        args.statK = args.statK * max1
    else:
        raise ValueError("statN and statK must match in length")

import os
import itertools
import subprocess
import multiprocessing as mp

def make_command(rungtag, sample, skimmer, 
                 objsyst, wtsyst, what, 
                 original_statN, statN, statK):
    cmd = [
        'python', 'scripts/merge_statsplit.py',
        rungtag, sample, skimmer,
        objsyst, wtsyst, what,
        str(original_statN),
        '--statN', str(statN),
        '--statK', str(statK),
        '--firstN', str(args.firstN),
        '--nboot', str(args.nboot),
        '--boot_per_file', str(args.boot_per_file),
        '--reweight', args.reweight if args.reweight else '',
        '--r123type', args.r123type if args.r123type else '',
    ] 
    if args.oldbinning:
        cmd.append('--oldbinning')
    if args.force:
        cmd.append('--force')
    return cmd

if __name__ == '__main__':
    def run_command(command):
        subprocess.run(command, check=True)

    systargs = []
    for objsyst in args.objsyst:
        systargs.append((objsyst, 'nominal'))
    for wtsyst in args.wtsyst:
        systargs.append(('nominal', wtsyst))

    commands = []
    for sample in args.sample:
        for syst in systargs:
            for what in args.what:
                for statN, statK in zip(args.statN, args.statK):
                    commands.append(make_command(
                        args.Runtag, sample, args.Skimmer,
                        syst[0], syst[1], what,
                        args.original_statN, statN, statK
                    ))

    print("Total commands to run:", len(commands))

    from tqdm import tqdm
    with mp.Pool(args.jobs) as pool:
        list(tqdm(pool.imap(run_command, commands), total=len(commands), desc="Merging histograms"))

