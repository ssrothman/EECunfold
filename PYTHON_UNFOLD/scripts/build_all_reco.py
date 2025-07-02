import argparse
import fasteigenpy as eigen

parser = argparse.ArgumentParser(description='Build EEC reco histograms')
parser.add_argument('Tag', type=str)
parser.add_argument('Sample', type=str)
parser.add_argument('--nboot', type=int, nargs='+', default=[-1])
parser.add_argument('--statN', type=int, nargs='+', default=[-1])
parser.add_argument('--statK', type=int, nargs='+', default=[-1])
parser.add_argument('--firstN', type=int, nargs='+', default=[-1])
parser.add_argument('--wtsyst', type=str, nargs='+', default=['nominal'])
parser.add_argument('--objsyst', type=str, nargs='+', default=['nominal'])

parser.add_argument('--boot_per_file', type=int, nargs='+', default=[-1])
parser.add_argument('--reweight', type=str, default=None)

parser.add_argument('--force', action='store_true')
parser.add_argument('--projectAxes', type=str, nargs='*', default=None)


parser.add_argument('--clip_to_zero', type=float, default=1e-20)

args = parser.parse_args()

import subprocess

def get_command(nboot, statN, statK, firstN, objsyst, wtsyst):
    command = [
        'python', 'scripts/build_reco.py',
        args.Tag, args.Sample,
        '--nboot', str(nboot),
        '--statN', str(statN),
        '--statK', str(statK),
        '--firstN', str(firstN),
        '--wtsyst', wtsyst,
        '--objsyst', objsyst,
        '--boot_per_file', *[str(b) for b in args.boot_per_file],
    ]
    if args.reweight is not None:
        command += ['--reweight', args.reweight]
    if args.force:
        command.append('--force')
    if args.projectAxes is not None:
        command += ['--projectAxes'] + list(args.projectAxes)
    return command

def get_command_eig(nboot, statN, steatK, firstN, objsyst, wtsyst):
    command = [
        'python', 'scripts/compute_cov_eigendecomposition.py',
        args.Tag, args.Sample,
        '--nboot', str(nboot),
        '--statN', str(statN),
        '--statK', str(statK),
        '--firstN', str(firstN),
        '--wtsyst', wtsyst,
        '--objsyst', objsyst,
        '--clip_to_zero', str(args.clip_to_zero),
    ] 
    if args.force:
        command.append('--force')
    if args.projectAxes is not None:
        command += ['--projectAxes'] + list(args.projectAxes)
    return command

maxlen = max(len(args.statK), len(args.statN), len(args.firstN))
if len(args.statK) == 1:
    args.statK = args.statK * maxlen
if len(args.statN) == 1:
    args.statN = args.statN * maxlen
if len(args.firstN) == 1:
    args.firstN = args.firstN * maxlen
if len(args.statK) != len(args.statN) or len(args.statN) != len(args.firstN):
    raise ValueError("statK, statN, and firstN must have the same length or be scalars.")

maxlen = max(len(args.wtsyst), len(args.objsyst))
if len(args.wtsyst) == 1:
    args.wtsyst = args.wtsyst * maxlen
if len(args.objsyst) == 1:
    args.objsyst = args.objsyst * maxlen
if len(args.wtsyst) != len(args.objsyst):
    raise ValueError("wtsyst and objsyst must have the same length or be scalars.")

for statK, statN, firstN in zip(args.statK, args.statN, args.firstN):
    for objsyst, wtsyst in zip(args.objsyst, args.wtsyst):
        for nboot in args.nboot:
            print("Building reco for nboot=%d, statN=%d, statK=%d, firstN=%d, objsyst=%s, wtsyst=%s" % (
                nboot, statN, statK, firstN, objsyst, wtsyst))
            command = get_command(nboot, statN, statK, firstN, objsyst, wtsyst)
            subprocess.run(command, check=True)
            command_eig = get_command_eig(nboot, statN, statK, firstN, objsyst, wtsyst)
            subprocess.run(command_eig, check=True)
