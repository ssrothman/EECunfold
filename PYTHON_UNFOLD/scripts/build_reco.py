import argparse

parser = argparse.ArgumentParser(description='Build EEC reco histograms')
parser.add_argument('Tag', type=str)
parser.add_argument('Sample', type=str)
parser.add_argument('Skimmer', type=str)
parser.add_argument('--nboot', type=int, default=-1)
parser.add_argument('--statN', type=int, default=-1)
parser.add_argument('--statK', type=int, default=-1)
parser.add_argument('--firstN', type=int, default=-1)
parser.add_argument('--wtsyst', type=str, default='nominal')
parser.add_argument('--objsyst', type=str, default='nominal')

parser.add_argument('--boot_per_file', type=int, default=-1)
parser.add_argument('--reweight', type=str, default=None)
parser.add_argument('--r123type', type=str, default=None)

parser.add_argument('--force', action='store_true')

parser.add_argument('--projectAxes', type=str, nargs='*', default=None)

parser.add_argument('--oldbinning', action='store_true',)

parser.add_argument('--rebinning', type=str, default=None)

parser.add_argument('--what', type=str, default='reco')

parser.add_argument('--getcov', action='store_true',)
parser.add_argument('--invertcov', action='store_true')

args = parser.parse_args()

import fasteigenpy as eigen
import filenames
import datasets
import numpy as np
import os
import ioutil
import hist

capswhat = args.what.upper()

Hreco = filenames.get_full_hist(
    args.Tag, args.Sample, args.Skimmer, args.boot_per_file,
    args.statN, args.statK, args.firstN,
    args.objsyst, args.wtsyst, args.what,
    args.reweight, args.r123type,
    max_nboot=args.nboot,
    from_bkp=args.oldbinning,
)

if args.nboot >= 0:
    Hreco = Hreco[{'bootstrap' : slice(None, args.nboot+1)}]

actual_nboot = Hreco.axes['bootstrap'].size - 1

recofolder = filenames.reco_folder(
        args.Tag, args.Sample, args.Skimmer, actual_nboot,
        args.statN, args.statK, args.firstN,
        args.objsyst, args.wtsyst, 
        args.projectAxes, args.rebinning, args.what
)
if args.oldbinning:
    recofolder += '_oldbinning'

if os.path.exists(recofolder) and not args.force:
    print(f"Folder {recofolder} already exists. Use --force to overwrite.")
    import sys
    sys.exit(0)

os.makedirs(recofolder, exist_ok=True)

import indexing
RecoBinning = indexing.Binning()
RecoBinning.setup_from_histogram(Hreco[{'bootstrap' : 0}])
recovalues = Hreco.values(flow=True).reshape(Hreco.axes['bootstrap'].size, -1)
if args.rebinning is not None:
    print("Rebinning...")
    prebinned_sum = np.sum(recovalues, axis=1)
    recovalues, RecoBinning = RecoBinning.rebin(
            recovalues.T, 
            os.path.join('rebinnings', args.rebinning + '.json')
    )
    recovalues = recovalues.T
    print("\trebinned shape: ", recovalues.shape)
    postbinned_sum = np.sum(recovalues, axis=1)
    if not np.allclose(prebinned_sum, postbinned_sum):
        raise ValueError("Rebinning changed the total sum of the histogram, which is unexpected.")

if args.projectAxes is not None:
    preproject_sum = np.sum(recovalues, axis=1)
    axes_to_project = [ax for ax in RecoBinning.axis_names if ax not in args.projectAxes]
    for ax in axes_to_project:
        print("projecting out ", ax)
        recovalues, RecoBinning = RecoBinning.project_out(
            recovalues.T, ax,
        )
        recovalues = recovalues.T
        print("\tprojected shape: ", recovalues.shape)
    postproject_sum = np.sum(recovalues, axis=1)
    if not np.allclose(preproject_sum, postproject_sum):
        raise ValueError("Projection changed the total sum of the histogram, which is unexpected.")

ioutil.wrapped_write_np(os.path.join(recofolder, '%s.npy'%capswhat), recovalues[0])
RecoBinning.dump_to_file(os.path.join(recofolder, 'Binning.json'))

if args.getcov:
    import subprocess
    command = [
        'python', 'scripts/cov_from_direct.py',
        args.Tag, args.Sample, args.Skimmer,
        '--statN', str(args.statN),
        '--statK', str(args.statK),
        '--firstN', str(args.firstN),
        '--nboot', str(actual_nboot),
        '--objsyst', args.objsyst,
        '--wtsyst', args.wtsyst,
        '--what', args.what,
    ]
    if args.reweight is not None:
        command += ['--reweight', args.reweight]
    if args.r123type is not None:
        command += ['--r123type', args.r123type]
    if args.force:
        command.append('--force')
    if args.projectAxes:
        command += ['--projectAxes'] + args.projectAxes
    if args.rebinning is not None:
        command += ['--rebinning', args.rebinning]
    if args.oldbinning:
        command.append('--oldbinning')
    subprocess.run(command, check=True)

if args.invertcov:
    import subprocess
    command = [
        'python', 'scripts/compute_cov_eigendecomposition.py',
        args.Tag, args.Sample, args.Skimmer,
        '--statN', str(args.statN),
        '--statK', str(args.statK),
        '--nboot', str(actual_nboot),
        '--objsyst', args.objsyst,
        '--wtsyst', args.wtsyst,
        '--what', args.what,
    ]
    if args.force:
        command.append('--force')
    if args.projectAxes:
        command += ['--projectAxes'] + args.projectAxes
    if args.rebinning is not None:
        command += ['--rebinning', args.rebinning]
    if args.oldbinning:
        command.append('--oldbinning')
    subprocess.run(command, check=True)
