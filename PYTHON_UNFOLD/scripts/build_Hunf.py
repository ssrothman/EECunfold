import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('Rundir', type=str)

parser.add_argument("--out_nboot", type=int, default=15000)

mutually_exclusive = parser.add_mutually_exclusive_group(required=False)
mutually_exclusive.add_argument('--statonly', action='store_true')
mutually_exclusive.add_argument('--conditionOne', type=int, default=None)
mutually_exclusive.add_argument('--conditionRange', nargs=2, type=int, default=None)

parser.add_argument('--force', action='store_true')

parser.add_argument('--clipLowestN', type=int, default=0)
parser.add_argument('--forcePositive', action='store_true')

parser.add_argument('--clip_wrt_corr', action='store_true')

args = parser.parse_args()

if args.Rundir[-1] == '/':
    args.Rundir = args.Rundir[:-1]

import os

outname = 'Hunf_boot%d' % args.out_nboot
if args.statonly:
    outname += '_statonly'
elif args.conditionOne is not None:
    outname += '_cond%d' % args.conditionOne
elif args.conditionRange is not None:
    outname += '_cond%d-%d' % (args.conditionRange[0], args.conditionRange[1])

clipname = 'clip%d' % args.clipLowestN
if args.forcePositive:
    clipname += '_forcePos'
if args.clip_wrt_corr:
    clipname += '_clipCorr'

outname += '_' + clipname + '.npy'
resultpath = os.path.join(args.Rundir, 'minimization_result', outname)

if os.path.exists(resultpath) and not args.force:
    print(f"File {resultpath} already exists. Use --force to overwrite.")
    import sys
    sys.exit(0)

import filenames
import datasets
import minimizer
import ioutil
import hist
import indexing

res = minimizer.read_minimization_result(
    os.path.join(args.Rundir, 'minimization_result'),
)

x = res[0].x
reco = res[1]

Hinv = ioutil.wrapped_read_np(
    os.path.join(args.Rundir, 'minimization_result', 'HESS_EIGINV_%s.npy' % clipname)
)
L = ioutil.wrapped_read_np(
    os.path.join(args.Rundir, 'minimization_result', 'HESS_EIGINV_L_%s.npy' % clipname)
)

import statutil

if args.statonly:
    raise NotImplementedError()
elif args.conditionOne is not None:
    raise NotImplementedError()
elif args.conditionRange is not None:
    raise NotImplementedError()
else:
    print("Not conditioning out any systematics.")
    print("\tx shape:", x.shape)
    print("\tHinv shape:", Hinv.shape)
    print("\tL shape:", L.shape)

print("Marginalizing out the rest of the systematics.")
x = x[:len(reco)]
Hinv = Hinv[:len(reco), :len(reco)]
L = L[:len(reco), :len(reco)]
print("\tx shape:", x.shape)
print("\tHinv shape:", Hinv.shape)
print("\tL shape:", L.shape)

import minimizer
minimizer.dump_result(x, L, reco, args.out_nboot, resultpath)
