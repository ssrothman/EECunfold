import argparse

parser = argparse.ArgumentParser(description='Run the minimizer for EEC reconstruction')
parser.add_argument('RecoTag', type=str)
parser.add_argument('RecoSample', type=str)
parser.add_argument('--reco_nboot', type=int, default=2000)
parser.add_argument('--reco_statN', type=int, default=-1)
parser.add_argument('--reco_statK', type=int, default=-1)
parser.add_argument('--reco_wtsyst', type=str, default='nominal')
parser.add_argument('--reco_objsyst', type=str, default='nominal')

parser.add_argument('GenTag', type=str)
parser.add_argument('GenSample', type=str)
parser.add_argument('--gen_nboot', type=int, default=2000)
parser.add_argument('--gen_statN', type=int, default=-1)
parser.add_argument('--gen_statK', type=int, default=-1)

parser.add_argument('--run2d', action='store_true')

mutually_exclusive = parser.add_mutually_exclusive_group(required=False)
mutually_exclusive.add_argument('--statonly', action='store_true')
mutually_exclusive.add_argument('--conditionOne', type=int, default=None)
mutually_exclusive.add_argument('--conditionRange', nargs=2, type=int, default=None)

args = parser.parse_args()

import os
import datasets
import pickle
import numpy as np

lossname = 'LOSS_%s_%s_%s' % ('2d' if args.run2d else '1d',
                              args.GenTag, args.GenSample)
lossname += '_boot%d' % args.gen_nboot
if args.reco_statN > 0:
    lossname += '_%dstat%d' % (args.gen_statN, args.gen_statK)

reconame = 'RECO' 
reconame += '_boot%d' % args.reco_nboot
if args.reco_statN > 0:
    reconame += '_%dstat%d' % (args.reco_statN, args.reco_statK)

respath = os.path.join(datasets.basedir, args.RecoTag, args.RecoSample,
                       'EECres4tee', 'UNFOLDED',
                       reconame, lossname,
                       'minimization_result.pkl')

print("Reading minimization result from", respath)
with open(respath, 'rb') as f:
    res_tuple = pickle.load(f)

reco = res_tuple[1]
res = res_tuple[0]

import minimizer

Htemplate = datasets.get_pickled_histogram(args.RecoTag, args.RecoSample,
                                           'EECres4tee',
                                           'nominal', 'nominal',
                                           'reco',
                                           statN=args.reco_statN,
                                           statK=args.reco_statK,
                                           max_nboot=0,
                                           reweight=None)


x = res.x
invhess = res.invhess
import statistics

syststart = reco.shape[0]
systend = x.shape[0]
Nsyst = systend - syststart
if args.statonly:
    print("Conditioning out all systematics...")
    x, invhess = statistics.condition(x, invhess, 
                                      syststart, systend,
                                      np.zeros(Nsyst))
    print("\tx shape:", x.shape)
    print("\tinvhess shape:", invhess.shape)
elif args.conditionOne is not None:
    print("Conditioning out systematic at index %d"%(args.conditionOne))
    x, invhess = statistics.condition(x, invhess, 
                                      syststart+args.conditionOne, 
                                      syststart+args.conditionOne + 1,
                                      np.zeros(1))
    print("\tx shape:", x.shape)
    print("\tinvhess shape:", invhess.shape)
elif args.conditionRange is not None:
    print("Conditioning out systematics from indices %d to %d"%(args.conditionRange[0], args.conditionRange[1]))
    x, invhess = statistics.condition(x, invhess, 
                                      syststart+args.conditionRange[0], 
                                      syststart+args.conditionRange[1],
                                      np.zeros(args.conditionRange[1] - args.conditionRange[0]))
    print("\tx shape:", x.shape)
    print("\tinvhess shape:", invhess.shape)
else:
    print("Not conditioning out any systematics.")
    print("\tx shape:", x.shape)
    print("\tinvhess shape:", invhess.shape)

syststart = reco.shape[0]
systend = x.shape[0]
print("Marginalizing out the rest of the systematics.")
x, invhess = statistics.marginalize(x, invhess, syststart, systend)
print("\tx shape:", x.shape)
print("\tinvhess shape:", invhess.shape)

outname = 'Hunf'
if args.statonly:
    outname += '_statonly'
elif args.conditionOne is not None:
    outname += '_cond%d' % args.conditionOne
elif args.conditionRange is not None:
    outname += '_cond%d-%d' % (args.conditionRange[0], args.conditionRange[1])
outname += '.pkl'

import minimizer
minimizer.dump_result(x, invhess, reco, Htemplate, 2000,
                      os.path.join(datasets.basedir,
                                   args.RecoTag,
                                   args.RecoSample,
                                   'EECres4tee', 
                                   'UNFOLDED',
                                   reconame, lossname,
                                   outname))
