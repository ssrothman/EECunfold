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

parser.add_argument('--MCstatstep', type=int, default=100)

args = parser.parse_args()

import os
import datasets
import pickle
import numpy as np
import sys
import shlex
import subprocess

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
res = res_tuple[0]
reco = res_tuple[1]
Nsyst = res.x.shape[0] - reco.shape[0]

thecommand = f'scripts/build_Hunf.py {args.RecoTag} {args.RecoSample} ' \
        f'--reco_nboot {args.reco_nboot} ' \
        f'--reco_statN {args.reco_statN} ' \
        f'--reco_statK {args.reco_statK} ' \
        f'--reco_wtsyst {args.reco_wtsyst} ' \
        f'--reco_objsyst {args.reco_objsyst} ' \
        f'{args.GenTag} {args.GenSample} ' \
        f'--gen_nboot {args.gen_nboot} ' \
        f'--gen_statN {args.gen_statN} ' \
        f'--gen_statK {args.gen_statK} ' 
if args.run2d:
    thecommand += '--run2d '

python = sys.executable
thecommand = python + ' ' + thecommand

subprocs = []
print("Running command:", thecommand)
if args.gen_nboot == -1:
    NMCstat = 4500
else:
    NMCstat = args.gen_nboot

for i in range(NMCstat, Nsyst):
    print(f'syst {i}')
    subprocs.append(subprocess.run(shlex.split(thecommand + 
        f' --conditionOne {i}')))
        #stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))

for i in range(args.MCstatstep, NMCstat, args.MCstatstep):
    print(f"MCstat 0 to {i}")
    subprocs.append(subprocess.run(shlex.split(thecommand + 
        f' --conditionRange 0 {i}')))
        #stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))

print("Statonly")
subprocs.append(subprocess.run(shlex.split(thecommand + ' --statonly ')))
        #stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
print("Nominal")
subprocs.append(subprocess.run(shlex.split(thecommand)))
        #stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))

print("Waiting for completion...")
for sp in subprocs:
    sp.wait()
    if sp.returncode != 0:
        print("Error in subprocess:", sp)
        sys.exit(sp.returncode)
    else:
        print("Subprocess completed successfully:", sp)
