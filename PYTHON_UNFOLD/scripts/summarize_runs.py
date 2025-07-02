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

parser.add_argument('--projectAxes', type=str, nargs='*', default=None,)
parser.add_argument('--smoothed', action='store_true',)

args = parser.parse_args()

import os
import ioutil
import filenames
import minimizer
import numpy as np

reco_folder = filenames.reco_folder(
    args.RecoTag, args.RecoSample, args.reco_nboot,
    args.reco_statN, args.reco_statK, args.reco_firstN,
    args.reco_objsyst, args.reco_wtsyst, args.projectAxes
)
loss_folder = filenames.loss_folder(
    args.GenTag, args.GenSample, args.gen_nboot,
    args.gen_statN, args.gen_statK, args.gen_firstN,
    args.systlist, args.projectAxes, args.smoothed
)
loss_name = os.path.basename(loss_folder)

base_folder = os.path.join(reco_folder, loss_name)

runs = os.listdir(base_folder)
runs = filter(lambda x: x.startswith('RUN'), runs)

print()
print("%-23s  %-6s  %-6s  %-5s  %-12s  %-10s  %-10s  %-5s" % ('RUN', 'Loss', '|grad|', 'Run2D', 'RecoErrMode', 'X0Mode', 'FreezeMode', 'Hess?'))
for run in runs:
    runconfig = ioutil.wrapped_read_json(os.path.join(base_folder, run, 'config.json'), silent=True)

    finished = os.path.exists(os.path.join(base_folder, run, 'minimization_result'))

    if finished:
        res = minimizer.read_minimization_result(os.path.join(base_folder, run, 'minimization_result'), silent=True)

        fun = "%6.3g"%res[0].fun
        gnorm = "%6.4g"%np.sum(np.square(res[0].grad))
        if 'freezeMode' not in runconfig:
            runconfig['freezeMode'] = 'no'
        elif runconfig['freezeMode'] == 'freezeAllNuisances':
            runconfig['freezeMode'] = 'all'

        has_hess = os.path.exists(os.path.join(base_folder, run, 'minimization_result', 'HESSIAN.npy'))
        has_invhess = os.path.exists(os.path.join(base_folder, run, 'minimization_result', 'INVHESS.npy'))

        if has_invhess:
            has_hess = 'inv'
        elif has_hess:
            has_hess = 'yes'
        else:
            has_hess = 'no'
        has_hess = '%-5s' % has_hess

    else:
        fun = "%-6s" % '-'
        gnorm = "%-6s" % '-'
        has_hess = '%-5s' % '-'

    print("%-23s  %s  %s  %-5s  %-12s  %-10s  %-10s  %s" % (run, fun, gnorm, runconfig['run2d'], runconfig['recoerr_mode'], runconfig['x0mode'], runconfig['freezeMode'], has_hess))
    if finished:
        Hunfs = os.listdir(os.path.join(base_folder, run, 'minimization_result'))
        Hunfs = list(filter(lambda x: x.startswith('Hunf'), Hunfs))
        if len(Hunfs) > 0:
            for Hunf in Hunfs:
                print("\t", Hunf)
        Hfwds = os.listdir(os.path.join(base_folder, run, 'minimization_result'))
        Hfwds = list(filter(lambda x: x.startswith('Hfwd'), Hfwds))
        if len(Hfwds) > 0:
            for Hfwd in Hfwds:
                print("\t", Hfwd)

print()
print("Basedir:")
print(base_folder)
