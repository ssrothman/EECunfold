import argparse

parser = argparse.ArgumentParser(description='Build EEC reco histograms')
parser.add_argument('Tag', type=str)
parser.add_argument('Sample', type=str)
parser.add_argument("Skimmer", type=str)
parser.add_argument('--nboot', type=int, default=-1)
parser.add_argument('--statN', type=int, default=-1)
parser.add_argument('--statK', type=int, default=-1)
parser.add_argument('--firstN', type=int, default=-1)
parser.add_argument('--wtsyst', type=str, default='nominal')
parser.add_argument('--objsyst', type=str, default='nominal')

parser.add_argument('--force', action='store_true')

parser.add_argument('--projectAxes', type=str, nargs='*', default=None)

parser.add_argument('--oldbinning', action='store_true',)

parser.add_argument('--rebinning', type=str, default=None)

parser.add_argument('--what', type=str, default='reco')

parser.add_argument('--axes', type=str, nargs='+', required=True)

args = parser.parse_args()

import filenames
import os

capswhat = args.what.upper()

recofolder = filenames.reco_folder(
        args.Tag, args.Sample, args.Skimmer,
        args.nboot,
        args.statN, args.statK, args.firstN,
        args.objsyst, args.wtsyst, 
        args.projectAxes, args.rebinning, args.what
)
if args.oldbinning:
    recofolder += '_oldbinning'

axstr = '-'.join(args.axes)
fluxpath = os.path.join(
    recofolder, 'FLUXES_%s.npy' % (axstr)
)
shapepath = os.path.join(
    recofolder, 'SHAPES_%s.npy' % (axstr)
)
fluxbinningpath = os.path.join(
    recofolder, 'FLUXBINNING_%s.json' % (axstr)
)
cov_fluxflux_path = os.path.join(
    recofolder, 'COV_FLUXES-FLUXES_%s.npy' % (axstr)
)
cov_fluxshape_path = os.path.join(
    recofolder, 'COV_FLUXES-SHAPES_%s.npy' % (axstr)
)
cov_shapeshape_path = os.path.join(
    recofolder, 'COV_SHAPES-SHAPES_%s.npy' % (axstr)
)

if os.path.exists(fluxpath) and os.path.exists(shapepath) and \
        os.path.exists(fluxbinningpath) and \
        os.path.exists(cov_fluxflux_path) and \
        os.path.exists(cov_fluxshape_path) and \
        os.path.exists(cov_shapeshape_path) and \
        not args.force:
    print("%s fluxes and shapes already exist, skipping." % capswhat)
    import sys
    sys.exit(0)

import indexing
import ioutil
import statutil

reco = ioutil.wrapped_read_np(os.path.join(
    recofolder, '%s.npy'%capswhat
))
covreco = ioutil.wrapped_read_np(
    os.path.join(recofolder, 'COV_DIRECT.npy')
)   
binning = indexing.Binning()
binning.load_from_file(
    os.path.join(recofolder, 'Binning.json')
)

fluxes, shapes, covflux, covshapes, covfluxshape, _, _, fluxbinning = statutil.flux_and_shape_covariance(
    reco, covreco, None,
    binning, args.axes
)

ioutil.wrapped_write_np(fluxpath, fluxes)
ioutil.wrapped_write_np(shapepath, shapes)
ioutil.wrapped_write_np(cov_fluxflux_path, covflux)
ioutil.wrapped_write_np(cov_fluxshape_path, covfluxshape)
ioutil.wrapped_write_np(cov_shapeshape_path, covshapes)
fluxbinning.dump_to_file(fluxbinningpath) 
