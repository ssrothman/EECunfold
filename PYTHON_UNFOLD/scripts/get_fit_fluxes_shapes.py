import argparse

parser = argparse.ArgumentParser(description='Run the minimizer for EEC reconstruction')
parser.add_argument('Rundir', type=str)

parser.add_argument('--axes', type=str, nargs='+', required=True)

parser.add_argument('--force', action='store_true')

args = parser.parse_args()

if args.Rundir.endswith('/'):
    args.Rundir = args.Rundir[:-1]

import os
axstr = '-'.join(args.axes)

unf_fluxpath = os.path.join(
    args.Rundir, 'minimization_result', 'UNFOLDED_FLUXES_%s.npy' % axstr
)
unf_shapepath = os.path.join(
    args.Rundir, 'minimization_result', 'UNFOLDED_SHAPES_%s.npy' % axstr
)
unf_fluxbinningpath = os.path.join(
    args.Rundir, 'minimization_result', 'UNFOLDED_FLUXBINNING_%s.json' % axstr
)

covunf_fluxflux_path = os.path.join(
    args.Rundir, 'minimization_result', 'COV_UNFOLDED_FLUXES-FLUXES_%s.npy' % axstr
)
covunf_fluxshape_path = os.path.join(
    args.Rundir, 'minimization_result', 'COV_UNFOLDED_FLUXES-SHAPES_%s.npy' % axstr
)
covunf_systflux_path = os.path.join(
    args.Rundir, 'minimization_result', 'COV_UNFOLDED_SYSTEMATIC-FLUXES_%s.npy' % axstr
)
covunf_shapeshape_path = os.path.join(
    args.Rundir, 'minimization_result', 'COV_UNFOLDED_SHAPES-SHAPES_%s.npy' % axstr
)
covunf_systshape_path = os.path.join(
    args.Rundir, 'minimization_result', 'COV_UNFOLDED_SYSTEMATIC-SHAPES_%s.npy' % axstr
)
covunf_systsyst_path = os.path.join(
    args.Rundir, 'minimization_result', 'COV_UNFOLDED_SYSTEMATIC-SYSTEMATIC_%s.npy' % axstr
)

fwd_fluxpath = os.path.join(
    args.Rundir, 'minimization_result', 'FORWARD_FLUXES_%s.npy' % axstr
)
fwd_shapepath = os.path.join(
    args.Rundir, 'minimization_result', 'FORWARD_SHAPES_%s.npy' % axstr
)
fwd_fluxbinningpath = os.path.join(
    args.Rundir, 'minimization_result', 'FORWARD_FLUXBINNING_%s.json' % axstr
)

if os.path.exists(unf_fluxpath) and os.path.exists(unf_shapepath) and os.path.exists(unf_fluxbinningpath) and \
    os.path.exists(fwd_fluxpath) and os.path.exists(fwd_shapepath) and os.path.exists(fwd_fluxbinningpath) and \
    os.path.exists(covunf_fluxflux_path) and os.path.exists(covunf_fluxshape_path) and \
    os.path.exists(covunf_systflux_path) and os.path.exists(covunf_shapeshape_path) and \
    os.path.exists(covunf_systshape_path) and os.path.exists(covunf_systsyst_path) and \
    not args.force:

    print('Fluxes and shapes already exist, skipping.')
    import sys
    sys.exit(0)

import indexing
import datasets
import filenames
import ioutil
import statutil

lossname = os.path.basename(os.path.dirname(args.Rundir))
skimmer = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(args.Rundir)))))
losstag, losssample, _, _, _, _, _, _, _, _, _ = filenames.parse_loss_name(lossname)
losspath = os.path.join(datasets.basedir, losstag, losssample, 
                        skimmer, 'CONSTRUCTED_LOSSES', 
                        lossname)

binning = indexing.GenRecoBinning()
binning.load_from_file(os.path.join(
    losspath, 'Binning.json'
))

unf = ioutil.wrapped_read_np(os.path.join(
    args.Rundir, 'minimization_result', 'UNFOLDED.npy'
))
covunf = ioutil.wrapped_read_np(os.path.join(
    args.Rundir, 'minimization_result', 'COV_UNFOLDED_SYST.npy'
))

unf_fluxes, unf_shapes, unf_covflux, unf_covshapes, unf_covfluxshape, unf_covtflux, unf_covtshapes, unf_fluxbinning = statutil.flux_and_shape_covariance(
    unf, covunf[:unf.shape[0], :unf.shape[0]], covunf[unf.shape[0]:, :unf.shape[0]],
    binning.genbinning, args.axes
)

ioutil.wrapped_write_np(unf_fluxpath, unf_fluxes)
ioutil.wrapped_write_np(unf_shapepath, unf_shapes)

ioutil.wrapped_write_np(covunf_fluxflux_path, unf_covflux)
ioutil.wrapped_write_np(covunf_shapeshape_path, unf_covshapes)
ioutil.wrapped_write_np(covunf_fluxshape_path, unf_covfluxshape)

ioutil.wrapped_write_np(covunf_systflux_path, unf_covtflux)
ioutil.wrapped_write_np(covunf_systshape_path, unf_covtshapes)
ioutil.wrapped_write_np(covunf_systsyst_path, covunf[unf.shape[0]:, unf.shape[0]:])

unf_fluxbinning.dump_to_file(unf_fluxbinningpath)

fwd = ioutil.wrapped_read_np(os.path.join(
    args.Rundir, 'minimization_result', 'FORWARD.npy'
))
fwd_fluxes, fwd_shapes, fwd_fluxbinning = binning.genbinning.get_fluxes_shapes(
    fwd, args.axes    
)
ioutil.wrapped_write_np(fwd_fluxpath, fwd_fluxes)
ioutil.wrapped_write_np(fwd_shapepath, fwd_shapes)
fwd_fluxbinning.dump_to_file(fwd_fluxbinningpath)
