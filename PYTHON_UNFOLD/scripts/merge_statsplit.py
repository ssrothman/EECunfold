import argparse

parser = argparse.ArgumentParser()

parser.add_argument("Runtag")
parser.add_argument('Sample')
parser.add_argument("Skimmer")
parser.add_argument("Objsyst")
parser.add_argument("Wtsyst")
parser.add_argument("what")
parser.add_argument('original_statN', type=int)

parser.add_argument('--statN', type=int, default=-1)
parser.add_argument('--statK', type=int, default=-1)
parser.add_argument('--firstN', type=int, default=-1)
parser.add_argument('--nboot', type=int, default=0)

parser.add_argument('--boot_per_file', type=int, default=-1)

parser.add_argument('--reweight', type=str, default=None)
parser.add_argument('--r123type', type=str, default=None)

parser.add_argument('--oldbinning', action='store_true')

parser.add_argument('--force', action='store_true',
                    help='Force overwrite of existing output file')

args = parser.parse_args()

import os
import datasets

outfile = '%s_%s_%s'%(args.what, args.Objsyst, args.Wtsyst)

if args.nboot > 0:
    outfile += '_nboot%d'%args.nboot
if args.statN > 0:
    outfile += '_%dstat%d'%(args.statN, args.statK)
if args.firstN > 0:
    outfile += '_firstN%d'%args.firstN
if args.reweight is not None:
    outfile += '_reweight%s'%args.reweight
if args.r123type is not None:
    outfile += '_r123type%s'%args.r123type
outfile += '.pkl'

basepath = os.path.join(datasets.basedir, args.Runtag, 
                        args.Sample, args.Skimmer)
options = os.scandir(basepath)
options = list(filter(lambda x: x.is_dir() and x.name.startswith('hists_'), options))
if len(options) == 0:
    raise ValueError("No histogram directories found in %s" % basepath)
if len(options) > 1:
    raise ValueError("Multiple histogram directories found in %s: %s" % (
        basepath, ', '.join([x.name for x in options]))
    )
hists_dir = options[0].name

if args.oldbinning:
    output_path = os.path.join(datasets.basedir, args.Runtag, 
                               args.Sample, args.Skimmer, 
                               hists_dir, 'hists_bkp',
                               outfile)
else:
    output_path = os.path.join(datasets.basedir, args.Runtag, 
                               args.Sample, args.Skimmer, 
                               hists_dir, args.Objsyst,
                               outfile)

if os.path.exists(output_path) and not args.force:
    print(f"File {output_path} already exists. Use --force to overwrite.")
    import sys
    sys.exit(0)

import hist
import filenames
Hacc = None

if args.statN > 0 and args.original_statN % args.statN != 0:
    raise ValueError(
        f"original_statN {args.original_statN} must be divisible by statN {args.statN}."
    )
if args.original_statN < args.statN:
    raise ValueError(
        f"original_statN {args.original_statN} must be greater than or equal to statN {args.statN}."
    )

for original_statK in range(args.statK, args.original_statN, args.statN):
    print("Trying to get histogram for statK=%d..." % original_statK)
    Hnext = filenames.get_full_hist(
        args.Runtag, args.Sample, args.boot_per_file,
        args.original_statN, original_statK, args.firstN,
        args.Objsyst, args.Wtsyst, args.what,
        args.reweight, args.r123type,
        max_nboot = args.nboot,
        from_bkp=args.oldbinning,
        silent=True
    )
    if Hacc is None:
        Hacc = Hnext
    else:
        Hacc = Hacc + Hnext
    if type(Hacc) is hist.Hist:
        print("\trunning total...", Hacc.sum(flow=True))
    else:
        print("\trunning total...", Hacc.sum())

os.makedirs(os.path.dirname(output_path), exist_ok=True)
import ioutil
ioutil.wrapped_write_pickle(output_path, Hacc)
