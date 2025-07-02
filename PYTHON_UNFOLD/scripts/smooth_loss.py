import argparse

parser = argparse.ArgumentParser(description='Build EEC loss functions')
parser.add_argument('Tag', type=str)
parser.add_argument('Sample', type=str)

parser.add_argument('--nboot', type=int, default=-1)

parser.add_argument('--statN', type=int, default=-1)
parser.add_argument('--statK', type=int, default=-1)
parser.add_argument('--firstN', type=int, default=-1)

parser.add_argument('--systlist', type=str, nargs='*',
                    default=['scale', 'isosf', 'idsf', 'triggersf',
                             'PU', 'PDF', 'aS', 'PDFaS',
                             'ISR', 'FSR',
                             'CH', 'JES', 'JER', 'UNCLUSTERED',
                             'TRK_EFF'])

parser.add_argument('--projectAxes', type=str, nargs='*', default=None)

parser.add_argument('--force', action='store_true')

args = parser.parse_args()

import filenames
import os

loss_folder = filenames.loss_folder(
    args.Tag, args.Sample, args.nboot,
    args.statN, args.statK, args.firstN,
    args.systlist, args.projectAxes, False
)

if args.nboot < 0:
    loss_name = os.path.basename(loss_folder)
    import re
    m = re.search(r'nboot(\d+)', loss_name)
    if m:
        args.nboot = int(m.group(1))
    else:
        args.nboot = 0

destination = filenames.loss_folder(
    args.Tag, args.Sample, args.nboot,
    args.statN, args.statK, args.firstN,
    args.systlist, args.projectAxes, True
)
if os.path.exists(destination) and not args.force:
    print(f"Folder {destination} already exists. Use --force to overwrite.")
    import sys
    sys.exit(0)

import loss
LOSS = loss.FullLoss()
LOSS.read_from_disk(loss_folder)

LOSS = loss.smooth_the_loss(LOSS)
LOSS.write_to_disk(destination)
