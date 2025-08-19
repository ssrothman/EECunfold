import argparse

parser = argparse.ArgumentParser()

parser.add_argument("Runtag")
parser.add_argument("Skimmer")

#systematics
parser.add_argument('--skipNominal', action='store_true')
parser.add_argument('--objsysts', type=str, nargs='*', 
                    default=['JES_UP', 'JES_DN', 
                             'JER_UP', 'JER_DN', 
                             'UNCLUSTERED_UP', 'UNCLUSTERED_DN',
                             'TRK_EFF',
                             'CH_UP', 'CH_DN'])
parser.add_argument('--wtsysts', type=str, nargs='*', 
                    default=[#'idsfUp', 'idsfDown', 
                             'aSUp', 'aSDown',
                             #'isosfUp', 'isosfDown', 
                             #'triggersfUp', 'triggersfDown',
                             #'prefireUp', 'prefireDown',
                             'PDFaSUp', 'PDFaSDown',
                             'scaleUp', 'scaleDown',
                             'PUUp', 'PUDown', 
                             'PDFUp', 'PDFDown',
                             'ISRDown', 'ISRUp',
                             'FSRDown', 'FSRUp'])

#histograms
parser.add_argument('--whats', type=str, nargs='+',
                    default=['reco', 'unmatchedReco', 'untransferedReco',
                             'gen', 'unmatchedGen','untransferedGen',
                             'transfer'])

#genuine options
parser.add_argument('--statN', type=int, default=-1)
parser.add_argument('--statK', type=int, default=-1)
parser.add_argument('--firstN', type=int, default=-1)

parser.add_argument('--boot_per_file', type=int, default=-1)
parser.add_argument('--max_nboot', type=int, default=-1)
parser.add_argument('--reweight', type=str, default=None)

parser.add_argument('--outputtag', type=str, default='Pythia_HTsum')

parser.add_argument('--oldbinning', action='store_true')

parser.add_argument('-j', '--jobs', type=int, default=1)

args= parser.parse_args()

import subprocess
import shlex

def make_command(objsyst, wtsyst, what):
    command = [
        'python', 'scripts/build_HTsum_hist.py',
        args.Runtag, args.Skimmer,
        objsyst, wtsyst, what,
        '--statN', str(args.statN),
        '--statK', str(args.statK),
        '--firstN', str(args.firstN),
        '--boot_per_file', str(args.boot_per_file),
        '--max_nboot', str(args.max_nboot),
        '--outputtag', args.outputtag,
        '--mute'
    ]
    if args.reweight is not None:
        command += ['--reweight', args.reweight]
    if args.oldbinning:
        command.append('--oldbinning')
    return command

# setup parallel execution
if __name__ == "__main__":
    from multiprocessing import Pool

    def run_command(command):
        subprocess.run(command, check=True)

    from tqdm import tqdm
    with Pool(args.jobs) as pool:
        commands = []
        for thewhat in args.whats:
            if not args.skipNominal:
                commands.append(make_command('nominal', 'nominal', thewhat))
            for objsyst in args.objsysts:
                commands.append(make_command(objsyst, 'nominal', thewhat))
            for wtsyst in args.wtsysts:
                commands.append(make_command('nominal', wtsyst, thewhat))

        results = list(tqdm(pool.imap(run_command, commands), total=len(commands), desc="Building HTsum histograms"))

