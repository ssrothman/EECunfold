import subprocess

samples = ["Pythia_inclusive", "Herwig_inclusive"]
reweights = ['Pythia_Zkinweight', 'Herwig_Zkinweight']
statKs = [0, 1, -1]
statNs = [2, 2, -1]
nboots = [500, 1000, 2000, 2500, 3000, 3500, 4000, -1]

for sample, reweight in zip(samples, reweights):
    for statN, statK in zip(statNs, statKs):
        for nboot in nboots:
            q = subprocess.run(['python', 'scripts/build_reco.py', 
                                'Apr_23_2025', sample, 
                            '--statN', str(statN), '--statK', str(statK),
                            '--max_nboot', str(nboot), '--reweight', reweight])
            if q.returncode != 0:
                print(f"Error occurred while processing {sample} with statN={statN}, statK={statK}, nboot={nboot}")
                import sys
                sys.exit(q.returncode)
