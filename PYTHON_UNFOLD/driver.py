from datasets import get_pickled_histogram, get_pickled_histogram_sum
import json

with open("config/datasets.json", 'rb') as f:
    datasets = json.load(f) 

#tags = [datasets['DatasetsMC'][key]['tag'] for key in datasets['DatasetsMC'] if 'HT-' in key]
#xsecs = [datasets['DatasetsMC'][key]['xsec'] for key in datasets['DatasetsMC'] if 'HT-' in key]

#HTsum = {}
#HTinc = {}

#for what in ['reco', 'unmatchedReco', 'untransferedReco', 'gen', 'unmatchedGen', 'untransferedGen', 'transfer']:
#    HTsum[what] = get_pickled_histogram_sum(tags, xsecs, 'Apr_23_2025', 'EECres4tee', 'nominal', 'nominal', what)
#    HTinc[what] = get_pickled_histogram('Apr_23_2025', 'Pythia_inclusive', 'EECres4tee', 'nominal', 'nominal', what)

sample = "Pythia_inclusive"

hists0 = {
    "reco" : {},
    "unmatchedReco" : {},
    "untransferedReco" : {},
    "gen" : {},
    "unmatchedGen" : {},
    "untransferedGen" : {},
    "transfer" : {}
}
hists1 = {
    "reco" : {},
    "unmatchedReco" : {},
    "untransferedReco" : {},
    "gen" : {},
    "unmatchedGen" : {},
    "untransferedGen" : {},
    "transfer" : {}
}
for key in hists0.keys():
    hists0[key]['nominal'] = get_pickled_histogram("Apr_23_2025", "Pythia_inclusive", "EECres4tee", "nominal", "nominal", key, statN = 2, statK = 0)
    hists1[key]['nominal'] = get_pickled_histogram("Apr_23_2025", "Pythia_inclusive", "EECres4tee", "nominal", "nominal", key, statN = 2, statK = 1)

    for systwt in ['ISR', 'FSR']:
        hists0[key][systwt] = (
            get_pickled_histogram("Apr_23_2025", "Pythia_inclusive", "EECres4tee", "nominal", systwt+"Up", key, statN = 2, statK = 0),
            get_pickled_histogram("Apr_23_2025", "Pythia_inclusive", "EECres4tee", "nominal", systwt+"Down", key, statN = 2, statK = 0)  
        )

        hists1[key][systwt] = (
            get_pickled_histogram("Apr_23_2025", "Pythia_inclusive", "EECres4tee", "nominal", systwt+"Up", key, statN = 2, statK = 1),
            get_pickled_histogram("Apr_23_2025", "Pythia_inclusive", "EECres4tee", "nominal", systwt+"Down", key, statN = 2, statK = 1)  
        )
