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

sample = "Herwig_inclusive"

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
    for wtsyst in ['nominal', 
                 'scaleUp', 'scaleDown', 
                   #'isosfUp', 'isosfDown', 
                   #'idsfUp', 'idsfDown',
                   #'triggersfUp', 'triggersfDown',
                 'PUUp', 'PUDown',
                   #'PDFUp', 'PDFDown',
                   #'aSUp', 'aSDown',
                 'PDFaSUp', 'PDFaSDown']:
        hists0[key][wtsyst] = get_pickled_histogram("Apr_23_2025", sample, "EECres4tee", "nominal", wtsyst, key, statN = 2, statK = 0)
        hists1[key][wtsyst] = get_pickled_histogram("Apr_23_2025", sample, "EECres4tee", "nominal", wtsyst, key, statN = 2, statK = 1)
    for objsyst in ['CH_UP', 'CH_DN', 
                    'JES_UP', 'JES_DN', 
                    #"JER_UP", "JER_DN", 
                    #"UNCLUSTERED_UP", 'UNCLUSTERED_DN',
                    'TRK_EFF']:
        if objsyst.endswith('_UP'):
            name = objsyst[:-3]+'Up'
        elif objsyst.endswith('_DN'):
            name = objsyst[:-3]+'Down'
        else:
            name = objsyst

        hists0[key][name] = get_pickled_histogram("Apr_23_2025", sample, "EECres4tee", objsyst, "nominal", key, statN = 2, statK = 0)
        hists1[key][name] = get_pickled_histogram("Apr_23_2025", sample, "EECres4tee", objsyst, "nominal", key, statN = 2, statK = 1)


import minimizer
import numpy as np
import hist
from importlib import reload
