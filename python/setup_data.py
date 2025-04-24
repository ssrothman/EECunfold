import datasets
import numpy as np

Hreco = datasets.get_pickled_histogram(
        "Apr_01_2025", "Pythia_inclusive", "EECres4tee",
        "nominal", "nominal", 
        "reco"
)
HunmatchedReco = datasets.get_pickled_histogram(
        "Apr_01_2025", "Pythia_inclusive", "EECres4tee",
        "nominal", "nominal", 
        "unmatchedReco"
)
HuntransferedReco = datasets.get_pickled_histogram(
        "Apr_01_2025", "Pythia_inclusive", "EECres4tee",
        "nominal", "nominal", 
        "untransferedReco"
)

Hgen = datasets.get_pickled_histogram(
        "Apr_01_2025", "Pythia_inclusive", "EECres4tee",
        "nominal", "nominal", 
        "gen"
)
HunmatchedGen = datasets.get_pickled_histogram(
        "Apr_01_2025", "Pythia_inclusive", "EECres4tee",
        "nominal", "nominal", 
        "unmatchedGen"
)
HuntransferedGen = datasets.get_pickled_histogram(
        "Apr_01_2025", "Pythia_inclusive", "EECres4tee",
        "nominal", "nominal", 
        "untransferedGen"
)

Htransfer = datasets.get_pickled_histogram(
        "Apr_01_2025", "Pythia_inclusive", "EECres4tee",
        "nominal", "nominal", 
        "transfer"
)

axes = ['bootstrap', 'pt', 'R', 'r', 'c']
transferaxes = ['pt_reco', 'R_reco', 'r_reco', 'c_reco',
                'pt_gen', 'R_gen', 'r_gen', 'c_gen']

#force order of dimensions to be as we want
Hreco = Hreco.project(*axes)
HunmatchedReco = HunmatchedReco.project(*axes)
HuntransferedReco = HuntransferedReco.project(*axes)

Hgen = Hgen.project(*axes)
HunmatchedGen = HunmatchedGen.project(*axes)
HuntransferedGen = HuntransferedGen.project(*axes)

Htransfer = Htransfer.project(*transferaxes)

import os
os.makedirs("pythia_data", exist_ok=True)

np.save("pythia_data/reco.npy", np.ascontiguousarray(Hreco.values(flow=True)))
np.save("pythia_data/gen.npy",  np.ascontiguousarray(Hgen.values(flow=True)))
np.save("pythia_data/unmatchedReco.npy", np.ascontiguousarray(HunmatchedReco.values(flow=True)))
np.save("pythia_data/unmatchedGen.npy", np.ascontiguousarray(HunmatchedGen.values(flow=True)))
np.save("pythia_data/untransferedReco.npy", np.ascontiguousarray(HuntransferedReco.values(flow=True)))
np.save("pythia_data/untransferedGen.npy", np.ascontiguousarray(HuntransferedGen.values(flow=True)))
np.save("pythia_data/transfer.npy", np.ascontiguousarray(Htransfer.values(flow=True)))

Hreco = datasets.get_pickled_histogram(
        "Apr_01_2025", "Herwig_inclusive", "EECres4tee",
        "nominal", "nominal", 
        "reco"
)
HunmatchedReco = datasets.get_pickled_histogram(
        "Apr_01_2025", "Herwig_inclusive", "EECres4tee",
        "nominal", "nominal", 
        "unmatchedReco"
)
HuntransferedReco = datasets.get_pickled_histogram(
        "Apr_01_2025", "Herwig_inclusive", "EECres4tee",
        "nominal", "nominal", 
        "untransferedReco"
)

Hgen = datasets.get_pickled_histogram(
        "Apr_01_2025", "Herwig_inclusive", "EECres4tee",
        "nominal", "nominal", 
        "gen"
)
HunmatchedGen = datasets.get_pickled_histogram(
        "Apr_01_2025", "Herwig_inclusive", "EECres4tee",
        "nominal", "nominal", 
        "unmatchedGen"
)
HuntransferedGen = datasets.get_pickled_histogram(
        "Apr_01_2025", "Herwig_inclusive", "EECres4tee",
        "nominal", "nominal", 
        "untransferedGen"
)

Htransfer = datasets.get_pickled_histogram(
        "Apr_01_2025", "Herwig_inclusive", "EECres4tee",
        "nominal", "nominal", 
        "transfer"
)

axes = ['bootstrap', 'pt', 'R', 'r', 'c']
transferaxes = ['pt_reco', 'R_reco', 'r_reco', 'c_reco',
                'pt_gen', 'R_gen', 'r_gen', 'c_gen']

#force order of dimensions to be as we want
Hreco = Hreco.project(*axes)
HunmatchedReco = HunmatchedReco.project(*axes)
HuntransferedReco = HuntransferedReco.project(*axes)

Hgen = Hgen.project(*axes)
HunmatchedGen = HunmatchedGen.project(*axes)
HuntransferedGen = HuntransferedGen.project(*axes)

Htransfer = Htransfer.project(*transferaxes)

import os
os.makedirs("herwig_data", exist_ok=True)

np.save("herwig_data/reco.npy", np.ascontiguousarray(Hreco.values(flow=True)))
np.save("herwig_data/gen.npy",  np.ascontiguousarray(Hgen.values(flow=True)))
np.save("herwig_data/unmatchedReco.npy", np.ascontiguousarray(HunmatchedReco.values(flow=True)))
np.save("herwig_data/unmatchedGen.npy", np.ascontiguousarray(HunmatchedGen.values(flow=True)))
np.save("herwig_data/untransferedReco.npy", np.ascontiguousarray(HuntransferedReco.values(flow=True)))
np.save("herwig_data/untransferedGen.npy", np.ascontiguousarray(HuntransferedGen.values(flow=True)))
np.save("herwig_data/transfer.npy", np.ascontiguousarray(Htransfer.values(flow=True)))

HrecoA = datasets.get_pickled_histogram(
        "Apr_01_2025", "DATA_2018A", "EECres4tee",
        "nominal", "nominal", 
        "reco"
)
HrecoB = datasets.get_pickled_histogram(
        "Apr_01_2025", "DATA_2018B", "EECres4tee",
        "nominal", "nominal", 
        "reco"
)
HrecoC = datasets.get_pickled_histogram(
        "Apr_01_2025", "DATA_2018C", "EECres4tee",
        "nominal", "nominal", 
        "reco"
)
HrecoD = datasets.get_pickled_histogram(
        "Apr_01_2025", "DATA_2018D", "EECres4tee",
        "nominal", "nominal", 
        "reco"
)
Hreco = HrecoA + HrecoB + HrecoC + HrecoD

axes = ['bootstrap', 'pt', 'R', 'r', 'c']

#force order of dimensions to be as we want
Hreco = Hreco.project(*axes)

import os
os.makedirs("data_data", exist_ok=True)

np.save("data_data/reco.npy", np.ascontiguousarray(Hreco.values(flow=True)))
