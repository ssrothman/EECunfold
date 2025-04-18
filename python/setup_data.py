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

HrecoPure = Hreco - HunmatchedReco - HuntransferedReco
HgenPure = Hgen - HunmatchedGen - HuntransferedGen

print("Sum of transfer:", Htransfer.sum(flow=True))
print("Sum of recopure:", HrecoPure[{'bootstrap' : 0}].sum(flow=True))

transfer = Htransfer.values(flow=True)
denom = HgenPure[{'bootstrap' : 0}].values(flow=True)
denom[denom == 0] = 1
transfer = transfer / denom[None, :]

np.save("data/recoPure.npy", HrecoPure.values(flow=True))
np.save("data/genPure.npy", HgenPure.values(flow=True))
np.save("data/transfer.npy", transfer)
