import numpy as np
import datasets
import os
import pickle

Htemplate_reco = datasets.get_pickled_histogram(
        "Apr_01_2025", "Pythia_inclusive", "EECres4tee",
        "nominal", "nominal", 
        "reco"
)
Htemplate_gen = datasets.get_pickled_histogram(
        "Apr_01_2025", "Pythia_inclusive", "EECres4tee",
        "nominal", "nominal", 
        "gen"
)
Htemplate_transfer = datasets.get_pickled_histogram(
        "Apr_01_2025", "Pythia_inclusive", "EECres4tee",
        "nominal", "nominal", 
        "transfer"
)

options = os.listdir("data")

for option in options:
    if not option.endswith(".npy"):
        continue

    print(option)
    thename = os.path.splitext(option)[0]
    print(thename)

    vals = np.load(os.path.join("data", option))
    vals[vals<=0] = 0
    if 'transfer' in thename:
        Hvals = Htemplate_transfer.copy().reset()
    elif 'unfolded' in thename:
        Hvals = Htemplate_gen.copy().reset()
    elif 'forward' in thename:
        Hvals = Htemplate_reco.copy().reset()
    else:
        print("Unknown type for "+thename)
        continue

    Hvals += vals

    with open(os.path.join("data", thename+".pkl"), "wb") as f:
        pickle.dump(Hvals, f)
