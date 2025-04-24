import numpy as np
import datasets
import os
import pickle

Htemplate = datasets.get_pickled_histogram(
        "Apr_01_2025", "Pythia_inclusive", "EECres4tee",
        "nominal", "nominal", 
        "reco"
)

options = os.listdir("data")
print(options)

for option in options:
    if not option.endswith(".npy"):
        continue

    print(option)
    thename = os.path.splitext(option)[0]
    print(thename)

    vals = np.load(os.path.join("data", option))
    vals[vals<=0] = 0
    Hvals = Htemplate.copy().reset()
    Hvals += vals

    with open(os.path.join("data", thename+".pkl"), "wb") as f:
        pickle.dump(Hvals, f)
