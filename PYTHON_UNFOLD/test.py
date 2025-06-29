import datasets

Hreco = datasets.get_pickled_histogram("Apr_23_2025", "Pythia_inclusive", 
                                       "EECres4tee", "nominal", "nominal",
                                       "reco", statN = 2, statK = 1,
                                       boot_per_file = 500, 
                                       reweight='Pythia_Zkinweight')

HunmatchedReco = datasets.get_pickled_histogram("Apr_23_2025", "Pythia_inclusive", 
                                       "EECres4tee", "nominal", "nominal",
                                       "unmatchedReco", statN = 2, statK = 1,
                                       boot_per_file = 500, 
                                       reweight='Pythia_Zkinweight')

HuntransferedReco = datasets.get_pickled_histogram("Apr_23_2025", "Pythia_inclusive", 
                                       "EECres4tee", "nominal", "nominal",
                                       "untransferedReco", statN = 2, statK = 1,
                                       boot_per_file = 500, 
                                       reweight='Pythia_Zkinweight')

Hgen = datasets.get_pickled_histogram("Apr_23_2025", "Pythia_inclusive", 
                                       "EECres4tee", "nominal", "nominal",
                                       "gen", statN = 2, statK = 1,
                                       boot_per_file = 500, 
                                       reweight='Pythia_Zkinweight')

HunmatchedGen = datasets.get_pickled_histogram("Apr_23_2025", "Pythia_inclusive", 
                                       "EECres4tee", "nominal", "nominal",
                                       "unmatchedGen", statN = 2, statK = 1,
                                       boot_per_file = 500, 
                                       reweight='Pythia_Zkinweight')

HuntransferedGen = datasets.get_pickled_histogram("Apr_23_2025", "Pythia_inclusive", 
                                       "EECres4tee", "nominal", "nominal",
                                       "untransferedGen", statN = 2, statK = 1,
                                       boot_per_file = 500, 
                                       reweight='Pythia_Zkinweight')

HrecoBkg = HunmatchedReco + HuntransferedReco
HgenBkg = HunmatchedGen + HuntransferedGen
HrecoPure = Hreco - HrecoBkg
HgenPure = Hgen - HgenBkg

reco= Hreco[{'bootstrap': 0}].values(flow=True).ravel()
gen = Hgen[{'bootstrap': 0}].values(flow=True).ravel()
recoBkg = HrecoBkg[{'bootstrap': 0}].values(flow=True).ravel()
genBkg = HgenBkg[{'bootstrap': 0}].values(flow=True).ravel()
recoPure = HrecoPure[{'bootstrap': 0}].values(flow=True).ravel()
genPure = HgenPure[{'bootstrap': 0}].values(flow=True).ravel()

import pickle
with open("/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Pythia_inclusive/EECres4tee/CONSTRUCTED_LOSSES/LOSS_2d_boot2000_2stat1.pkl", 'rb') as f:
    LOSS = pickle.load(f)

with open("/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Pythia_inclusive/EECres4tee/CONSTRUCTED_LOSSES/MCx0_boot2000_2stat1.pkl", 'rb') as f:
    MCx0 = pickle.load(f)
