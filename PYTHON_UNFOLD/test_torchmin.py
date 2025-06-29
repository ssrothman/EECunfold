import torchmin
import torchmin.function
import torch
import loss
import pickle
import numpy as np

torch.set_default_dtype(torch.float64)

LOSS = loss.FullLoss()
LOSS.read_from_disk('/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Pythia_HTsum/EECres4tee/CONSTRUCTED_LOSSES/LOSS_Apr_23_2025_Pythia_HTsum_boot1000_2stat0')
LOSS.torch()
LOSS.set_1d()

reco = torch.ones(LOSS.nBeta)
recoerr = torch.ones_like(reco)
x0 = torch.ones(LOSS.nBeta + LOSS.nTheta)

theloss = LOSS.one_parameter_loss(reco, recoerr)

SF = torchmin.function.ScalarFunction(theloss, x0.shape, hessp=True)
SF2 = torchmin.function.ScalarFunctionPT2(theloss, x0.shape, hessp=True,
                                        torchcompile=False)
SF3 = torchmin.function.ScalarFunctionPT2(theloss, x0.shape, hessp=True, 
                                          torchcompile=True, 
                                          torchcompile_options={})

# warmup for compiled function
for i in range(10):
    SF3.closure(x0)

# benchmark performance
def time_eval(func, x):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    func(x)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)
