import numpy as np
import unc
from scipy.optimize import minimize
from tqdm import tqdm
import torchmin
import torch
import loss
from scipy.stats import multivariate_normal
import indexing

#cut = {'pt' : slice(None,None,sum)}
#tcut = {'pt_reco' : slice(None,None,sum), 'pt_gen' : slice(None,None,sum)}

torch.set_default_dtype(torch.float64)

def get_arrs(histdict, syst, iboot, basebinning, 
             rebinning_path_reco, rebinning_path_gen, axes_to_project):

    the_cut = {}
    the_tcut = {}
    if iboot is not None:
        the_tcut['bootstrap'] = iboot
        the_cut['bootstrap'] = iboot
    else:
        the_tcut['bootstrap'] = 0

    transfer = histdict['transfer'][syst][the_tcut].values(flow=True)

    gen = histdict['gen'][syst][the_cut].values(flow=True)
    reco = histdict['reco'][syst][the_cut].values(flow=True)

    genBkg = (histdict['unmatchedGen'][syst] + histdict['untransferedGen'][syst])[the_cut].values(flow=True)
    recoBkg = (histdict['unmatchedReco'][syst] + histdict['untransferedReco'][syst])[the_cut].values(flow=True)

    if iboot is None:
        reco = reco.reshape((reco.shape[0], -1))
        gen = gen.reshape((gen.shape[0], -1))
        genBkg = genBkg.reshape((genBkg.shape[0], -1))
        recoBkg = recoBkg.reshape((recoBkg.shape[0], -1))

        recoshape = reco.shape[1]
        genshape = gen.shape[1]
    else:
        reco = reco.ravel()
        gen = gen.ravel()
        genBkg = genBkg.ravel()
        recoBkg = recoBkg.ravel()

        recoshape = reco.shape[0] 
        genshape = gen.shape[0]

    transfer = transfer.reshape((recoshape, genshape))

    print("Rebinning...")

    reco, _ = basebinning.rebin(reco.T, 'reco', rebinning_path_reco)
    recoBkg, _ = basebinning.rebin(recoBkg.T, 'reco', rebinning_path_reco)
    gen, _ = basebinning.rebin(gen.T, 'gen', rebinning_path_gen)
    genBkg, _ = basebinning.rebin(genBkg.T, 'gen', rebinning_path_gen)
    reco = reco.T
    recoBkg = recoBkg.T
    gen = gen.T
    genBkg = genBkg.T

    transfer, basebinning = basebinning.rebin_transfer2d(
        transfer, rebinning_path_reco, rebinning_path_gen
    )

    print("after rebinning, shapes are")
    print("\treco: ", reco.shape)
    print("\trecoBkg: ", recoBkg.shape)
    print("\tgen: ", gen.shape)
    print("\tgenBkg: ", genBkg.shape)
    print("\ttransfer: ", transfer.shape)
    
    if axes_to_project is not None:
        for ax in axes_to_project:
            print("projecting out", ax)

            reco = basebinning.project_out(reco.T, 'reco', ax)[0].T
            recoBkg = basebinning.project_out(recoBkg.T, 'reco', ax)[0].T
            gen = basebinning.project_out(gen.T, 'gen', ax)[0].T
            genBkg = basebinning.project_out(genBkg.T, 'gen', ax)[0].T

            transfer, basebinning = basebinning.project_out_transfer2d(transfer, ax)

    tdenom = gen - genBkg
    tdenom = np.where(tdenom==0, 1, tdenom)
    if iboot is None:
        tdenom = tdenom[0]
    transfer = transfer/tdenom[None, :]

    Gdenom = np.where(gen==0, 1, gen)
    gamma = genBkg / Gdenom

    Rdenom = np.where(reco-recoBkg==0, 1, reco-recoBkg)
    rho = recoBkg / Rdenom

    #check
    if iboot == 0:
        purefwd = transfer @ (gen - genBkg)
        print("Checking transfer: ", np.allclose(purefwd, reco - recoBkg))

    return reco, gen, rho, gamma, transfer

def setup_loss(histdict, 
               thebinning,
               Nboot=-1,
               two_sided_systs=[],
               one_sided_systs=[], 
               basebinning=None,
               rebinning_path_reco=None,
               rebinning_path_gen=None,
               axes_to_project=None):
    #nominal
    reco0, gen0, rho0, gamma0, transfer0 = get_arrs(histdict, 'nominal', 0,
                                                    basebinning, 
                                                    rebinning_path_reco,
                                                    rebinning_path_gen,
                                                    axes_to_project)

    print("RECO0", reco0.sum(), reco0.shape)
    print("\t", reco0)
    print("GEN0", gen0.sum(), gen0.shape)
    print("\t", gen0)
    print("GenBkg0", (gen0*gamma0).sum(), gamma0.shape)
    print("\t", gamma0)
    print("TRANSFER0", transfer0.sum(), transfer0.shape)
    print("\t", transfer0)

    rhoVariations = []
    gammaVariations = []
    transferVariations = []
    transferVarIndices = []

    #stat variations
    if Nboot <= 0:
        Nboot = histdict['reco']['nominal'].axes['bootstrap'].size - 1

    namedNuisances = {}

    print("Building stat templates...")
    _, _, rhoboot, gammaboot, _ = get_arrs(histdict, 'nominal', None,
                                           basebinning, 
                                           rebinning_path_reco,
                                           rebinning_path_gen,
                                           axes_to_project)
    for iboot in tqdm(range(1, Nboot+1)):
        rhoVariations.append((rhoboot[iboot] - rho0)/Nboot)
        gammaVariations.append((gammaboot[iboot] - gamma0)/Nboot)

    #syst variations
    print("Buiding two-sided systs...")
    for syst in tqdm(two_sided_systs):
        print()
        print("FOR SYST", syst)
        print()
        _, _, rho_up, gamma_up, transfer_up = get_arrs(histdict, '%sUp'%syst, 0,
                                                       basebinning, 
                                                       rebinning_path_reco,
                                                       rebinning_path_gen,
                                                       axes_to_project)
        _, _, rho_dn, gamma_dn, transfer_dn = get_arrs(histdict, '%sDown'%syst, 0,
                                                       basebinning, 
                                                       rebinning_path_reco,
                                                       rebinning_path_gen,
                                                       axes_to_project)
        rhoVariations.append(0.5*(rho_up - rho_dn))
        gammaVariations.append(0.5*(gamma_up - gamma_dn))
        transferVariations.append(0.5*(transfer_up - transfer_dn))
        transferVarIndices.append(len(rhoVariations)-1)
        namedNuisances[len(rhoVariations)-1] = syst

    print("Building one-sided systs...")
    for syst in tqdm(one_sided_systs):
        _, _, rho_up, gamma_up, transfer_up = get_arrs(histdict, syst, 0,
                                                       basebinning, 
                                                       rebinning_path_reco,
                                                       rebinning_path_gen,
                                                       axes_to_project)
        rhoVariations.append(rho_up - rho0)
        gammaVariations.append(gamma_up - gamma0)
        transferVariations.append(transfer_up - transfer0)
        transferVarIndices.append(len(rhoVariations)-1)
        namedNuisances[len(rhoVariations)-1] = syst

    rhoVariations = np.asarray(rhoVariations)
    gammaVariations = np.asarray(gammaVariations)
    transferVariations = np.asarray(transferVariations)
    transferVarIndices = np.asarray(transferVarIndices)

    print("reco0: ", reco0.shape)
    print("gen0: ", gen0.shape)
    print("gamma0: ", gamma0.shape)
    print("rho0: ", rho0.shape)
    print("transfer0: ", transfer0.shape)
    print()
    print("gammaVariations", gammaVariations.shape)
    print("rhoVariations", rhoVariations.shape)
    print("transferVariations", transferVariations.shape)
    print("transferVarIndices: ", transferVarIndices.shape)
    print("\t", transferVarIndices)
    print("namedNuisances: ")
    for key in namedNuisances:
        print("\t", key, namedNuisances[key])

    LOSS = loss.FullLoss()
    LOSS.setup(transfer0, transferVariations, 
               transferVarIndices,
               gamma0, gammaVariations, 
               rho0, rhoVariations,
               gen0, np.asarray(reco0.sum()),
               binning = thebinning,
               namedNuisances = namedNuisances)

    return LOSS

def compute_hessian(LOSS, reco, recoErr, run2d, x,
                    device='cuda',
                    frozen_mask=None,
                    frozen_vals=None,
                    fluxes_and_shapes=None):

    if fluxes_and_shapes is not None:
        fluxes, shapes, fluxbinning = LOSS.binning.genbinning.get_fluxes_shapes(
            x[:LOSS.nBeta]*LOSS.genBaseline*reco.sum()/LOSS.baselineRecoFlux,
            fluxes_and_shapes
        )

    if type(device) is str:
        device = torch.device(device)

    if run2d and len(recoErr.shape) != 2:
        raise ValueError("for run2d, recoErr must be 2d invcov array")
    if not run2d and len(recoErr.shape) != 1:
        raise ValueError("for run1d, recoErr must be 1d stderr array")

    if run2d:
        LOSS.set_2d()
    else:
        LOSS.set_1d()

    LOSS.set_rescaled(fluxes_and_shapes is not None)

    if type(x) is not torch.Tensor:
        x = torch.from_numpy(x)
    if type(reco) is not torch.Tensor:
        reco = torch.from_numpy(reco)
    if type(recoErr) is not torch.Tensor:
        recoErr = torch.from_numpy(recoErr)
    if LOSS.device == 'numpy':
        LOSS = LOSS.torch()

    if frozen_mask is not None and type(frozen_mask) is not torch.Tensor:
        frozen_mask = torch.from_numpy(frozen_mask)
    if frozen_vals is not None and type(frozen_vals) is not torch.Tensor:
        frozen_vals = torch.from_numpy(frozen_vals)

    LOSS = LOSS.to(device)
    x = x.to(device)
    reco = reco.to(device)
    recoErr = recoErr.to(device)
    if frozen_mask is not None:
        frozen_mask = frozen_mask.to(device)
        frozen_vals = frozen_vals.to(device)

    if frozen_mask is not None:
        newx = torch.zeros(LOSS.nBeta + LOSS.nTheta, dtype=x.dtype, device=device)
        newx[~frozen_mask] = x
        newx[frozen_mask] = frozen_vals
        x = newx

    if fluxes_and_shapes is None: 
        theloss = LOSS.one_parameter_loss(reco, recoErr, None, None)
        print("X sum:", x.sum().item())
    else:
        fluxes = torch.from_numpy(fluxes).to(device)
        shapes = torch.from_numpy(shapes).to(device)

        if LOSS.nTheta == 0:
            theloss = lambda a, b: LOSS.loss_from_fluxes_shapes(
                    a, b, fluxbinning, 
                    torch.tensor([], device=a.device),
                    reco, recoErr
            )
            x = (fluxes, shapes)
        else:
            theloss = lambda a, b, c: LOSS.loss_from_fluxes_shapes(
                    a, b, fluxbinning, 
                    c,
                    reco, recoErr
            )
            x = (fluxes, shapes, x[LOSS.nBeta:])

        print("Fluxes sum:", fluxes.sum().item())
        print("Shapes sum:", shapes.sum().item())
    
    result = torch.autograd.functional.hessian(theloss, x, vectorize=False)

    if fluxes_and_shapes is None:
        H = result.cpu().detach().numpy()

        return 0.5 * (H + H.T)
    else:
        if LOSS.nTheta > 0:
            NS = LOSS.nBeta
            NF = fluxes.shape[0] 
            NSF = NS+NF
            NH = NSF+LOSS.nTheta
            H = np.zeros((NH, NH), dtype=np.float64)
            H[:NS, :NS]        = result[1][1].cpu().detach().numpy()
            H[NS:NSF, :NS]     = result[0][1].cpu().detach().numpy()
            H[NSF:, :NS]       = result[2][1].cpu().detach().numpy()
            H[:NS, NS:NSF]     = result[1][0].cpu().detach().numpy()
            H[NS:NSF, NS:NSF]  = result[0][0].cpu().detach().numpy()
            H[NSF:, NS:NSF]    = result[2][0].cpu().detach().numpy()
            H[:NS, NSF:]       = result[1][2].cpu().detach().numpy()
            H[NS:NSF, NSF:]    = result[0][2].cpu().detach().numpy()
            H[NSF:, NSF:]      = result[2][2].cpu().detach().numpy()
        else:
            HS = LOSS.nBeta
            NF = fluxes.shape[0]
            NSF = HS + NF
            H = np.zeros((NSF, NSF), dtype=np.float64)
            H[:HS, :HS]        = result[1][1].cpu().detach().numpy()
            H[HS:NSF, :HS]     = result[0][1].cpu().detach().numpy()
            H[:HS, HS:NSF]     = result[1][0].cpu().detach().numpy()
            H[HS:NSF, HS:NSF]  = result[0][0].cpu().detach().numpy()

        return 0.5 * (H + H.T)

def setup_minimizer_from_run(rundir):
    import ioutil
    import os

    if rundir.endswith('/'):
        rundir = rundir[:-1]

    configdict = ioutil.wrapped_read_json(os.path.join(rundir, 'config.json'))

    lossname = os.path.basename(os.path.dirname(rundir))

    skimmer = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(rundir)))))

    import datasets
    import filenames
    losstag, losssample, _, _, _, _, _, _, _, _, _ = filenames.parse_loss_name(lossname)
    losspath = os.path.join(datasets.basedir, losstag, losssample, 
                            skimmer, 'CONSTRUCTED_LOSSES', 
                            lossname)

    import loss
    LOSS = loss.FullLoss()
    LOSS.read_from_disk(losspath)

    if os.path.exists(os.path.join(rundir, 'minimization_result')):
        completed=True
        x = read_minimization_result(os.path.join(rundir, 'minimization_result'))
    else:
        completed=False
        #find most recent checkpoint
        import os
        checkpointdir = os.path.join(rundir, 'checkpoints')
        checkpoints = os.listdir(checkpointdir)
        checkpoints = filter(lambda x: x.startswith('cpt_') and x.endswith('.npy'), checkpoints)
        cpt_ids = [int(x.split('_')[1].split('.')[0]) for x in checkpoints]
        maxid = max(cpt_ids)
        print("Loading most recent checkpoint #%d"%maxid)
        x = ioutil.wrapped_read_np(os.path.join(checkpointdir, 'cpt_%03d.npy' % maxid))
        x = (x, maxid)

    return completed, LOSS, configdict, x

def run_minimization(LOSS, reco, recoErr,
                     run2d=False,
                     method='scan', 
                     x0=None,
                     device='cuda',
                     frozen_mask=None,
                     frozen_vals=None,
                     logpath=None,
                     cpt_interval=50,
                     cpt_start=0,
                     rescaled = False,
                     **kwargs):

    if type(device) is str:
        device = torch.device(device)

    if run2d and len(recoErr.shape) != 2:
        raise ValueError("for run2d, recoErr must be 2d invcov array")
    if not run2d and len(recoErr.shape) != 1:
        raise ValueError("for run1d, recoErr must be 1d stderr array")

    if x0 is None:
        x0 = np.ones(LOSS.nBeta, dtype=np.float64)
        t0 = np.zeros(LOSS.nTheta, dtype=np.float64)
        x0 = np.concatenate((x0, t0), axis=0)
    elif len(x0) == LOSS.nBeta:
        t0 = np.zeros(LOSS.nTheta, dtype=np.float64)
        x0 = np.concatenate((x0, t0), axis=0)

    if run2d:
        LOSS.set_2d()
    else:
        LOSS.set_1d()

    LOSS.set_rescaled(rescaled)

    if type(x0) is not torch.Tensor:
        x0 = torch.from_numpy(x0)
    if type(reco) is not torch.Tensor:
        reco = torch.from_numpy(reco)
    if type(recoErr) is not torch.Tensor:
        recoErr = torch.from_numpy(recoErr)
    if LOSS.device == 'numpy':
        LOSS = LOSS.torch()

    if frozen_mask is not None and type(frozen_mask) is not torch.Tensor:
        frozen_mask = torch.from_numpy(frozen_mask)
    if frozen_vals is not None and type(frozen_vals) is not torch.Tensor:
        frozen_vals = torch.from_numpy(frozen_vals)

    LOSS = LOSS.to(device)
    x0 = x0.to(device)
    reco = reco.to(device)
    recoErr = recoErr.to(device)

    if frozen_mask is not None:
        frozen_mask = frozen_mask.to(device)
        frozen_vals = frozen_vals.to(device)

    if frozen_mask is not None:
        x0 = x0[~frozen_mask]

    if method == 'scan':
        methodlist = ['bfgs', 'l-bfgs', 'cg', 'newton-cg', 'newton-exact', 
                      'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']
    else:
        methodlist = [method]
        
    theloss_tofit = LOSS.one_parameter_loss(reco, recoErr,
                                            frozen_mask=frozen_mask,
                                            frozen_vals=frozen_vals)

    if logpath is None:
        cpt_path = None
    else:
        import os
        cpt_path = os.path.join(logpath, 'checkpoints')

    for method in methodlist:
        try:
            from time import time
            print(method)
            print("starting minimization")
            print("initial loss = %g"%theloss_tofit(x0).item())
            t0 = time()
            res = torchmin.minimize(
                    theloss_tofit, x0 = x0,
                    method = method,
                    callback = StatusCallback(theloss_tofit,
                                              cpt_interval=cpt_interval,
                                              cpt_path=cpt_path,
                                              cpt_start=cpt_start),
                    options = kwargs,
            )
            print("\tt =", time()-t0)
            print("\tSuccess: ", res.success)
            print("\tStatus: ", res.status)
            print("\tMessage: ", res.message)
            print("\tL = %g"%res.fun.cpu().detach().item())
        except Exception as e:
            print(f"Method {method} failed with error: {e}")
            print("Stack trace:")
            import traceback
            traceback.print_exc()
            continue

    res_to_npy(res)
    reco = reco.cpu().detach().numpy()
    recoErr = recoErr.cpu().detach().numpy()
    LOSS = LOSS.cpu().detach().numpy()
    x0 = x0.cpu().detach().numpy()

    res.namedNuisances = LOSS.namedNuisances

    return res, reco, recoErr, x0

def res_to_npy(res): 
    for key in res.keys():
        if type(res[key]) is torch.Tensor:
            res[key] = res[key].cpu().detach().numpy()

class StatusCallback:
    def __init__(self, lossfunc, cpt_interval=1000, cpt_start=0, cpt_path=None):
        self.lossfunc = lossfunc
        self.counter = 0
        self.cpt_interval = cpt_interval
        self.cpt_path = cpt_path
        self.cpt_start = cpt_start
        if cpt_path is not None:
            import os
            os.makedirs(cpt_path, exist_ok=True)

    def __call__(self, x):
        import os
        import ioutil

        print("LOSS:", self.lossfunc(x).item())
        self.counter += 1
        if self.cpt_path is not None and self.counter % self.cpt_interval == 0:
            print("\tcheckpointing...")
            icpt = self.counter // self.cpt_interval
            icpt += self.cpt_start
            ipath = os.path.join(self.cpt_path, 'cpt_%03d.npy' % icpt)
            ioutil.wrapped_write_np(ipath, x.cpu().detach().numpy())

def write_minimization_result(res, reco, recoErr, x0, destination):
    import ioutil
    import os
    arrays = []
    structs = []
    os.makedirs(destination, exist_ok=True)
    for key in res.keys():
        if isinstance(res[key], np.ndarray) and np.prod(res[key].shape) > 0:
            arrays.append(key)
        else:
            structs.append(key)

    for name in arrays:
        ioutil.wrapped_write_np(
            os.path.join(destination, name + '.npy'), res[name]
        )

    ioutil.wrapped_write_np(
        os.path.join(destination, 'RECO.npy'), reco
    )
    ioutil.wrapped_write_np(
        os.path.join(destination, 'RECOERR.npy'), recoErr
    )
    ioutil.wrapped_write_np(
        os.path.join(destination, 'X0.npy'), x0
    )

    structdict = {}
    for name in structs:
        structdict[name] = res[name]

    structdict['arrays'] = arrays
    ioutil.wrapped_write_json(
        os.path.join(destination, 'structs.json'), structdict
    )

def read_minimization_result(destination, silent=False):
    from collections import namedtuple
    import os
    import ioutil
    structs = ioutil.wrapped_read_json(
        os.path.join(destination, 'structs.json'),
        silent=silent
    )
    arrays = {}
    for name in structs['arrays']:
        arrays[name] = ioutil.wrapped_read_np(
            os.path.join(destination, name + '.npy'),
            silent=silent
        )

    reco = ioutil.wrapped_read_np(
        os.path.join(destination, 'RECO.npy'),
        silent=silent
    )
    recoErr = ioutil.wrapped_read_np(
        os.path.join(destination, 'RECOERR.npy'),
        silent=silent
    )
    x0 = ioutil.wrapped_read_np(
        os.path.join(destination, 'X0.npy'),
        silent=silent
    )

    res = {}
    for name in structs:
        if name == 'arrays':
            continue
        res[name] = structs[name]
    for name in arrays:
        res[name] = arrays[name]

    res = namedtuple('MinimizationResult', res.keys())(*res.values())
    return res, reco, recoErr, x0

def build_template_hist(Htemplate, nboot, ptoverflow):
    import hist
    axes = []
    for ax in Htemplate.axes:
        if ax.name == 'bootstrap':
            axes.append(hist.axis.Integer(
                0, nboot+1, 
                label='bootstrap',
                name='bootstrap',
                underflow=False, overflow=False
            ))
        elif ax.name == 'pt':
            if ptoverflow is not None:
                if ptoverflow == 'merge':
                    axes.append(hist.axis.Variable(
                        ax.edges[:-1], overflow=True, underflow=True,
                        name='pt', label='pt',
                    ))
                elif ptoverflow == 'drop':
                    axes.append(hist.axis.Variable(
                        ax.edges, underflow=True, overflow=False,
                        name='pt', label='pt',
                    ))
                else:
                    raise ValueError("ptoverflow must be 'merge' or 'drop', got %s"%ptoverflow)
            else:
                axes.append(ax)
        else:
            axes.append(ax)

    return hist.Hist(
        *axes,
        storage=hist.storage.Double(),         
    )

def dump_Hfwd(LOSS, x, invhess_L, reco, Nboot, destination,
              device='cuda'):
    import hist
    import pickle

    beta = x[:LOSS.nBeta] * reco
    theta = x[LOSS.nBeta:] 

    beta = torch.from_numpy(beta).to(device)
    theta = torch.from_numpy(theta).to(device)
    reco = torch.from_numpy(reco).to(device)
    LOSS = LOSS.torch().to(device)

    fwd = LOSS.forward(beta, theta).cpu().detach().numpy()

    result = np.zeros((Nboot+1, fwd.shape[0]), dtype=fwd.dtype)
    result[0] += fwd

    import statutil
    print("Generating toys from multivariate gaussian...")
    samples = statutil.multivariate_gaussian_rvs(x, invhess_L, Nboot)
    samples = torch.from_numpy(samples).to(device)
    from tqdm import tqdm
    for iboot in tqdm(range(Nboot)):
        beta = samples[iboot, :LOSS.nBeta] * reco
        beta[beta< 0] = 0
        theta = samples[iboot, LOSS.nBeta:]
        fwd = LOSS.forward(beta, theta).cpu().detach().numpy()
        result[iboot+1] += fwd

    import ioutil
    ioutil.wrapped_write_np(destination, result)

def dump_result(x, invhess_L, reco, Nboot, destination):
    if type(x) is torch.Tensor:
        x = x.cpu().detach().numpy()
    if type(reco) is torch.Tensor:
        reco = reco.cpu().detach().numpy()

    nominal = x*reco[None, :]

    print("Generating toys from multivariate gaussian...")
    import statutil
    samples = statutil.multivariate_gaussian_rvs(x, invhess_L, Nboot)
    samples[samples < 0] = 0  # Ensure non-negative samples

    boots = samples * reco[None, :]

    result = np.concatenate((nominal, boots), axis=0)

    import ioutil
    ioutil.wrapped_write_np(destination, result)
