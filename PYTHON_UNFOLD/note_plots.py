
gen_herwig_teepath = '/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Herwig_inclusive/EECres4tee/CONSTRUCTED_GEN/Apr_23_2025_Herwig_inclusive_boot0_REBINNING-gen-proposed1_nominal_nominal/'
gen_pythia_teepath = '/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Pythia_HTsum/EECres4tee/CONSTRUCTED_GEN/Apr_23_2025_Pythia_HTsum_boot0_REBINNING-gen-proposed1_nominal_nominal/'
gen_pythia_teepath_alt = '/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Pythia_HTsum/EECres4tee/CONSTRUCTED_GEN/Apr_23_2025_Pythia_HTsum_boot0_REBINNING-gen-proposed2_nominal_nominal/'

reco_herwig_teepath = '/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Herwig_inclusive/EECres4tee/CONSTRUCTED_RECO/Apr_23_2025_Herwig_inclusive_boot0_REBINNING-reco-proposed1_nominal_nominal/'
reco_pythia_teepath = '/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Pythia_HTsum/EECres4tee/CONSTRUCTED_RECO/Apr_23_2025_Pythia_HTsum_boot0_REBINNING-reco-proposed1_nominal_nominal/'
reco_pythia_teepath_alt = '/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Pythia_HTsum/EECres4tee/CONSTRUCTED_RECO/Apr_23_2025_Pythia_HTsum_boot0_REBINNING-reco-proposed4_nominal_nominal/'

losspath_tee = '/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Pythia_HTsum/EECres4tee/CONSTRUCTED_LOSSES/Apr_23_2025_Pythia_HTsum_boot0_2stat0_REBINNINGRECO-reco-proposed1_REBINNINGGEN-gen-proposed1_SYST-scale-isosf-idsf-triggersf-PU-prefire-PDFaS-ISR-FSR-CH-JES-JER-UNCLUSTERED-TRK_EFF/'
losspath_tee_alt = '/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Pythia_HTsum/EECres4tee/CONSTRUCTED_LOSSES/Apr_23_2025_Pythia_HTsum_boot0_2stat0_REBINNINGRECO-reco-proposed4_REBINNINGGEN-gen-proposed2_SYST-scale-isosf-idsf-triggersf-PU-prefire-PDFaS-ISR-FSR-CH-JES-JER-UNCLUSTERED-TRK_EFF/'

trainingpath_tee = '/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Pythia_HTsum/EECres4tee/CONSTRUCTED_RECO/Apr_23_2025_Pythia_HTsum_boot0_2stat0_REBINNING-reco-proposed1_nominal_nominal/Apr_23_2025_Pythia_HTsum_boot0_2stat1_REBINNINGRECO-reco-proposed1_REBINNINGGEN-gen-proposed1_SYST-scale-isosf-idsf-triggersf-PU-prefire-PDFaS-ISR-FSR-CH-JES-JER-UNCLUSTERED-TRK_EFF/RUN_2025_07_23-10_41_02'

gen_herwig_dipolepath = '/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Herwig_inclusive/EECres4dipole/CONSTRUCTED_GEN/Apr_23_2025_Herwig_inclusive_boot0_REBINNING-gen-proposed1_nominal_nominal/'
gen_pythia_dipolepath = '/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Pythia_HTsum/EECres4dipole/CONSTRUCTED_GEN/Apr_23_2025_Pythia_HTsum_boot0_REBINNING-gen-proposed1_nominal_nominal/'

reco_herwig_dipolepath = '/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Herwig_inclusive/EECres4dipole/CONSTRUCTED_RECO/Apr_23_2025_Herwig_inclusive_boot0_REBINNING-reco-proposed1_nominal_nominal/'
reco_pythia_dipolepath = '/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Pythia_HTsum/EECres4dipole/CONSTRUCTED_RECO/Apr_23_2025_Pythia_HTsum_boot0_REBINNING-reco-proposed1_nominal_nominal/'

losspath_dipole = '/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Pythia_HTsum/EECres4dipole/CONSTRUCTED_LOSSES/Apr_23_2025_Pythia_HTsum_boot0_2stat0_REBINNINGRECO-reco-proposed1_REBINNINGGEN-gen-proposed1_SYST-scale-isosf-idsf-triggersf-PU-prefire-PDFaS-ISR-FSR-CH-JES-JER-UNCLUSTERED-TRK_EFF/'

trainingpath_dipole = '/ceph/submit/data/group/cms/store/user/srothman/EEC/Apr_23_2025/Pythia_HTsum/EECres4dipole/CONSTRUCTED_RECO/Apr_23_2025_Pythia_HTsum_boot0_2stat0_REBINNING-reco-proposed1_nominal_nominal/Apr_23_2025_Pythia_HTsum_boot0_2stat1_REBINNINGRECO-reco-proposed1_REBINNINGGEN-gen-proposed1_SYST-scale-PU-PDFaS-ISR-FSR-CH-JES-JER-UNCLUSTERED-TRK_EFF/RUN_2025_08_04-18_19_38/'

import fasteigenpy as eigen
import indexing
import loss
import plotting.projected
import ioutil
import statutil
from importlib import reload
import os
import numpy as np

binning = indexing.GenRecoBinning()
binning.load_from_file(
    os.path.join(losspath_tee, 'Binning.json')
)

LOSS = loss.FullLoss()
LOSS.read_from_disk(losspath_tee)

LOSS_alt = loss.FullLoss()
LOSS_alt.read_from_disk(losspath_tee_alt)


gen_pythia_tee = ioutil.wrapped_read_np(
    os.path.join(gen_pythia_teepath, 'GEN.npy')
)
gen_covpythia_tee = ioutil.wrapped_read_np(
    os.path.join(gen_pythia_teepath, 'COV_DIRECT.npy')
)
gen_pythia_tee_alt = ioutil.wrapped_read_np(
    os.path.join(gen_pythia_teepath_alt, 'GEN.npy')
)
gen_covpythia_tee_alt = ioutil.wrapped_read_np(
    os.path.join(gen_pythia_teepath_alt, 'COV_DIRECT.npy')
)

gen_herwig_tee = ioutil.wrapped_read_np(
    os.path.join(gen_herwig_teepath, 'GEN.npy')
)
gen_covherwig_tee = ioutil.wrapped_read_np(
    os.path.join(gen_herwig_teepath, 'COV_DIRECT.npy')
)

reco_pythia_tee = ioutil.wrapped_read_np(
    os.path.join(reco_pythia_teepath, 'RECO.npy')
)
reco_covpythia_tee = ioutil.wrapped_read_np(
    os.path.join(reco_pythia_teepath, 'COV_DIRECT.npy')
)
reco_pythia_tee_alt = ioutil.wrapped_read_np(
    os.path.join(reco_pythia_teepath_alt, 'RECO.npy')
)
reco_covpythia_tee_alt = ioutil.wrapped_read_np(
    os.path.join(reco_pythia_teepath_alt, 'COV_DIRECT.npy')
)

unf_tee = ioutil.wrapped_read_np(
    os.path.join(trainingpath_tee, 'minimization_result', 'UNFOLDED.npy')
)
theta_tee = ioutil.wrapped_read_np(
    os.path.join(trainingpath_tee, 'minimization_result', 'UNFOLDED_SYST.npy')
)
covunf_tee = ioutil.wrapped_read_np(
    os.path.join(trainingpath_tee, 'minimization_result', 'COV_UNFOLDED_SYST.npy')
)
beta_full_tee = np.concatenate([unf_tee, theta_tee])



gen_pythia_dipole = ioutil.wrapped_read_np(
    os.path.join(gen_pythia_dipolepath, 'GEN.npy')
)
gen_covpythia_dipole = ioutil.wrapped_read_np(
    os.path.join(gen_pythia_dipolepath, 'COV_DIRECT.npy')
)

gen_herwig_dipole = ioutil.wrapped_read_np(
    os.path.join(gen_herwig_dipolepath, 'GEN.npy')
)
gen_covherwig_dipole = ioutil.wrapped_read_np(
    os.path.join(gen_herwig_dipolepath, 'COV_DIRECT.npy')
)

reco_pythia_dipole = ioutil.wrapped_read_np(
    os.path.join(reco_pythia_dipolepath, 'RECO.npy')
)
reco_covpythia_dipole = ioutil.wrapped_read_np(
    os.path.join(reco_pythia_dipolepath, 'COV_DIRECT.npy')
)

unf_dipole = ioutil.wrapped_read_np(
    os.path.join(trainingpath_dipole, 'minimization_result', 'UNFOLDED.npy')
)
theta_dipole = ioutil.wrapped_read_np(
    os.path.join(trainingpath_dipole, 'minimization_result', 'UNFOLDED_SYST.npy')
)
covunf_dipole = ioutil.wrapped_read_np(
    os.path.join(trainingpath_dipole, 'minimization_result', 'COV_UNFOLDED_SYST.npy')
)
beta_full_dipole = np.concatenate([unf_dipole, theta_dipole])


plotting.projected.plot_cov_2d(
    reco_covpythia_tee_alt,
    savefig='CMSnote/covreco_pythia_tee',
)

plotting.projected.plot_transfer_2d(
    LOSS_alt, logz=True,
    savefig='CMSnote/transfer_pythia_tee',
)

ptbins = [50, 88, 150, 254, 408, np.inf]
Rbins = [0.4, 0.5]

ptslices = [
    (50, 88), 
    (88, 150), 
    (150, 254), 
    (254, 408), 
    (408, np.inf)
]
Rslices = [(0.4,0.5)] * 6
labels = [
    '$50 < p_T < 88$ GeV',
    '$88 < p_T < 150$ GeV',
    '$150 < p_T < 254$ GeV',
    '$254 < p_T < 408$ GeV',
    '$p_T > 408$ GeV',
]

cbins = []
for block in LOSS.binning.genbinning.blocks:
    if 'c' in block.axis_names:
        cbins += block.ax_details['c']['edges']
cbins = sorted(list(set(cbins)))

plotting.projected.compare_flux_projection(
    [gen_pythia_tee] * 6,
    [gen_covpythia_tee] * 6,
    labels,
    binning.genbinning,
    ptslices,
    Rslices,
    what = 'angular_average',
    isData=False,
    logy=True,
    logx=True,
    rbin=None,
    savefig='CMSnote/tee_pythia_angular_profile',
    extratext='Pythia8'
)

plotting.projected.compare_flux_projection(
    [gen_herwig_tee] * 6,
    [gen_covherwig_tee] * 6,
    labels,
    binning.genbinning,
    ptslices,
    Rslices,
    what = 'angular_average',
    isData=False,
    logy=True,
    logx=True,
    rbin=None,
    savefig='CMSnote/tee_herwig_angular_profile',
    extratext='Herwig7'
)

nuisance_groups = [
    ['TRK_EFF'],
    ['CH'],
    ['JES', 'JER', 'UNCLUSTERED'],
    ['PU'],
    ['scale', 'ISR', 'FSR', 'PDFaS'],
    ['prefire', 'isosf', 'triggersf', 'idsf']
]
nuisance_labels = [
    'Tracking efficiency',
    'Track energy scale',
    'Jet reconstruction',
    'Pileup rate',
    'Theory',
    'Muon reconstruction'
]

for ipt in range(len(ptbins) - 1):
    for iR in range(len(Rbins) - 1):
        plotting.projected.plot_uncertainty_contributions(
            LOSS, beta_full_tee, covunf_tee,
            nuisances_l = nuisance_groups,
            labels_l = nuisance_labels,
            binning=LOSS.binning.genbinning,
            cut = {
                'pt' : (ptbins[ipt], ptbins[ipt+1]),
                'R' : (Rbins[iR], Rbins[iR + 1]),
                'c' : (cbins[0], cbins[1]),
            },
            savefig='CMSnote/unc_impact_tee_pt%d_R%d' % (ipt, iR),
            ratio=False,
        )

        plotting.projected.plot_teedipole_2d(
            gen_pythia_tee, gen_covpythia_tee,
            binning.genbinning,
            (ptbins[ipt], ptbins[ipt + 1]),
            (Rbins[iR], Rbins[iR + 1]),
            what='flux',
            savefig = 'CMSnote/tee2d_pythia_pt%d_R%d' % (ipt, iR),
            vmin=1e-2, vmax=2e1,
            extratext='Pythia8'
        )

        plotting.projected.plot_teedipole_2d(
            gen_pythia_tee, gen_covpythia_tee,
            binning.genbinning,
            (ptbins[ipt], ptbins[ipt + 1]),
            (Rbins[iR], Rbins[iR + 1]),
            what='angular_effect',
            savefig = 'CMSnote/tee2d_pythia_pt%d_R%d' % (ipt, iR),
            vmin=0.5, vmax=1.5,
            extratext='Pythia8'
        )

        plotting.projected.plot_teedipole_2d(
            gen_herwig_tee, gen_covherwig_tee,
            binning.genbinning,
            (ptbins[ipt], ptbins[ipt + 1]),
            (Rbins[iR], Rbins[iR + 1]),
            what='flux',
            savefig = 'CMSnote/tee2d_herwig_pt%d_R%d' % (ipt, iR),
            vmin=1e-2, vmax=2e1,
            extratext='Herwig7'
        )

        plotting.projected.plot_teedipole_2d(
            gen_herwig_tee, gen_covherwig_tee,
            binning.genbinning,
            (ptbins[ipt], ptbins[ipt + 1]),
            (Rbins[iR], Rbins[iR + 1]),
            what='angular_effect',
            vmin=0.5, vmax=1.5,
            savefig = 'CMSnote/tee2d_herwig_pt%d_R%d' % (ipt, iR),
            extratext='Herwig7'
        )

        plotting.projected.plot_teedipole_2d(
            [gen_pythia_tee, gen_herwig_tee], 
            [gen_covpythia_tee, gen_covherwig_tee],
            binning.genbinning,
            (ptbins[ipt], ptbins[ipt + 1]),
            (Rbins[iR], Rbins[iR + 1]),
            vmin=0.5, vmax=1.5,
            what='ratio_flux',
            cbarlabel='Pythia / Herwig',
            savefig = 'CMSnote/tee2d_PYTHIAratioHERWIG_pt%d_R%d' % (ipt, iR),
        )


        plotting.projected.plot_teedipole_2d(
            gen_pythia_dipole, gen_covpythia_dipole,
            binning.genbinning,
            (ptbins[ipt], ptbins[ipt + 1]),
            (Rbins[iR], Rbins[iR + 1]),
            what='flux',
            vmin=1e-2, vmax=2e1,
            savefig = 'CMSnote/dipole2d_pythia_pt%d_R%d' % (ipt, iR),
            extratext='Pythia8'
        )

        plotting.projected.plot_teedipole_2d(
            gen_herwig_dipole, gen_covherwig_dipole,
            binning.genbinning,
            (ptbins[ipt], ptbins[ipt + 1]),
            (Rbins[iR], Rbins[iR + 1]),
            what='flux',
            vmin=1e-2, vmax=2e1,
            savefig = 'CMSnote/dipole2d_herwig_pt%d_R%d' % (ipt, iR),
            extratext='Herwig7'
        )

        plotting.projected.plot_teedipole_2d(
            [gen_pythia_dipole, gen_herwig_dipole], 
            [gen_covpythia_dipole, gen_covherwig_dipole],
            binning.genbinning,
            (ptbins[ipt], ptbins[ipt + 1]),
            (Rbins[iR], Rbins[iR + 1]),
            vmin=0.5, vmax=1.5,
            what='ratio_flux',
            cbarlabel='Pythia / Herwig',
            savefig = 'CMSnote/dipole2d_PYTHIAratioHERWIG_pt%d_R%d' % (ipt, iR),
        )
