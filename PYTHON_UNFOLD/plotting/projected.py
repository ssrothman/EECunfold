from tqdm import tqdm
import statutil
import os
import matplotlib.pyplot as plt
import hist
from matplotlib.colors import LogNorm, Normalize
import numpy as np
import awkward as ak
from scipy.optimize import curve_fit
import contextlib
import mplhep as hep
from scipy.special import erf

plt.style.use(hep.style.CMS)

import json
with open("config/config.json", 'r') as f:
    config = json.load(f)

def wrapped_savefig(path):
    print("Saving figure to", path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=config['DPI'], bbox_inches='tight', format='png')

def get_vals_errs(vals, cov, normalize=False, what='value'):
    if normalize:
        vals, cov = statutil.normalize_distribution(vals, cov)

    err1D = np.sqrt(np.diag(cov))

    if what == 'value' or what == 'valuePull':
        pass
    elif what == 'error':
        vals = err1D
        err1D = np.zeros_like(vals)
    elif what == 'relativeError':
        vals = err1D / vals
        err1D = np.zeros_like(vals)
    else:
        raise ValueError("Invalid 'what' parameter: %s" % what)

    return vals, err1D

def make_chi2_latextable(chi2_l, label_l, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('\\begin{tabular}{l|c}\n')
        f.write('Sample & $\\chi^2$ \\\\\n')
        f.write('\\hline\n')
        for chi2, label in zip(chi2_l, label_l):
            f.write(f'{label} & {chi2:.7g} \\\\\n')
        f.write('\\end{tabular}\n')
    print(f"Latex table saved to {path}")

def get_ratio_vals_errs(vals_num, vals_denom, 
                        cov_num, cov_denom,
                        normalize=False, what='value'):
    '''
    Assume independent num and denom for now
    '''

    if normalize:
        vals_num, cov_num, vals_denom, cov_denom, _ = \
                statutil.conormalize_distributions(
                    vals_num, cov_num, vals_denom, cov_denom,
                    cov12 = None
                )

    if what == 'value' or what == 'valuePull':
        ratio, covratio = statutil.quotient_distribution(
                vals_num, cov_num,
                vals_denom, cov_denom,
                cov12=None
        )
    elif what == 'error' or what =='relativeError':
        err_num = np.sqrt(np.diag(cov_num))
        err_denom = np.sqrt(np.diag(cov_denom))

        if what == 'relativeError':
            err_num /= vals_num
            err_denom /= vals_denom

        ratio = err_num / err_denom
        covratio = np.zeros((*err_num.shape, *err_num.shape))

        ratio = err_num / err_denom
        covratio = np.zeros((*err_num.shape, *err_num.shape))
    else:
        raise ValueError("Invalid 'what' parameter: %s" % what)

    err1D = np.sqrt(np.diag(covratio))
    if what == 'valuePull':
        ratio = (ratio - 1) / err1D
        err1D = np.ones_like(err1D)

    return ratio, err1D

def plot_transfer_2d(LOSS, isCMS=True, savefig=None, variation=None, logz=False):
    fig = plt.figure(figsize=config['Figure_Size'])
    try:
        ax = fig.add_subplot(111)
        if isCMS:
            hep.cms.label(ax=ax, data=False, label=config['Approval_Text'])

        if variation is None:
            T = LOSS.transfer0
            name = 'nominal'
            cmap = 'Reds'
            if logz:
                norm = LogNorm()
            else:
                norm = Normalize()
        else:
            print("taking variation", variation)
            if str(variation) in LOSS.namedNuisances:
                name = LOSS.namedNuisances[str(variation)]
                print("Variation name:", name)
            else:
                name = 'Variation %s' % variation

            T = LOSS.transferVariations[variation] / LOSS.transfer0
            if logz:
                T = np.abs(T)
                norm = LogNorm()
                cmap = 'Reds'
            else:
                cmap = 'coolwarm'
                #maxval = np.nanmax(np.abs(T)[np.isfinite(T)])
                maxval = 0.1
                print(maxval)
                norm = Normalize(vmin=-maxval, vmax=maxval)

        q = ax.pcolormesh(T, cmap=cmap, norm=norm)
        fig.colorbar(q, ax=ax, pad=0.01)
        ax.set_xlabel("Gen")
        ax.set_ylabel("Reco")

        ax.text(0.05, 0.95, name,
                transform=ax.transAxes, fontsize=46,
                bbox=dict(facecolor='white', alpha=0.5),
                verticalalignment='top', horizontalalignment='left')

        plt.tight_layout()

        if savefig is not None:
            if logz:
                savefig += '_logz'
            if variation is not None:
                filename = savefig + '_transfer_variation%s.png' % variation
            else:
                filename = savefig + '_transfer.png'
            wrapped_savefig(filename)
        else:
            plt.show()
    finally:
        plt.close(fig)

def plot_transfer_scale(LOSS, isCMS=True, savefig=None, binning=None, cut=None, variation=None):
    fig = plt.figure(figsize=config['Figure_Size'])
    try:
        ax = fig.add_subplot(111)
        if isCMS:
            hep.cms.label(ax=ax, data=False, label=config['Approval_Text'])

        if variation is not None:
            if str(variation) in LOSS.namedNuisances:
                name = LOSS.namedNuisances[str(variation)]
            else:
                name = 'Variation %s' % variation
            print("Taking variation", variation, "with name", name)
            T = LOSS.transferVariations[variation]
        else:
            print("Taking nominal transfer")
            T = LOSS.transfer0
            name = 'nominal'

        scale = np.sum(T, axis=0)

        if cut is not None:
            scale = binning.get_slice(scale.T, **cut).T
            
            labeltext = ''
            for key, value in cut.items():
                labeltext += '%g < %s < %g\n' % (value[0], key, value[1])
            labeltext = labeltext[:-1]
    
        ax.errorbar(np.arange(len(scale)) + 0.5, scale, xerr=0.5, fmt='o')
        ax.set_xlabel("Gen Bin")
        ax.set_ylabel("Transfer scale factor")

        ax.text(0.05, 0.95, name,
                transform=ax.transAxes, fontsize=46,
                bbox=dict(facecolor='white', alpha=0.5),
                verticalalignment='top', horizontalalignment='left')

        if cut is not None:
            ax.text(0.05, 0.05, labeltext,
                    transform=ax.transAxes, fontsize=46,
                    bbox=dict(facecolor='white', alpha=0.5))

        plt.tight_layout()

        if savefig is not None:
            if variation is not None:
                filename = savefig + '%s_transfer_scale_variation%s.png'%(cut, variation)
            else:
                filename = savefig + '_transfer_scale.png'
            wrapped_savefig(filename)
        else:
            plt.show()
    finally:
        plt.close(fig)

def plot_bkg_templates(LOSS, isCMS=True, savefig=None, binning=None, cut=None, variation=None):
    if cut is not None and binning is None:
        raise ValueError("If 'cut' is provided, 'binning' must also be provided.")

    fig = plt.figure(figsize=config['Figure_Size'])
    try:
        ax = fig.add_subplot(111)
        if isCMS:
            hep.cms.label(ax=ax, data=False, label=config['Approval_Text'])

        if variation is not None:
            if str(variation) in LOSS.namedNuisances:
                name = LOSS.namedNuisances[str(variation)]
            else:
                name = 'Variation %s' % variation
            print("Taking variation", variation, "with name", name)
            R0 = LOSS.rhoVariations[variation]
            G0 = LOSS.gammaVariations[variation]
        else:
            print("Taking nominal background templates")
            R0 = LOSS.rho0
            G0 = LOSS.gamma0
            name = 'nominal'
    
        if cut is not None:
            R0 = binning.get_slice(R0.T, **cut).T
            G0 = binning.get_slice(G0.T, **cut).T

            labeltext = ''
            fnametext = ''
            for key, value in cut.items():
                labeltext += '%g < %s < %g\n' % (value[0], key, value[1])
                fnametext += '_%g-%s-%g' % (value[0], key, value[1])
            labeltext = labeltext[:-1]
        else:
            labeltext = None
            fnametext = ''

        x = np.arange(len(R0), dtype=np.float64) + 0.5

        ax.errorbar(x, R0, xerr=0.5, fmt='o', label='Reco')
        ax.errorbar(x, G0, xerr=0.5, fmt='o', label='Gen')

        ax.text(0.05, 0.95, name,
                transform=ax.transAxes, fontsize=46,
                bbox=dict(facecolor='white', alpha=0.5),
                verticalalignment='top', horizontalalignment='left')

        if labeltext is not None:
            ax.text(0.05, 0.05, labeltext,
                    transform=ax.transAxes, fontsize=46,
                    bbox=dict(facecolor='white', alpha=0.5))

        ax.set_xlabel("Bin")
        ax.set_ylabel("Background template")
        ax.legend(loc='best')
        plt.tight_layout()

        if savefig is not None:
            if variation is not None:
                filename = savefig + '%s_bkg_templates_variation%s.png'%(fnametext, variation)
            else:
                filename = savefig + '%s_bkg_templates.png'%fnametext

            wrapped_savefig(filename)
        else:
            plt.show()
    finally:
        plt.close(fig)

def plot_purity_stability(LOSS, isCMS=True, savefig=None, binning=None, cut=None):
    if cut is not None and binning is None:
        raise ValueError("If 'cut' is provided, 'binning' must also be provided.")

    fig = plt.figure(figsize=config['Figure_Size'])
    try:
        ax = fig.add_subplot(111)
        if isCMS:
            hep.cms.label(ax=ax, data=False, label=config['Approval_Text'])

        T = LOSS.transfer0

        if cut is not None:
            T = binning.get_slice(T.T, **cut)
            T = binning.get_slice(T.T, **cut)

            labeltext = ''
            fnametext = ''
            for key, value in cut.items():
                labeltext += '%g < %s < %g\n' % (value[0], key, value[1])
                fnametext += '_%g-%s-%g' % (value[0], key, value[1])
            labeltext = labeltext[:-1]
        else:
            labeltext = None
            fnametext = ''

        x = np.arange(T.shape[0], dtype=np.float64) + 0.5

        purity = np.diag(T) / np.sum(T, axis=1)
        stability = np.diag(T) / np.sum(T, axis=0)
        ax.errorbar(x, purity, xerr=0.5, fmt='o', label='Purity')
        ax.errorbar(x, stability, xerr=0.5, fmt='o', label='Stability')
        ax.set_xlabel("Bin")
        ax.set_ylabel("Purity / Stability")
        if labeltext is not None:
            ax.text(0.05, 0.05, labeltext,
                    transform=ax.transAxes, fontsize=46,
                    bbox=dict(facecolor='white', alpha=0.5))

        ax.legend(loc='best')
        ax.axhline(1, color='black', linestyle='--', linewidth=1.)
        ax.set_ylim(0, 1.1)
        ax.axhline(0.5, color='red', linestyle='--', linewidth=1.)
        plt.tight_layout()

        if savefig is not None:
            filename = savefig + '%s_purity_stability.png'%fnametext
            wrapped_savefig(filename)
        else:
            plt.show()
    finally:
        plt.close(fig)

def plot_cov_2d(cov, isCMS=True, data=False, correl=True,
                ticklabels=None,
                logz=False, savefig=None):
    fig = plt.figure(figsize=config['Figure_Size'])
    try:
        ax = fig.add_subplot(111)
        if isCMS:
            hep.cms.label(ax=ax, data=data, label=config['Approval_Text'])

        if correl:
            err = np.sqrt(np.diag(cov))
            corr = cov / np.outer(err, err)

            if logz:
                q = ax.pcolormesh(np.abs(corr), cmap='Reds', norm=LogNorm())
                fig.colorbar(q, ax=ax, pad=0.01, label='|Correlation|')
            else:
                q = ax.pcolormesh(corr, cmap='coolwarm', vmin=-1, vmax=1)
                fig.colorbar(q, ax=ax, pad=0.01, label='Correlation')

        else:
            if logz:
                q = ax.pcolormesh(np.abs(corr), cmap='Reds', norm=LogNorm())
                fig.colorbar(q, ax=ax, pad=0.01, label='|Covariance|')
            else:
                r = np.max(np.abs(cov))
                q = ax.pcolormesh(cov, cmap='coolwarm', vmin=-r, vmax=r)
                fig.colorbar(q, ax=ax, pad=0.01, label='Covariance')

        if ticklabels is not None:
            ax.set_xticks(np.arange(len(ticklabels)) + 0.5)
            ax.set_xticklabels(ticklabels, rotation=45, ha='right', rotation_mode='anchor')
            ax.set_yticks(np.arange(len(ticklabels)) + 0.5)
            ax.set_yticklabels(ticklabels, rotation=45, ha='right', rotation_mode='anchor')
        else:
            ax.set_xlabel("Bin")
            ax.set_ylabel("Bin")

        plt.tight_layout()

        if savefig is not None:
            if correl:
                filename = savefig + '_correl.png'
            else:
                filename = savefig + '_cov.png'
            wrapped_savefig(filename)
        else:
            plt.show()
    finally:
        plt.close(fig)

def plot_eigvals(eigvals_l, label_l=None, 
                 isCMS=True, isData=False, 
                 savefig=None):
    if type(eigvals_l) is not list:
        eigvals_l = [eigvals_l]
    if type(label_l) is not list:
        label_l = [label_l] * len(eigvals_l)

    fig = plt.figure(figsize=config['Figure_Size'])
    try:
        ax = fig.add_subplot(111)
        if isCMS:
            hep.cms.label(ax=ax, data=isData, label=config['Approval_Text'])

        any_labels=False
        for eigvals, label in zip(eigvals_l, label_l):
            x = np.arange(len(eigvals), dtype=np.float64) + 0.5
            ax.errorbar(x, eigvals, xerr=0.5, fmt='o', label=label)
            if label is not None:
                any_labels = True
        ax.set_xlabel("Eigenvalue Index")
        ax.set_ylabel("Eigenvalue")
        ax.set_yscale('log')

        if any_labels:
            plt.legend(loc='best')
        plt.tight_layout()

        if savefig is not None:
            filename = savefig + '_eigvals.png'
            wrapped_savefig(filename)
        else:
            plt.show()

    finally:
        plt.close(fig)

def compare_1d(vals_l, covs_l, label_l, 
               normalize=False, what = 'value',
               isCMS=True, isData=False, 
               logy=True, xoffset=0.1,
               binning=None, cut=None, 
               savefig=None, calculate_chi2=False):

    fig = plt.figure(figsize=config['Figure_Size'])
    try:
        (ax_main, ax_ratio) = fig.subplots(
                2, 1, sharex=True, 
                height_ratios=(1, config['Ratiopad_Height'])
        )

        if isCMS:
            hep.cms.label(ax=ax_main, data=isData, label=config['Approval_Text'])

        labeltext = ''
        if cut is not None:
            for key, value in cut.items():
                labeltext += '%g < %s < %g\n' % (value[0], key, value[1])
            labeltext = labeltext[:-1]

        main_artists = []
        for i, (val, cov, label) in enumerate(zip(vals_l, covs_l, label_l)):
            if cut is not None:
                val = binning.get_slice(val.T, **cut).T
                cov = binning.get_slice(cov.T, **cut)
                cov = binning.get_slice(cov.T, **cut)

            x = np.arange(len(val), dtype=np.float64) + 0.5
            x += xoffset * i  # Offset each dataset for visibility

            nom, err1D = get_vals_errs(val, cov, 
                                       normalize=normalize,
                                       what=what)

            main_artists.append(
                    ax_main.errorbar(x, nom, xerr=0.5, yerr=err1D, fmt='o',
                             label=label)
            )

        val0 = vals_l[0]
        cov0 = covs_l[0]
        if cut is not None:
            val0 = binning.get_slice(val0.T, **cut).T
            cov0 = binning.get_slice(cov0.T, **cut)
            cov0 = binning.get_slice(cov0.T, **cut)

        if calculate_chi2:
            chi2s = []
        for i, (val,cov,artist) in enumerate(zip(vals_l[1:], covs_l[1:], main_artists[1:])):
            if cut is not None:
                val = binning.get_slice(val.T, **cut).T
                cov = binning.get_slice(cov.T, **cut)
                cov = binning.get_slice(cov.T, **cut)

            ratio, ratioerr = get_ratio_vals_errs(val0, val,
                                                  cov0, cov,
                                                  normalize=normalize,
                                                  what=what)
            if calculate_chi2:
                chi2s.append(
                    statutil.get_chi2(val0, val, cov0, cov)[0]
                )

            x = np.arange(len(ratio), dtype=np.float64) + 0.5
            x += xoffset * (i+1)

            ax_ratio.errorbar(x, ratio, xerr=0.5, yerr=ratioerr, fmt='o',
                              color=artist[0].get_color(),)

        ax_ratio.set_xlabel("Bin")
        if what == 'value' or what == 'valuePull':
            if normalize:
                ax_main.set_ylabel("Normalized Value")
            else:
                ax_main.set_ylabel("Value")
        elif what == 'error':
            ax_main.set_ylabel("Uncertainty")
        elif what == 'relativeError':
            ax_main.set_ylabel("Relative Uncertainty")

        if what == 'valuePull':
            ax_ratio.set_ylabel("Pulls")
            ax_ratio.fill_between(
                ax_ratio.get_xlim(), -1, 1, color='gray', alpha=0.2
            )
            ax_ratio.axhline(0, color='black', linestyle='--', linewidth=1.)
        else:
            ax_ratio.set_ylabel("Ratio")
            ax_ratio.axhline(1, color='black', linestyle='--', linewidth=1.)

        if logy:
            ax_main.set_yscale('log')

        ax_main.legend(loc='best')

        if labeltext:
            ax_main.text(0.05, 0.05, labeltext,
                         fontsize=32,
                         transform=ax_main.transAxes,
                         bbox=dict(facecolor='white', alpha=0.5))
        if calculate_chi2:
            chi2text = 'Chi2 (%d bins):\n'%(val0.shape[0])
            for chi2, label in zip(chi2s, label_l[1:]):
                chi2text += '%s: %.5g\n' % (label, chi2)
            chi2text = chi2text[:-1]
            ax_main.text(
                0.95, 0.05, chi2text,
                transform=ax_main.transAxes, fontsize=32,
                bbox=dict(facecolor='white', alpha=0.5),
                verticalalignment='bottom', horizontalalignment='right'
            )

        plt.tight_layout()

        if savefig is not None:
            filename = savefig + '_%s.png' % what
            wrapped_savefig(filename)
        else:
            plt.show()
    finally:
        plt.close(fig)

def plot_pulls(LOSS, x, invhess, data=False, isCMS=True,
               savefig=None, names=False):
    fig = plt.figure(figsize=config['Figure_Size'])
    try:
        pulls = x[LOSS.nBeta:]
        pullerr = np.diag(np.sqrt(invhess[LOSS.nBeta:, LOSS.nBeta:]))
        ax = fig.add_subplot(111)

        if isCMS:
            hep.cms.label(ax=ax, data=data, label=config['Approval_Text'])

        x = np.arange(len(pulls)) + 0.5

        ax.errorbar(x, pulls, xerr=0.5,
                    yerr=pullerr, fmt='o', 
                    label='Pulls', color='black',
                    ecolor='gray')

        if names:
            names = []
            for i in range(len(pulls)):
                if str(i) in LOSS.namedNuisances:
                    names.append(LOSS.namedNuisances[str(i)])
                else:
                    names.append('Nuisance %d' % i)
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=45, ha='right', rotation_mode='anchor')
        else:
            ax.set_xlabel('Nuisance index')

        ax.axhline(0, color='red', linestyle='--')
        ax.fill_between(ax.get_xlim(), -1, 1, color='gray', alpha=0.2)
        ax.set_ylabel('Pulls')

        plt.tight_layout()
        if savefig is not None:
            filename = savefig + '_pulls.png'
            wrapped_savefig(filename)
        else:
            plt.show()
    finally:
        plt.close(fig)

def plot_impact(LOSS, x, invhess, whichnuisance,
                normalize=False, what='value', 
                isCMS=True, isData=False,
                logy=True, xoffset=0.1, 
                binning=None, cut=None,
                savefig=None, calculate_chi2=False):

    if str(whichnuisance) in LOSS.namedNuisances:
        nuisance_name = LOSS.namedNuisances[str(whichnuisance)]
    else:
        nuisance_name = 'Nuisance %s' % whichnuisance

    print("Plotting impacts for nuisance:", nuisance_name)

    _, _, xC, HC = statutil.nuisance_impact(x, invhess, LOSS.nBeta + whichnuisance)

    compare_1d([x[:LOSS.nBeta], xC[:LOSS.nBeta]], 
               [invhess[:LOSS.nBeta, :LOSS.nBeta], 
                HC[:LOSS.nBeta, :LOSS.nBeta]], 
               ['With %s' % nuisance_name, 
                'Without %s' % nuisance_name],
               normalize=normalize, what=what,
               isCMS=isCMS, isData=isData,
               logy=logy, xoffset=xoffset,
               binning=binning, cut=cut,
               savefig=savefig, calculate_chi2=calculate_chi2)

def plot_pull_correlations(LOSS, x, invhess, isCMS=True, data=False, correl=True,
                           logz=False, savefig=None):

    C = invhess[LOSS.nBeta:, LOSS.nBeta:]
    names = []
    for i in range(C.shape[0]):
        if str(i) in LOSS.namedNuisances:
            names.append(LOSS.namedNuisances[str(i)])
        else:
            names.append('Nuisance %d' % i)


    if savefig is not None:
        savefig += '_pull_correlations'

    plot_cov_2d(C, isCMS=isCMS, data=data, correl=correl,
                ticklabels=names, logz=logz, savefig=savefig)
