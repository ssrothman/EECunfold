from tqdm import tqdm
import statutil
import os
import matplotlib.pyplot as plt
import hist
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
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
    plt.savefig(path+'.pdf', dpi=config['DPI'], bbox_inches='tight', format='pdf')
    plt.savefig(path+'.png', dpi=config['DPI'], bbox_inches='tight', format='png')

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
                        normalize=False, 
                        what='value',
                        whatpad='ratio'):
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
        if whatpad == 'ratio':
            ratio, covratio = statutil.quotient_distribution(
                    vals_num, cov_num,
                    vals_denom, cov_denom,
                    cov12=None
            )
        elif whatpad == 'difference':
            ratio, covratio = statutil.difference_distribution(
                    vals_num, cov_num,
                    vals_denom, cov_denom,
                    cov12=None
            )
        else:
            raise ValueError("Invalid 'whatpad' parameter: %s" % whatpad)

    elif what == 'error' or what =='relativeError':
        err_num = np.sqrt(np.diag(cov_num))
        err_denom = np.sqrt(np.diag(cov_denom))

        if what == 'relativeError':
            err_num /= vals_num
            err_denom /= vals_denom

        if whatpad == 'ratio':
            ratio = err_num / err_denom
            covratio = np.zeros((*err_num.shape, *err_num.shape))
        elif whatpad == 'difference' : 
            ratio = err_num - err_denom
            covratio = np.zeros((*err_num.shape, *err_num.shape))
        else:
            raise ValueError("Invalid 'whatpad' parameter: %s" % whatpad)

    else:
        raise ValueError("Invalid 'what' parameter: %s" % what)

    err1D = np.sqrt(np.diag(covratio))
    if what == 'valuePull':
        if whatpad == 'ratio':
            ratio = (ratio - 1) / err1D
        elif whatpad == 'difference':
            ratio = ratio / err1D
        else:
            raise ValueError("Invalid 'whatpad' parameter: %s" % whatpad)

        err1D = np.ones_like(err1D)

    return ratio, err1D

def plot_transfer_2d(LOSS, isCMS=True, savefig=None, 
                     variation=None, logz=False):
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
                norm = Normalize(vmin=-maxval, vmax=maxval)

        q = ax.pcolormesh(T, cmap=cmap, norm=norm, rasterized=True)
        fig.colorbar(q, ax=ax, pad=0.01)
        ax.set_xlabel("Gen $(p_T \\otimes R \\otimes r \\otimes \\phi)$ Bin Index")
        ax.set_ylabel("Reco $(p_T \\otimes R \\otimes r \\otimes \\phi)$ Bin Index")

        if name != 'nominal':
            ax.text(0.05, 0.95, name,
                    transform=ax.transAxes, fontsize=46,
                    bbox=dict(facecolor='white', alpha=0.5),
                    verticalalignment='top', horizontalalignment='left')

        plt.tight_layout()

        if savefig is not None:
            if logz:
                savefig += '_logz'
            if variation is not None:
                filename = savefig + '_transfer_variation-%s' % name
            else:
                filename = savefig + '_transfer'
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
    
        ax.errorbar(np.arange(len(scale)) + 0.5, scale, xerr=0.5, fmt='o', rasterized=True)
        ax.set_xlabel("Gen Bin Index")
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
                filename = savefig + '%s_transfer_scale_variation%s'%(cut, variation)
            else:
                filename = savefig + '_transfer_scale'
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
            R0 = LOSS.rhoVariations[variation] / LOSS.rho0
            G0 = LOSS.gammaVariations[variation] / LOSS.gamma0
        else:
            print("Taking nominal background templates")
            R0 = LOSS.rho0
            G0 = LOSS.gamma0
            name = 'nominal'
    
        if cut is not None:
            R0 = binning.get_slice(R0.T, **cut).T
            G0 = binning.get_slice(G0.T, **cut).T

            labeltext = ''
            for key, value in cut.items():
                labeltext += '%g < %s < %g\n' % (value[0], key, value[1])
            labeltext = labeltext[:-1]
        else:
            labeltext = None

        x = np.arange(len(R0), dtype=np.float64) + 0.5

        ax.errorbar(x, R0, xerr=0.5, fmt='o', label='Reco', rasterized=True)
        ax.errorbar(x, G0, xerr=0.5, fmt='o', label='Gen', rasterized=True)

        ax.text(0.05, 0.95, name,
                transform=ax.transAxes, fontsize=46,
                bbox=dict(facecolor='white', alpha=0.5),
                verticalalignment='top', horizontalalignment='left')

        if labeltext is not None:
            ax.text(0.05, 0.05, labeltext,
                    transform=ax.transAxes, fontsize=46,
                    bbox=dict(facecolor='white', alpha=0.5))

        ax.set_xlabel("Bin Index")
        ax.set_ylabel("Background template")
        ax.legend(loc='best')
        ax.ticklabel_format(useOffset=False, axis='both')
        plt.tight_layout()

        if savefig is not None:
            if variation is not None:
                filename = savefig + 'bkg_templates_variation-%s'%(name)
            else:
                filename = savefig + 'bkg_templates'

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
            for key, value in cut.items():
                labeltext += '%g < %s < %g\n' % (value[0], key, value[1])
            labeltext = labeltext[:-1]
        else:
            labeltext = None

        x = np.arange(T.shape[0], dtype=np.float64) + 0.5

        purity = np.diag(T) / np.sum(T, axis=1)
        stability = np.diag(T) / np.sum(T, axis=0)
        ax.errorbar(x, purity, xerr=0.5, fmt='o', label='Purity', rasterized=True)
        ax.errorbar(x, stability, xerr=0.5, fmt='o', label='Stability', rasterized=True)
        ax.set_xlabel("Bin Index")
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
            filename = savefig + 'purity_stability'
            wrapped_savefig(filename)
        else:
            plt.show()
    finally:
        plt.close(fig)

def plot_cov_2d(cov, isCMS=True, data=False, correl=True,
                ticklabelsA=None, ticklabelsB=None,
                errA=None, errB=None,
                logz=False, savefig=None):
    fig = plt.figure(figsize=config['Figure_Size'])
    try:
        ax = fig.add_subplot(111)
        if isCMS:
            hep.cms.label(ax=ax, data=data, label=config['Approval_Text'])

        if correl:
            if errA is None:
                errA = np.sqrt(np.diag(cov))
            if errB is None:
                errB = np.sqrt(np.diag(cov))

            corr = cov / np.outer(errA, errB)

            if logz:
                q = ax.pcolormesh(corr, cmap='coolwarm', 
                                  norm=SymLogNorm(
                                      vmin=-1,
                                      vmax=+1,
                                      linthresh=1e-2,
                                      linscale=1e-1
                                  ),
                                  rasterized=True)
                fig.colorbar(q, ax=ax, pad=0.01, label='Correlation')
            else:
                q = ax.pcolormesh(corr, cmap='coolwarm', vmin=-1, vmax=1, rasterized=True)
                fig.colorbar(q, ax=ax, pad=0.01, label='Correlation')

        else:
            if logz:
                q = ax.pcolormesh(np.abs(corr), cmap='Reds', norm=LogNorm(), rasterized=True)
                fig.colorbar(q, ax=ax, pad=0.01, label='|Covariance|')
            else:
                r = np.max(np.abs(cov))
                q = ax.pcolormesh(cov, cmap='coolwarm', vmin=-r, vmax=r, rasterized=True)
                fig.colorbar(q, ax=ax, pad=0.01, label='Covariance')

        if ticklabelsB is not None:
            ax.set_xticks(np.arange(len(ticklabelsB)) + 0.5)
            ax.set_xticklabels(ticklabelsB, rotation=45, ha='right', rotation_mode='anchor')
        else:
            ax.set_xlabel("$(p_T \\otimes R \\otimes r \\otimes \\phi)$ Bin Index")
        if ticklabelsA is not None:
            ax.set_yticks(np.arange(len(ticklabelsA)) + 0.5)
            ax.set_yticklabels(ticklabelsA, rotation=45, ha='right', rotation_mode='anchor')
        else:
            ax.set_ylabel("$(p_T \\otimes R \\otimes r \\otimes \\phi)$ Bin Index")

        plt.tight_layout()

        if savefig is not None:
            if correl:
                filename = savefig + '_correl'
            else:
                filename = savefig + '_cov'
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
            ax.errorbar(x, eigvals, xerr=0.5, fmt='o', label=label, rasterized=True)
            if label is not None:
                any_labels = True
        ax.set_xlabel("Eigenvalue Index")
        ax.set_ylabel("Eigenvalue")
        ax.set_yscale('log')

        if any_labels:
            plt.legend(loc='best')
        plt.tight_layout()

        if savefig is not None:
            filename = savefig + '_eigvals'
            wrapped_savefig(filename)
        else:
            plt.show()

    finally:
        plt.close(fig)

def compare_flux_projection(vals_l, covs_l, label_l,
                            binning, ptslice_l, Rslice_l, 
                            what='angular_average',
                            whatpad='none',
                            rbin=None,
                            normalize=True, 
                            jacobian=True,
                            logy=True, logx=True,
                            isCMS=True, isData=False,
                            extratext=None,
                            savefig=None):

    fig = plt.figure(figsize=config['Figure_Size'])
    try:
        if len(vals_l) == 1 or whatpad == 'none':
            ax_main = fig.add_subplot(111)
        else:
            (ax_main, ax_ratio) = fig.subplots(
                2, 1, sharex=True,
                height_ratios=(1, config['Ratiopad_Height'])
            )

        if isCMS:
            hep.cms.label(ax=ax_main, data=isData, label=config['Approval_Text'])

        allSamePt = True
        for ptslice in ptslice_l:
            if ptslice != ptslice_l[0]:
                allSamePt = False
                break
        allSameR = True
        for Rslice in Rslice_l:
            if Rslice != Rslice_l[0]:
                allSameR = False
                break

        axtext = ''
        if extratext is not None:
            axtext += extratext.strip() + '\n'
        if allSamePt:
            axtext += '$%g < p_t \\text{ [GeV]} < %g$\n' % (
                ptslice_l[0][0], ptslice_l[0][1]
            )
        if allSameR:
            axtext += '$%g < R < %g$\n' % (
                Rslice_l[0][0], Rslice_l[0][1]
            )
        if rbin is not None:
            axtext += '$%g < r < %g$\n' % (0.1*rbin, 0.1*(rbin+1))

        axtext = axtext.strip()
        if axtext:
            ax_main.text(0.05, 0.05, axtext,
                         transform=ax_main.transAxes, fontsize=32, 
                         bbox=dict(facecolor='white', alpha=0.8),
                         verticalalignment='bottom', 
                         horizontalalignment='left')

        main_artists = []
        fluxes = []
        covfluxes = []
        for vals, covs, label, ptslice, Rslice, in zip(vals_l, covs_l, label_l, ptslice_l, Rslice_l):
            if what == 'angular_average':
                flux, covflux, r_edges, c_edges = statutil.angular_averaged_flux(
                    vals, covs, binning, ptslice, Rslice,
                    normalize=normalize, jacobian=jacobian
                )
                x = 0.5 * (r_edges[:-1] + r_edges[1:])
                xerr = 0.5 * (r_edges[1:] - r_edges[:-1])

            elif what == 'radial_sum':
                flux, covflux, r_edges, c_edges = statutil.radial_summed_flux(
                    vals, covs, binning, ptslice, Rslice,
                    normalize=normalize, jacobian=jacobian
                )
                x = 0.5 * (c_edges[:-1] + c_edges[1:])
                xerr = 0.5 * (c_edges[1:] - c_edges[:-1])
            elif what == 'radial_slice':
                flux, covflux, r_edges, c_edges = statutil.radial_slice_flux(
                    vals, covs, binning, ptslice, Rslice, rbin,
                    normalize=normalize, jacobian=jacobian
                )
                x = 0.5 * (c_edges[:-1] + c_edges[1:])
                xerr = 0.5 * (c_edges[1:] - c_edges[:-1])

            main_artists.append(
                ax_main.errorbar(x, flux, xerr=xerr, yerr=np.sqrt(np.diag(covflux)),
                                 fmt='o', rasterized=True, label=label)
            )

            fluxes.append(flux)
            covfluxes.append(covflux)

        if whatpad != 'none':
            for i in range(1, len(vals_l)):
                ratio, covratio = statutil.quotient_distribution(
                    fluxes[0], covfluxes[0],
                    fluxes[i], covfluxes[i],
                    None
                )

                ax_ratio.errorbar(x, ratio, xerr=xerr, yerr=np.sqrt(np.diag(covratio)),
                                  fmt='o', color=main_artists[i][0].get_color(),
                                  rasterized=True)
            ax_ratio.axhline(1, color='black', linestyle='--', linewidth=1.)

        if logx:
            ax_main.set_xscale('log')
        if logy:
            ax_main.set_yscale('log')

        if what == 'angular_average':
            if len(vals_l) > 1 and whatpad != 'none':
                ax_ratio.set_xlabel("$r$")
            else:
                ax_main.set_xlabel("$r$")
            #ax_main.set_ylabel("Angular-Averaged Flux")
            ax_main.set_ylabel('$\\frac{1}{\\sigma}\\frac{d \\text{EEC}}{r\\,dr}$', fontsize=42)
        elif what == 'radial_sum':
            if len(vals_l) > 1 and whatpad != 'none':
                ax_ratio.set_xlabel("$\\phi$")
            else:
                ax_main.set_xlabel('$\\phi$')
            ax_main.set_ylabel("Radially-Summed Flux")

        if what=='radial_slice':
            print("WARNING: r binning is hard-coded")
            text = '%g < r < %g'%(0.1*rbin, 0.1*(rbin+1))
        else:
            text = None
    
        allcuts_same = True
        if len(vals_l) != 1:
            for i in range(1, len(vals_l)):
                if ptslice_l[i] != ptslice_l[0] or Rslice_l[i] != Rslice_l[0]:
                    allcuts_same = False
                    break
        if allcuts_same:
            if text is None:
                text = ''

            text = '%g < pt [GeV] < %g\n%g < R < %g\n' % (
                ptslice_l[0][0], ptslice_l[0][1],
                Rslice_l[0][0], Rslice_l[0][1]
            ) + text

            text = text.strip()

        if text is not None:
            ax_main.text(0.95, 0.95, text,
                         transform=ax_main.transAxes, fontsize=32,
                         bbox=dict(facecolor='white', alpha=0.8),
                         verticalalignment='top', horizontalalignment='right')
        #labeltext = '%g < pt [GeV] < %g\n' % (ptslice[0], ptslice[1])
        #labeltext += '%g < R < %g\n' % (Rslice[0], Rslice[1])
        #labeltext = labeltext[:-1]
        #ax_main.text(0.95, 0.95, labeltext,
        #        transform=ax_main.transAxes, fontsize=24,
        #        bbox=dict(facecolor='white', alpha=0.4),
        #        verticalalignment='top', horizontalalignment='right')

        if len(vals_l) > 1:
            ax_main.legend(loc='best')

        plt.tight_layout()
        if savefig is not None:
            savefig += '_%s' % what
            wrapped_savefig(savefig)
        else:
            plt.show()
    finally:
        plt.close(fig)

def plot_teedipole_2d(
                vals, covs,
                binning, ptslice, Rslice,
                what='flux',
                normalize=True, 
                jacobian=True,
                logz=None, cmap=None,
                isCMS=True, isData=False,
                cbarlabel=None,
                extratext=None,
                vmin=None, vmax=None,
                savefig=None):
    fig = plt.figure(figsize=config['Figure_Size'])
    try:
        ax = fig.add_subplot(111, projection='polar')
        if isCMS:
            hep.cms.label(ax=ax, data=isData, label=config['Approval_Text'], pad=0.05)

        if ptslice[0] == -np.inf:
            labeltext = '$30 < p_T \\text{ [GeV]} < %g$\n' % ptslice[1]
        elif ptslice[1] == np.inf:
            labeltext = '$%g < p_T \\text{ [GeV]}$\n' % ptslice[0]
        else:
            labeltext = '$%g < p_T \\text{ [GeV]} < %g$\n' % (ptslice[0], ptslice[1])
        if Rslice[0] == -np.inf:
            labeltext += '$0 < R < %g$\n' % Rslice[1]
        elif Rslice[1] == np.inf:
            labeltext += '$%g < R$\n' % Rslice[0]
        else:
            labeltext += '$%g < R < %g$\n' % (Rslice[0], Rslice[1])
        labeltext = labeltext[:-1]
        if extratext is not None:
            labeltext = extratext.strip() + '\n' + labeltext

        if what in ['flux', 'angular_effect']:
            flux, _, r_edges, c_edges = statutil.compute_flux(
                vals, covs, binning, ptslice, Rslice,
                normalize=normalize, jacobian=jacobian,
            )
            flux = flux.reshape((len(r_edges)-1, len(c_edges)-1))

            if what == 'angular_effect':
                angular_avg, _, _, _, = statutil.angular_averaged_flux(
                    vals, covs, binning, ptslice, Rslice,
                    normalize=normalize, jacobian=jacobian
                )
                flux = flux / angular_avg[:, None]
        elif what in ['ratio_flux', 'ratio_angular_effect']:
            flux1, _, r_edges, c_edges = statutil.compute_flux(
                vals[0], covs[0], binning, ptslice, Rslice,
                normalize=normalize, jacobian=jacobian,
            )
            flux2, _, _, _ = statutil.compute_flux(
                vals[1], covs[1], binning, ptslice, Rslice,
                normalize=normalize, jacobian=jacobian,
            )
            flux1 = flux1.reshape((len(r_edges)-1, len(c_edges)-1))
            flux2 = flux2.reshape((len(r_edges)-1, len(c_edges)-1))

            if what == 'ratio_flux':
                flux = flux1/flux2
            elif what == 'ratio_angular_effect':
                angular_avg1 = statutil.angular_averaged_flux(
                    vals[0], covs[0], binning, ptslice, Rslice,
                    normalize=normalize, jacobian=jacobian
                )[0]
                angular_avg2 = statutil.angular_averaged_flux(
                    vals[1], covs[1], binning, ptslice, Rslice,
                    normalize=normalize, jacobian=jacobian
                )[0]
                flux1 = flux1 / angular_avg1[:, None]
                flux2 = flux2 / angular_avg2[:, None]

                flux = flux1 / flux2

        if what == 'flux':
            if cmap is None:
                cmap = 'inferno'
            if logz is None:
                logz = True
            if cbarlabel is None:
                #cbarlabel = 'Flux'
                cbarlabel = '$\\frac{1}{\\sigma}\\frac{d^2\\text{EEC}}{r\\,dr\\,d\\phi}$'
        elif what == 'angular_effect':
            if cmap is None:
                cmap = 'coolwarm'
            if logz is None:
                logz = False
            if cbarlabel is None:
                #cbarlabel = 'Angular Modification'
                cbarlabel = '$\\frac{2 \\pi r}{\\text{EEC}(r)}\\frac{d\\text{EEC}}{d\\phi}$'
        elif what.startswith('ratio'):
            if cmap is None:
                cmap = 'coolwarm'
            if logz is None:
                logz = False
            if cbarlabel is None:
                cbarlabel = 'Ratio'

        if logz:
            if vmin is None:
                vmin = flux[flux > 0].min()
            if vmax is None:
                vmax = flux.max()
            norm = LogNorm(vmin = vmin, vmax=vmax)
        else:
            if cmap == 'coolwarm':
                q = np.max(np.abs(flux-1))
                if vmin is None:
                    vmin = 1 - q
                if vmax is None:
                    vmax = 1 + q
                norm = Normalize(vmin=vmin, vmax=vmax)
            else:
                if vmin is None:
                    vmin = 0
                if vmax is None:
                    vmax = flux.max()
                norm = Normalize(vmin=vmin, vmax=vmax)

        pc1 = ax.pcolormesh(
            c_edges, r_edges, flux,
            shading='auto', rasterized=True,
            cmap=cmap, norm=norm, 
        )
        pc2 = ax.pcolormesh(
            np.pi-c_edges, r_edges, flux,
            shading='auto', rasterized=True,
            cmap=cmap, norm=norm,
        )
        pc3 = ax.pcolormesh(
            np.pi+c_edges, r_edges, flux,
            shading='auto', rasterized=True,
            cmap=cmap, norm=norm,
        )
        pc4 = ax.pcolormesh(
            2*np.pi-c_edges, r_edges, flux,
            shading='auto', rasterized=True,
            cmap=cmap, norm=norm,
        )

        cb = fig.colorbar(pc1, ax=ax, pad=0.05)
        cb.set_label(cbarlabel,
                     fontsize=42)

        ax.text(0.00, 1.00, labeltext,
                transform=ax.transAxes, fontsize=32,
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top', horizontalalignment='left')

        plt.tight_layout()
        
        if savefig is not None:
            filename = savefig + '_%s' % what
            wrapped_savefig(filename)
        else:
            plt.show()
    finally:
        plt.close(fig)

def compare_1d(vals_l, covs_l, label_l, 
               normalize=False, what = 'value',
               whatpad='ratio',
               isCMS=True, isData=False, 
               logy=True, xoffset=0.1,
               binning=None, cut=None, 
               savefig=None, calculate_chi2=False):

    fig = plt.figure(figsize=config['Figure_Size'])
    try:
        if len(vals_l) == 1:
            ax_main = fig.add_subplot(111)
        else:
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
                             label=label, rasterized=True)
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
                                                  what=what,
                                                  whatpad=whatpad)
            if calculate_chi2:
                chi2s.append(
                    statutil.get_chi2(val0, val, cov0, cov, normalize=normalize)[0]
                )

            x = np.arange(len(ratio), dtype=np.float64) + 0.5
            x += xoffset * (i+1)

            ax_ratio.errorbar(x, ratio, xerr=0.5, yerr=ratioerr, fmt='o',
                              color=artist[0].get_color(), rasterized=True,)

        if len(vals_l) == 1:
            ax_main.set_xlabel("Bin Index")
        else:
            ax_ratio.set_xlabel("Bin Index")

        if what == 'value' or what == 'valuePull':
            if normalize:
                ax_main.set_ylabel("Normalized Value")
            else:
                ax_main.set_ylabel("Value")
        elif what == 'error':
            ax_main.set_ylabel("Uncertainty")
        elif what == 'relativeError':
            ax_main.set_ylabel("Relative Uncertainty")

        if len(vals_l) != 1:
            if what == 'valuePull':
                if whatpad == 'ratio':
                    ax_ratio.set_ylabel("Ratio Pulls")
                elif whatpad == 'difference':
                    ax_ratio.set_ylabel("Difference Pulls")
                else:
                    raise ValueError("Invalid 'whatpad' parameter: %s" % whatpad)

                ax_ratio.fill_between(
                    ax_ratio.get_xlim(), -1, 1, color='gray', alpha=0.2
                )
                ax_ratio.axhline(0, color='black', linestyle='--', linewidth=1.)
            else:
                if whatpad == 'ratio':
                    ax_ratio.set_ylabel("Ratio")
                    ax_ratio.axhline(1, color='black', linestyle='--', linewidth=1.)
                elif whatpad=='difference':
                    ax_ratio.set_ylabel("Difference")
                    ax_ratio.axhline(0, color='black', linestyle='--', linewidth=1.)
                else:
                    raise ValueError("Invalid 'whatpad' parameter: %s" % whatpad)

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
            filename = savefig + '_%s' % what
            wrapped_savefig(filename)
        else:
            plt.show()
    finally:
        plt.close(fig)

def plot_pulls(LOSS, x, invhess, data=False, isCMS=True,
               savefig=None, names=True):
    fig = plt.figure(figsize=config['Figure_Size'])
    try:
        pulls = x[LOSS.nBeta:]
        pullerr = np.sqrt(np.diag(invhess[LOSS.nBeta:, LOSS.nBeta:]))
        ax = fig.add_subplot(111)

        if isCMS:
            hep.cms.label(ax=ax, data=data, label=config['Approval_Text'])

        x = np.arange(len(pulls)) + 0.5

        ax.errorbar(x, pulls, xerr=0.5,
                    yerr=pullerr, fmt='o', 
                    label='Pulls', color='black',
                    ecolor='gray', rasterized=True)

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
            filename = savefig + '_pulls'
            wrapped_savefig(filename)
        else:
            plt.show()
    finally:
        plt.close(fig)

def plot_impact(LOSS, x, invhess, whichnuisance,
                normalize=False, what='value', 
                whatpad='ratio',
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

    if savefig is not None:
        savefig+='_variation-%s' % nuisance_name

    compare_1d([x[:LOSS.nBeta], xC[:LOSS.nBeta]], 
               [invhess[:LOSS.nBeta, :LOSS.nBeta], 
                HC[:LOSS.nBeta, :LOSS.nBeta]], 
               ['With %s' % nuisance_name, 
                'Without %s' % nuisance_name],
               normalize=normalize, what=what,
               whatpad=whatpad,
               isCMS=isCMS, isData=isData,
               logy=logy, xoffset=xoffset,
               binning=binning, cut=cut,
               savefig=savefig, calculate_chi2=calculate_chi2)

def plot_uncertainty_contributions(LOSS, x, covx, 
                                   nuisances_l=None,
                                   labels_l=None,
                                   relative=True, ratio=True,
                                   isCMS=True, isData=False, logy=True,
                                   binning=None, cut=None,
                                   savefig=None):
    if nuisances_l is None:
        nuisances_l = [int(k) for k in LOSS.namedNuisances.keys()]

    fig = plt.figure(figsize=config['Figure_Size'])
    try:
        ax = fig.add_subplot(111)
        
        if isCMS:
            hep.cms.label(ax=ax, data=isData, label=config['Approval_Text'])

        total_unc = np.sqrt(np.diag(covx)[:LOSS.nBeta])
        if relative: 
            total_unc = total_unc / x[:LOSS.nBeta]
        if cut is not None:
            total_unc = binning.get_slice(total_unc, **cut)


        xstatonly = x.copy()
        covstatonly = covx.copy()
        for i in range(LOSS.nBeta, LOSS.nBeta + LOSS.nTheta):
            _, _, xstatonly, covstatonly = statutil.nuisance_impact(
                xstatonly, covstatonly, len(xstatonly)-1
            )
        statonly_unc = np.sqrt(np.diag(covstatonly)[:LOSS.nBeta])
        if relative:
            statonly_unc = statonly_unc / x[:LOSS.nBeta]
        if cut is not None:
            statonly_unc = binning.get_slice(statonly_unc, **cut)
        if ratio:
            statonly_unc = statonly_unc / total_unc

        cmap = plt.get_cmap('hsv')
        inv_map = {v: k for k, v in LOSS.namedNuisances.items()}
        for i, (whichnuisance, label) in enumerate(zip(nuisances_l, labels_l)):
            if type(whichnuisance) not in [tuple, list]:
                whichnuisance = [whichnuisance]

            wn_ints = []

            for wn in whichnuisance:
                if type(wn) is str:
                    wn = int(inv_map[wn])
                wn_ints.append(wn)

            wn_ints = sorted(wn_ints)[::-1]
            print(wn_ints)
            for wn in wn_ints:
                xshift, covshift, _, _ = statutil.nuisance_impact(
                    x, covx, LOSS.nBeta + wn
                )
            
            unc_contrib = np.sqrt(np.diag(-covshift)[:LOSS.nBeta])
            if relative:
                unc_contrib = unc_contrib / x[:LOSS.nBeta]

            if cut is not None:
                unc_contrib = binning.get_slice(unc_contrib, **cut)

            if ratio:
                unc_contrib = unc_contrib / total_unc

            ax.errorbar(np.arange(unc_contrib.shape[0])+0.5, unc_contrib, 
                        xerr=0.5, fmt='o',
                        color=cmap(i / len(nuisances_l)),
                        label=label)

        ax.errorbar(np.arange(statonly_unc.shape[0])+0.5, statonly_unc,
                    xerr=0.5, fmt='o', color='gray',
                    label='Stat', rasterized=True)
        if ratio:
            ax.axhline(1, color='black', linestyle='--', linewidth=1.)
        else:
            ax.errorbar(np.arange(total_unc.shape[0])+0.5, total_unc, 
                        xerr=0.5, fmt='o', color='black',
                        label='Total', rasterized=True)

        ax.legend(bbox_to_anchor=(1., 1), loc='upper left',
                  frameon=True, fontsize=16, ncol=1)

        plt.tight_layout()
        if len(cut.keys()) == 3:
            if 'r' not in cut.keys():
                print("WARNING: r binning is hard-coded")
                ax.set_xlabel('r')
                ax.set_xticks([0., 2., 4., 6., 8., 10.],
                              [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        else:
            ax.set_xlabel("Bin index")

        if ratio:
            ax.set_ylabel("Proportional uncertainty contribution")
        elif relative:
            ax.set_ylabel("Relative Uncertainty")
        else:
            ax.set_ylabel("Uncertainty")

        if logy:
            ax.set_yscale('log')
        else:
            ax.set_ylim(0, None)

        if cut is not None:
            cutlabel = ''
            for key, value in cut.items():
                if key == 'c':
                    key = '\\phi'
                cutlabel += '$%g < %s < %g$\n' % (value[0], key, value[1])
            cutlabel = cutlabel[:-1]
            ax.text(1.05, 0.05, cutlabel,
                    transform=ax.transAxes, fontsize=26,
                    bbox=dict(facecolor='white', alpha=0.5),
                    verticalalignment='bottom', horizontalalignment='left'
            )

        plt.tight_layout()

        if savefig is not None:
            savefig += '_uncertainty_contributions'
            wrapped_savefig(savefig)
        else:
            plt.show()
    finally:
        plt.close(fig)

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
                ticklabelsA=names, 
                ticklabelsB=names,
                logz=logz, savefig=savefig)
