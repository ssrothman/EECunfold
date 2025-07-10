from tqdm import tqdm
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

def get_vals_errs(H, normalize=False):
    Ys = H.values(flow=True).reshape(H.axes['bootstrap'].size, -1)
    if Ys.shape[0] == 1:
        # If there's only one bootstrap sample, we can't compute an error
        return Ys[0], np.zeros_like(Ys[0])

    nom = Ys[0]
    boots = Ys[1:]

    if normalize:
        nom /= nom.sum()
        boots /= boots.sum(axis=1, keepdims=True)

    DY = boots - nom[None, :]
    cov = DY.T @ DY / DY.shape[0]
    err1D = np.sqrt(np.diag(cov))
    return nom, err1D

def make_chi2_latextable(chi2_l, label_l, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('\\begin{tabular}{l|c}\n')
        f.write('Sample & $\\chi^2$ \\\\\n')
        f.write('\\hline\n')
        for chi2, label in zip(chi2_l, label_l):
            f.write(f'{label} & {chi2:.4g} \\\\\n')
        f.write('\\end{tabular}\n')
    print(f"Latex table saved to {path}")

def get_chi2(H1, H2, normalize=False):
    Ys1 = H1.values(flow=True).reshape(H1.axes['bootstrap'].size, -1)
    Ys2 = H2.values(flow=True).reshape(H2.axes['bootstrap'].size, -1)

    if normalize:
        Ys1 /= Ys1.sum(axis=1, keepdims=True)
        Ys2 /= Ys2.sum(axis=1, keepdims=True)
    
    nom1 = Ys1[0]
    nom2 = Ys2[0]
    boots1 = Ys1[1:]
    boots2 = Ys2[1:]
    DY1 = boots1 - nom1[None, :]
    DY2 = boots2 - nom2[None, :]


    cov1 = DY1.T @ DY1 / DY1.shape[0]
    cov2 = DY2.T @ DY2 / DY2.shape[0]

    if DY1.shape[0] == 0:
        cov1 = np.zeros_like(cov2)
    if DY2.shape[0] == 0:
        cov2 = np.zeros_like(cov1)

    covdiff = cov1 + cov2
    diff = nom1 - nom2

    import statutil
    import fasteigenpy as eigen
    solver = eigen.SelfAdjointEigenSolver(covdiff)
    invcov, _ = statutil.inverse_from_eigenspectrum(solver) 
    chi2 = diff @ invcov @ diff
    return chi2

def get_ratio_vals_errs(Hnum, Hdenom, normalize=False, what='value'):
    Ys_num = Hnum.values(flow=True).reshape(Hnum.axes['bootstrap'].size, -1)
    Ys_denom = Hdenom.values(flow=True).reshape(Hdenom.axes['bootstrap'].size, -1)

    if what == 'value':
        if Ys_num.shape[0] < Ys_denom.shape[0]:
            Ys_denom = Ys_denom[:Ys_num.shape[0], :]
        elif Ys_num.shape[0] > Ys_denom.shape[0]:
            Ys_num = Ys_num[:Ys_denom.shape[0], :]

    if normalize:
        Ys_num /= Ys_num.sum(axis=1, keepdims=True)
        Ys_denom /= Ys_denom.sum(axis=1, keepdims=True)

    if what == 'value':
        ratios = Ys_num / Ys_denom

        nom = ratios[0]
        boots = ratios[1:]
        if boots.shape[0] == 0:
            err1D = np.zeros_like(nom)
        else:
            DY = boots - nom[None, :]
            cov = DY.T @ DY / DY.shape[0]
            err1D = np.sqrt(np.diag(cov))
    elif what == 'error':
        nom_num = Ys_num[0]
        nom_denom = Ys_denom[0]
        boots_num = Ys_num[1:]
        boots_denom = Ys_denom[1:]
        DY_num = boots_num - nom_num[None, :]
        DY_denom = boots_denom - nom_denom[None, :]
        cov_num = DY_num.T @ DY_num / DY_num.shape[0]
        cov_denom = DY_denom.T @ DY_denom / DY_denom.shape[0]

        nom = np.sqrt(np.diag(cov_num)) / np.sqrt(np.diag(cov_denom))
        err1D = np.zeros_like(nom)
    elif what == 'relativeError':
        nom_num = Ys_num[0]
        nom_denom = Ys_denom[0]
        boots_num = Ys_num[1:]
        boots_denom = Ys_denom[1:]
        DY_num = boots_num - nom_num[None, :]
        DY_denom = boots_denom - nom_denom[None, :]
        cov_num = DY_num.T @ DY_num / DY_num.shape[0]
        cov_denom = DY_denom.T @ DY_denom / DY_denom.shape[0]

        err_num = np.sqrt(np.diag(cov_num)) / nom_num
        err_denom = np.sqrt(np.diag(cov_denom)) / nom_denom

        nom = err_num / err_denom
        err1D = np.zeros_like(nom)

    return nom, err1D

def plot_transfer_2d(LOSS, isCMS=True, savefig=None):
    fig = plt.figure(figsize=config['Figure_Size'])
    try:
        ax = fig.add_subplot(111)
        if isCMS:
            hep.cms.label(ax=ax, data=False, label=config['Approval_Text'])
        q = ax.pcolormesh(LOSS.transfer0, cmap='Reds')
        fig.colorbar(q, ax=ax, pad=0.01)
        ax.set_xlabel("Gen")
        ax.set_ylabel("Reco")

        plt.tight_layout()

        if savefig is not None:
            filename = savefig + '_transfer.png'
            wrapped_savefig(filename)
        else:
            plt.show()
    finally:
        plt.close(fig)

def plot_transfer_scale(LOSS, isCMS=True, savefig=None):
    fig = plt.figure(figsize=config['Figure_Size'])
    try:
        ax = fig.add_subplot(111)
        if isCMS:
            hep.cms.label(ax=ax, data=False, label=config['Approval_Text'])
        scale = np.sum(LOSS.transfer0, axis=0)
        ax.errorbar(np.arange(len(scale)) + 0.5, scale, xerr=0.5, fmt='o')
        ax.set_xlabel("Gen Bin")
        ax.set_ylabel("Transfer scale factor")
        plt.tight_layout()

        if savefig is not None:
            filename = savefig + '_transfer_scale.png'
            wrapped_savefig(filename)
        else:
            plt.show()
    finally:
        plt.close(fig)

def plot_bkg_templates(LOSS, isCMS=True, savefig=None):
    fig = plt.figure(figsize=config['Figure_Size'])
    try:
        ax = fig.add_subplot(111)
        if isCMS:
            hep.cms.label(ax=ax, data=False, label=config['Approval_Text'])
        x = np.arange(len(LOSS.rho0), dtype=np.float64) + 0.5
        ax.errorbar(x, LOSS.rho0, xerr=0.5, fmt='o', label='Reco')
        ax.errorbar(x, LOSS.gamma0, xerr=0.5, fmt='o', label='Gen')
        ax.set_xlabel("Bin")
        ax.set_ylabel("Background template")
        ax.legend(loc='best')
        plt.tight_layout()

        if savefig is not None:
            filename = savefig + '_bkg_templates.png'
            wrapped_savefig(filename)
        else:
            plt.show()
    finally:
        plt.close(fig)

def plot_purity_stability(LOSS, isCMS=True, savefig=None):
    fig = plt.figure(figsize=config['Figure_Size'])
    try:
        ax = fig.add_subplot(111)
        if isCMS:
            hep.cms.label(ax=ax, data=False, label=config['Approval_Text'])

        x = np.arange(len(LOSS.rho0), dtype=np.float64) + 0.5
        T = LOSS.transfer0
        purity = np.diag(T) / np.sum(T, axis=1)
        stability = np.diag(T) / np.sum(T, axis=0)
        ax.errorbar(x, purity, xerr=0.5, fmt='o', label='Purity')
        ax.errorbar(x, stability, xerr=0.5, fmt='o', label='Stability')
        ax.set_xlabel("Bin")
        ax.set_ylabel("Purity / Stability")
        ax.legend(loc='best')
        ax.axhline(1, color='black', linestyle='--', linewidth=0.5)
        ax.set_ylim(0, 1.1)
        plt.tight_layout()

        if savefig is not None:
            filename = savefig + '_purity_stability.png'
            wrapped_savefig(filename)
        else:
            plt.show()
    finally:
        plt.close(fig)

def plot_cov_2d(cov, isCMS=True, data=False, correl=True,
                savefig=None):
    fig = plt.figure(figsize=config['Figure_Size'])
    try:
        ax = fig.add_subplot(111)
        if isCMS:
            hep.cms.label(ax=ax, data=data, label=config['Approval_Text'])

        if correl:
            err = np.sqrt(np.diag(cov))
            corr = cov / np.outer(err, err)

            q = ax.pcolormesh(corr, cmap='coolwarm', vmin=-1, vmax=1)

            fig.colorbar(q, ax=ax, pad=0.01, label='Correlation')
        else:
            r = np.max(np.abs(cov))
            q = ax.pcolormesh(cov, cmap='coolwarm', vmin=-r, vmax=r)

            fig.colorbar(q, ax=ax, pad=0.01, label='Covariance')

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

def compare_1d(H_l, label_l, normalize=False, isCMS=True, isData=False, 
               logy=True, xoffset=0.1, what='value', pulls=False,
               savefig=None, cut={}):
    fig = plt.figure(figsize=config['Figure_Size'])
    try:
        (ax_main, ax_ratio) = fig.subplots(
                2, 1, sharex=True, 
                height_ratios=(1, config['Ratiopad_Height'])
        )

        if isCMS:
            hep.cms.label(ax=ax_main, data=isData, label=config['Approval_Text'])

        main_artists = []
        for i, (H, label) in enumerate(zip(H_l, label_l)):
            nom, err1D = get_vals_errs(H[cut], normalize=normalize)
            x = np.arange(len(nom), dtype=np.float64) + 0.5
            x += xoffset * i  # Offset each dataset for visibility

            if what == 'value':
                main_artists.append(
                        ax_main.errorbar(x, nom, xerr=0.5, yerr=err1D, fmt='o',
                                 label=label)
                )
            elif what == 'error':
                main_artists.append(
                        ax_main.errorbar(x, err1D, xerr=0.5, fmt='o',
                                 label=label)
                )
            elif what == 'relativeError':
                main_artists.append(
                        ax_main.errorbar(x, err1D/nom, xerr=0.5, fmt='o',
                                 label=label)
                )


        for i, (H,artist) in enumerate(zip(H_l[1:], main_artists[1:])):
            ratio, ratioerr = get_ratio_vals_errs(H_l[0][cut], H[cut], normalize=normalize, what=what)
            x = np.arange(len(ratio), dtype=np.float64) + 0.5
            x += xoffset * (i+1)
            if pulls:
                ax_ratio.errorbar(x, (ratio-1)/ratioerr, xerr=0.5, yerr=1, fmt='o',
                                  color=artist[0].get_color(),)
            else:
                ax_ratio.errorbar(x, ratio, xerr=0.5, yerr=ratioerr, fmt='o',
                                  color=artist[0].get_color(),)

        ax_ratio.set_xlabel("Bin")
        if what == 'value':
            if normalize:
                ax_main.set_ylabel("Normalized Value")
            else:
                ax_main.set_ylabel("Value")
        elif what == 'error':
            ax_main.set_ylabel("Uncertainty")

        if pulls:
            ax_ratio.set_ylabel("Pulls")
            ax_ratio.fill_between(
                ax_ratio.get_xlim(), -1, 1, color='gray', alpha=0.2
            )
            ax_ratio.axhline(0, color='black', linestyle='--', linewidth=0.5)
        else:
            ax_ratio.set_ylabel("Ratio")
            ax_ratio.axhline(1, color='black', linestyle='--', linewidth=0.5)

        if logy:
            ax_main.set_yscale('log')

        ax_main.legend(loc='best')

        plt.tight_layout()

        if savefig is not None:
            if pulls:
                filename = savefig + '_%s_pulls.png' % what
            else:
                filename = savefig + '_%s.png' % what
            wrapped_savefig(filename)
        else:
            plt.show()
    finally:
        plt.close(fig)
