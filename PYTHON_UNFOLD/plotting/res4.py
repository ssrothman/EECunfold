import fasteigenpy as eigen
from tqdm import tqdm
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

def get_labeltext(Redges, ptedges, Rbin, ptbin):
    if type(Rbin) is int:
        Rtext = '$%0.1f < R_L < %0.1f$'%(Redges[Rbin], Redges[Rbin+1])
    elif Rbin == hist.overflow:
        Rtext = '$%0.1f < R_L < Inf$'%(Redges[-1])
    elif Rbin == hist.underflow:
        Rtext = '$0 < R_L < %0.1f$'%(Redges[0])
    else:
        raise ValueError("Invalid R bin: %s"%Rbin)

    if type(ptbin) is int:
        Ptext = '$%0.0f < p_T < %0.0f$'%(ptedges[ptbin], ptedges[ptbin+1])
    elif ptbin == hist.overflow:
        Ptext = '$%0.0f < p_T < Inf$'%(ptedges[-1])
    elif ptbin == hist.underflow:
        Ptext = '$30 < p_T < %0.0f$'%(ptedges[0])
    else:
        raise ValueError("Invalid pt bin: %s"%ptbin)

    return Rtext + "\n" + Ptext

import json
with open('config/config.json', 'r') as f:
    config = json.load(f)

def compute_flux(H, RLbin, ptbin, iToy,
                 jacobian=True,
                 normalize=True):
    H2d = H[{'R' : RLbin, 'pt' : ptbin, 'bootstrap' : iToy}]

    vals = H2d.values(flow=True)

    if jacobian:
        redges = H2d.axes['r'].edges
        cedges = H2d.axes['c'].edges
        area = 0.5 * (redges[1:]**2 - redges[:-1]**2)[:, None] \
                * (cedges[1:] - cedges[:-1])[None, :]
        vals = vals/area

    if normalize:
        vals = vals/np.sum(vals)

    return vals

def compute_angular_average(H, RLbin, ptbin, iToy,
                            jacobian=True,
                            normalize=True):
    flux = compute_flux(H, RLbin, ptbin, iToy,
                        jacobian=jacobian,
                        normalize=normalize)

    angular_avg = np.mean(flux, axis=1)

    return angular_avg

def compute_angular_fluctuation(H, RLbin, ptbin, iToy,
                                jacobian=True,
                                normalize=True):
    flux = compute_flux(H, RLbin, ptbin, iToy,
                        jacobian=jacobian,
                        normalize=normalize)

    angular_avg = compute_angular_average(H, RLbin, ptbin, iToy,
                                          jacobian=jacobian,
                                          normalize=normalize)

    angular_fluctuation = flux/angular_avg[:,None]

    return angular_fluctuation

def compute_differnce(H1, H2, RLbin, ptbin, iToy,
                      jacobian=True,
                      normalize=True,
                      relative=False,
                      what='flux'):
    if what == 'flux':
        flux1 = compute_flux(H1, RLbin, ptbin, iToy,
                             jacobian=jacobian,
                             normalize=normalize)
        flux2 = compute_flux(H2, RLbin, ptbin, iToy,
                             jacobian=jacobian,
                             normalize=normalize)
    elif what == 'angular_avg':
        flux1 = compute_angular_average(H1, RLbin, ptbin, iToy,
                                        jacobian=jacobian,
                                        normalize=normalize)
        flux2 = compute_angular_average(H2, RLbin, ptbin, iToy,
                                        jacobian=jacobian,
                                        normalize=normalize)
    elif what == 'angular_fluctuation':
        flux1 = compute_angular_fluctuation(H1, RLbin, ptbin, iToy,
                                            jacobian=jacobian,
                                            normalize=normalize)
        flux2 = compute_angular_fluctuation(H2, RLbin, ptbin, iToy,
                                            jacobian=jacobian,
                                            normalize=normalize)
    else:
        raise ValueError("Invalid what: %s" % what)

    if relative:
        return np.nan_to_num((flux2 - flux1)/flux1)
    else:
        return flux2 - flux1

def compute_ratio(H1, H2, RLbin, ptbin, iToy,
                  jacobian=True,
                  normalize=True,
                  what='flux'):

    if what == 'flux':
        flux1 = compute_flux(H1, RLbin, ptbin, iToy,
                             jacobian=jacobian,
                             normalize=normalize)
        flux2 = compute_flux(H2, RLbin, ptbin, iToy,
                             jacobian=jacobian,
                             normalize=normalize)
    elif what == 'angular_avg':
        flux1 = compute_angular_average(H1, RLbin, ptbin, iToy,
                                        jacobian=jacobian,
                                        normalize=normalize)
        flux2 = compute_angular_average(H2, RLbin, ptbin, iToy,
                                        jacobian=jacobian,
                                        normalize=normalize)
    elif what == 'angular_fluctuation':
        flux1 = compute_angular_fluctuation(H1, RLbin, ptbin, iToy,
                                            jacobian=jacobian,
                                            normalize=normalize)
        flux2 = compute_angular_fluctuation(H2, RLbin, ptbin, iToy,
                                            jacobian=jacobian,
                                            normalize=normalize)
    else:
        raise ValueError("Invalid what: %s" % what)

    return np.nan_to_num(flux2/flux1)

def errs_from_toys(fluxfunc, pulls=False, **kwargs):

    nominal = fluxfunc(**kwargs, iToy=0)
    
    if 'H' in kwargs:
        ntoys = kwargs['H'].axes['bootstrap'].size - 1
    elif 'H1' in kwargs:
        ntoys = np.min((kwargs['H1'].axes['bootstrap'].size - 1,
                        kwargs['H2'].axes['bootstrap'].size - 1))
    else:
        raise ValueError("No H or H1 provided to determine number of toys")

    if ntoys == 0:
        return nominal, np.zeros_like(nominal)

    sumdiff = np.zeros_like(nominal)
    sumdiff2 = np.zeros_like(nominal)

    for iToy in range(1, ntoys+1):
        nextflux = fluxfunc(**kwargs, iToy=iToy)

        sumdiff += (nextflux - nominal)
        sumdiff2 += (nextflux - nominal)**2

    sumdiff /= ntoys
    sumdiff2 /= ntoys

    std_diff = np.sqrt(sumdiff2 - sumdiff**2)

    if pulls:
        return nominal/std_diff, np.ones_like(std_diff)
    else:
        return nominal, std_diff

def flux_vals_errs(H, RLbin, ptbin,
                   jacobian=True,
                   normalize=True,
                   pulls=False):
    return errs_from_toys(compute_flux,
                          H=H,
                          RLbin=RLbin,
                          ptbin=ptbin,
                          jacobian=jacobian,
                          normalize=normalize,
                          pulls=pulls)

def angular_avg_vals_errs(H, RLbin, ptbin,
                          jacobian=True,
                          normalize=True,
                          pulls=False):
    return errs_from_toys(compute_angular_average,
                          H=H,
                          RLbin=RLbin,
                          ptbin=ptbin,
                          jacobian=jacobian,
                          normalize=normalize,
                          pulls=pulls)

def angular_fluctuation_vals_errs(H, RLbin, ptbin,
                                  jacobian=True,
                                  normalize=True,
                                  pulls=False):
    return errs_from_toys(compute_angular_fluctuation,
                          H=H,
                          RLbin=RLbin,
                          ptbin=ptbin,
                          jacobian=jacobian,
                          normalize=normalize,
                          pulls=pulls)

def unc_vals_errs(H, RLbin, ptbin,
                  jacobian=True,
                  normalize=True,
                  relative=False):
    val, err = flux_vals_errs(H, RLbin, ptbin,
                              normalize=normalize,
                              jacobian=jacobian)
    if relative:
        return err/val, np.zeros_like(err)
    else:
        return err, np.zeros_like(err)
        
def difference_vals_errs(H1, H2, RLbin, ptbin,
                         jacobian=True,
                         normalize=True,
                         pulls=False,
                         relative=False,
                         what='flux'):
    return errs_from_toys(compute_differnce,
                          H1=H1,
                          H2=H2,
                          RLbin=RLbin,
                          ptbin=ptbin,
                          jacobian=jacobian,
                          normalize=normalize,
                          pulls=pulls,
                          relative=relative,
                          what=what)

def ratio_vals_errs(H1, H2, RLbin, ptbin,
                    jacobian=True,
                    normalize=True,
                    what='flux'):
    return errs_from_toys(compute_ratio,
                          H1=H1,
                          H2=H2,
                          RLbin=RLbin,
                          ptbin=ptbin,
                          jacobian=jacobian,
                          normalize=normalize,
                          what=what)

def plot_flux_2d(H, RLbin, ptbin,
                 jacobian=True,
                 normalize=True,
                 logz = True,
                 cmap='inferno',
                 isCMS=True,
                 label=None):
    flux = compute_flux(H, RLbin, ptbin, 0,
                        jacobian=jacobian,
                        normalize=normalize)

    ptedges = H.axes['pt'].edges
    Redges = H.axes['R'].edges
    redges = H.axes['r'].edges
    cedges = H.axes['c'].edges

    fig = plt.figure(figsize=config['Figure_Size'])
    try:
        ax = fig.add_subplot(111, projection='polar')
        if isCMS:
            hep.cms.label(ax=ax, data=False, label=config['Approval_Text'], pad=0.05)

        if logz:
            norm = LogNorm(vmin=flux[flux>0].min(), vmax=flux.max())
        else:
            norm = Normalize(vmin=flux.min(), vmax=flux.max())

        pc1 = ax.pcolormesh(cedges, redges,
                            flux,
                            cmap=cmap,
                            shading='auto',
                            norm=norm)
        pc2 = ax.pcolormesh(np.pi-cedges, redges,
                            flux,
                            cmap=cmap,
                            shading='auto',
                            norm=norm)
        pc3 = ax.pcolormesh(np.pi+cedges, redges,
                            flux,
                            cmap=cmap,
                            shading='auto',
                            norm=norm)
        pc4 = ax.pcolormesh(2*np.pi-cedges, redges,
                            flux,
                            cmap=cmap,
                            shading='auto',
                            norm=norm)

        plt.colorbar(pc1, ax=ax)

        thetext = get_labeltext(RLedges, ptedges, RLbin, ptbin)
        if label is not None:
            thetext = label + '\n\n' + thetext
        ax.text(0.1, 0.8, thetext, 
                horizontalalignment='center',
                verticalalignment='bottom',
                transform=ax.transAxes,
                fontsize=24,
                bbox=dict(boxstyle="round,pad=0.3", 
                          edgecolor='black', 
                          facecolor='white')
        )

        plt.tight_layout()
        plt.show()
    finally:
        plt.close(fig)

def plot_angular_avg(H, RLbin, ptbin,
                     jacobian=True,
                     normalize=True,
                     logx=False,
                     logy=True, 
                     isCMS=True,
                     label=None):

    ptedges = H.axes['pt'].edges
    Redges = H.axes['R'].edges
    redges = H.axes['r'].edges
    cedges = H.axes['c'].edges

    fig = plt.figure(figsize=config['Figure_Size'])
    try:
        val, err = angular_avg_vals_errs(H, RLbin, ptbin, 
                                         jacobian=jacobian,
                                         normalize=normalize)

        rmid = (redges[1:] + redges[:-1])/2
        rwidths = redges[1:] - redges[:-1]

        ax = fig.add_subplot(111)
        if isCMS:
            hep.cms.label(ax=ax, data=False, label=config['Approval_Text'], pad=0.05)
        ax.errorbar(rmid, val,
                    xerr = rwidths/2,
                    yerr=err,
                    fmt='o')
        if logx:
            ax.set_xscale('log')
        if logy:
            ax.set_yscale('log')
        ax.set_xlabel('r')
        ax.set_ylabel('Angular average')

        thetext = get_labeltext(RLedges, ptedges, RLbin, ptbin)
        if label is not None:
            thetext = label + '\n\n' + thetext
        ax.text(0.80, 0.75, thetext, 
                horizontalalignment='center',
                verticalalignment='bottom',
                transform=ax.transAxes,
                fontsize=24,
                bbox=dict(boxstyle="round,pad=0.3", 
                          edgecolor='black', 
                          facecolor='white')
        )

        plt.tight_layout()
        plt.show()
    finally:
        plt.close(fig)

def plot_angular_fluctuation_2d(H, RLbin, ptbin,
                                jacobian=True,
                                normalize=True,
                                cmap='coolwarm',
                                isCMS=True,
                                label=None):

    ptedges = H.axes['pt'].edges
    Redges = H.axes['R'].edges
    redges = H.axes['r'].edges
    cedges = H.axes['c'].edges

    flux = compute_angular_fluctuation(H, RLbin, ptbin, 0,
                                        jacobian=jacobian,
                                        normalize=normalize)

    fig = plt.figure(figsize=config['Figure_Size'])
    try:
        ax = fig.add_subplot(111, projection='polar')
        if isCMS:
            hep.cms.label(ax=ax, data=False, label=config['Approval_Text'], pad=0.05)

        d1 = flux-1
        maxvar = np.max(np.abs(d1))

        norm = Normalize(vmin=1-maxvar, vmax=1+maxvar)

        pc1 = ax.pcolormesh(cedges, redges,
                            flux,
                            cmap=cmap,
                            shading='auto',
                            norm=norm)
        pc2 = ax.pcolormesh(np.pi-cedges, redges,
                            flux,
                            cmap=cmap,
                            shading='auto',
                            norm=norm)
        pc3 = ax.pcolormesh(np.pi+cedges, redges,
                            flux,
                            cmap=cmap,
                            shading='auto',
                            norm=norm)
        pc4 = ax.pcolormesh(2*np.pi-cedges, redges,
                            flux,
                            cmap=cmap,
                            shading='auto',
                            norm=norm)
        plt.colorbar(pc1, ax=ax)

        thetext = get_labeltext(RLedges, ptedges, RLbin, ptbin)
        if label is not None:
            thetext = label + '\n\n' + thetext
        ax.text(0.1, 0.8, thetext, 
                horizontalalignment='center',
                verticalalignment='bottom',
                transform=ax.transAxes,
                fontsize=24,
                bbox=dict(boxstyle="round,pad=0.3", 
                          edgecolor='black', 
                          facecolor='white')
        )

        plt.tight_layout()
        plt.show()
    finally:
        plt.close(fig)

def plot_angular_fluctuation_1d(H, RLbin, ptbin,
                                jacobian=True,
                                normalize=True,
                                rstep=5,
                                rstop=15,
                                rebin=1,
                                isCMS=True,
                                label=None):

    theH = H[{'r' : slice(None, None, hist.rebin(rebin))}]

    ptedges = theH.axes['pt'].edges
    Redges = theH.axes['R'].edges
    redges = theH.axes['r'].edges
    cedges = theH.axes['c'].edges

    fig = plt.figure(figsize=config['Figure_Size'])
    try:
        val, err = angular_fluctuation_vals_errs(theH, RLbin, ptbin, 
                                                 jacobian=jacobian,
                                                 normalize=normalize)
        cmid = (cedges[1:] + cedges[:-1])/2
        cwidths = cedges[1:] - cedges[:-1]

        ax = fig.add_subplot(111)
        if isCMS:
            hep.cms.label(ax=ax, data=False, label=config['Approval_Text'], pad=0.05)
        for i in range(0, min(val.shape[0], rstop), rstep):
            q = ax.errorbar(cmid, val[i],
                        xerr = cwidths/2,
                        yerr=err[i],
                        fmt='o',
                        label=f'%0.2f < r < %0.2f' % (redges[i], redges[i+1]))
            ax.errorbar(np.pi-cmid, val[i],
                        xerr = cwidths/2,
                        yerr=err[i],
                        fmt='o', color=q[0].get_color())
            ax.errorbar(np.pi+cmid, val[i],
                        xerr = cwidths/2,
                        yerr=err[i],
                        fmt='o', color=q[0].get_color())
            ax.errorbar(2*np.pi-cmid, val[i],
                        xerr = cwidths/2,
                        yerr=err[i],
                        fmt='o', color=q[0].get_color())
            
        ax.axhline(1, color='black', linestyle='--')
        ax.axvline(np.pi/2, color='gray', linestyle='--')
        ax.axvline(np.pi, color='gray', linestyle='--')
        ax.axvline(3*np.pi/2, color='gray', linestyle='--')

        ax.set_xlabel('$\\phi$')
        ax.set_ylabel('Angular fluctuation')
        ax.legend(loc='lower right', frameon=True)

        thetext = get_labeltext(RLedges, ptedges, RLbin, ptbin)
        if label is not None:
            thetext = label + '\n\n' + thetext
        ax.text(0.8, 0.75, thetext, 
                horizontalalignment='center',
                verticalalignment='bottom',
                transform=ax.transAxes,
                fontsize=24,
                bbox=dict(boxstyle="round,pad=0.3", 
                          edgecolor='black', 
                          facecolor='white')
        )

        plt.tight_layout()
        plt.show()
    finally:
        plt.close(fig)

def compare_1d(H1, H2, 
               label1, label2,
               RLbin, ptbin,
               jacobian=True, normalize=True, 
               what = 'flux', shiftx=0,
               r=0, c=0,
               savefig=None):

    if what == 'flux':
        val1, err1 = flux_vals_errs(H1, RLbin, ptbin,
                                    jacobian=jacobian,
                                    normalize=normalize)
        val2, err2 = flux_vals_errs(H2, RLbin, ptbin,
                                    jacobian=jacobian,
                                    normalize=normalize)

        ratio, ratioerr = ratio_vals_errs(H1, H2, RLbin, ptbin,
                                          jacobian=jacobian,
                                          normalize=normalize)

    elif what == 'angular_avg':
        val1, err1 = angular_avg_vals_errs(H1, RLbin, ptbin,
                                           jacobian=jacobian,
                                           normalize=normalize)
        val2, err2 = angular_avg_vals_errs(H2, RLbin, ptbin,
                                           jacobian=jacobian,
                                           normalize=normalize)

        ratio, ratioerr = ratio_vals_errs(H1, H2, RLbin, ptbin,
                                          jacobian=jacobian,
                                          normalize=normalize,
                                          what='angular_avg')

    elif what == 'angular_fluctuation':
        val1, err1 = angular_fluctuation_vals_errs(H1, RLbin, ptbin,
                                                   jacobian=jacobian,
                                                   normalize=normalize)
        val2, err2 = angular_fluctuation_vals_errs(H2, RLbin, ptbin,
                                                   jacobian=jacobian,
                                                   normalize=normalize)

        ratio, ratioerr = ratio_vals_errs(H1, H2, RLbin, ptbin,
                                          jacobian=jacobian,
                                          normalize=normalize,
                                          what='angular_fluctuation')
    elif what == 'uncertainty':
        val1, err1 = unc_vals_errs(H1, RLbin, ptbin,
                                    jacobian=jacobian,
                                    normalize=normalize,
                                    relative=False)
        val2, err2 = unc_vals_errs(H2, RLbin, ptbin,
                                   jacobian=jacobian,
                                   normalize=normalize,
                                   relative=False)
        
        ratio = np.nan_to_num(val2/val1)
        ratioerr = np.zeros_like(ratio)

    elif what == 'relative_uncertainty':
        val1, err1 = unc_vals_errs(H1, RLbin, ptbin,
                                    jacobian=jacobian,
                                    normalize=normalize,
                                    relative=True)
        val2, err2 = unc_vals_errs(H2, RLbin, ptbin,
                                   jacobian=jacobian,
                                   normalize=normalize,
                                   relative=True)

        ratio = np.nan_to_num(val2/val1)
        ratioerr = np.zeros_like(ratio)
    else:
        raise ValueError("Invalid what: %s" % what)

    RLedges = H1.axes['R'].edges
    ptedges = H1.axes['pt'].edges

    if what != 'angular_fluctuation':
        fig = plt.figure(figsize=config['Figure_Size'])
        try:
            (ax_main, ax_ratio) = fig.subplots(
                    2, 1, sharex=True,
                    height_ratios=(1, config['Ratiopad_Height'])
            )
            hep.cms.label(ax=ax_main, data=False, label=config['Approval_Text'])

            redges = H1.axes['r'].edges
            rmid = (redges[1:] + redges[:-1])/2
            rwidths = redges[1:] - redges[:-1]

            if what == 'angular_avg':
                ax_main.errorbar(rmid+shiftx*rwidths/4, val1,
                                xerr = rwidths/2,
                                yerr=err1,
                                fmt='o', label=label1)
                ax_main.errorbar(rmid-shiftx*rwidths/4, val2,
                                xerr = rwidths/2,
                                yerr=err2,
                                fmt='o', label=label2)
                ax_main.set_ylabel("Angular average flux")

                ax_ratio.errorbar(rmid, ratio,
                                xerr = rwidths/2,
                                yerr=ratioerr,
                                fmt='o', label='Ratio')
            else:
                ctedges = H1.axes['c'].edges
                phimin = ctedges[c]
                phimax = ctedges[c+1]

                phitext = ' - $%0.2f < \\phi < %0.2f$' % (phimin, phimax)
                ax_main.errorbar(rmid+shiftx*rwidths/4, val1[:,c],
                                xerr = rwidths/2,
                                yerr=err1[:,c],
                                fmt='o', 
                                label=label1+phitext)
                q = ax_main.errorbar(rmid-shiftx*rwidths/4, val2[:,c],
                                xerr = rwidths/2,
                                yerr=err2[:,c],
                                fmt='o', 
                                label=label2+phitext)

                ax_ratio.errorbar(rmid, ratio[:,c],
                                xerr = rwidths/2,
                                yerr=ratioerr[:,c],
                                fmt='o', label='Ratio',
                                color = q[0].get_color())

                if what == 'flux':
                    ax_main.set_ylabel("Flux")
                elif what == 'uncertainty':
                    ax_main.set_ylabel("Uncertainty")
                elif what == 'relative_uncertainty':
                    ax_main.set_ylabel("Relative uncertainty")

            ax_main.legend()

            ax_ratio.axhline(1, color='black', linestyle='--')
            ax_ratio.set_xlabel('r')
            if what != 'relative_uncertainty':
                ax_main.set_yscale('log')
            #ax_ratio.set_ylabel("%s / %s" % (label2, label1))
            ax_ratio.set_ylabel("Ratio")

            thetext = get_labeltext(RLedges, ptedges, RLbin, ptbin)
            ax_main.text(0.15, 0.1, thetext, 
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    transform=ax_main.transAxes,
                    fontsize=24,
                    bbox=dict(boxstyle="round,pad=0.3", 
                              edgecolor='black', 
                              facecolor='white')
            )

            plt.tight_layout()
            plt.subplots_adjust(hspace=0, wspace=0)

            if savefig is not None:
                plt.savefig("%s_R%d_PT%d_r%d_c%d_vsr.png"%(savefig, RLbin, ptbin, r, c), format='png',
                            bbox_inches='tight')
                plt.clf()
            else:
                plt.show()
        finally:
            plt.close(fig)

    if what != 'angular_avg':
        fig = plt.figure(figsize=config['Figure_Size'])
        try:
            (ax_main, ax_ratio) = fig.subplots(
                    2, 1, sharex=True,
                    height_ratios=(1, config['Ratiopad_Height'])
            )
            hep.cms.label(ax=ax_main, data=False, label=config['Approval_Text'])

            ctedges = H1.axes['c'].edges
            cmid =   (ctedges[1:] + ctedges[:-1])/2
            cwidths = ctedges[1:] - ctedges[:-1]

            redges = H1.axes['r'].edges
            rmin = redges[r]
            rmax = redges[r+1]

            rtext = ' - $%0.2f < r < %0.2f$' % (rmin, rmax)

            ax_main.errorbar(cmid+shiftx*cwidths/4, val1[r,:],
                            xerr = cwidths/2,
                            yerr=err1[r,:],
                            fmt='o', 
                            label=label1+rtext)
            q = ax_main.errorbar(cmid-shiftx*cwidths/4, val2[r,:],
                            xerr = cwidths/2,
                            yerr=err2[r,:],
                            fmt='o', 
                            label=label2+rtext)

            ax_ratio.errorbar(cmid, ratio[r,:],
                            xerr = cwidths/2,
                            yerr=ratioerr[r,:],
                            fmt='o', label='Ratio',
                            color = q[0].get_color())

            ax_main.legend()

            ax_ratio.axhline(1, color='black', linestyle='--')
            ax_ratio.set_xlabel('$\\phi$')
            ax_ratio.set_ylabel("Ratio")
            #ax_ratio.set_ylabel("%s / %s" % (label2, label1))

            if what == 'flux':
                ax_main.set_ylabel("Flux")
            elif what == 'uncertainty':
                ax_main.set_ylabel("Uncertainty")
            elif what == 'relative_uncertainty':
                ax_main.set_ylabel("Relative uncertainty")
            elif what == 'angular_fluctuation':
                ax_main.set_ylabel("Angular fluctuation")

            thetext = get_labeltext(RLedges, ptedges, RLbin, ptbin)
            ax_main.text(0.1, 0.15, thetext, 
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    transform=ax_main.transAxes,
                    fontsize=24,
                    bbox=dict(boxstyle="round,pad=0.3", 
                              edgecolor='black', 
                              facecolor='white')
            )

            plt.tight_layout()
            plt.subplots_adjust(hspace=0, wspace=0)

            if savefig is not None:
                plt.savefig("%s_R%d_PT%d_r%d_c%d_vsc.png"%(savefig, RLbin, ptbin, r, c), format='png',
                            bbox_inches='tight')
                plt.clf()
            else:
                plt.show()
        finally:
            plt.close(fig)



def compare(H1, H2, RLbin, ptbin,
            mode='diff',
            jacobian=True, normalize=True,
            plot2d=True, plotr1d=True, plotphi1d=True, plothist=True,
            savefig=None, show=True):

    ptedges = H1.axes['pt'].edges
    if plot2d:
        fig = plt.figure(figsize=config['Figure_Size'])
        try:
            if mode.startswith('diff'):
                relative = mode.endswith('relative') or mode.endswith('pulls')
                flux = compute_differnce(H1, H2, RLbin, ptbin, 0,
                                          jacobian=jacobian,
                                          normalize=normalize,
                                          relative=relative)
            elif mode == 'ratio':
                flux = compute_ratio(H1, H2, RLbin, ptbin, 0,
                                     jacobian=jacobian,
                                     normalize=normalize)
            else:
                raise ValueError("Invalid mode: %s" % mode)

            ax = fig.add_subplot(111, projection='polar')
            hep.cms.label(ax=ax, data=False, label=config['Approval_Text'], pad=0.05)

            if mode.startswith('diff'):
                d1 = flux
                maxvar = np.max(np.abs(d1))

                norm = Normalize(vmin=-maxvar, vmax=maxvar)
            elif mode == 'ratio':
                d1 = flux - 1
                maxvar = np.max(np.abs(d1))
                norm = Normalize(vmin=1-maxvar, vmax=1+maxvar)

            pc1 = ax.pcolormesh(ctedges_teedipole, redges_teedipole,
                                flux,
                                cmap='coolwarm',
                                shading='auto',
                                norm=norm)
            pc2 = ax.pcolormesh(np.pi-ctedges_teedipole, redges_teedipole,
                                flux,
                                cmap='coolwarm',
                                shading='auto',
                                norm=norm)
            pc3 = ax.pcolormesh(np.pi+ctedges_teedipole, redges_teedipole,
                                flux,
                                cmap='coolwarm',
                                shading='auto',
                                norm=norm)
            pc4 = ax.pcolormesh(2*np.pi-ctedges_teedipole, redges_teedipole,
                                flux,
                                cmap='coolwarm',
                                shading='auto',
                                norm=norm)
            cb = plt.colorbar(pc1, ax=ax)

            if mode.startswith('difference'):
                if relative:
                    cb.set_label('Relative difference')
                else:
                    cb.set_label('Difference')
            elif mode =='ratio':
                cb.set_label('Ratio')

            thetext = get_labeltext(RLedges, ptedges, RLbin, ptbin)
            ax.text(0.1, 0.8, thetext, 
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    transform=ax.transAxes,
                    fontsize=24,
                    bbox=dict(boxstyle="round,pad=0.3", 
                              edgecolor='black', 
                              facecolor='white')
            )
            plt.tight_layout()

            if savefig is not None:
                plt.savefig(savefig + '_2d.png', dpi=300, format='png', bbox_inches='tight')
            if show:
                plt.show()
        finally:
            plt.close(fig)

    if plotr1d or plotphi1d:
        if mode.startswith('diff'):
            pulls = mode.endswith('pulls')
            relative = pulls or mode.endswith('relative')

            val, err = difference_vals_errs(H1, H2, RLbin, ptbin,
                                            jacobian=jacobian,
                                            normalize=normalize,
                                            pulls=pulls,
                                            relative=relative)
        elif mode == 'ratio':
            val, err = ratio_vals_errs(H1, H2, RLbin, ptbin,
                                       jacobian=jacobian,
                                       normalize=normalize)

    if plotr1d:
        fig = plt.figure(figsize=config['Figure_Size'])
        try:
            rmid = (redges_teedipole[1:] + redges_teedipole[:-1])/2
            rwidths = redges_teedipole[1:] - redges_teedipole[:-1]

            rmid_b = np.broadcast_to(rmid[:, None], val.shape)
            rwidths_b = np.broadcast_to(rwidths[:, None], val.shape)

            ctmid = (ctedges_teedipole[1:] + ctedges_teedipole[:-1])/2
            cmap = plt.get_cmap('Reds')

            ax = fig.add_subplot(111)
            hep.cms.label(ax=ax, data=False, label=config['Approval_Text'], pad=0.05)

            for i in range(val.shape[1]):
                jitter = ((-val.shape[1]/2 + i)/val.shape[1]) * rwidths/3
                ax.errorbar(rmid_b[:,i]+jitter, 
                            val[:,i],
                            xerr = rwidths_b[:,i]/2,
                            yerr=err[:,i],
                            color=cmap(ctmid[i]/ctmid.max()),
                            fmt='o')
            ax.set_xlabel('r')

            if mode.startswith('diff'):
                ax.axhline(0, color='black', linestyle='--')
                if pulls:
                    ax.set_ylabel('Pulls')
                elif relative:
                    ax.set_ylabel('Relative difference')
                else:
                    ax.set_ylabel('Difference')
            elif mode == 'ratio':
                ax.set_ylabel('Ratio')
                ax.axhline(1, color='black', linestyle='--')


            plt.tight_layout()
            if savefig is not None:
                plt.savefig(savefig + '_r1d.png', dpi=300, format='png', bbox_inches='tight')
            if show:
                plt.show()
        finally:
            plt.close(fig)

    if plotphi1d:
        fig = plt.figure(figsize=config['Figure_Size'])
        try:
            tmid = (ctedges_teedipole[1:] + ctedges_teedipole[:-1])/2
            twidths = ctedges_teedipole[1:] - ctedges_teedipole[:-1]

            tmid_b = np.broadcast_to(tmid[None, :], val.shape)
            twidths_b = np.broadcast_to(twidths[None, :], val.shape)

            jitter = np.random.uniform(-twidths/5, twidths/5, size=tmid_b.shape)
            rmid_b = np.broadcast_to(rmid[:, None], val.shape)
            cmap = plt.get_cmap('Reds')
            colors = cmap(rmid_b.ravel())

            ax = fig.add_subplot(111)
            hep.cms.label(ax=ax, data=False, label=config['Approval_Text'], pad=0.05)

            for i in range(val.shape[0]):
                jitter = ((-val.shape[0]/2 + i)/val.shape[0]) * twidths/3
                ax.errorbar(tmid_b[i]+jitter, val[i],
                            xerr = twidths_b[i]/2,
                            yerr=err[i],
                            color=cmap(rmid[i]/rmid.max()),
                            fmt='o')
            ax.set_xlabel('$\\phi$')
            if mode.startswith('diff'):
                ax.axhline(0, color='black', linestyle='--')
                if pulls:
                    ax.set_ylabel('Pulls')
                elif relative:
                    ax.set_ylabel('Relative difference')
                else:
                    ax.set_ylabel('Difference')
            elif mode == 'ratio':
                ax.set_ylabel('Ratio')
                ax.axhline(1, color='black', linestyle='--')


            plt.tight_layout()
            if savefig is not None:
                plt.savefig(savefig + '_phi1d.png', dpi=300, format='png', bbox_inches='tight')
            if show:
                plt.show()
        finally:
            plt.close(fig)

    if plothist:
        fig = plt.figure(figsize=config['Figure_Size'])
        try:
            ax = fig.add_subplot(111)
            hep.cms.label(ax=ax, data=False, label=config['Approval_Text'], pad=0.05)

            ax.hist(val.ravel(), bins=21, histtype='step')
            if mode.startswith('diff'):
                ax.axvline(0, color='black', linestyle='--')
                if pulls:
                    ax.set_xlabel('Pulls')
                elif relative:
                    ax.set_xlabel('Relative difference')
                else:
                    ax.set_xlabel('Difference')
            elif mode == 'ratio':
                ax.set_xlabel('Ratio')
                ax.axvline(1, color='black', linestyle='--')

            ax.set_ylabel('Counts')

            plt.tight_layout()
            if savefig is not None:
                plt.savefig(savefig + '_hist.png', dpi=300, format='png', bbox_inches='tight')
            if show: 
                plt.show()
        finally:
            plt.close(fig)

def plot_transfer_2d(Htransfer, wrt, cuts={}):
    cuts_transfer = {}
    for key in cuts.keys():
        cuts_transfer[key+"_reco"] = cuts[key]
        cuts_transfer[key+"_gen"] = cuts[key]

    transfer = Htransfer[cuts_transfer]

    if wrt is not None:
        transfer = transfer.project(wrt+'_reco', wrt+'_gen')
        transfer = transfer.values(flow=True)
    else:
        genaxes = []
        recoaxes = []
        genshape = 1
        recoshape = 1
        for axis in transfer.axes:
            if axis.name.endswith('_gen'):
                genaxes.append(axis.name)
                genshape *= axis.extent
            elif axis.name.endswith('_reco'):
                recoaxes.append(axis.name)
                recoshape *= axis.extent
        transfer = transfer.project(*recoaxes, *genaxes)
        transfer = transfer.values(flow=True).reshape((recoshape, genshape))

    denom = transfer.sum(axis=0)
    transfer = transfer/denom[None, :]

    if wrt is not None:
        plt.xlabel(wrt+'_gen')
        plt.ylabel(wrt+'_reco')
    else:
        plt.xlabel('reco')
        plt.ylabel('gen')

    plt.pcolormesh(transfer, cmap="Reds", norm=LogNorm())
    plt.colorbar()
    plt.show()

def plot_transfer_1d(Htransfer, wrt, thebin, gen, cuts={}, cuts_reco=None,
                     show=True):
    cuts_transfer = {}
    thetext = ''

    if cuts_reco is None:
        for key in cuts.keys():
            cuts_transfer[key+"_reco"] = cuts[key]
            cuts_transfer[key+"_gen"] = cuts[key]
            theax = Htransfer.axes[key+'_reco']
            edges = theax.bin(cuts[key])
            thetext += '$%g < %s < %g$\n' % (edges[0], key, edges[1])
    else:
        for key in cuts.keys():
            cuts_transfer[key+"_gen"] = cuts[key]
            theax = Htransfer.axes[key+'_gen']
            edges = theax.bin(cuts[key])
            thetext += '$%g < %s < %g$\n' % (edges[0], key+'_{gen}', edges[1])
        for key in cuts_reco.keys():
            cuts_transfer[key+"_reco"] = cuts_reco[key]
            theax = Htransfer.axes[key+'_reco']
            edges = theax.bin(cuts_reco[key])
            thetext += '$%g < %s < %g$\n' % (edges[0], key+'_{reco}', edges[1])

    transfer = Htransfer[cuts_transfer].project(wrt+'_reco', wrt+'_gen')

    transfer = transfer.values(flow=True)

    if gen:
        denom = transfer.sum(axis=0)
        transfer = transfer/denom[None, :]

        toplot_x = np.arange(transfer.shape[0]+1) -0.5
        toplot_y = transfer[:, thebin]

        plt.xlabel(wrt+'_gen')
    else:
        denom = transfer.sum(axis=1)
        transfer = transfer/denom[:, None]

        toplot_x = np.arange(transfer.shape[1]+1) -0.5
        toplot_y = transfer[thebin, :]

        plt.xlabel(wrt+'_reco')

    if thetext != '':
        plt.text(0.1, 0.1, thetext[:-1], 
                horizontalalignment='center',
                verticalalignment='bottom',
                transform=plt.gca().transAxes,
                fontsize=24,
                bbox=dict(boxstyle="round,pad=0.3", 
                          edgecolor='black', 
                          facecolor='white')
        )
    
    plt.stairs(toplot_y, toplot_x)
    plt.ylabel('Transfer')
    plt.axvline(thebin, color='red', linestyle='--')


    if show:
        plt.show()

def c_smearing_model(centralbins, themodel):
    nbins = 15
    modelsize = themodel.shape[0]//2
    
    result = np.zeros(nbins)

    for i in range(-modelsize, modelsize+1):
        target = centralbins + i
        target = np.where(target<0, -1-target, target)
        target = np.where(target>=nbins, 2*nbins-1-target, target)

        result[target] += themodel[i+modelsize]

    return result

def c_smearing_loss(themodel, c2d):
    loss = 0
    
    #MSE LOSS
    for iGen in range(c2d.shape[1]):
        if c2d[:,iGen].sum() == 0:
            continue

        thehist = c2d[:,iGen]
        pred = c_smearing_model(iGen, themodel)
        loss += np.sum(np.square(thehist - pred))

    #MONOTONICITY LOSS
    CM = 10
    middle = len(themodel)//2
    for i in range(1, middle+1):
        if themodel[middle+i] - themodel[middle+(i-1)] > 0:
            loss += CM * (themodel[middle+i] - themodel[middle+(i-1)])
        if themodel[middle-i] - themodel[middle-(i-1)] > 0:
            loss += CM * (themodel[middle-i] - themodel[middle-(i-1)])

    #SYMMETRY LOSS
    CS = 10000
    for i in range(1, middle+1):
        loss += CS * np.square(themodel[middle+i] - themodel[middle-i])

    #NORMALIZATION LOSS
    CN = 100
    norm = np.sum(themodel)
    loss += CN * np.square(norm - 1)

    return loss

def fit_c_smearing(Htransfer, cuts, cuts_reco=None):
    cuts_transfer = {}
    if cuts_reco is None:
        for key in cuts.keys():
            cuts_transfer[key+"_reco"] = cuts[key]
            cuts_transfer[key+"_gen"] = cuts[key]
    else:
        for key in cuts.keys():
            cuts_transfer[key+"_gen"] = cuts[key]
        for key in cuts_reco.keys():
            cuts_transfer[key+"_reco"] = cuts_reco[key]

    transfer = Htransfer[cuts_transfer].project('c_reco', 'c_gen')
    transfer = transfer.values(flow=True)

    denom = transfer.sum(axis=0)
    transfer = transfer/denom[None, :]

    import scipy.optimize as opt
    p0 = np.zeros(15)
    res = opt.minimize(
        c_smearing_loss, p0,
        args = transfer,
        method='L-BFGS-B',
        bounds = [(0, 1)]*15,
    )
    
    for i in [0, 2, 4, 6, 8, 10, 12, 14]:
        plot_transfer_1d(Htransfer, 'c', i, True, cuts, cuts_reco, show=False)
        themodel = c_smearing_model(i, res.x)
        plt.errorbar(np.arange(15), themodel, xerr=0.5, fmt='o', label='fit')

    plt.show()
    return res

def fit_all_c_smearings(Htransfer):
    import scipy.optimize as opt
    from tqdm import tqdm
    result = np.zeros_like(Htransfer.values(flow=True))
    for pt_reco in tqdm(range(Htransfer.axes['pt_reco'].extent)):
        for R_reco in range(Htransfer.axes['R_reco'].extent):
            for r_reco in range(Htransfer.axes['r_reco'].extent):
                for pt_gen in range(Htransfer.axes['pt_gen'].extent):
                    for R_gen in range(Htransfer.axes['R_gen'].extent):
                        for r_gen in range(Htransfer.axes['r_gen'].extent):
                            print("%d %d %d %d %d %d" % (pt_reco, R_reco, r_reco, pt_gen, R_gen, r_gen))
                            c2d = Htransfer.values(flow=True)[pt_reco, R_reco, r_reco, :, pt_gen, R_gen, r_gen, :]
                            if c2d.sum() == 0:
                                continue

                            denom = c2d.sum(axis=0)
                            denom = np.where(denom == 0, 1, denom)
                            c2d = c2d/denom[None, :]

                            p0 = np.zeros(15)
                            res = opt.minimize(
                                c_smearing_loss, p0,
                                args = c2d,
                                method='L-BFGS-B',
                                bounds = [(0, 1)]*15,
                            )
                            if not res.success:
                                print("MINIMIZATION FAILED")
                                print(res)
                            else:
                                for iGen in range(c2d.shape[1]):
                                    pred = c_smearing_model(iGen, res.x)
                                    result[pt_reco, R_reco, r_reco, :, pt_gen, R_gen, r_gen, iGen] = pred
    return result

def plot_purity_stability(Htransfer, wrt, cuts):
    cuts_transfer = {}
    for key in cuts.keys():
        cuts_transfer[key+"_reco"] = cuts[key]
        cuts_transfer[key+"_gen"] = cuts[key]

    transfer = Htransfer[cuts_transfer]
    if wrt is not None:
        transfer = transfer.project(wrt+'_reco', wrt+'_gen')
        transfer = transfer.values(flow=True)
    else:
        genaxes = []
        recoaxes = []
        genshape = 1
        recoshape = 1
        for axis in transfer.axes:
            if axis.name.endswith('_gen'):
                genaxes.append(axis.name)
                genshape *= axis.extent
            elif axis.name.endswith('_reco'):
                recoaxes.append(axis.name)
                recoshape *= axis.extent
        transfer = transfer.project(*recoaxes, *genaxes)
        transfer = transfer.values(flow=True).reshape((recoshape, genshape))

    denom1 = transfer.sum(axis=0)
    purity = np.diag(transfer/denom1[None, :])

    denom2 = transfer.sum(axis=1)
    stability = np.diag(transfer/denom2[:, None])

    x = np.arange(transfer.shape[0])

    plt.errorbar(x, purity, yerr=0, xerr = 0.5, label='Purity', fmt='o')
    plt.errorbar(x, stability, yerr=0, xerr = 0.5, label='Stability', fmt='o')
    if wrt is not None:
        plt.xlabel(wrt)

    plt.ylabel('Purity/Stability')
    plt.axhline(1, color='red', linestyle='--')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.show()

def plot_correlations(invhess, data=False, isCMS=True,
                      savefig=None):
    fig = plt.figure(figsize=config['Figure_Size'])
    try:
        err = np.sqrt(np.diag(invhess))
        corrmat = invhess / np.outer(err, err)
        ax = fig.add_subplot(111)
        if isCMS:
            hep.cms.label(ax=ax, data=data, label=config['Approval_Text'])

        im = ax.imshow(corrmat, cmap='coolwarm', vmin=-1, vmax=1)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Correlation Coefficient')
        #ax.set_xticks(np.arange(corrmat.shape[0]))
        #ax.set_yticks(np.arange(corrmat.shape[1]))
        #ax.set_xticklabels(np.arange(corrmat.shape[0]))
        #ax.set_yticklabels(np.arange(corrmat.shape[1]))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        ax.set_xlabel('Bins')
        ax.set_ylabel('Bins')

        plt.tight_layout()

        if savefig is not None:
            plt.savefig(savefig, format='png', bbox_inches='tight', dpi=300)
            plt.clf()
        else:
            plt.show()
    finally:
        plt.close(fig)

def plot_pulls(res, data=False, isCMS=True,
               savefig=None):
    fig = plt.figure(figsize=config['Figure_Size'])
    try:
        pulls = res.x[7875:]
        pullerr = np.diag(np.sqrt(res.invhess[7875:, 7875:]))
        ax = fig.add_subplot(111)

        if isCMS:
            hep.cms.label(ax=ax, data=data, label=config['Approval_Text'])

        ax.errorbar(np.arange(len(pulls)), pulls, yerr=pullerr, fmt='o', label='Pulls', color='black', ecolor='gray')
        ax.axhline(0, color='red', linestyle='--')
        ax.set_xlabel('Nuisance index')
        ax.set_ylabel('Pulls')

        plt.tight_layout()
        if savefig is not None:
            plt.savefig(savefig, format='png', bbox_inches='tight', dpi=300)
            plt.clf()
        else:
            plt.show()
    finally:
        plt.close(fig)

def plot_named_pulls(res, data=False, isCMS=True,
               savefig=None):
    fig = plt.figure(figsize=config['Figure_Size'])
    try:
        pulls = res.x[7875:]
        pullerr = np.diag(np.sqrt(res.invhess[7875:, 7875:]))

        pullidxs = []
        pullnames = []
        pullvals = []
        pullerrs = []
        for key in res.namedNuisances.keys():
            pullidxs.append(key)
            pullnames.append(res.namedNuisances[key])
            pullvals.append(pulls[key])
            pullerrs.append(pullerr[key])

        ax = fig.add_subplot(111)

        if isCMS:
            hep.cms.label(ax=ax, data=data, label=config['Approval_Text'])

        ax.errorbar(np.arange(len(pullvals)), pullvals, yerr=pullerrs, fmt='o', label='Pulls', color='black', ecolor='gray')
        ax.axhline(0, color='red', linestyle='--')
        ax.set_xticks(np.arange(len(pullvals)))
        ax.set_xticklabels(pullnames, rotation=45, ha='right', rotation_mode='anchor')
        ax.set_ylabel('Pulls')

        plt.tight_layout()
        if savefig is not None:
            plt.savefig(savefig, format='png', bbox_inches='tight', dpi=300)
            plt.clf()
        else:
            plt.show()
    finally:
        plt.close(fig)

def chi2_1d(H1, H2):
    nom1 = H1[{'bootstrap' : 0}].values(flow=True).ravel()
    nom2 = H2[{'bootstrap' : 0}].values(flow=True).ravel()

    vals1 = H1.values(flow=True).reshape((H1.axes['bootstrap'].size, -1))
    vals2 = H2.values(flow=True).reshape((H2.axes['bootstrap'].size, -1))

    sums1 = np.sum(vals1, axis=1)
    sums2 = np.sum(vals2, axis=1)

    vals1 = vals1 * sums1[0] / sums1[:, None]
    vals2 = vals2 * sums2[0] / sums2[:, None]

    vals1 = vals1[1:]
    vals2 = vals2[1:]

    var1 = vals1.T @ vals1
    var2 = vals2.T @ vals2

    vardiff = var1 + var2

    errdiff = np.sqrt(np.diag(vardiff))

    chi2 = np.sum(np.square(nom1 - nom2) / errdiff**2)
    return chi2

def chi2_2d(H1, H2):
    nom1 = H1[{'bootstrap' : 0}].values(flow=True).ravel()
    nom2 = H2[{'bootstrap' : 0}].values(flow=True).ravel()
    err= nom2-nom1

    vals1 = H1.values(flow=True).reshape((H1.axes['bootstrap'].size, -1))
    vals2 = H2.values(flow=True).reshape((H2.axes['bootstrap'].size, -1))

    sums1 = np.sum(vals1, axis=1)
    sums2 = np.sum(vals2, axis=1)

    vals1 = vals1 * sums1[0] / sums1[:, None]
    vals2 = vals2 * sums2[0] / sums2[:, None]

    vals1 = vals1[1:]
    vals2 = vals2[1:]

    var1 = (vals1.T @ vals1) / vals1.shape[0]
    var2 = (vals2.T @ vals2) / vals2.shape[0]

    vardiff = var1 + var2

    #print("Computing LDLT...")
    ldlt = eigen.LDLT(vardiff)
    if ldlt.info() != eigen.ComputationInfo.Success:
        raise RuntimeError("LDLT decomposition failed: %s" % ldlt.info())
    chi2 = err @ ldlt.solve(err)
    return chi2

def covmat_trace(H, step=500, savefig=None):
    vals = H.values(flow=True).reshape((H.axes['bootstrap'].size, -1))

    sums = np.sum(vals, axis=1)
    vals = vals * sums[0] / sums[:, None]

    boots = vals[1:]
    nom = vals[0]

    DY = boots - nom[None, :]

    accumulator = np.zeros((DY.shape[1], DY.shape[1]), dtype=DY.dtype)

    traces = []
    ends = []
    for i in tqdm(range(0, DY.shape[0], step)):
        end = min(i+step, DY.shape[0])
        accumulator += DY[i:end].T @ DY[i:end]
        traces.append(np.trace(accumulator) / end)
        ends.append(end)

    fig = plt.figure(figsize=config['Figure_Size'])
    try:
        ax = fig.add_subplot(111)
        hep.cms.label(ax=ax, data=False, label=config['Approval_Text'], pad=0.05)

        ax.plot(ends, traces, marker='o')
        ax.set_xlabel('Bootstrap samples')
        ax.set_ylabel('Covariance trace')

        plt.tight_layout()

        if savefig is not None:
            plt.savefig(savefig + '_covmat_trace.png', dpi=300, format='png', bbox_inches='tight')
            plt.clf()
        else:
            plt.show()
    finally:
        plt.close(fig)

def squared_error_vs_Nboot(basepath, minboot, maxboot, bootstep, 
                           num_lines=5, savefig=None):
    import os
    invcovs = []
    covs = []
    boots = []
    for boot in range(minboot, maxboot+bootstep, bootstep):
        print(boot)
        invcovs.append(np.load(os.path.join(basepath.replace("REPLACEHERE", "boot%d"%boot), "INVCOV.npy")))
        covs.append(np.load(os.path.join(basepath.replace("REPLACEHERE", "boot%d"%boot), "COV.npy")))
        boots.append(boot)

    np.random.seed(12)  # For reproducibility

    fig = plt.figure(figsize=config['Figure_Size'])
    try:
        ax = fig.add_subplot(111)
        hep.cms.label(ax=ax, data=False, label=config['Approval_Text'], pad=0.05)

        for i in range(num_lines):
            err = np.random.random(invcovs[0].shape[0])
            vals = [err @ invcov @ err for invcov in invcovs]
            ax.plot(boots, vals, marker='o')
        ax.set_xlabel('Number of bootstrap samples')
        ax.set_ylabel('Squared error')
        ax.set_yscale('log')

        plt.tight_layout()

        if savefig is not None:
            plt.savefig(savefig + '_squared_error_vs_Nboot.png', dpi=300, format='png', bbox_inches='tight')
            plt.clf()
        else:
            plt.show()
    finally:
        plt.close(fig)

    return covs, invcovs

def eigenvals_vs_Nboot(basepath, minboot, maxboot, bootstep, savefig=None, clip_to_zero=1e-20):
    import os
    eigvals = []
    boots = []
    for boot in range(minboot, maxboot+bootstep, bootstep):
        print(boot)
        eigvals.append(np.load(os.path.join(basepath.replace("REPLACEHERE", "boot%d"%boot), "COV_EIGVALS.npy")))
        boots.append(boot)

    for i in range(len(eigvals)):
        eigvals[i][eigvals[i] < clip_to_zero] = 0

    fig = plt.figure(figsize=config['Figure_Size'])
    try:
        ax = fig.add_subplot(111)
        hep.cms.label(ax=ax, data=False, label=config['Approval_Text'], pad=0.05)

        for i, boot in enumerate(boots):
            ax.plot(np.arange(len(eigvals[i])), np.flip(eigvals[i]), marker='o', label=f'Boot {boot}')

        ax.set_yscale('log')
        ax.set_xlabel('Eigenvalue index')
        ax.set_ylabel('Eigenvalue')
        ax.legend()

        plt.tight_layout()

        if savefig is not None:
            plt.savefig(savefig + '_eigenvals_vs_Nboot.png', dpi=300, format='png', bbox_inches='tight')
            plt.clf()
        else:
            plt.show()
    finally:
        plt.close(fig)

    return eigvals
