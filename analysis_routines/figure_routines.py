"""
This file contains the routines used to make figures
either used to diagnose the analysis so far or to make final
figures for the paper.
"""
import numpy as np
from helper_functions import *
from models import *
import os, sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import blinding
Blinding_amp, lam_exp, z_exp = blinding.get_blinding_variables()

plt.rc("text", usetex=True)
plt.rc("font", size=20)
plt.rc("font", family="serif")
plt.rc("errorbar", capsize=3)

DSlabel = r"$\Delta\Sigma\ [{\rm M_\odot/pc^2}]$"
Rlabel  = r"$R\ [{\rm Mpc}]$"

y1zlabels = [r"$z\in[0.2;0.35)$", r"$z\in[0.35;0.5)$", r"$z\in[0.5;0.65)$"]
y1llabels = [r"$\lambda\in[5;10)$",r"$\lambda\in[10;14)$",r"$\lambda\in[14;20)$",
             r"$\lambda\in[20;30)$",r"$\lambda\in[30;45)$",r"$\lambda\in[45;60)$",
             r"$\lambda\in[60;\infty)$"]
svzlabels = [r"$z\in[0.2,0.4)$",r"$z\in[0.4,0.6)$",r"$z\in[0.6,0.8)$"]
svllabels = [r"$\lambda\in[5;10)$",r"$\lambda\in[10;14)$",r"$\lambda\in[14;20)$",
             r"$\lambda\in[20;35)$",r"$\lambda\in[35;180)$"]

#Set up the assumptions
cosmo = get_cosmo_default()
h = cosmo['h'] #Hubble constant
om = cosmo['om']
defaults = get_model_defaults(h)
model_name = "M" #Mfree, Afixed, cfixed

#Calculate all parts of the delta sigma model
#Output units are all Msun/pc^2 and Mpc physical
def calc_DS_model(params, args):
    lM, c, tau, fmis, Am, B0, Rs, sigb = params

    Rp, Sigma, Sigma_mis, DeltaSigma, DeltaSigma_mis, full_DeltaSigma, ave_DeltaSigma, boost_model = get_delta_sigma(params, args)
    return Rp, full_DeltaSigma, DeltaSigma, DeltaSigma_mis, boost_model, ave_DeltaSigma

def fix_errorbars(ds, err):
    """
    Find locations where the errorbars are larger than the measurement.
    Correct the lower part of the bar to be at 10^-2, which is below
    the lower limit of the plot.
    """
    bad = err>ds
    errout = np.vstack((err, err))
    errout[0,bad] = ds[bad]-1e-2
    return errout

def plot_DS_in_bin(params, args, i, j):
    lM, c, tau, fmis, Am, B0, Rs, sigb = params
    Rdata = args['Rdata']
    cuts = args['cuts']
    cov = args['cov']
    z = args['z']
    lo,hi = cuts
    good = (lo<Rdata)*(Rdata<hi)
    bad  = (lo>Rdata)+(Rdata>hi)
    dserr = np.sqrt(np.diag(cov))
    dserr_fixed = fix_errorbars(ds, dserr)
    Rmodel, DSfull, DSc, DSm, boost, aDS = calc_DS_model(params, args)
    #Convert to Mpc physical
    Rmodel /= h*(1+z)
    DSfull *= h*(1+z)**2
    DSc *= h*(1+z)**2
    DSm *= h*(1+z)**2
    aDS *= h*(1+z)**2

    gs = gridspec.GridSpec(3, 6)
    axarr = [plt.subplot(gs[0:2, 0:3]), plt.subplot(gs[0:2, 3:]), plt.subplot(gs[-1, 0:3]), plt.subplot(gs[-1, 3:])]
    axarr[0].errorbar(Rdata[good], ds[good], dserr_fixed[:,good], c='k', marker='o', ls='', markersize=3, zorder=1)
    axarr[0].errorbar(Rdata[bad], ds[bad], dserr_fixed[:,bad], c='k', marker='o', mfc='w', markersize=3, ls='', zorder=1)
    #First plot DeltaSigma
    axarr[0].loglog(Rmodel, DSfull, c='r', zorder=0)
    axarr[0].loglog(Rmodel, DSm, c='b', ls='--', zorder=-1)
    axarr[0].loglog(Rmodel, DSc, c='k', ls='-.', zorder=-3)
    axarr[0].loglog(Rmodel, DSfull*boost, c='g', ls='-', zorder=-2)
    axarr[0].set_ylabel(DSlabel)

    
    pd = (ds - aDS)/aDS
    pde = dserr/aDS
    axarr[2].errorbar(Rdata[good], pd[good], pde[good], c='k', marker='o', ls='', markersize=3, zorder=1)
    axarr[2].errorbar(Rdata[bad], pd[bad], pde[bad], c='k', marker='o', mfc='w', markersize=3, ls='', zorder=1)
    axarr[2].axhline(0, ls='-', c='k')
    
    Rb = args['Rb']
    Bp1 = args['Bp1']
    Berr = np.sqrt(np.diag(args['Bcov']))
    boost_Rb = get_boost_model(B0, Rs*(h*(1+z)), Rb)
    good = (lo<Rb)*(Rb<hi)
    bad  = (lo>Rb)+(Rb>hi)
    axarr[1].errorbar(Rb[good], Bp1[good], Berr[good], c='k', marker='o', ls='', markersize=3, zorder=1)
    axarr[1].errorbar(Rb[bad], Bp1[bad], Berr[bad], c='k', marker='o', ls='', markersize=3, zorder=1, mfc='w')
    axarr[1].plot(Rmodel, boost, c='r')
    axarr[1].axhline(1, ls='-', c='k')
    axarr[1].set_yticklabels([])
    axarr[1].get_yaxis().set_visible(False)
    axarr[1].set_ylim(.9, 1.8)
    axtwin = axarr[1].twinx()
    axtwin.set_ylabel(r"$1-f_{\rm cl}$")
    axtwin.set_ylim(axarr[1].get_ylim())

    pd = (Bp1 - boost_Rb)/(boost_Rb-1)
    pde = Berr/(boost_Rb-1)
    axarr[3].errorbar(Rb[good], pd[good], pde[good], c='k', marker='o', ls='', markersize=3, zorder=1)
    axarr[3].errorbar(Rb[bad], pd[bad], pde[bad], c='k', marker='o', mfc='w', markersize=3, ls='', zorder=1)
    axarr[3].axhline(0, ls='-', c='k')
    axarr[3].set_yticklabels([])

    ylim = 1.2
    axarr[2].set_ylim(-ylim, ylim)
    axarr[3].set_ylim(-ylim, ylim)

    axarr[2].set_ylabel(r"\% Diff")#, fontsize=14)
    axarr[2].set_xlabel(Rlabel)
    axarr[3].set_xlabel(Rlabel)    
    plt.subplots_adjust(hspace=0.0, wspace=0.0, bottom=0.15, left=0.17, right=0.87)
    axarr[0].set_ylim(0.1, 1e3)
    for axinds in range(4):
        axarr[axinds].set_xscale('log')        
        axarr[axinds].set_xlim(0.1, 30.)
    axarr[0].set_xticklabels([])
    axarr[1].set_xticklabels([])
    axarr[3].set_xticks([1, 10])
    if usey1: zlabel, llabel = y1zlabels[i], y1llabels[j]
    else: zlabel, llabel = svzlabels[i], svllabels[j]
    axarr[1].text(.8, 1.65, zlabel, fontsize=18)
    axarr[1].text(.8, 1.5,  llabel, fontsize=18)
    #plt.suptitle("%s %s"%(zlabel, llabel))
    plt.show()

if __name__ == '__main__':
    usey1 = True
    blinded = True
    zs, lams = get_zs_and_lams(usey1=usey1)
    Rlams = (lams/100.0)**0.2 #Mpc/h; richness radius
    SCIs = get_sigma_crit_inverses(usey1)

    #This specifies which analysis we are doing
    #Name options are full, fixed, boostfixed or Afixed
    if blinded: bstatus  = "blinded" #blinded or unblinded
    else: bstatus = "unblinded"
    if usey1: name = "y1"
    else:
        name = "sv"
        bstatus = "unblinded"
    basesuffix = bstatus+"_"+name+"_z%d_l%d"    
    bestfitbase = "bestfits/bf_%s.txt"%basesuffix
    chainbase   = "chains/chain_%s.txt"%basesuffix

    #The bin edges in Mpc physical
    Nbins = 15
    Redges = np.logspace(np.log(0.0323), np.log(30.), num=Nbins+1, base=np.e)

    #Loop over bins
    for i in xrange(2, -1, -1): #z bins
        if i <2: continue
        for j in xrange(6, -1, -1): #lambda bins
            if j > 4 or j < 4: continue
            print "Working at z%d l%d for %s"%(i,j,name)
            #Read in everything
            z    = zs[i,j]
            lam  = lams[i,j]
            Rlam = Rlams[i,j]
            sigma_crit_inv = SCIs[i,j]*(h*(1+z)**2) #Convert to pc^2/hMsun comoving
            k, Plin, Pnl = get_power_spectra(i, j, usey1=usey1)

            Rmodel = np.logspace(-2, 3, num=1000, base=10) 
            xi_mm = clusterwl.xi.xi_mm_at_R(Rmodel, k, Pnl)
            #Xi_mm MUST be evaluated to higher than BAO for correct accuracy
            #Group everything up for convenience
            Redges = get_Redges(usey1 = usey1)*h*(1+z) #Mpc/h comoving
            Rdata, ds, icov, cov, inds = get_data_and_icov(i, j, usey1=usey1, alldata=True)
            Rb, Bp1, iBcov, Bcov = get_boost_data_and_cov(i, j, usey1=usey1, alldata=True)

            cuts = get_cuts(i, j, usey1=usey1)

            blinding_factor = 0
            if blinded: blinding_factor = np.log10(Blinding_amp) +  np.log10((lam/30.0)**lam_exp) + np.log10(((1+z)/1.5)**z_exp)
            
            args = {"z":z, "lam":lam, "Rlam":Rlam, "Rdata":Rdata, "ds":ds, "cov":cov, "icov":icov, "Rb":Rb, "Bp1":Bp1, "Bcov":Bcov, "iBcov":iBcov, "k":k, "Plin":Plin, "Pnl":Pnl, "Rmodel":Rmodel, "xi_mm":xi_mm, "Redges":Redges, "inds":inds, "sigma_crit_inv":sigma_crit_inv, "model_name":model_name, "cuts":cuts}

            params = np.loadtxt(bestfitbase%(i,j))
            params = model_swap(params, z, blinding_factor, "full")
            plot_DS_in_bin(params, args, i, j)
