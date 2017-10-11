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
plt.rc("text", usetex=True)
plt.rc("font", size=24)
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
defaults = get_model_defaults(h)
model_name = "Mfree" #Mfree, Afixed, cfixed

#Calculate all parts of the delta sigma model
#Output units are all Msun/pc^2 and Mpc physical
def calc_DS_model(params, args):
    z, lam, Rlam, Rdata, ds, icov, cov, Rb, Bp1, iBcov, Bcov, cuts, cosmo, k, Plin, Pnl, Rmodel, xi_mm, R_edges, sigma_crit_inv, model_name, usey1 = args
    lM, c, tau, fmis, Am, B0, Rs, sigb = params

    Rp, Sigma, Sigma_mis, DeltaSigma, DeltaSigma_mis, full_DeltaSigma, ave_DeltaSigma, boost_model = get_delta_sigma_all_parts(params, z, Rlam, cosmo, k, Plin, Pnl, Rmodel, xi_mm, Redges, sigma_crit_inv, model_name)
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
    z, lam, Rlam, Rdata, ds, icov, cov, Rb, Bp1, iBcov, Bcov, cuts, cosmo, k, Plin, Pnl, Rmodel, xi_mm, R_edges, sigma_crit_inv, model_name, usey1 = args

    h = cosmo['h']
    lo,hi = cuts
    good = (lo<Rdata)*(Rdata<hi)
    bad  = (lo>Rdata)+(Rdata>hi)
    dserr = np.sqrt(np.diag(cov))
    dserr = fix_errorbars(ds, dserr)
    Rmodel, DSfull, DSc, DSm, boost, aDS = calc_DS_model(params, args)
    #Convert to Mpc physical
    Rmodel /= (h*(1+z))
    DSfull *= (h*(1+z)**2)
    DSc *= (h*(1+z)**2)
    DSm *= (h*(1+z)**2)

    Nplots = 1
    fig, axarr = plt.subplots(Nplots, sharex=True)
    if Nplots == 1: axarr = [axarr]
    axarr[0].errorbar(Rdata[good], ds[good], dserr[:,good], c='k', marker='o', 
                 ls='', markersize=3, zorder=1)
    axarr[0].errorbar(Rdata[bad], ds[bad], dserr[:,bad], c='k', marker='o', mfc='w', markersize=3, ls='', zorder=1)
    axarr[0].loglog(Rmodel, DSfull, c='r', zorder=0)
    axarr[0].loglog(Rmodel, DSm, c='b', ls='--', zorder=-1)
    axarr[0].loglog(Rmodel, DSc, c='k', ls='-.', zorder=-3)
    axarr[0].loglog(Rmodel, DSfull*boost, c='g', ls='-', zorder=-2)
    axarr[0].set_ylabel(DSlabel)
    if Nplots == 2:
        axarr[1].errorbar(Rb, Bp1, np.sqrt(np.diag(Bcov)), c='b')
        axarr[1].plot(Rmodel, boost, c='k')
    #axarr[0].set_xlabel(Rlabel)
    #plt.text(2, 100, "%s\n%s"%(zlabels[i], llabels[j]))
    #plt.title(r"SV $z%d\lambda%d$"%(i,j))
    plt.subplots_adjust(bottom=0.17, left=0.2, hspace=0.0)
    axarr[0].set_ylim(0.1, 1e3)
    axarr[0].set_xlim(0.1, 30.)
    if usey1:
        axarr[0].text(2, 300, y1zlabels[i])
        axarr[0].text(2, 100, y1llabels[j])
        axarr[0].get_xaxis().set_visible(False)
        axarr[0].get_yaxis().set_visible(False)
        axarr[0].get_xaxis().set_ticks([])
        axarr[0].get_yaxis().set_ticks([])

    else:
        axarr[0].text(3, 300, svzlabels[i])
        axarr[0].text(3, 100, svllabels[j])
    plt.show()

if __name__ == '__main__':
    usey1 = True
    zs, lams = get_zs_and_lams(usey1=usey1)
    Rlams = (lams/100.0)**0.2 #Mpc/h; richness radius
    SCIs = get_sigma_crit_inverses(usey1)

    #This specifies which analysis we are doing
    #Name options are full, fixed, boostfixed or Afixed
    if usey1: name = "y1"
    else:  name = "sv" 
    bstatus  = "blinded" #blinded or unblinded
    basesuffix = bstatus+"_"+name+"_z%d_l%d"    
    bestfitbase = "bestfits/bf_%s.txt"%basesuffix
    chainbase   = "chains/chain_%s.txt"%basesuffix

    #The bin edges in Mpc physical
    Nbins = 15
    Redges = np.logspace(np.log(0.0323), np.log(30.), num=Nbins+1, base=np.e)

    #Loop over bins
    for i in xrange(2, -1, -1): #z bins
        if i > 0: continue
        for j in xrange(6, -1, -1): #lambda bins
            if j > 3 or j < 3: continue
            print "Working at z%d l%d for %s"%(i,j,name)
            #Read in everything
            z    = zs[i,j]
            lam  = lams[i,j]
            Rlam = Rlams[i,j]
            sigma_crit_inv = SCIs[i,j]/(h*(1+z)**2) #Convert to Msun h/pc^2 comoving
            k, Plin, Pnl = get_power_spectra(i, j, usey1=usey1)

            Rmodel = np.logspace(-2, 3, num=1000, base=10) 
            xi_mm = clusterwl.xi.xi_mm_at_R(Rmodel, k, Pnl)
            #Xi_mm MUST be evaluated to higher than BAO for correct accuracy
            #Group everything up for convenience
            Redges = get_Redges(usey1 = usey1)*h*(1+z) #Mpc/h comoving
            Rdata, ds, icov, cov, inds = get_data_and_icov(i, j, usey1=usey1, alldata=True)
            Rb, Bp1, iBcov, Bcov = get_boost_data_and_cov(i, j, usey1=usey1)

            cuts = get_cuts(i, j, usey1=usey1)
            args = (z, lam, Rlam, Rdata, ds, icov, cov, Rb, Bp1, iBcov, Bcov, cuts, cosmo, k, Plin, Pnl, Rmodel, xi_mm, Redges, sigma_crit_inv, model_name, usey1)
            
            params = np.loadtxt(bestfitbase%(i,j))

            plot_DS_in_bin(params, args, i, j)
