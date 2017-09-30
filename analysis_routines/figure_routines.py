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
zlabels = [r"$z\in[0.2;0.35)$", r"$z\in[0.35;0.5)$", r"$z\in[0.5;0.65)$"]
llabels = [r"$\lambda\in[5;10)$",r"$\lambda\in[10;14)$",r"$\lambda\in[14;20)$",
           r"$\lambda\in[20;30)$",r"$\lambda\in[30;45)$",r"$\lambda\in[45;60)$",
           r"$\lambda\in[60;\infty)$"]

#Set up the assumptions
cosmo = get_cosmo_default()
h = cosmo['h'] #Hubble constant
defaults = get_model_defaults(h)
model_name = "Mfree" #Mfree, Afixed, cfixed

#Calculate all parts of the delta sigma model
#Output units are all Msun/pc^2 and Mpc physical
def calc_DS_model(params, args):
    z, lam, Rlam, Rdata, ds, icov, cov, Rb, Bp1, iBcov, Bcov, cuts, cosmo, k, Plin, Pnl, Rmodel, xi_mm, R_edges, indices, model_name = args
    lM, c, tau, fmis, Am, B0, Rs, sigb = params

    Rp, Sigma, Sigma_mis, DeltaSigma, DeltaSigma_mis, full_DeltaSigma, ave_DeltaSigma, boost_model = get_delta_sigma_all_parts(params, z, Rlam, cosmo, k, Plin, Pnl, Rmodel, xi_mm, Redges, model_name)

    #Convert to Mpc physical
    Rp /= (h*(1+z))
    #Convert to Msun/pc^2 physical
    full_DeltaSigma *= h*(1+z)**2
    DeltaSigma *= h*(1+z)**2
    DeltaSigma_mis *= h*(1+z)**2
    ave_DeltaSigma *= h*(1+z)**2

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
    z, lam, Rlam, Rdata, ds, icov, cov, Rb, Bp1, iBcov, Bcov, cuts, cosmo, k, Plin, Pnl, Rmodel, xi_mm, R_edges, indices, model_name = args

    h = cosmo['h']
    lo,hi = cuts
    good = (lo<Rdata)*(Rdata<hi)
    bad  = (lo>Rdata)+(Rdata>hi)
    dserr = np.sqrt(np.diag(cov))
    dserr = fix_errorbars(ds, dserr)
    Rmodel, DSfull, DSc, DSm, boost, aDS = calc_DS_model(params, args)
    #Swap units and such
    Rmodel /= (h*(1+z))
    DSfull *= (h*(1+z)**2)
    DSc *= (h*(1+z)**2)
    DSm *= (h*(1+z)**2)

    X = ds - aDS
    chi2 = np.dot(X, np.dot(icov, X))
    print "chi2 = ", chi2

    Nplots = 2
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
    axarr[0].set_xlim(0.03, 50.)
    plt.show()

if __name__ == '__main__':
    zs, lams = get_zs_and_lams()
    Rlams = (lams/100.0)**0.2 #Mpc/h; richness radius

    #This specifies which analysis we are doing
    #Name options are full, fixed, boostfixed or Afixed
    name = "sv" 
    bstatus  = "unblinded" #blinded or unblinded
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
            k, Plin, Pnl = get_power_spectra(i, j)

            Rmodel = np.logspace(-2, 3, num=1000, base=10) 
            xi_mm = clusterwl.xi.xi_mm_at_R(Rmodel, k, Pnl)
            #Xi_mm MUST be evaluated to higher than BAO for correct accuracy
            #Group everything up for convenience
            cuts = (0.2, 21.5) #Radial cuts, Mpc physical, 20 just for SV
            Redges = np.logspace(np.log10(0.02), np.log10(30.), num=Nbins+1)
            Rmeans = 2./3. * (Redges[1:]**3 - Redges[:-1]**3)/(Redges[1:]**2 - Redges[:-1]**2) #Mpc physical
            Redges *= h*(1+z) #Mpc/h comoving
            indices = (Rmeans > cuts[0])*(Rmeans < cuts[1])
            Rdata, ds, icov, cov = get_data_and_icov(i, j, alldata=True)
            Rb, Bp1, iBcov, Bcov = get_boost_data_and_cov(i, j, highcut=Rlam*1.5/h/(1+z))

            args = (z, lam, Rlam, Rdata, ds, icov, cov, Rb, Bp1, iBcov, Bcov, cuts, cosmo, k, Plin, Pnl, Rmodel, xi_mm, Redges, indices, model_name)
            
            params = np.loadtxt(bestfitbase%(i,j))

            plot_DS_in_bin(params, args, i, j)
