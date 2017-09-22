"""
This file contains the routines used to make figures
either used to diagnose the analysis so far or to make final
figures for the paper.
"""
import numpy as np
from get_all_data import *
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

#Calculate all parts of the delta sigma model
#Output units are all Msun/pc^2 and Mpc physical
def calc_DS_all(params, name, defaults, z, lam, extras):
    ds_params, k, Plin, Pnl, cosmo = extras
    results = get_delta_sigma(params, name, ds_params, k, Plin, Pnl, cosmo, defaults)
    lM, c, Rmis, fmis, A, B0, Cl, Dz, ER = model_swap(params, name, defaults)
    result = get_delta_sigma(params, name, ds_params, k, Plin, Pnl, 
                             cosmo, defaults)

    #Convert to Mpc physical
    R = result['R']/(h*(1+z))
    #Convert to Msun/pc^2 physical
    dsc = result['delta_sigma']*h*(1+z)**2 
    dsm = result['miscentered_delta_sigma']*h*(1+z)**2
    boost_model = get_boost_model(params, lam, z, R, name, defaults) 
    dsfull = A*(dsc*(1.-fmis) + fmis*dsm)/boost_model
    return [R, dsc, dsm, boost_model, dsfull]

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

def plot_DS_in_bin(params, name, defaults, z, lam, R, ds, cov, extras, cuts, i,j):
    lo,hi = cuts
    good = (lo<R)*(R<hi)
    bad  = (lo>R)+(R>hi)
    dserr = np.sqrt(np.diag(cov))
    dserr = fix_errorbars(ds, dserr)
    Rmodel, dsc, dsm, boost, dsfull = calc_DS_all(params,name,
                                                  defaults,z,lam,extras)
    plt.errorbar(R[good], ds[good], dserr[:,good], c='k', marker='o', 
                 ls='', markersize=3, zorder=1)
    plt.errorbar(R[bad], ds[bad], dserr[:,bad], c='k', marker='o', mfc='w', 
                 markersize=3, ls='', zorder=1)
    plt.loglog(Rmodel, dsfull, c='r', zorder=0)
    plt.loglog(Rmodel, dsm, c='b', ls='--', zorder=-1)
    plt.loglog(Rmodel, dsc, c='k', ls='-.', zorder=-3)
    plt.loglog(Rmodel, dsfull*boost, c='g', ls='-', zorder=-2)
    plt.ylabel(DSlabel)
    plt.xlabel(Rlabel)
    plt.text(2, 100, "%s\n%s"%(zlabels[i], llabels[j]))
    plt.subplots_adjust(bottom=0.17, left=0.2)
    plt.ylim(0.1, 1e3)
    plt.xlim(0.03, 50.)
    plt.show()

if __name__ == '__main__':
    #This specifies which analysis we are doing
    #Name options are full, fixed or Afixed
    name = "boostfixed" 
    bstatus  = "blinded" #blinded or unblinded

    #These are the basic paths to the data
    #They are completed when the redshift/richness bin is specified
    #as well as the blinding status
    base  = "/home/tmcclintock/Desktop/des_wl_work/Y1_work/data_files/"
    base2 = base+"%s_tamas_files/"%bstatus
    database     = base2+"full-mcal-raw_y1subtr_l%d_z%d_profile.dat"
    covbase      = base2+"full-mcal-raw_y1subtr_l%d_z%d_dst_cov.dat"
    boostbase    = base2+"full-mcal-raw_y1clust_l%d_z%d_pz_boost.dat"
    boostcovbase = "alsothis" #DOESN'T EXIST YET
    kpath        = "P_files/k.txt"
    Plinpath     = "P_files/plin_z%d_l%d.txt"
    Pnlpath      = "P_files/pnl_z%d_l%d.txt"
    
    #Output suffix to be appended on things
    basesuffix = bstatus+"_"+name+"_z%d_l%d"
    
    #Read in the redshifts and richnesses
    zs    = np.genfromtxt(base+"Y1_meanz.txt")
    lams  = np.genfromtxt(base+"Y1_meanl.txt")
    Rlams = 1.0*(lams/100.0)**0.2 #Mpc/h; richness radius

    bestfitbase = "bestfits/bf_%s.txt"%basesuffix
    chainbase   = "chains/chain_%s.txt"%basesuffix

    for i in xrange(0, 3): #z bins
        if i < 2: continue
        for j in xrange(0, 7): #lambda bins
            if j < 3: continue
            print "Working at z%d l%d for %s"%(i,j,name)
            #Read in everything
            z    = zs[i,j]
            lam  = lams[i,j]
            Rlam = Rlams[i,j]
            datapath     = database%(j,i)
            covpath      = covbase%(j,i)
            bestfitpath  = bestfitbase%(i,j)
            R, ds, icov, cov = get_data_and_icov(datapath, covpath, alldata=True)
            k    = np.genfromtxt(kpath)
            Plin = np.genfromtxt(Plinpath%(i,j))
            Pnl  = np.genfromtxt(Pnlpath%(i,j))
            ds_params = get_default_ds_params(z, h)
            cuts = (0.2, 999) #Radial cuts, Mpc physical
            extras = [ds_params, k, Plin, Pnl, cosmo]

            #Find the best fit model
            params = np.loadtxt(bestfitpath)
            plot_DS_in_bin(params, name, defaults, z, lam, R, ds, cov, extras, cuts, i, j)
