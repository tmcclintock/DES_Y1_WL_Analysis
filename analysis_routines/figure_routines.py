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

#Calculate all parts of the delta sigma model
#Output units are all Msun/pc^2 and Mpc physical
def calc_DS_all(params, bf_args):
    lM, c, tau, fmis, Am, B0, Rs, sigb = model_swap(params, model_name)
    z, lam, Rlam, Rdata, ds, icov, Rb, Bp1, iBcov, cuts, cosmo, k, Plin, Pnl, Rmodel, xi_mm, R_edges, indices, model_name = bf_args

    Rp, full_DeltaSigma, ave_DeltaSigma, full_boost_model = get_delta_sigma(params, z, Rlam, cosmo, k, Plin, Pnl, Rmodel, xi_mm, Redges, model_name)

    #Convert to Mpc physical
    Rp /= (h*(1+z))
    #Convert to Msun/pc^2 physical
    full_DeltaSigma *= h*(1+z)**2
    return Rp, full_DeltaSigma

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
    print "working..."


    #plot_DS_in_bin(params, name, defaults, z, lam, R, ds, cov, extras, cuts, i, j)
