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
import clusterwl
Blinding_amp, lam_exp, z_exp = blinding.get_blinding_variables()

plt.rc("text", usetex=True*0)
plt.rc("font", size=20, family="serif")
plt.rc("errorbar", capsize=3)

DSlabel = r"$\Delta\Sigma$ [ M$_\odot$/pc$^2$]"
Rlabel  = r"R [Mpc]$"

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
def calc_DS_model(params, args):
    lM, c, tau, fmis, Am, B0, Rs = params
    Rp, Sigma, Sigma_mis, DeltaSigma, DeltaSigma_mis, full_DeltaSigma, ave_DeltaSigma, boost_model = get_delta_sigma(params, args)
    return Rp, full_DeltaSigma, DeltaSigma, DeltaSigma_mis, boost_model, ave_DeltaSigma

def fix_errorbars(ds, err):
    bad = err>ds
    errout = np.vstack((err, err))
    errout[0,bad] = ds[bad]-1e-2
    return errout

def plot_just_DS(params, args, i, j):
    lM, c, tau, fmis, Am, B0, Rs = params
    print "MASS HERE:",lM
    Rdata = args['Rdata']
    cuts = args['cuts']
    cov = args['cov']
    z = args['z']
    lo,hi = cuts
    good = (lo<Rdata)*(Rdata<hi)
    bad  = (lo>Rdata)+(Rdata>hi)
    ds = args['ds']
    dserr = np.sqrt(np.diag(cov))
    dserr_fixed = fix_errorbars(ds, dserr)
    Rmodel, DSfull, DSc, DSm, boost, aDS = calc_DS_model(params, args)
    #Convert to Mpc physical
    Rmodel /= h*(1+z)
    DSfull *= h*(1+z)**2
    DSc *= h*(1+z)**2
    DSm *= h*(1+z)**2
    aDS *= h*(1+z)**2
    plt.errorbar(Rdata[good], ds[good], dserr_fixed[:,good], c='k', marker='o', ls='', markersize=3, zorder=1)
    plt.errorbar(Rdata[bad], ds[bad], dserr_fixed[:,bad], c='k', marker='o', mfc='w', markersize=3, ls='', zorder=1)
    #First plot DeltaSigma
    plt.loglog(Rmodel, DSfull, c='r', zorder=0)
    plt.loglog(Rmodel, DSm, c='b', ls='--', zorder=-1)
    plt.loglog(Rmodel, DSc, c='k', ls='-.', zorder=-3)
    plt.loglog(Rmodel, DSfull*boost, c='g', ls='-', zorder=-2)
    plt.ylabel(DSlabel)
    plt.xlabel(Rlabel)
    plt.ylim(0.1, 1e3)
    plt.xlim(0.1, 30.)
    if usey1: zlabel, llabel = y1zlabels[i], y1llabels[j]
    else: zlabel, llabel = svzlabels[i], svllabels[j]
    plt.text(3, 3e2, zlabel, fontsize=18)
    plt.text(3, 1e2,  llabel, fontsize=18)
    plt.subplots_adjust(bottom=0.15, left=0.18)
    #plt.gcf().savefig("figures/deltasigma_z%d_l%d.png"%(i,j))
    plt.show()


if __name__ == '__main__':
    usey1 = True
    blinded = True
    cal = False
    useJK = False
    zs, lams = get_zs_and_lams(usey1=usey1)
    Rlams = (lams/100.0)**0.2 #Mpc/h; richness radius
    SCIs = get_sigma_crit_inverses(usey1)

    #This specifies which analysis we are doing
    #Name options are full, fixed, boostfixed or Afixed
    name = "y1"
    model_name = "full"
    if useJK: covname = "JK"
    else: covname = "SAC"

    #Loop over bins
    for i in xrange(0, -1, -1): #z bins
        #if i <2: continue
        for j in xrange(6, -1, -1): #lambda bins
            if j > 3 or j < 3: continue
            paths, args = get_args_and_paths(name, i, j, model_name, blinded, cal, useJK)
            bfpath, chainpath, likespath = paths
            chain = np.loadtxt(chainpath)
            args['cuts'] = (0.2, 999.)
            print "Working at z%d l%d for %s"%(i,j,name)

            params = np.loadtxt(bfpath)
            means = np.mean(chain,0)
            print "lM mean (blinded)",means[0]
            print means
            params = means
            params = model_swap(params, args)
            plot_just_DS(params, args, i, j)
            #plot_boost_and_resid(params, args, i, j)
            #plot_fourpanels(params, args, i, j)
