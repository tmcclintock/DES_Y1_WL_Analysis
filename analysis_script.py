"""
This is the script that needs to be run in order to actually
do the analysis.

The functions in this file include:
- a function to get the data
- a function to get the covariance matrix
- a function to get the boost factor data + covariance

The main function at the bottom contains the actual script.

NOTE: because DES Y1 hasn't had a public data release yet,
all paths here are hard-coded in, since the data cannot be included
in this repository yet.
"""
import numpy as np
from likelihood_functions import *
from get_all_data import *
import figure_routines
import sys

#Set up the assumptions
cosmo = get_cosmo_default()
h = cosmo['h'] #Hubble constant
defaults = get_model_defaults(cosmo['h'])

def find_best_fit(bf_args, name, bestfitpath):
    #Take out all of the arguments
    ds_args, boost_args, P_args, cuts, z, lam, Rlam, zs, lams, cosmo = bf_args
    R, ds, icov, ds_params = ds_args
    Rb, Bp1, Be = boost_args
    k, Plin, Pnl = P_args
    #Switch between which model we are working with
    if name is "full":
        guess = [defaults['lM']+np.log(lam/30.)*1.12/np.log(10), 
                 defaults['c'],
                 Rlam*np.exp(defaults['Rmis']),
                 defaults['fmis'], defaults['A'], defaults['B0'],
                 defaults['Cl'], defaults['Dz'], defaults['ER']]
    elif name is 'fixed':
        guess = [defaults['lM']+np.log(lam/30.)*1.12/np.log(10), 
                 defaults['c']]
        defaults['Rmis'] = Rlam*np.exp(defaults['Rmis'])
    elif name is "boostfixed":
        guess = [defaults['lM']+np.log(lam/30.)*1.12/np.log(10), 
                 defaults['c'],
                 Rlam*np.exp(defaults['Rmis']),
                 defaults['fmis'], defaults['A']]
    else: #'Afixed'
        guess = [defaults['lM']+np.log(lam/30.)*1.12/np.log(10), 
                 defaults['c'], 
                 Rlam*np.exp(defaults['Rmis']),
                 defaults['fmis'], defaults['B0'],
                 defaults['Cl'], defaults['Dz'], defaults['ER']]
    #Perform a max-likelihood analysis to find the best parameters to start the MCMC
    import scipy.optimize as op
    lnprob_args = (name, ds, icov, Rb, Bp1, Be, z, lam, Rlam, 
                   zs, lams, defaults, cuts, (ds_params, k, Plin, Pnl, cosmo))
    nll = lambda *args: -lnprob(*args)
    result = op.minimize(nll, guess, args=lnprob_args, tol=1e-1)
    print "Best fit being saved at :\n%s"%bestfitpath
    print "\tresults: ",result['x']
    print "\tsuccess = %s"%result['success']
    #print result
    np.savetxt(bestfitpath, result['x'])
    defaults['Rmis'] = -1.12631563312 #Reset this value
    return 

def do_mcmc():
    import emcee
    return 0

if __name__ == '__main__':
    #This specifies which analysis we are doing
    #Name options are full, fixed, boostfixed or Afixed
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
    Rlams = (lams/100.0)**0.2 #Mpc/h; richness radius

    bestfitbase = "bestfits/bf_%s.txt"%basesuffix
    chainbase   = "chains/chain_%s.txt"%basesuffix

    #Loop over bins
    for i in xrange(0, 3): #z bins
        if i < 2: continue
        for j in xrange(0, 7): #lambda bins
            if j > 7: continue
            print "Working at z%d l%d for %s"%(i,j,name)
            #Read in everything
            z    = zs[i,j]
            lam  = lams[i,j]
            Rlam = Rlams[i,j]
            suffix = basesuffix%(i,j)
            datapath     = database%(j,i)
            covpath      = covbase%(j,i)
            boostpath    = boostbase
            boostcovpath = boostcovbase%()
            bestfitpath  = bestfitbase%(i,j)
            chainpath    = chainbase%(i,j)
            #Note: convert Rlam to Mpc physical when we get the data for the cuts
            R, ds, icov, cov = get_data_and_icov(datapath, covpath)
            Rb, Bp1, Be = get_boost_data_and_cov(boostpath, boostcovpath, zs, lams, Rlams*1.5/h/(1+zs))
            k    = np.genfromtxt(kpath)
            Plin = np.genfromtxt(Plinpath%(i,j))
            Pnl  = np.genfromtxt(Pnlpath%(i,j))
            ds_params = get_default_ds_params(z, h)

            #Group everything up for convenience
            ds_args = (R, ds, icov, ds_params)
            boost_args = (Rb, Bp1, Be)
            P_args = (k, Plin, Pnl)
            cuts = (0.2, 999) #Radial cuts, Mpc physical
            bf_args = (ds_args, boost_args, P_args, cuts, z, lam, Rlam, zs, lams, cosmo)

            #Flow control for whatever you want to do
            find_best_fit(bf_args, name, bestfitpath)
