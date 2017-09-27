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
from helper_functions import *
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
    zs, lams = get_zs_and_lams()
    Rlams = (lams/100.0)**0.2 #Mpc/h; richness radius

    #This specifies which analysis we are doing
    #Name options are full, fixed, boostfixed or Afixed
    name = "fixed" 
    bstatus  = "blinded" #blinded or unblinded
    basesuffix = bstatus+"_"+name+"_z%d_l%d"    
    bestfitbase = "bestfits/bf_%s.txt"%basesuffix
    chainbase   = "chains/chain_%s.txt"%basesuffix

    

    #Loop over bins
    for i in xrange(2, -1, -1): #z bins
        if i < 2: continue
        for j in xrange(6, 5, -1): #lambda bins
            if j < 6: continue
            print "Working at z%d l%d for %s"%(i,j,name)
            #Read in everything
            z    = zs[i,j]
            lam  = lams[i,j]
            Rlam = Rlams[i,j]
            k, Plin, Pnl = get_power_sepctra(i, j)
            #Calculate xi_mm here...
            #Note: convert Rlam to Mpc physical when we get the data for the cuts
            Rdata, ds, icov, cov = get_data_and_icov(i, j)
            Rb, Bp1, Be = get_boost_data_and_cov(boostpath, boostcovpath, Rlam*1.5/h/(1+z))
            #ds_params = get_default_ds_params(z, h) # NOT NEEDED ANYMORE

            bfpath    = bestfitbase %(i,j)
            chainpath = chainbase%(i,j)

            #Group everything up for convenience
            cuts = (0.2, 999) #Radial cuts, Mpc physical
            args = (z, lam, Rdata, ds, icov, Rb, Bp1, Be, cuts, cosmo, k, Plin, Pnl)#xi_mm
            
            #bf_args = (ds_args, boost_args, P_args, cuts, z, lam, Rlam, zs, lams, cosmo)

            #Flow control for whatever you want to do
            #find_best_fit(bf_args, name, bestfitpath)
