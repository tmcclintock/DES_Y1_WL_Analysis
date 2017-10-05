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
import models
import sys
import clusterwl #Used to get xi_mm(R) from P(k)

#Set up the assumptions
cosmo = get_cosmo_default()
h = cosmo['h'] #Hubble constant

model_name = "Mc" #Mfree, Afixed, cfixed

def find_best_fit(bf_args, bestfitpath):
    z, lam, Rlam, Rdata, ds, icov, cov, Rb, Bp1, iBcov, Bcov, cosmo, k, Plin, Pnl, Rmodel, xi_mm, Redges, inds, Am_prior, Am_prior_var, model_name = bf_args
    guess = get_model_start(model_name, lam, h)
    import scipy.optimize as op
    nll = lambda *args: -lnprob(*args)
    result = op.minimize(nll, guess, args=(bf_args,), tol=1e-2)
    print "Best fit being saved at :\n%s"%bestfitpath
    print result
    print "\tresults: ",result['x']
    print "\tsuccess = %s"%result['success']
    lM = result['x'][0]
    print "lM = %.3f"%(lM - np.log10(h))
    outmodel = models.model_swap(result['x'], model_name)
    #np.savetxt(bestfitpath, outmodel)
    return 

def do_mcmc():
    import emcee
    return 0

if __name__ == '__main__':
    usey1 = False
    zs, lams = get_zs_and_lams(usey1 = usey1)
    Rlams = (lams/100.0)**0.2 #Mpc/h; richness radius

    #This specifies which analysis we are doing
    #Name options are full, fixed, boostfixed or Afixed
    name = "sv" 
    bstatus  = "unblinded" #blinded or unblinded
    basesuffix = bstatus+"_"+name+"_z%d_l%d"    
    bestfitbase = "bestfits/bf_%s.txt"%basesuffix
    chainbase   = "chains/chain_%s.txt"%basesuffix

    import matplotlib.pyplot as plt
    #Loop over bins
    for i in xrange(0, -1, -1): #z bins
        if i > 0: continue
        for j in xrange(3, 2, -1): #lambda bins
            if j > 3 or j < 3: continue
            print "Working at z%d l%d for %s"%(i,j,name)
            z    = zs[i,j]
            lam  = lams[i,j]
            Rlam = Rlams[i,j]
            k, Plin, Pnl = get_power_spectra(i, j, usey1)
            Rmodel = np.logspace(-2, 3, num=1000, base=10) 
            xi_mm = clusterwl.xi.xi_mm_at_R(Rmodel, k, Pnl)
            #Xi_mm MUST be evaluated to higher than BAO for correct accuracy

            #Note: convert Rlam to Mpc physical when we specificy the cuts
            Rdata, ds, icov, cov, inds = get_data_and_icov(i, j, usey1=usey1)

            Rb, Bp1, iBcov, Bcov = get_boost_data_and_cov(i, j, usey1=usey1)
            bfpath    = bestfitbase%(i,j)
            chainpath = chainbase%(i,j)

            #Multiplicative prior
            Am_prior, Am_prior_var = get_Am_prior(i, j)

            #Group everything up for convenience
            Redges = get_Redges(usey1 = usey1)*h*(1+z) #Mpc/h comoving
            args = (z, lam, Rlam, Rdata, ds, icov, cov, Rb, Bp1, iBcov, Bcov, cosmo, k, Plin, Pnl, Rmodel, xi_mm, Redges, inds, Am_prior, Am_prior_var, model_name)

            #Flow control for whatever you want to do
            find_best_fit(args, bfpath)
