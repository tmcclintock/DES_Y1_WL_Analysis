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

model_name = "full" #Mfree, Afixed, cfixed

def find_best_fit(bf_args, bestfitpath):
    z, lam, Rlam, Rdata, ds, icov, cov, Rb, Bp1, iBcov, Bcov, cuts, cosmo, k, Plin, Pnl, Rmodel, xi_mm, Redges, indices, model_name = bf_args
    guess = get_model_start(model_name, lam, cosmo['h'])
    import scipy.optimize as op
    nll = lambda *args: -lnprob(*args)
    result = op.minimize(nll, guess, args=(bf_args,), tol=1e-2)
    print "Best fit being saved at :\n%s"%bestfitpath
    print result
    print "\tresults: ",result['x']
    print "\tsuccess = %s"%result['success']
    outmodel = models.model_swap(result['x'], model_name)
    np.savetxt(bestfitpath, outmodel)
    return 

def do_mcmc():
    import emcee
    return 0

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

    import matplotlib.pyplot as plt
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

            #Note: convert Rlam to Mpc physical when we specificy the cuts
            Rdata, ds, icov, cov = get_data_and_icov(i, j)

            Rb, Bp1, iBcov, Bcov = get_boost_data_and_cov(i, j, highcut=Rlam*1.5/h/(1+z))
            bfpath    = bestfitbase%(i,j)
            chainpath = chainbase%(i,j)

            #Group everything up for convenience
            cuts = (0.2, 21.5) #Radial cuts, Mpc physical, 20 just for SV
            Redges = np.logspace(np.log10(0.02), np.log10(30.), num=Nbins+1)
            Rmeans = 2./3. * (Redges[1:]**3 - Redges[:-1]**3)/(Redges[1:]**2 - Redges[:-1]**2) #Mpc physical
            Redges *= h*(1+z) #Mpc/h comoving
            indices = (Rmeans > cuts[0])*(Rmeans < cuts[1])
            args = (z, lam, Rlam, Rdata, ds, icov, cov, Rb, Bp1, iBcov, Bcov, cuts, cosmo, k, Plin, Pnl, Rmodel, xi_mm, Redges, indices, model_name)

            #Flow control for whatever you want to do
            find_best_fit(args, bfpath)
