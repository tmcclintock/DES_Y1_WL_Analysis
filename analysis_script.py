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

#The cosmology used in this analysis
cosmo = {'h'      : 0.7,
         'om'     : 0.3,
         'ode'    : 0.7,
         'ob'     : 0.05,
         'ok'     : 0.0,
         'sigma8' : 0.8,
         'ns'     : 0.96}
h = cosmo['h'] #Hubble constant

#Dictionary of default starting points for the best fit
defaults = {'lM'   : 13.5,
           'c'    : 4.0,
           'Rmis' : 0.3,
           'fmis' : 0.22,
           'A'    : 1.0,
           'B0'   : -0.056,
           'Cl'   : 0.495,
           'Dz'   : -5.16,
           'ER'   : -0.85}

def find_best_fit(bf_args, name, bestfitpath):
    #Take out all of the arguments
    ds_args, boost_args, P_args, cuts, z, lam, Rlam, zs, lams, cosmo = bf_args
    R, ds, icov, ds_params = ds_args
    Rb, Bp1, Be = boost_args
    k, Plin, Pnl = P_args
    #Switch between which model we are working with
    if name is "full":
        guess = [defaults['lM'], defaults['c'], defaults['Rmis'],
                 defaults['fmis'], defaults['A'], defaults['B0'],
                 defaults['Cl'], defaults['Dz'], defaults['ER']]
    elif name is 'fixed':
        guess = [defaults['lM'], defaults['c']]
    else: #'Afixed'
        guess = [defaults['lM'], defaults['c'], defaults['Rmis'],
                 defaults['fmis'], defaults['B0'],
                 defaults['Cl'], defaults['Dz'], defaults['ER']]
    #Perform a max-likelihood analysis to find the best parameters to start the MCMC
    import scipy.optimize as op
    lnprob_args = (name, R, ds, icov, Rb, Bp1, Be, z, lam, Rlam, 
                   zs, lams, defaults, cuts, (ds_params, k, Plin, Pnl, cosmo))
    nll = lambda *args: -lnprob(*args)
    result = op.minimize(nll, guess, args=lnprob_args, tol=1e-3)
    print "Best fit being saved at :\n%s"%bestfitpath
    print "\tresults: ",result['x']
    print "\tsucces = %s"%result['success']
    #print result
    np.savetxt(bestfitpath, result['x'])
    return 

def do_mcmc():
    import emcee
    return 0

if __name__ == '__main__':
    #This specifies which analysis we are doing
    #Name options are full, fixed or Afixed
    name = "full" 
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

    #Loop over bins
    for i in xrange(0, 3): #z bins
        if i > 0: continue
        for j in xrange(0, 7): #lambda bins
            if j < 0: continue
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
            R, ds, icov = get_data_and_icov(datapath, covpath)
            Rb, Bp1, Be = get_boost_data_and_cov(boostpath, boostcovpath, zs, lams, Rlams*1.5/h/(1+zs))
            k    = np.genfromtxt(kpath)
            Plin = np.genfromtxt(Plinpath%(i,j))
            Pnl  = np.genfromtxt(Pnlpath%(i,j))

            #DeltaSigma module parameters
            ds_params = {'NR'        : 300,
                         'Rmin'      : 0.01,
                         'Rmax'      : 200.0,
                         'Nbins'     : 15,
                         'R_bin_min' : 0.0323*h*(1+z), #Mpc/h comoving
                         'R_bin_max' : 30.0*h*(1+z), #Mpc/h comoving
                         'delta'     : 200,
                         'miscentering' : 1,
                         'averaging'    : 1}

            #Group everything up for convenience
            ds_args = (R, ds, icov, ds_params)
            boost_args = (Rb, Bp1, Be)
            P_args = (k, Plin, Pnl)
            cuts = (0.2, 999) #Radial cuts, Mpc comoving
            bf_args = (ds_args, boost_args, P_args, cuts, z, lam, Rlam, zs, lams, cosmo)

            #Flow control for whatever you want to do
            find_best_fit(bf_args, name, bestfitpath)
