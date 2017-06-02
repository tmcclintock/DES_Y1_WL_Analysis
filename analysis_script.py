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
import figure_routines
import sys

#The cosmology used in this analysis
cosmo = {'h'      : 0.7,
         'Om'     : 0.3,
         'Ode'    : 0.7,
         'Ob'     : 0.05,
         'Ok'     : 0.0,
         'sigma8' : 0.8,
         'ns'     : 0.96}
h = cosmo['h'] #Hubble constant

def get_data_and_cov(datapath, covpath, lowcut = 0.2, highcut = 999):
    #lowcut is the lower cutoff, assumed to be 0.2 Mpc physical
    #highcut might not be implemented in this analysis
    R, ds, dse, dsx, dsxe = np.genfromtxt(datapath, unpack=True)
    cov = np.genfromtxt(covpath)
    indices = (R > lowcut)*(R < highcut)
    R   = R[indices]
    ds  = ds[indices]
    cov = cov[indices]
    cov = cov[:,indices]
    return R, ds, cov

def get_boost_data_and_cov(boostpath, boostcovpath, lowcut=0.2, highcut = 999):
    #Radii, 1+B, B error
    #Note: the boost factors don't have the same number of radial bins
    #as deltasigma. This doesn't matter, because all we do is
    #de-boost the model, which fits to the boost factors independently.
    R, Bp1, Be = np.genfromtxt(boostpath, unpack=True)
    indices = (R > lowcut)*(R < highcut)
    R   = R[indices]
    Bp1 = Bp1[indices]
    Be  = Be[indices]
    return R, Bp1, Be   

def find_best_fit():
    import scipy.optimize as op
    return 0

def do_mcmc():
    import emcee
    return 0

if __name__ == '__main__':
    #This specifies which analysis we are doing
    analysis = "full" #"fixed", "Afixed"
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
    
    #Output suffix to be appended on things
    basesuffix = bstatus+"_"+analysis+"_z%d_l%d"
    
    #Read in the redshifts and richnesses
    zs   = np.genfromtxt(base+"Y1_meanz.txt")
    lams = np.genfromtxt(base+"Y1_meanl.txt")

    bestfitbase = "bestfits/bf_%s.txt"%basesuffix
    chainbase   = "chains/chain_%s.txt"%basesuffix

    #Loop over bins
    for i in xrange(0, 3): #z bins
        for j in xrange(0, 7): #lambda bins
            #Read in everything
            z = zs[i,j]
            lam = lams[i,j]
            suffix = basesuffix%(i,j)
            datapath     = database%(j,i)
            covpath      = covbase%(j,i)
            boostpath    = boostbase%(j,i)
            boostcovpath = boostcovbase%()
            bestfitpath  = bestfitbase%(i,j)
            chainpath    = chainbase%(i,j)
            R, ds, cov = get_data_and_cov(datapath, covpath)
            Rb, Bp1, Be = get_boost_data_and_cov(boostpath, boostcovpath)
