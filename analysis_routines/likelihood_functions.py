"""
This contains the likelihood functions used in the analysis, 
including priors, as well as our model of the boost factors.

NOTE: I need to move the dependenceo of py_Delta_Sigma into the models.py file.
"""
import numpy as np
import os, sys
from models import *

"""
Log prior
First gets parameters
Second applies prior on the mass and concentration
Third applies prior on cluster stack parameters (Rmis, fmis, A)
Fourth (not implemented) applies priors to the boost factor parameters
"""
def lnprior(params, name, defaults, Rlam):
    lM, c, Rmis, fmis, A, B0, Cl, Dz, ER = model_swap(params, name, defaults)
    if lM < 11.0 or lM > 18.0 or c <= 0.0 or c > 20.0 or A <= 0.0 or Rmis <= 0.0: return -np.inf
    #if B0 >= 0.0 or Cl < 0.0 or Dz > 0.0 or ER > 0.0: return -np.inf
    #Priors on Rmis, fmix, A, B0, Cl, Dz, ER
    #NOTE: priors from SV used. The one on A is wrong and hasn't been
    #calculated yet for Y1. At that time I'll figure out 
    #a clever way of specifying which prior to use.
    #These are the priors on the lensing parameters
    #Note: Rlam is Mpc/h here
    LPfmis = (0.22 - fmis)**2/0.11**2
    LPRmis = (-1.12631563312 - np.log(Rmis/Rlam))**2/0.223613966662**2
    LPA    = (1.02 - A)**2/0.038**2 #NEEDS TO BE UPDATED FROM SV
    #Priors on boost factor parameters, if we want to use them.
    #these might come from doing the independent boost factor analysis.
    return -0.5*(LPfmis + LPRmis + LPA)

"""
Log posterior of the boost factor model
First gets parameters
Second loops through each redshift/richness bin, calculating the boost model and finding -chi^2/2
"""
def lnlike_boost(params, name, Rb, Bp1, Be, zs, lams, defaults):
    LLboost = 0.0
    for i in range(len(Bp1)): #Loop over all boost measurements
        for j in xrange(0,len(Bp1[i])):
            Bmodel = get_boost_model(params, lams[i,j], zs[i,j], Rb[i][j], 
                                     name, defaults)
            LLboost += np.sum(-0.5*(Bp1[i][j]-Bmodel)**2/Be[i][j]**2)
    return LLboost

"""
Log posterior of the DeltaSigma model
First gets parameters
Second sets up and calls the DeltaSigma module (pyDS)
Third calculates the boost model for this redshift/richness bin
Fourth makes cuts at the correct scales
Fifth finds -chi^2/2
"""
def lnlike_DS(params, name, ds, icov, z, lam, defaults, cuts, extras):
    lM, c, Rmis, fmis, A, B0, Cl, Dz, ER = model_swap(params, name, defaults)
    lowcut, highcut = cuts
    LLDS = 0
    ds_params, k, Plin, Pnl, cosmo = extras
    result = get_delta_sigma(params, name, ds_params, k, Plin, Pnl, 
                             cosmo, defaults)
    #Convert to Mpc physical
    Rbins = result['Rbins']/(cosmo['h']*(1+z))
    #Convert to Msun/pc^2 physical
    dsc = result['ave_delta_sigma']*cosmo['h']*(1+z)**2 
    dsm = result['ave_miscentered_delta_sigma']*cosmo['h']*(1+z)**2
    boost_model = get_boost_model(params, lam, z, Rbins, name, defaults) 
    #Cuts are in Mpc physical
    boost_model = boost_model[(Rbins > lowcut)*(Rbins < highcut)]
    dsc = dsc[(Rbins > lowcut)*(Rbins < highcut)]
    dsm = dsm[(Rbins > lowcut)*(Rbins < highcut)]
    X = ds - A*(dsc*(1.-fmis) + fmis*dsm)/boost_model
    return -0.5*np.dot(X, np.dot(icov, X))

"""
Log posterior probability, what we are probing
First calls the prior
Second calls the log likelihood of the boost factor model
Third calls the log likelihood of the DeltaSigma model
"""
def lnprob(params, name, ds, icov, Rb, Bp1, Be, z, lam, Rlam, zs, lams, defaults, cuts, extras):
    lpr = lnprior(params, name, defaults, Rlam)
    if not np.isfinite(lpr): return -np.inf
    return lpr + \
        lnlike_boost(params, name, Rb, Bp1, Be, zs, lams, defaults) + \
        lnlike_DS(params, name, ds, icov, z, lam, defaults, cuts, extras)