"""
This contains the likelihood functions used in the analysis, 
including priors.
"""
import numpy as np
#import DeltaSigma

def lnprior(params, name):
    if name is "full":
        lM, c, Rmis, fmis, A, B0, Cl, Dz, ER = params
        #Prior on A
        #Priors on Rmis, fmix, A, B0, Cl, Dz, ER
    elif name is "fixed":
        lM, c = params
    else: #Afixed
        lM, c, Rmis, fmis, B0, Cl, Dz, ER = params
        #Priors on Rmis, fmix, A, B0, Cl, Dz, ER
    if 12.0 > lM  or lM > 17.0 or c <= 0.0: return -np.inf
    return 0

def lnlike(params, name, data, cov, R, boost, lam, z, defaults):
    if name is "full":
        lM, c, Rmis, fmis, A, B0, Cl, Dz, ER = params
    elif name is "fixed":
        lM, c = params
        Rmis, fmis, A, B0, Cl, Dz, ER = defaults
    else: #Afixed
        lM, c, Rmis, fmis, B0, Cl, Dz, ER = params
        A = defaults
    #Now get the model...
    return 0
