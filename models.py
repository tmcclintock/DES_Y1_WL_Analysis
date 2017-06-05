"""
This file contains interfaces for the boost factor model and the
DeltaSigma model.
"""
import numpy as np
import os, sys
sys.path.insert(0, "../Delta-Sigma/src/wrapper/")
import py_Delta_Sigma as pyDS

#Swap between whatever model type we are working with and return
#the parameters, including their default values.
def model_swap(params, name, defaults):
    lM, c, Rmis, fmis, A, B0, Cl, Dz, ER = [defaults['lM'], defaults['c'], defaults['Rmis'], defaults['fmis'], defaults['A'], defaults['B0'], defaults['Cl'], defaults['Dz'], defaults['ER']]
    if name is "full":
        lM, c, Rmis, fmis, A, B0, Cl, Dz, ER = params
    elif name is "fixed":
        lM, c = params
    else: #Afixed
        lM, c, Rmis, fmis, B0, Cl, Dz, ER = params
        A = defaults['A']
    return [lM, c, Rmis, fmis, A, B0, Cl, Dz, ER]

#Boost factor model
def get_boost_model(params, l, z, R, name, defaults):
    lM, c, Rmis, fmis, A, B0, Cl, Dz, ER = model_swap(params, name, defaults)
    #Pivots are 30 richness, 0.5 redshift, 500 kpc physical radius
    return 1.0 - B0 * (l/30.0)**Cl * ((1.+z)/1.5)**Dz * (R/0.5)**ER

def get_delta_sigma(params, name, ds_params, k, Plin, Pnl, cosmo, defaults):
    lM, c, Rmis, fmis, A, B0, Cl, Dz, ER = model_swap(params, name, defaults)
    ds_params['Mass'] = 10**lM #Msun/h
    ds_params['concentration'] = c
    ds_params['Rmis'] = Rmis
    ds_params['fmis'] = fmis
    result = pyDS.calc_Delta_Sigma(k, Plin, k, Pnl, cosmo, ds_params)
    return result
