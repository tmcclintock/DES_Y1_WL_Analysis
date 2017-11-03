"""
This contains the likelihood functions used in the analysis, 
including priors, as well as our model of the boost factors.

NOTE: I need to move the dependenceo of py_Delta_Sigma into the models.py file.
"""
import numpy as np
import os, sys
import helper_functions as hf
from models import *
import clusterwl
cosmo = hf.get_cosmo_default()
h = cosmo['h']

def lnprior(params, Am_prior, Am_prior_var):
    lM, c, tau, fmis, Am, B0, Rs, sigb = params
    if lM < 11.0 or lM > 18.0 or c <= 0.0 or c > 20.0 or Am <= 0.0 or tau <= 0.0 or Rs <=0.0 or B0 < 0.0 or sigb < 0.0 or fmis < 0.0 or fmis > 1.0: return -np.inf
    LPfmis = (0.32 - fmis)**2/0.05**2 #Y1
    LPtau  = (0.153 - tau)**2/0.03**2 #Y1
    LPA    = (Am_prior - Am)**2/Am_prior_var #Y1
    return -0.5*(LPfmis + LPtau + LPA)

def lnlike(params, args):
    lM, c, tau, fmis, Am, B0, Rs, sigb = params

    Rp, Sigma, Sigma_mis, DScen, DSmis, full_DeltaSigma, ave_DeltaSigma, full_boost_model = get_delta_sigma(params, args)
    inds = args['inds']
    z = args['z']
    ds = args['ds']
    icov = args['icov']
    ds_model = ave_DeltaSigma[inds]
    ds_model *= h*(1+z)**2 #convert to Msun/pc^2 physical
    X = ds - ds_model
    LLDS = -0.5*np.dot(X, np.dot(icov, X))
    
    Rb = args['Rb']
    Bp1 = args['Bp1']
    iBcov = args['iBcov']
    Berr = np.sqrt(np.diag(args['Bcov']))
    boost_model = clusterwl.boostfactors.boost_nfw_at_R(Rb, B0, Rs)
    Xb = Bp1 - boost_model
    LLboost = -0.5*np.dot(Xb, np.dot(iBcov, Xb))
    return LLDS + LLboost

def lnprob(params, args):
    z = args['z']
    blinding_factor = args['blinding_factor']
    model_name = args['model_name']
    Am_prior = args['Am_prior']
    Am_prior_var = args['Am_prior_var']
    if 'bf_defaults' in args:
        pars = model_swap(params, z, blinding_factor, model_name, args['bf_defaults'])
    else:
        pars = model_swap(params, z, blinding_factor, model_name)
    lpr = lnprior(pars, Am_prior, Am_prior_var)
    if not np.isfinite(lpr): return -np.inf
    return lpr + lnlike(pars, args)
