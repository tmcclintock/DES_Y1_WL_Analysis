"""
This contains the likelihood functions used in the analysis, 
including priors, as well as our model of the boost factors.

NOTE: I need to move the dependenceo of py_Delta_Sigma into the models.py file.
"""
import numpy as np
import os, sys
import helper_functions as hf
from models import *
cosmo = hf.get_cosmo_default()
h = cosmo['h']

def lnprior(params, Am_prior, Am_prior_var, name):
    lM, c, tau, fmis, Am, B0, Rs, sigb = model_swap(params, name)
    if lM < 11.0 or lM > 18.0 or c <= 0.0 or c > 20.0 or Am <= 0.0 or tau <= 0.0 or Rs <=0.0 or B0 < 0.0 or sigb < 0.0 or fmis < 0.0 or fmis > 1.0: return -np.inf
    LPfmis = (0.32 - fmis)**2/0.05**2 #Y1
    LPtau  = (0.153 - tau)**2/0.03**2 #Y1
    LPA    = (Am_prior - Am)**2/Am_prior_var #Y1
    return -0.5*(LPfmis + LPtau + LPA)

def lnlike(params, args):
    z, lam, Rlam,  ds, icov, cov, Rb, Bp1, iBcov, Bcov, k, Plin, Pnl, Rmodel, xi_mm, Redges, indices, Am_prior, Am_prior_var, sigma_crit_inv, model_name = args
    lM, c, tau, fmis, Am, B0, Rs, sigb = model_swap(params, model_name)

    Rp, full_DeltaSigma, ave_DeltaSigma, full_boost_model = get_delta_sigma(params, z, Rlam, k, Plin, Pnl, Rmodel, xi_mm, Redges, sigma_crit_inv, model_name)
    ds_model = ave_DeltaSigma[indices]
    ds_model *= h*(1+z)**2 #physical
    X = ds - ds_model
    LLDS = -0.5*np.dot(X, np.dot(icov, X))

    boost_model = get_boost_model(B0, Rs, Rb)
    Xb = Bp1 - boost_model
    LLboost = -0.5*np.dot(Xb, np.dot(iBcov, Xb))

    return LLDS + LLboost

def lnprob(params, args):
    z, lam, Rlam, ds, icov, cov, Rb, Bp1, iBcov, Bcov, k, Plin, Pnl, Rmodel, xi_mm, Redges, indices, Am_prior, Am_prior_var, sigma_crit_inv, model_name = args
    lpr = lnprior(params, Am_prior, Am_prior_var, model_name)
    if not np.isfinite(lpr): return -np.inf
    return lpr + lnlike(params, args)
