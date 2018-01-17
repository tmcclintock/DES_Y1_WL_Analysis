"""
This contains the likelihood functions used in the analysis, 
including priors, as well as our model of the boost factors.
"""
import numpy as np
from models import *

def lnprior(params, Am_prior, Am_prior_var):
    lM, c, tau, fmis, Am, B0, Rs = params
    if lM < 11.0 or lM > 18.0 or c <= 0.0 or c > 20.0 or Am <= 0.0 or tau <= 0.0 or Rs <=0.0 or B0 < 0.0 or fmis < 0.0 or fmis > 1.0: return -np.inf
    LPfmis = (0.32 - fmis)**2/0.05**2 #Y1
    LPtau  = (0.153 - tau)**2/0.03**2 #Y1
    LPA    = (Am_prior - Am)**2/Am_prior_var #Y1
    return -0.5*(LPfmis + LPtau + LPA)

def lnlike(params, args):
    z = args['z']
    inds = args['inds']
    ds = args['ds']
    icov = args['icov']
    h = args['h']

    Rp, Sigma, Sigma_mis, DScen, DSmis, full_DeltaSigma, ave_DeltaSigma, boost_model_at_Rmodel = get_delta_sigma(params, args)
    ds_model = ave_DeltaSigma[inds] #Scale cuts
    ds_model *= h*(1+z)**2 #convert to Msun/pc^2 physical
    X = ds - ds_model
    LLDS = -0.5*np.dot(X, np.dot(icov, X))
    
    Bp1 = args['Bp1']
    iBcov = args['iBcov']
    boost_model = get_boost_model(params, args)
    Xb = Bp1 - boost_model
    LLboost = -0.5*np.dot(Xb, np.dot(iBcov, Xb))
    return LLDS + LLboost

def lnprob(params, args):
    z = args['z']
    Am_prior = args['Am_prior']
    Am_prior_var = args['Am_prior_var']
    pars = model_swap(params, args)
    lpr = lnprior(pars, Am_prior, Am_prior_var)
    if not np.isfinite(lpr): return -np.inf
    return lpr + lnlike(pars, args)
