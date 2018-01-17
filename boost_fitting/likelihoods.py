import numpy as np
from model import *

def lnprior(params, args):
    B0, Rs, alpha = params
    if Rs <=0.0 or B0 < 0.0:
        return -np.inf
    return 0

def lnlike(params, args):
    Bp1 = args['Bp1'] #1 + boost
    iBcov = args['iBcov'] #C_{boost}^{-1}
    boost_model = get_boost_model(params, args)
    Xb = Bp1 - boost_model
    LLboost = -0.5*np.dot(Xb, np.dot(iBcov, Xb))
    return LLboost

def lnprob(params, args):
    pars = swap(params, args)
    lpr = lnprior(pars, args)
    if not np.isfinite(lpr):
        return -1e99 #a big negative number
    return lpr + lnlike(pars, args)


