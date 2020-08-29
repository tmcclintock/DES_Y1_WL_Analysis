"""
This file contains interfaces for the boost factor model and the
DeltaSigma model.
"""
import numpy as np
import os, sys
import helper_functions as hf
import clusterwl
cal = hf.get_calTF()
cosmo = hf.get_cosmo_default(cal)
h = cosmo['h']
om = cosmo['om']
defaults = hf.get_model_defaults(h)

#R perpendicular
Rp = np.logspace(-2, 2.4, 1000, base=10) #Mpc/h

conc_spline = hf.get_concentration_spline(cal)

#Swap between whatever model type we are working with and return
#the parameters, including their default values.
def model_swap(params, z, blinding_factor, name, bf_defaults=None):
    if bf_defaults is not None:
        lM0, c, tau, fmis, Am, B0, Rs = bf_defaults
        sigb = defaults['sig_b'] #Not used in fits at this time
    else:
        c, tau, fmis, Am, B0, Rs, sigb = [defaults['conc'], defaults['tau'], defaults['fmis'], defaults['Am'], defaults['B0'], defaults['Rs'], defaults['sig_b']]
    if name == "full":
        lM, c, tau, fmis, Am, B0, Rs = params
    elif name == "Afixed":
        lM, c, tau, fmis, B0, Rs = params
    elif name == "Mc":
        lM, c = params
    elif name == "cfixed":
        lM, tau, fmis, Am, B0, Rs = params
        c = conc_spline(10**(lM-blinding_factor), z)
    elif name == "M":
        lM = params
        c = conc_spline(10**(lM-blinding_factor), z)
    return [lM-blinding_factor, c, tau, fmis, Am, B0, Rs, sigb]

#Boost factor variance model
def boost_variance_model(sigma, R):
    return (sigma/R)**2 #R is in Mpc, pivot is 1 Mpc

def get_delta_sigma(params, args):
    lM, c, tau, fmis, Am, B0, Rs, sigb = params
    Rmodel = args['Rmodel']
    k = args['k']
    Plin = args['Plin']
    xi_mm = args['xi_mm']
    Rlam = args['Rlam']
    z = args['z']
    sigma_crit_inv = args['sigma_crit_inv']
    Redges = args['Redges']
    M = 10**lM
    xi_nfw   = clusterwl.xi.xi_nfw_at_R(Rmodel, M, c, om)
    bias = clusterwl.bias.bias_at_M(M, k, Plin, om)
    xi_2halo = clusterwl.xi.xi_2halo(bias, xi_mm)
    xi_hm    = clusterwl.xi.xi_hm(xi_nfw, xi_2halo)
    Rp = np.logspace(-2, 2.4, 1000, base=10) #Mpc/h
    Sigma  = clusterwl.deltasigma.Sigma_at_R(Rp, Rmodel, xi_hm, M, c, om)
    DeltaSigma = clusterwl.deltasigma.DeltaSigma_at_R(Rp, Rp, Sigma, M, c, om)
    Rmis = tau*Rlam #Mpc/h
    Sigma_mis  = clusterwl.miscentering.Sigma_mis_at_R(Rp, Rp, Sigma, M, c, om, Rmis, kernel="exponential")
    DeltaSigma_mis = clusterwl.miscentering.DeltaSigma_mis_at_R(Rp, Rp, Sigma_mis)

    full_Sigma = (1-fmis)*Sigma + fmis*Sigma_mis
    full_DeltaSigma = (1-fmis)*DeltaSigma + fmis*DeltaSigma_mis
    full_DeltaSigma *= Am #multiplicative bias
    #Note: Rs is default in Mpc physical
    boost_model = clusterwl.boostfactors.boost_nfw_at_R(Rp, B0, Rs*h*(1+z))

    full_DeltaSigma /= boost_model #de-boost the model
    full_DeltaSigma /= (1-full_Sigma*sigma_crit_inv) #Reduced shear
    
    ave_DeltaSigma = np.zeros((len(Redges)-1))
    clusterwl.averaging.average_profile_in_bins(Redges, Rp, full_DeltaSigma, ave_DeltaSigma)
    return Rp, Sigma, Sigma_mis, DeltaSigma, DeltaSigma_mis, full_DeltaSigma, ave_DeltaSigma, boost_model