"""
This file contains interfaces for the boost factor model and the DeltaSigma model.
"""
import numpy as np
import cluster_toolkit as ct

#R perpendicular
Rp = np.logspace(-2, 2.4, 1000, base=10) #Mpc/h

def model_swap(params, args):
    name = args['model_name']
    z = args['z']
    defaults = args['defaults']
    blinding_factor = args['blinding_factor']
    c, tau, fmis, Am, B0, Rs = [defaults['conc'], defaults['tau'], defaults['fmis'], defaults['Am'], defaults['B0'], defaults['Rs']]
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
        conc_spline = args['cspline']
        c = conc_spline(10**(lM-blinding_factor), z)
    return [lM-blinding_factor, c, tau, fmis, Am, B0, Rs]

def get_delta_sigma(params, args):
    lM, c, tau, fmis, Am, B0, Rs = params
    Rmodel = args['r'] #3d r; Mpc/h
    k = args['k'] #h/Mpc
    Plin = args['Plin'] #(Mpc/h)^3
    xi_mm = args['xi_nl'] #can also choose xi_lin
    Rlam = args['Rlam'] #Mpc/h
    z = args['z']
    h = args['h']
    om = args['Omega_m']
    Sigma_crit_inv = args['Sigma_crit_inv'] #pc^2/hMsun comoving
    Redges = args['Redges'] #Mpc/h comoving
    M = 10**lM #Msun/h
    xi_nfw   = ct.xi.xi_nfw_at_R(Rmodel, M, c, om)
    bias = ct.bias.bias_at_M(M, k, Plin, om)
    xi_2halo = ct.xi.xi_2halo(bias, xi_mm)
    xi_hm    = ct.xi.xi_hm(xi_nfw, xi_2halo)
    Rp = np.logspace(-2, 2.4, 1000, base=10) #Mpc/h
    Sigma  = ct.deltasigma.Sigma_at_R(Rp, Rmodel, xi_hm, M, c, om)
    DeltaSigma = ct.deltasigma.DeltaSigma_at_R(Rp, Rp, Sigma, M, c, om)
    Rmis = tau*Rlam #Mpc/h
    Sigma_mis  = ct.miscentering.Sigma_mis_at_R(Rp, Rp, Sigma, M, c, om, Rmis, kernel="exponential")
    DeltaSigma_mis = ct.miscentering.DeltaSigma_mis_at_R(Rp, Rp, Sigma_mis)

    full_Sigma = (1-fmis)*Sigma + fmis*Sigma_mis
    full_DeltaSigma = (1-fmis)*DeltaSigma + fmis*DeltaSigma_mis
    full_DeltaSigma *= Am #multiplicative bias
    #Note: Rs is default in Mpc physical
    boost_model = ct.boostfactors.boost_nfw_at_R(Rp, B0, Rs*h*(1+z))

    full_DeltaSigma /= boost_model #de-boost the model
    full_DeltaSigma /= (1-full_Sigma*Sigma_crit_inv) #Reduced shear
    #Here, DeltaSigma is in Msun h/pc^2 comoving
    
    ave_DeltaSigma = ct.averaging.average_profile_in_bins(Redges, Rp, full_DeltaSigma)
    return Rp, Sigma, Sigma_mis, DeltaSigma, DeltaSigma_mis, full_DeltaSigma, ave_DeltaSigma, boost_model

def get_boost_model(params, Rb):
    lM, c, tau, fmis, Am, B0, Rs = params
    #Rb is Mpc physical; same units as Rs
    return ct.boostfactors.boost_nfw_at_R(Rb, B0, Rs)
