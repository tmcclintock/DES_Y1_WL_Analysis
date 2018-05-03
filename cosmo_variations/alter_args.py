import numpy as np
from classy import Class
import cluster_toolkit as ct

#This function call fixes the power spectrum and correlation functions
#To update: k, Plin, Pnl, xi_nl, xi_lin, xi_nl2

def update_args(args, zi, lj, H0, Om):
    h = H0/100.
    
    z = args['z']
    Ob = 0.05
    Ocdm = Om - Ob
    params = {
        'output': 'mPk',
        "h":h,
        "A_s":1.9735e-9, #Was set by hand...
        #"sigma8":0.8,
        "n_s":0.96,
        "Omega_b":Ob,
        "Omega_cdm":Ocdm,
        'YHe':0.24755048455476272,#By hand, default value
        'P_k_max_1/Mpc':1000.,
        'z_max_pk':1.0,
        'non linear':'halofit'}
    print h, Om
    cosmo = Class()
    cosmo.set(params)
    print "computing"
    cosmo.compute()
    print "sigma8 is:", cosmo.sigma8()

    k = args['k']*0.7 #In units of 1/Mpc
    Pnl  = np.array([cosmo.pk(ki, z) for ki in k]) 
    Plin = np.array([cosmo.pk_lin(ki, z) for ki in k])
    k /= h
    Pnl *= h**3
    Plin *= h**3
    args['k'] = k
    args['Plin'] = Plin
    args['Pnl'] = Pnl
    
    Rmodel = np.logspace(-2, 3, num=1000, base=10) #Mpc/h comoving
    xi_nl2  = ct.xi.xi_mm_at_R(Rmodel, k, Pnl, N=200)
    xi_nl  = ct.xi.xi_mm_at_R(Rmodel, k, Pnl)
    xi_lin = ct.xi.xi_mm_at_R(Rmodel, k, Plin)
    args['xi_nl'] = xi_nl
    args['xi_nl2'] = xi_nl2
    args['xi_lin'] = xi_lin
    return args
