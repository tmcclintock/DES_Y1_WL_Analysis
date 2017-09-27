"""
This file contains functions used to make the analysis script easier to read. This includes file IO and loading various things.
"""
import numpy as np

base  = "/home/tmcclintock/Desktop/des_wl_work/Y1_work/data_files/"
base2 = base+"%s_tamas_files/"%bstatus
database     = base2+"full-mcal-raw_y1subtr_l%d_z%d_profile.dat"%(lj, zi)
covbase      = base2+"full-mcal-raw_y1subtr_l%d_z%d_dst_cov.dat"%(lj, zi)
boostbase    = base2+"full-mcal-raw_y1clust_l%d_z%d_pz_boost.dat"
boostcovbase = "alsothis" #DOESN'T EXIST YET

def get_zs_and_lams():

    zs    = np.genfromtxt(base+"Y1_meanz.txt")
    lams  = np.genfromtxt(base+"Y1_meanl.txt")
    return zs, lams

def get_power_spectra(zi, lj):
    k    = np.genfromtxt("P_files/k.txt")
    Plin = np.genfromtxt("P_files/plin_z%d_l%d.txt"%(i, j))
    Pnl  = np.genfromtxt("P_files/pnl_z%d_l%d.txt"%(i, j))
    return k, Plin, Pnl
    
def get_data_and_icov(zi, lj, lowcut = 0.2, highcut = 999, alldata=False):
    #lowcut is the lower cutoff, assumed to be 0.2 Mpc physical
    #highcut might not be implemented in this analysis
    R, ds, dse, dsx, dsxe = np.genfromtxt(datapath, unpack=True)
    cov = np.genfromtxt(covpath)
    indices = (R > lowcut)*(R < highcut)
    if not alldata: #If we need to make the cuts
        R   = R[indices]
        ds  = ds[indices]
        cov = cov[indices]
        cov = cov[:,indices]
    return R, ds, np.linalg.inv(cov), cov

def get_boost_data_and_cov(zi, lj, highcut, lowcut=0.2):
    boostpath = boostbase%(zi, lj)
    bcovpath  = boostcovbase%(zi, lj)
    Rb, Bp1, Be = np.genfromtxt(boostpath%(lj, zi), unpack=True)
    Bp1 = Bp1[Be > 1e-3]
    Rb  = Rb[Be > 1e-3]
    Be  = Be[Be > 1e-3]
    indices = (Rb > lowcut)*(Rb < highcut)
    Bp1 = Bp1[indices]
    Rb  = Rb[indices]
    Be  = Be[indices]
    #Radii, 1+B, B error
    #Note: high cuts are Rlams*1.5 where Rlams are now in Mpc physical
    #Note: the boost factors don't have the same number of radial bins
    #as deltasigma. This doesn't matter, because all we do is
    #de-boost the model, which fits to the boost factors independently.
    return Rb, Bp1, Be   

def get_default_ds_params(z, h):
    #DeltaSigma module parameters
    ds_params = {'NR'        : 300,
                 'Rmin'      : 0.01,
                 'Rmax'      : 200.0,
                 'Nbins'     : 15,
                 'R_bin_min' : 0.0323*h*(1+z), #Mpc/h comoving
                 'R_bin_max' : 30.0*h*(1+z), #Mpc/h comoving
                 'delta'     : 200,
                 'miscentering' : 1,
                 'averaging'    : 1,
                 'single_miscentering': 0}
    return ds_params

def get_model_defaults(h):
    #Dictionary of default starting points for the best fit
    defaults = {'lM'   : 14.37+np.log10(h),
                'c'    : 5.0,
                'Rmis' : -1.12631563312, #Need to do Rlam*exp(this)
                'fmis' : 0.22,
                'A'    : 1.02,
                'B0'   : -0.056,
                'Cl'   : 0.495,
                'Dz'   : -5.16,
                'ER'   : -0.85}
    return defaults

def get_cosmo_default():
    #The cosmology used in this analysis
    cosmo = {'h'      : 0.7,
             'om'     : 0.3,
             'ode'    : 0.7,
             'ob'     : 0.05,
             'ok'     : 0.0,
             'sigma8' : 0.8,
             'ns'     : 0.96}
    return cosmo
