"""
This file contains two functions to read in the weak lensing
and boost factor data.
"""
import numpy as np

def get_data_and_icov(datapath, covpath, lowcut = 0.2, highcut = 999, alldata=False):
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

def get_boost_data_and_cov(boostpath, boostcovpath, zs, lams, highcuts, lowcut=0.2):
    #Radii, 1+B, B error
    #Note: high cuts are Rlams*1.5 where Rlams are now in Mpc physical
    #Note: the boost factors don't have the same number of radial bins
    #as deltasigma. This doesn't matter, because all we do is
    #de-boost the model, which fits to the boost factors independently.
    Bp1 = []
    Be  = []
    Rb  = []
    for i in range(len(zs)):
        Bp1i  = []
        Bei = []
        Rbi    = []
        for j in xrange(0,len(zs[i])):
            Rbij, Bp1ij, Beij = np.genfromtxt(boostpath%(j, i), unpack=True)
            Bp1ij = Bp1ij[Beij > 1e-3]
            Rbij  = Rbij[Beij > 1e-3]
            Beij  = Beij[Beij > 1e-3]
            indices = (Rbij > lowcut)*(Rbij < highcuts[i,j])
            Bp1ij = Bp1ij[indices]
            Rbij  = Rbij[indices]
            Beij  = Beij[indices]
            Bp1i.append(Bp1ij)
            Bei.append(Beij)
            Rbi.append(Rbij)
        Bp1.append(Bp1i)
        Be.append(Bei)
        Rb.append(Rbi)
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
