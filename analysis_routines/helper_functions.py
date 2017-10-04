"""
This file contains functions used to make the analysis script easier to read. This includes file IO and loading various things.
"""
import numpy as np

do_y1 = False
if do_y1:
    bstatus  = "blinded" #blinded or unblinded
    base = "/home/tmcclintock/Desktop/des_wl_work/DATA_FILES/y1_data_files/"
    #base  = "/home/tmcclintock/Desktop/des_wl_work/Y1_work/data_files/"
    base2 = base+"%s_tamas_files/"%bstatus
    database     = base2+"full-mcal-raw_y1subtr_l%d_z%d_profile.dat"
    covbase      = base2+"full-mcal-raw_y1subtr_l%d_z%d_dst_cov.dat"
    boostbase    = base2+"full-mcal-raw_y1clust_l%d_z%d_pz_boost.dat"
    boostcovbase = "alsothis" #DOESN'T EXIST YET
    zspath   = base+"Y1_meanz.txt"
    lamspath = base+"Y1_meanl.txt"
else:
    #For SV test we will only do z0 l3
    base = "/home/tmcclintock/Desktop/des_wl_work/DATA_FILES/sv_data_files/"
    database = base+"profile_z0_l3.dat"
    covbase  = base+"cov_t_z0_l3.dat"
    boostbase = base+"SV_boost_factors.txt"
    boostcovbase = "need"
    zspath   = base+"SV_meanz.txt"
    lamspath = base+"SV_meanl.txt"

def get_zs_and_lams():
    zs    = np.genfromtxt(zspath)
    lams  = np.genfromtxt(lamspath)
    return zs, lams

def get_lams(fory1):
    if fory1: lampath="/home/tmcclintock/Desktop/des_wl_work/DATA_FILES/y1_data_files/Y1_meanl.txt"
    else: lampath="/home/tmcclintock/Desktop/des_wl_work/DATA_FILES/sv_data_files/SV_meanl.txt"
    return np.genfromtxt(lamspath)

def get_power_spectra(zi, lj):
    k    = np.genfromtxt(base+"P_files/k.txt")
    Plin = np.genfromtxt(base+"P_files/plin_z%d_l%d.txt"%(zi, lj))
    Pnl  = np.genfromtxt(base+"P_files/pnl_z%d_l%d.txt"%(zi, lj))
    return k, Plin, Pnl
    
def get_data_and_icov(zi, lj, lowcut = 0.2, highcut = 999, alldata=False):
    #lowcut is the lower cutoff, assumed to be 0.2 Mpc physical
    #highcut might not be implemented in this analysis
    if do_y1:
        datapath = database%(zi, lj)
        R, ds, dse, dsx, dsxe = np.genfromtxt(datapath, unpack=True)
        cov = np.genfromtxt(covpath)
        indices = (R > lowcut)*(R < highcut)
    else: #use_sv
        highcut=21.5 #Just for z0 l3 in SV
        R, ds, dse, dsx, dsxe = np.genfromtxt(database, unpack=True)
        cov = np.genfromtxt(covbase)
        indices = (R > lowcut)*(R < highcut)
    if not alldata: #If we need to make the cuts
        R   = R[indices]
        ds  = ds[indices]
        cov = cov[indices]
        cov = cov[:,indices]
    return R, ds, np.linalg.inv(cov), cov

def get_boost_data_and_cov(zi, lj, lowcut = 0.2, highcut = 999):
    if do_y1:
        boostpath = boostbase%(zi, lj)
        bcovpath  = boostcovbase%(zi, lj)
        Rb, Bp1, Be = np.genfromtxt(boostpath%(lj, zi), unpack=True)
        Bp1 = Bp1[Be > 1e-5]
        Rb  = Rb[Be > 1e-5]
        Be  = Be[Be > 1e-5]
        indices = (Rb > lowcut)*(Rb < highcut)
        Bp1 = Bp1[indices]
        Rb  = Rb[indices]
        Be  = Be[indices]
        #Note: high cuts are Rlams*1.5 where Rlams are now in Mpc physical
        #Note: the boost factors don't have the same number of radial bins
        #as deltasigma. This doesn't matter, because all we do is
        #de-boost the model, which fits to the boost factors independently.
        #NEED TO BE ABLE TO RETURN A COVARIANCE MATRIX
        return Rb, Bp1, Be
    else: #use_sv
        Bp1, Rb = np.genfromtxt(boostbase, unpack=True, skip_header=1)
        #SV didn't have boost errors. We construct them instead
        del2 = 10**-4.09 #BF result from SV
        Be = np.sqrt(del2/Rb**2)
        highcut = 20. #only for SV z0 l3
        indices = (Rb > lowcut)*(Rb < highcut)
        Bp1 = Bp1[indices]
        Rb  = Rb[indices]
        Be  = Be[indices]
        Bcov = np.diag(Be**2)
        return Rb, Bp1, np.linalg.inv(Bcov), Bcov

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
    defaults = {'lM'   : 14.37+np.log10(h), #Result of SV relation
                'conc'    : 5.0, #Arbitrary
                'tau' : 0.153, #Y1
                'fmis' : 0.32, #Y1
                'Am'    : 1.02, #SV result still...
                'B0'   : 0.07, #Y1
                'Rs'   : 2.49,  #Y1; Mpc physical
                'sig_b': 0.005} #Y1 boost scatter
    return defaults

def get_model_start(model_name, lam, h):
    defaults = get_model_defaults(h)
    #M is in Msun/h
    lM_guess = defaults['lM']+np.log(lam/30.)*1.12/np.log(10)
    if model_name is "full":
        guess = [lM_guess,
                 defaults['conc'],
                 defaults['tau'],
                 defaults['fmis'], 
                 defaults['Am'],
                 defaults['B0'],
                 defaults['Rs']]
    elif model_name is "Mfree":
        guess = [lM_guess]
    return guess

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
