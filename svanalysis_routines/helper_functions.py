"""
This file contains functions used to make the analysis script easier to read. This includes file IO and loading various things.
"""
import numpy as np
import cluster_toolkit as ct
from scipy.interpolate import interp2d
import blinding

#fullbase = "/home/tmcclintock/Desktop/des_wl_work" #susie
fullbase = "/Users/tmcclintock/Data" #laptop
#fullbase = "/calvin1/tmcclintock/DES_DATA_FILES" #calvin

#SV paths
svbase = fullbase+"/DATA_FILES/sv_data_files/"
svdatabase = svbase+"profile_z%d_l%d.dat"
svcovbase  = svbase+"cov_t_z%d_l%d.dat"
svboostbase = svbase+"SV_boost_factors.txt"
svboostcovbase = "need_z%d_l%d"
svzspath   = svbase+"SV_meanz.txt"
svlamspath = svbase+"SV_meanl.txt"

def get_args_and_paths(name, zi, lj, model_name, blinded=True, cal=False, useJK=False):
    covname = "JK"
    cosmo = get_cosmo_default(cal)
    h = cosmo['h']
    om = cosmo['om']
    defaults = get_model_defaults(h)
    conc_spline = get_concentration_spline(cal)

    #First fix the paths
    basesuffix = name+"_"+covname+"_z%d_l%d"%(zi, lj)
    bfpath = "bestfits/bf_%s_%s.txt"%(model_name, basesuffix)
    chainpath   = "chains/chain_%s_%s.txt"%(model_name, basesuffix)
    likespath   = "chains/likes_%s_%s.txt"%(model_name, basesuffix)
    paths = [bfpath, chainpath, likespath]
    
    #Now prep the args
    zs, lams = get_zs_and_lams()
    Rlams = (lams/100.)**0.2 #Mpc/h
    print zs.shape, lams.shape, Rlams.shape
    z, lam, Rlam = zs[zi, lj], lams[zi, lj], Rlams[zi, lj]
    k, Plin, Pnl = get_power_spectra(zi, lj)
    Rmodel = np.logspace(-2, 3, num=1000, base=10) #Mpc/h comoving
    xi_nl  = ct.xi.xi_mm_at_R(Rmodel, k, Pnl)
    xi_lin = ct.xi.xi_mm_at_R(Rmodel, k, Plin)
    lowcut = 0.2 #Mpc physical
    #1degree cut is 21.5 Mpc for zi=0
    Rdata, ds, icov, cov, inds = get_data_and_icov(zi, lj, lowcut=lowcut)
    boostdata = get_boost_data()

    Am_prior, Am_prior_var = get_Am_prior(zi, lj)
    Redges = get_Redges(usey1 = usey1) * h*(1+z) #Mpc/h comoving
    Blinding_amp, lam_exp, z_exp = blinding.get_blinding_variables()
    blinding_factor = np.log10(Blinding_amp) +  np.log10((lam/30.0)**lam_exp) + np.log10(((1+z)/1.5)**z_exp)
    args = {"z":z, "lam":lam, "Rlam":Rlam, "k":k, "Plin":Plin, "Pnl":Pnl, "Rmodel":Rmodel, "xi_nl":xi_nl, "xi_lin":xi_lin, "Rdata":Rdata, "ds":ds, "cov":cov, "icov":icov, "Rb":Rb, "Bp1":Bp1, "Bcov":Bcov, "iBcov":iBcov, "Redges":Redges, "inds":inds, "Am_prior":Am_prior, "Am_prior_var":Am_prior_var, "sigma_crit_inv":SCI, "model_name":model_name, "zi":zi, "lj":lj, "blinding_factor":blinding_factor, 'h':h, 'om':om, 'defaults':defaults, 'cspline':conc_spline, 'xi_nl2':xi_nl2}
    return paths, args

def get_zs_and_lams():
    lams = np.genfromtxt(svlamspath)
    zs = np.genfromtxt(svzspath)
    return zs, lams

def get_power_spectra(zi, lj):
    base = svbase
    k    = np.genfromtxt(base+"P_files/k.txt")
    Plin = np.genfromtxt(base+"P_files/plin_z%d_l%d.txt"%(zi, lj))
    Pnl  = np.genfromtxt(base+"P_files/pnl_z%d_l%d.txt"%(zi, lj))
    return k, Plin, Pnl
    
def get_data_and_icov(zi, lj, lowcut = 0.2, highcut = 999, alldata=True):
    print "SV data z%d l%d"%(zi, lj)
    datapath = svdatabase%(zi, lj)
    covpath = svcovbase%(zi, lj)
    R, ds, dse, dsx, dsxe = np.genfromtxt(datapath, unpack=True)
    cov = np.genfromtxt(covpath)
    if zi == 0 and not usey1: highcut=21.5 #Just for z0 in SV
    indices = (R > lowcut)*(R < highcut)
    if alldata: indices = R > 0.0
    R   = R[indices]
    ds  = ds[indices]
    cov = cov[indices]
    cov = cov[:,indices]
    #APPLY THE HARTLAP CORRECTION HERE
    print "Hartlap applied"
    Njk = 100.
    D = len(R)
    cov = cov*((Njk-1.)/(Njk-D-2))
    return R, ds, np.linalg.inv(cov), cov, indices

def get_boost_data():
    boostpath = svbase+"/all_SV_boost_factors.txt"
    boostdata = np.loadtxt(boostpath)
    #print boostdata.shape
    #print boostdata[0]
    zi, _, _, _, R = boostdata.T
    inds = (R>0.2)
    boostdata =  boostdata[inds]
    #print boostdata.shape
    #print boostdata[0]
    zi, _, _, _, R = boostdata.T
    inds = np.invert((zi==0)*(R>21.5))
    boostdata =  boostdata[inds]
    #print boostdata.shape
    #print boostdata[:10]
    return boostdata


def get_model_defaults(h):
    #Dictionary of default starting points for the best fit
    defaults = {'lM'   : 14.37+np.log10(h), #Result of SV relation
                'lnc'  : -1.13
                'fmis' : 0.22, 
                'Am'    : 1.02,
                'lB0'  : -1.399,
                'cl'   : 0.92,
                'dz'   : -4.,
                'er': -0.98,
                'l'
    return defaults

def get_model_start(model_name, lam, h):
    defaults = get_model_defaults(h)
    #M is in Msun/h
    lM_guess = defaults['lM']+np.log(lam/30.)*1.12/np.log(10)
    if model_name == "full":
        guess = [lM_guess,
                 defaults['conc'],
                 defaults['tau'],
                 defaults['fmis'], 
                 defaults['Am'],
                 defaults['B0'],
                 defaults['Rs']]
    elif model_name == "Afixed":
        guess = [lM_guess,
                 defaults['conc'],
                 defaults['tau'],
                 defaults['fmis'], 
                 defaults['B0'],
                 defaults['Rs']]
    elif model_name == "cfixed":
        guess = [lM_guess,
                 defaults['tau'],
                 defaults['fmis'],
                 defaults['Am'],
                 defaults['B0'],
                 defaults['Rs']]
    elif model_name == "Mc":
        guess = [lM_guess, 4.5]
    elif model_name == "M":
        guess = lM_guess
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

def get_Am_prior(zi, lj):
    Am_prior = np.array([1.019, 1.020, 1.044])[zi]
    Am_prior_var = np.array([0.034**2, 0.038**2, 0.039**2])[zi]
    print "Am prior: ", zi, lj, Am_prior, np.sqrt(Am_prior_var)
    return Am_prior, Am_prior_var

def get_Redges():
    #The bin edges in Mpc physical
    Nbins = 15
    return np.logspace(np.log10(0.02), np.log10(30.), num=Nbins+1) #use_sv

#Set up the Concentration spline
def get_concentration_spline(cal=False):
    from colossus.halo import concentration
    from colossus.cosmology import cosmology
    cosmo = get_cosmo_default(cal)
    cos = {'flat':True,'H0':cosmo['h']*100.,'Om0':cosmo['om'],'Ob0':cosmo['ob'],'sigma8':cosmo['sigma8'],'ns':cosmo['ns']}
    cosmology.addCosmology('fiducial', cos)
    cosmology.setCosmology('fiducial')
    N = 20
    M = np.logspace(12, 17, N)
    z = np.linspace(0.2, 0.65, N)
    c_array = np.ones((N, N))
    for i in range(N):
        for j in range(N):
            c_array[j,i] = concentration.concentration(M[i],'200m',z=z[j],model='diemer15')
    return interp2d(M, z, c_array)
