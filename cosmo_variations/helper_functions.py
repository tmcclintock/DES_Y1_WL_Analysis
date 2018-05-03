"""
This file contains functions used to make the analysis script easier to read. This includes file IO and loading various things.
"""
import numpy as np
import cluster_toolkit as ct
from scipy.interpolate import interp2d

#fullbase = "/home/tmcclintock/Desktop/des_wl_work" #susie
fullbase = "/Users/tmcclintock/Data" #laptop
#fullbase = "/calvin1/tmcclintock/DES_DATA_FILES" #calvin
#Y1 paths
y1base = fullbase+"/DATA_FILES/y1_data_files/"
y1base2 = y1base+"FINAL_FILES/"
y1database     = y1base2+"full-unblind-v2-mcal-zmix_y1subtr_l%d_z%d_profile.dat"
y1JKcovbase      = y1base2+"full-unblind-v2-mcal-zmix_y1subtr_l%d_z%d_dst_cov.dat"
y1SACcovbase     = y1base2+"SACs/SAC_z%d_l%d.txt"
y1boostbase    = y1base+"FINAL_FILES/full-unblind-v2-mcal-zmix_y1clust_l%d_z%d_zpdf_boost.dat"
y1boostcovbase = y1base+"FINAL_FILES/full-unblind-v2-mcal-zmix_y1clust_l%d_z%d_zpdf_boost_cov.dat"

y1zspath   = y1base+"Y1_meanz.txt"
y1lamspath = y1base+"Y1_meanl.txt"
#SV paths
svbase = fullbase+"/DATA_FILES/sv_data_files/"
svdatabase = svbase+"profile_z%d_l%d.dat"
svcovbase  = svbase+"cov_t_z%d_l%d.dat"
svboostbase = svbase+"SV_boost_factors.txt"
svboostcovbase = "need_z%d_l%d"
svzspath   = svbase+"SV_meanz.txt"
svlamspath = svbase+"SV_meanl.txt"

#calibration paths

calbase = fullbase+"/DATA_FILES/calibration_data_files/"
caldatabase = calbase+"cal_ps25_z%d_l%d.txt"
calSACcovbase = y1SACcovbase
calboostbase = y1boostbase
calboostcovbase = y1boostcovbase
calzspath = calbase+"CAL_meanz.txt"
callamspath = calbase+"CAL_ps25_meanl.txt"

#Sigma crit inverse path
SCIpath = "../photoz_calibration/sigma_crit_inv.txt"

#First value rescales DeltaSigma and all covariances by the square
#Second value is the old Sigma_crit^-1 value
#Third value is the new Sigma_crit^-1 value
#Last value rescales the Radius
inpath = "input_files/full-unblind-v2-mcal-zmix_y1clust_l%d_z%d_cosmo_H0-%d_Om0-%.2f_scritinv.dat"

def get_args_and_paths(name, zi, lj, H0new, Omnew, model_name, cal=False, useJK=False):
    rDS, _, SCInew, rR = np.loadtxt(inpath%(lj,zi,H0new,Omnew))
    hnew = H0new/100.

    covname = "SAC"
    zmap = None #Only used for calibration
    if name == "y1" or name == "cal":
        usey1 = True
    else: #sv
        usey1 = False
    if useJK:
        covname = "JK"
    if name == "cal":
        if not cal: raise Exception("'cal' must be True if doing calibration")
        zmap = get_zmap() #Maps zi to y1zi for the calibration
    if name != "cal" and cal:
        raise Exception("'cal' specified but analysis is %s"%name)
    cosmo = get_cosmo_default(hnew, Omnew, cal)
    h = cosmo['h']
    om = cosmo['om']
    defaults = get_model_defaults(h)
    conc_spline = get_concentration_spline(h, om, cal)

    #First fix the paths
    basesuffix = name+"_"+covname+"_z%d_l%d"%(zi, lj)
    bfpath = "bestfits/bf_%s_%s_H0-%d_Om-%.2f.txt"%(model_name, basesuffix, H0new, Omnew)
    chainpath   = "chains/chain_%s_%s_H0-%d_Om-%.2f.txt"%(model_name, basesuffix, H0new, Omnew)
    likespath   = "chains/likes_%s_%s_H0-%d_Om-%.2f.txt"%(model_name, basesuffix, H0new, Omnew)
    paths = [bfpath, chainpath, likespath]
    
    #Now prep the args
    zs, lams = get_zs_and_lams(usey1, cal)
    Rlams = (lams/100.)**0.2 #Mpc/h
    SCIs = get_sigma_crit_inverses(usey1) #In pc^2/Msun physical
    #print "Z, lams, Rlams shapes: ",zs.shape, lams.shape, Rlams.shape
    z, lam, Rlam = zs[zi, lj], lams[zi, lj], Rlams[zi, lj]
    if cal: SCI = SCIs[zmap[zi], lj] * h*(1+z)**2
    else: SCI = SCIs[zi, lj] * h*(1+z)**2 #Convert to pc^2/hMsun comoving
    k, Plin, Pnl = get_power_spectra(zi, lj, usey1, cal)
    Rmodel = np.logspace(-2, 3, num=1000, base=10) #Mpc/h comoving
    xi_nl2  = ct.xi.xi_mm_at_R(Rmodel, k, Pnl, N=200)
    xi_nl  = ct.xi.xi_mm_at_R(Rmodel, k, Pnl)
    xi_lin = ct.xi.xi_mm_at_R(Rmodel, k, Plin)
    lowcut = 0.2 #Mpc physical
    Rdata, ds, icov, cov, inds = get_data_and_icov(zi, lj, lowcut=lowcut, usey1=usey1, useJK=useJK, cal=cal, alldata=True)
    Rdata *= rR
    ds *= rDS
    icov /= rDS**2
    cov *= rDS**2
    SCI = SCInew * hnew*(1+z)**2 #Convert to pc^2/hMsun comoving
    if cal:
        Rb, Bp1, iBcov, Bcov = get_boost_data_and_cov(zmap[zi], lj, usey1=usey1, diag_only=True)
    else:
        Rb, Bp1, iBcov, Bcov = get_boost_data_and_cov(zi, lj, lowcut=lowcut, usey1=usey1, diag_only=False)
    Rb *= rR
    print "Boost shapes:",Bp1.shape, iBcov.shape
    if cal: Am_prior, Am_prior_var = get_Am_prior(zmap[zi], lj)
    else: Am_prior, Am_prior_var = get_Am_prior(zi, lj)
    Redges = get_Redges(usey1 = usey1) * h*(1+z) #Mpc/h comoving
    Redges *= rR #Note: Rmodel doesn't change!
    print "Doing cal:",cal, "Hubble:",h, "Omega_m:",om, "Covariance type:",covname
    args = {"z":z, "lam":lam, "Rlam":Rlam, "k":k, "Plin":Plin, "Pnl":Pnl, "Rmodel":Rmodel, "xi_nl":xi_nl, "xi_lin":xi_lin, "Rdata":Rdata, "ds":ds, "cov":cov, "icov":icov, "Rb":Rb, "Bp1":Bp1, "Bcov":Bcov, "iBcov":iBcov, "Redges":Redges, "inds":inds, "Am_prior":Am_prior, "Am_prior_var":Am_prior_var, "sigma_crit_inv":SCI, "model_name":model_name, "zi":zi, "lj":lj, 'h':h, 'om':om, 'defaults':defaults, 'cspline':conc_spline, 'xi_nl2':xi_nl2}
    return paths, args

def get_calTF():
    #False if we aren't doing a calibration
    return False
def get_zmap():
    return np.array([1, 1, 0, 0]) #Maps zi to y1zi

zmap = get_zmap()

def get_zs_and_lams(usey1, cal=False):
    lams = get_lams(usey1)
    zs = get_zs(usey1)
    if cal:
        lams = np.genfromtxt(callamspath)
        zs = np.genfromtxt(calzspath)
    return zs, lams

def get_lams(usey1):
    if usey1: return np.genfromtxt(y1lamspath)
    else: return np.genfromtxt(svlamspath)

def get_zs(usey1):
    if usey1: return np.genfromtxt(y1zspath)
    else: return np.genfromtxt(svzspath)

def get_sigma_crit_inverses(usey1):
    if usey1: return np.loadtxt(SCIpath)
    else: return np.zeros((3,5))

def get_power_spectra(zi, lj, usey1, cal=False):
    if cal: #Use the calibration instead
        print "Using calibration P(k) zi=%d"%zi
        k = np.genfromtxt(calbase+"P_files/k.txt")
        Plin = np.genfromtxt(calbase+"P_files/plin_z%d.txt"%zi)
        Pnl  = np.genfromtxt(calbase+"P_files/pnl_z%d.txt"%zi)
        return k, Plin, Pnl
    if usey1: base = y1base
    else: base = svbase
    k    = np.genfromtxt(base+"P_files/k.txt")
    Plin = np.genfromtxt(base+"P_files/plin_z%d_l%d.txt"%(zi, lj))
    Pnl  = np.genfromtxt(base+"P_files/pnl_z%d_l%d.txt"%(zi, lj))
    return k, Plin, Pnl
    
def get_data_and_icov(zi, lj, lowcut = 0.2, highcut = 999, usey1=True, alldata=False, useJK=True, cal=False):
    #lowcut is the lower cutoff, assumed to be 0.2 Mpc physical
    #highcut might not be implemented in this analysis
    if usey1:
        print "Y1 data z%d l%d"%(zi, lj)
        datapath = y1database%(lj, zi)
        if useJK: covpath = y1JKcovbase%(lj, zi)
        else: covpath = y1SACcovbase%(zi, lj)
    else:
        print "SV data z%d l%d"%(zi, lj)
        datapath = svdatabase%(zi, lj)
        covpath = svcovbase%(zi, lj)
    if cal:
        print "Calibration used instead z%d l%d with zmap=%d"%(zi, lj, zmap[zi])
        datapath = caldatabase%(zi, lj)
        covpath = calSACcovbase%(zmap[zi], lj)
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
    if useJK:
        print "Hartlap applied"
        Njk = 100.
        D = len(R)
        cov = cov*((Njk-1.)/(Njk-D-2))
    else: print "Using SACs"
    return R, ds, np.linalg.inv(cov), cov, indices

def get_boost_data_and_cov(zi, lj, lowcut = 0.2, highcut = 999, usey1=True, alldata=False, diag_only=True):
    if usey1:
        boostpath = y1boostbase%(lj, zi)
        bcovpath  = y1boostcovbase%(lj, zi)
        Bcov = np.loadtxt(bcovpath)
        Rb, Bp1, Be = np.genfromtxt(boostpath, unpack=True)
        Becut = Be > 1e-6
        Bp1 = Bp1[Becut]
        Rb  = Rb[Becut]
        Be  = Be[Becut]
        Bcov = Bcov[Becut]
        Bcov = Bcov[:,Becut]
        if alldata: #Still make this cut though
            return Rb, Bp1, np.linalg.inv(Bcov), Bcov
        indices = (Rb > lowcut)*(Rb < highcut)
        Bp1 = Bp1[indices]
        Rb  = Rb[indices]
        Be  = Be[indices]
        Bcov = Bcov[indices]
        Bcov = Bcov[:,indices]
        Njk = 100.
        D = len(Rb)
        Bcov = Bcov*((Njk-1.)/(Njk-D-2)) #Hartlap correction
        if diag_only: Bcov = np.diag(Be**2)
        #Note: the boost factors don't have the same number of radial bins
        #as deltasigma. This doesn't matter, because all we do is
        #de-boost the model, which fits to the boost factors independently.
        print "Boost data shapes: ",Rb.shape, Bp1.shape, Be.shape, Bcov.shape
        return Rb, Bp1, np.linalg.inv(Bcov), Bcov
    else: #use_sv
        print "SV boosts"
        boostpath = svboostbase
        bcovpath  = svboostcovbase%(zi, lj) #doesn't exist
        if zi == 0: highcut = 21.5 #1 degree cut
        Bp1, Rb = np.genfromtxt(boostpath, unpack=True, skip_header=1)
        #SV didn't have boost errors. We construct them instead
        del2 = 10**-4.09 #BF result from SV
        Be = np.sqrt(del2/Rb**2)
        if alldata:
            Bcov = np.diag(Be**2)
            return Rb, Bp1, np.linalg.inv(Bcov), Bcov
        indices = (Rb > lowcut)*(Rb < highcut)
        Bp1 = Bp1[indices]
        Rb  = Rb[indices]
        Be  = Be[indices]
        Bcov = np.diag(Be**2)
        return Rb, Bp1, np.linalg.inv(Bcov), Bcov

def get_cuts(zi, lj, usey1=True):
    lo = 0.2 #Mpc physical
    hi = 999.
    if not usey1 and zi==0: hi = 21.5
    return [lo, hi]

def get_model_defaults(h):
    #Dictionary of default starting points for the best fit
    defaults = {'lM'   : 14.37+np.log10(h), #Result of SV relation
                'conc'    : 4.5, #Arbitrary
                'tau' : 0.153, #Y1
                'fmis' : 0.32, #Y1
                'Am'    : 1.02, #Y1 approx.
                'B0'   : 0.07, #Y1
                'Rs'   : 2.49,  #Y1; Mpc physical
                'sig_b': 0.005} #Y1 boost scatter
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

def get_mcmc_start(model, model_name):
    if model_name is "full":
        lM, c, tau, fmis, Am, B0, Rs = model
        return model
    elif model_name is "Afixed":
        lM, c, tau, fmis, B0, Rs = model
        return [lM, c, tau, fmis, B0, Rs]
    elif model_name is "cfixed":
        return [lM, tau, fmis, Am, B0, Rs]
    elif model_name is "Mc":
        lM, c = model
        return [lM, c]
    elif model_name is "M":
        lM = model
        return [lM,]
    
def get_cosmo_default(h, Om, cal=False):
    #The cosmology used in this analysis
    cosmo = {'h'      : h,
             'om'     : Om,
             'ode'    : 0.7,
             'ob'     : 0.05,
             'ok'     : 0.0,
             'sigma8' : 0.8,
             'ns'     : 0.96}
    if cal: #fox cosmology
        cosmo = {'h'      : 0.6704,
                 'om'     : 0.318,
                 'ode'    : 0.682,
                 'ob'     : 0.049,
                 'ok'     : 0.0,
                 'sigma8' : 0.835,
                 'ns'     : 0.962}
    return cosmo

def get_Am_prior(zi, lj):
    #Photoz calibration (1+delta)
    deltap1 = np.loadtxt("../photoz_calibration/Y1_deltap1.txt")[zi, lj]
    deltap1_var = np.loadtxt("../photoz_calibration/Y1_deltap1_var.txt")[zi, lj]
    #Shear calibration m
    m = 0.012
    m_var = 0.013**2
    Am_prior = deltap1 + m
    Am_prior_var = deltap1_var + m_var
    print "Am prior: ", zi, lj, Am_prior, np.sqrt(Am_prior_var)
    return Am_prior, Am_prior_var

def get_Redges(usey1):
    #The bin edges in Mpc physical
    Nbins = 15
    if usey1: return np.logspace(np.log10(0.0323), np.log10(30.), num=Nbins+1)
    else: return np.logspace(np.log10(0.02), np.log10(30.), num=Nbins+1) #use_sv

#Set up the Concentration spline
def get_concentration_spline(h, Om, cal=False):
    from colossus.halo import concentration
    from colossus.cosmology import cosmology
    cosmo = get_cosmo_default(h, Om, cal)
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

if __name__ == "__main__":
    model_name = "full"
    zi = 0
    lj = 3
    get_args_and_paths("cal", zi, lj, model_name, cal=True)
    get_args_and_paths("y1", zi, lj, model_name)
