"""
This file contains functions used to make the analysis script easier to read. This includes file IO and loading various things.
"""
import numpy as np
import blinding
import helper_tool

#Y1 paths
y1 = "../data_files/Y1_data/"
y1data = y1+"full-unblind-v2-mcal-zmix_y1subtr_l%d_z%d_profile.dat"
y1SAC = y1+"SACs/SAC_z%d_l%d.txt"
y1JK = y1+"full-unblind-v2-mcal-zmix_y1subtr_l%d_z%d_dst_cov.dat"
y1boost = y1+"full-unblind-v2-mcal-zmix_y1clust_l%d_z%d_zpdf_boost.dat"
y1boostJK = y1+"full-unblind-v2-mcal-zmix_y1clust_l%d_z%d_zpdf_boost_cov.dat"
#SV paths
sv = "../data_files/SV_data/"
svdata = sv+"profile_z%d_l%d.dat"
svJK = sv+"cov_t_z%d_l%d.dat"
svboost = sv+"SV_boost_factors.txt"
svboostJK = None

#redshifts and richnesses
y1zs = np.loadtxt(y1+"Y1_meanz.txt") #mean redshift of the stack
y1lams = np.loadtxt(y1+"Y1_meanl.txt") #lams == lambdas, or richnesses; mean of the stack
svzs = np.loadtxt(sv+"SV_meanz.txt")
svlams = np.loadtxt(sv+"SV_meanl.txt")

#Sigma_crit_inverses from Y1 analysis in pc^2/Msun physical
SCIs = np.loadtxt("../photoz_calibration/sigma_crit_inv.txt")

#Photo-z multiplicative biases from Y1 analysis; 1+delta(z)
delta_plus_1_all = np.loadtxt("../photoz_calibration/Y1_deltap1.txt")
delta_plus_1_all_var = np.loadtxt("../photoz_calibration/Y1_deltap1_var.txt")

#paths to old simulation data - commented out for now
"""
calbase = fullbase+"/DATA_FILES/calibration_data_files/"
caldatabase = calbase+"new_calDS_ps20_z%d_l%d.txt"
calSACcovbase = y1SACcovbase
calboostbase = y1boostbase
calboostcovbase = y1boostcovbase
calzspath = calbase+"CAL_meanz.txt"
callamspath = calbase+"CAL_ps25_meanl.txt"
"""

def get_output_paths(model_name, zi, lj, name="Y1", covkind="SAC", blinded=True):
    if model_name not in ["full", "Afixed", "cfixed", "Mc", "M"]:
        raise Exception("Invalid model name: %s"%model_name)
    if name not in ["Y1", "SV"]: #"fox_sim"]:
        raise Exception("'name':%s must be either Y1 or SV."%name)
    if covkind not in ["SAC", "JK"]:
        raise Exception("Covariance type %s not recognized. Use either SAC or JK."%covkind)
    suffix = "%s_%s_%s_z%d_l%d"%(model_name, name, covkind, zi, lj)
    bfpath = "bestfits/bf_%s.txt"%(suffix)
    chainpath = "chains/chain_%s"%(suffix)
    likespath = "chains/likes_%s"%(suffix)
    return bfpath, chainpath, likespath

def get_args(zi, lj, name="Y1", covkind="SAC", blinded=True, cuts=[0.2, 999.],
             boost_threshold=1e-6):
    if name not in ["Y1", "SV"]: #"fox_sim"]:
        raise Exception("'name':%s must be either Y1 or SV."%name)
    if covkind not in ["SAC", "JK"]:
        raise Exception("Covariance type %s not recognized. Use either SAC or JK."%covkind)
    if len(cuts) != 2:
        raise Exception("Scale cuts must be a list of two numbers: [lower cut, higher cut].")
    if cuts[0] >= cuts[1]:
        raise Exception("Scale cuts must be ascending.")
    
    #Branch to determind path names
    if name == "Y1":
        dpath = y1data%(lj,zi)
        if covkind == "SAC":
            cpath = y1SAC%(zi,lj)
            N_JK = None
        else: #covkind == "JK"
            cpath = y1JK%(lj,zk)
            N_JK = 100.
        bdpath = y1boost%(lj,zi)
        bcpath = y1boostJK%(lj,zi)
        z = y1zs[zi,lj]
        lam = y1lams[zi,lj]
        use_SV_boost = False
        cosmo_name = "Y1"
        SCI = SCIs[zi,lj]
        #Projected radial bin edges in Mpc physical
        Redges = np.logspace(np.log10(0.0323), np.log10(30.), num=15+1)
        #Photo-z multiplicative bias
        delta_plus_1 = delta_plus_1_all[zi,lj]
        delta_plus_1_var = delta_plus_1_all_var[zi,lj]
        #Shape noise
        m = 0.012
        m_var = 0.013**2
        #Power spectra
        k = np.loadtxt("../data_files/Y1_data/P_files/k.txt")
        Plin = np.loadtxt("../data_files/Y1_data/P_files/plin_z%d_l%d.txt"%(zi,lj))
        Pnl = np.loadtxt("../data_files/Y1_data/P_files/pnl_z%d_l%d.txt"%(zi,lj))
    else: #name == "SV"
        dpath = svdata%(zi,lj)
        if covkind == "SAC":
            print("No SACs for SV data. Re-assign a Y1 SAC later manually. For now, using JK.")
        cpath = svJK%(zi,lj)
        N_JK = 100.
        bdpath = svboost
        bcpath = None
        z = svzs[zi,lj]
        lam = svlams[zi,lj]
        use_SV_boost = True
        cosmo_name = "Y1" #same as SV
        SCI = 0 #No Sigma_crit_inverse correction used
        #Projected radial bin edges
        Redges = np.logspace(np.log10(0.02), np.log10(30.), num=15+1)
        #Photo-z multiplicative bias - approximately the same mapping as Y1
        delta_plus_1 = delta_plus_1_all[zi,lj]
        delta_plus_1_var = delta_plus_1_var_all[zi,lj]
        #Shape noise -- approximately the same as Y1
        m = 0.012
        m_var = 0.013**2
        #Power spectra
        k = np.loadtxt("../data_files/SV_data/P_files/k.txt")
        Plin = np.loadtxt("../data_files/SV_data/P_files/plin_z%d_l%d.txt"%(zi,lj))
        Pnl = np.loadtxt("../data_files/SV_data/P_files/pnl_z%d_l%d.txt"%(zi,lj))

    #Fetch dictionary entries by the helper tool
    helper = helper_tool.Helper()
    helper.get_lensing_data(dpath, cuts[0], cuts[1])
    helper.get_lensing_covariance(cpath, N_JK)
    helper.get_boost_data(bdpath, cuts[0], cuts[1], boost_threshold)
    helper.get_boost_covariance(bcpath, N_JK, use_SV_boost)
    helper.add_cosmology_dictionary(None, cosmo_name)
    helper.add_stack_data(z, lam, SCI)
    #comment out the following two lines and comment out the following
    #if you want power spectra computed at runtime
    #helper.compute_power_spectra(z)
    #helper.precompute_ximm(0,0,0, use_internal=True
    helper.precompute_ximm(k,Plin,Pnl)
    helper.create_concentration_spline()

    #Add analysis-specific entries to the dictionary here
    args = helper.args
    #Omega_m
    args['Omega_m'] = args['cosmology']['Omega_m']
    #Edges of the radial bins
    h = args['cosmology']['h']
    Redges *= h*(1+z) #Converted to Mpc/h comoving
    args['Redges'] = Redges
    #Prior on the multiplicative bias = Photo-z bias + shape bias (0.012)
    Am_prior = delta_plus_1 + m
    Am_prior_var = delta_plus_1_var + m_var
    args['Am_prior'] = Am_prior
    args['Am_prior_var'] = Am_prior_var
    #Blinding factors
    Blinding_amp, lam_exp, z_exp = blinding.get_blinding_variables()
    blinding_factor = np.log10(Blinding_amp) +  np.log10((lam/30.0)**lam_exp) + np.log10(((1+z)/1.5)**z_exp)
    if not blinding:
        blinding_factor *= 0
    args['blinding_factor'] = blinding_factor
    return args

def get_model_defaults(h):
    #Dictionary of default starting points for the best fit
    defaults = {'lM'   : 14.37+np.log10(h), #Result of SV relation
                'conc'    : 4.5, #Arbitrary
                'tau' : 0.153, #Y1
                'fmis' : 0.32, #Y1
                'Am'    : 1.02, #Y1 approx.
                'B0'   : 0.07, #Y1
                'Rs'   : 2.49,  #Y1; Mpc physical
                'sig_b': 0.005} #Y1 boost scatter - not used in the fiducial model
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

if __name__=="__main__":
    args = get_args(0, 3, name="Y1", covkind="SAC", blinded=True, cuts=[0.2, 999.], boost_threshold=1e-6)
    paths = get_output_paths("full", 0, 3, name="Y1", covkind="SAC", blinded=True)
