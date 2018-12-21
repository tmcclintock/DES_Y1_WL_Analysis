"""
A rebuild of the helper_function.py script used to create arguments
for the lensing analysis. This tool is now an object that can take in
arbitrary paths for things like data vectors and covariance matrices.
This makes it much easier to build the arguments dictionary.
"""
import os
import numpy as np
import cluster_toolkit as ct
import scipy.interpolate as interp
import blinding

class Helper(object):
    """
    Helper object used to create argument dictionaries.
    """
    def __init__(self):
        self.args = {}

    def get_lensing_data(self, datapath, lowcut=0.2, highcut=999):
        """
        Read in the data, assumed to be 5 columns
        with the first column being the central radial bin value
        and the second column the DeltaSigma tangential shear profile.
        The cuts are in Mpc (physical) and are cuts on which bins to keep.
        """
        R, ds, _, _, _ = np.genfromtxt(datapath, unpack=True)
        inds = (R > lowcut)*(R < highcut)
        #Save the radii and profiles both with and without cuts
        #also save the indicies of the bins to keep
        self.args["R_all"] = R
        self.args["R_cut"] = R[inds]
        self.args["DeltaSigma_all"] = ds
        self.args["DeltaSigma_cut"] = ds[inds]
        self.args["lensing_kept_indices"] = inds
        self.args["lensing_cuts"] = [lowcut, highcut]
        return

    def get_lensing_covariance(self, covpath, N_JK = None):
        """
        Read in the covariance matrix. get_data() must be called
        first, so that the indices for the cut have been saved.

        If the covariance was JK estimated, apply a Hartlap correction
        according to the value of N_JK to the cut covariance matrix.
        """
        if "lensing_kept_indices" not in self.args:
            raise Exception("Must read in data with get_lensing_data() "+
                            "before reading in the covariance.")
        inds = self.args["lensing_kept_indices"]
        cov = np.genfromtxt(covpath)
        cov_cut = cov[inds]
        cov_cut = cov_cut[:,inds]
        icov_cut = np.linalg.inv(cov_cut)
        Hartlap_factor = 1
        if N_JK is not None:
            Hartlap_factor = (N_JK-1.)/(N_JK-len(cov_cut)-2)
        #Save the covariance and inverse covariance
        self.args["C_all"] = cov
        self.args["C_cut"] = cov_cut * Hartlap_factor
        self.args["iC_cut"] = icov_cut / Hartlap_factor
        return

    def get_boost_data(self, boostpath, lowcut=0.2, highcut=999,
                       threshold=1e-6):
        """
        Read in boost factor data. Assumed to be a 3 or 2 column file with
        the format: R, Boost+1, Boosterr OR Boost+1, R.
        """
        try:
            Rb, Bp1, Be = np.genfromtxt(boostpath, unpack=True)
        except ValueError:
            #Note, these are SV boost factors and they have a header
            Bp1, Rb = np.genfromtxt(boostpath, unpack=True, skip_header=1)
            Be = np.sqrt(10**-4.09 / Rb**2) #SV result
        except:
            assert False, "Input boost factor path is wrong."

        threshold_cut = Be > threshold
        Rb = Rb[threshold_cut]
        Bp1 = Bp1[threshold_cut]
        Be = Be[threshold_cut]
        inds = (Rb > lowcut)*(Rb<highcut)
        #Save all boost factor data
        self.args['Rb_all'] = Rb
        self.args['Bp1_all'] = Bp1
        self.args['Be_all'] = Be
        self.args['Rb_cut'] = Rb[inds]
        self.args['Bp1_cut'] = Bp1[inds]
        self.args['Be_cut'] = Be[inds]
        self.args['threshold_cut'] = threshold_cut
        self.args['boost_kept_indices'] = inds
        return

    def get_boost_covariance(self, boostcpath, N_JK=None, use_SV_model=False):
        """
        Obtain the covariance matrix for the boost factors.
        """
        if "boost_kept_indices" not in self.args:
            raise Exception("Must read in data with get_boost_data() "+
                            "before reading in the covariance.")
        if use_SV_model:
            Bcov = np.diag(self.args['Be_cut']**2)
        else:
            Bcov = np.loadtxt(boostcpath)
        thc = self.args["threshold_cut"]
        inds = self.args["boost_kept_indices"]
        Bcov = Bcov[thc]
        Bcov = Bcov[:,thc]
        Bcov_cut = Bcov[inds]
        Bcov_cut = Bcov_cut[:,inds]
        Hartlap_factor = 1
        if N_JK is not None:
            Hartlap_factor = (N_JK-1.)/(N_JK-len(Bcov_cut)-2)
        self.args["Bcov_all"] = Bcov
        self.args["Bcov_cut"] = Bcov_cut * Hartlap_factor
        self.args["iBcov_cut"] = np.linalg.inv(Bcov_cut) / Hartlap_factor
        return

    def add_cosmology_dictionary(self, cosmo, name=None):
        """
        Attach the cosmology object to the arguments.
        """
        if name is not None:
            print("\t'name':%s supplied, using a pre-defined cosmology."%name)
            if name not in ['fox', 'Y1']:
                raise Exception("Cosmology %s not pre-defined."%name)
            if name is "Y1":
                cosmo = {'h'      : 0.7,
                         'Omega_m'     : 0.3,
                         'Omega_de'    : 0.7,
                         'Omega_b'     : 0.05,
                         'Omega_k'     : 0.0,
                         'sigma8' : 0.8,
                         'ns'     : 0.96}
            elif name is "fox":
                cosmo = {'h'      : 0.6704,
                         'Omega_m'     : 0.318,
                         'Omega_de'    : 0.682,
                         'Omega_b'     : 0.049,
                         'Omega_k'     : 0.0,
                         'sigma8' : 0.835,
                         'ns'     : 0.962}
        pars = ['h', 'Omega_m', 'Omega_b', 'Omega_de', 'Omega_k',
                'sigma8', 'ns']
        for p in pars:
            if p not in cosmo:
                raise Exception("%s missing from cosmology dictionary."%p)
        #Save the cosmology dictionaty
        self.args['cosmology'] = cosmo
        return
        
    def add_stack_data(self, z, richness, Sigma_crit_inv):
        """
        Add some data about the stack to the arguments.
        Note that the input units for Sigma_crit_inv is pc^2/Msun physical,
        and here it is converted to pc^2/h*Msun comoving.
        """
        if "cosmology" not in self.args:
            raise Exception("Must add a cosmology dictionary before adding "+\
                            "Sigma_crit_inv.")
        self.args['z'] = z
        self.args['lam'] = richness
        Rlam = (richness/100.)**0.2 #Mpc/h comoving
        self.args['Rlam'] = Rlam
        h = self.args['cosmology']['h']
        self.args['Sigma_crit_inv'] = Sigma_crit_inv*h*(1+z)**2
        return

    def compute_power_spectra(self, z):
        try:
            from classy import Class
        except ImportError:
            print("Cannot precompute power spectra because CLASS "+\
                  "is not installed.")
            return
        cos = self.args['cosmology']
        h = cos['h']
        params = {
            'output': 'mPk',
            "h":h,
            "sigma8":cos['sigma8'],
            "n_s":cos['ns'],
            "Omega_b":cos['Omega_b'],
            "Omega_cdm":cos['Omega_m'] - cos['Omega_b'],
            'P_k_max_1/Mpc':1000.,
            'z_max_pk':1.0,
            'non linear':'halofit'}
        class_cosmo = Class()
        class_cosmo.set(params)
        class_cosmo.compute()
        k = np.logspace(-5, 3, num=4000) #1/Mpc comoving
        kh = k/h #h/Mpc comoving
        #P(k) are in Mpc^3/h^3 comoving
        Pnl = np.array([class_cosmo.pk(ki, z) for ki in k])*h**3
        Plin = np.array([class_cosmo.pk_lin(ki, z) for ki in k])*h**3
        self.args['k'] = kh
        self.args['Plin'] = Plin
        self.args['Pnl'] = Pnl
        return

    def precompute_ximm(self, k, P_lin, P_nl, use_internal=False):
        """
        Precompute the matter correlation function, computed from
        either Plin or P_nonlin. Note that everything is comoving.
        If we want, we can use the powe spectra computed internally
        in this tool rather than the power spectra passed in.
        """
        r = np.logspace(-2, 3, num=1000) #Mpc/h comoving
        if use_internal:
            k = self.args['k']
            P_lin = self.args['Plin']
            P_nl = self.args['Pnl']
        xi_lin = ct.xi.xi_mm_at_R(r, k, P_lin)
        xi_nl  = ct.xi.xi_mm_at_R(r, k, P_nl)
        self.args['r'] = r
        self.args['xilin'] = xi_lin
        self.args['xinl'] = xi_nl
        self.args['k'] = k
        self.args['Plin'] = P_lin
        self.args['Pnl'] = P_nl
        return

    def create_concentration_spline(self):
        """
        Creates a spline for the concentration using colossus.
        """
        try:
            from colossus.halo import concentration
            from colossus.cosmology import cosmology
        except ImportError:
            print("colossus not installed. No concentration spline available.")
            return
        cosmo = self.args['cosmology']
        cos = {'flat':True,'H0':cosmo['h']*100.,'Om0':cosmo['Omega_m'],
               'Ob0':cosmo['Omega_b'],'sigma8':cosmo['sigma8'],'ns':cosmo['ns']}
        cosmology.addCosmology('fiducial', cos)
        cosmology.setCosmology('fiducial')
        z = self.args['z']
        M = np.logspace(12, 17, 50)
        c = np.zeros_like(M)
        for i in range(len(M)):
            c[i] = concentration.concentration(M[i],'200m',z=z,model='diemer15')
        self.args['cspline'] = interp.interp1d(M, c)
        return

if __name__ == "__main__":
    H = Helper()
    base = "../data_files/"
    dpath = base + "Y1_data/full-unblind-v2-mcal-zmix_y1subtr_l%d_z%d_profile.dat"%(3,0)
    cpath = base + "Y1_data/SACs/SAC_z%d_l%d.txt"%(0,3)
    H.get_lensing_data(dpath)
    H.get_lensing_covariance(cpath)
    print(H.args.keys())
    import matplotlib.pyplot as plt
    #plt.errorbar(H.args["R_cut"], H.args["DeltaSigma_cut"], np.sqrt(H.args["C_cut"].diagonal()))
    #plt.loglog()
    #plt.show()

    dpath = base + "Y1_data/full-unblind-v2-mcal-zmix_y1clust_l%d_z%d_zpdf_boost.dat"%(3,0)
    cpath = base + "Y1_data/full-unblind-v2-mcal-zmix_y1clust_l%d_z%d_zpdf_boost_cov.dat"%(3,0)
    H.get_boost_data(dpath)
    H.get_boost_covariance(cpath, 100.)
    plt.errorbar(H.args['Rb_cut'], H.args['Bp1_cut'], H.args['Be_cut'])
    plt.loglog()
    plt.show()
