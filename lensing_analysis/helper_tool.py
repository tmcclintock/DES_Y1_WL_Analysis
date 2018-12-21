"""
A rebuild of the helper_function.py script used to create arguments
for the lensing analysis. This tool is now an object that can take in
arbitrary paths for things like data vectors and covariance matrices.
This makes it much easier to build the arguments dictionary.
"""
import numpy as np
import cluster_toolkit as ct
from scipy.interpolate import interp2d
import blinding

class Helper(object):
    """
    Helper object used to create argument dictionaries.
    """
    def __init__(self):
        self.args = {}

    def get_data(self, datapath, lowcut=0.2, highcut=999):
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
        self.args["kept_indices"] = inds
        return

    def get_covariance(self, covpath, N_JK = None):
        """
        Read in the covariance matrix. get_data() must be called
        first, so that the indices for the cut have been saved.

        If the covariance was JK estimated, apply a Hartlap correction
        according to the value of N_JK to the cut covariance matrix.
        """
        if "kept_indices" not in self.args:
            raise Exception("Must read in data with get_data() before "+\
                            "reading in the covariance.")
        inds = self.args["kept_indices"]
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
        self.args["icov_cut"] = icov_cut / Hartlap_factor
        return

if __name__ == "__main__":
    H = Helper()
    base = "/Users/tmcclintock/Data/DATA_FILES/y1_data_files/FINAL_FILES/"
    dpath = base + "full-unblind-v2-mcal-zmix_y1subtr_l%d_z%d_profile.dat"%(3,0)
    cpath = base + "SACs/SAC_z%d_l%d.txt"%(0,3)
    H.get_data(dpath)
    H.get_covariance(cpath)
    print(H.args.keys())
    import matplotlib.pyplot as plt
    plt.errorbar(H.args["R_cut"], H.args["DeltaSigma_cut"], np.sqrt(H.args["C_cut"].diagonal()))
    plt.loglog()
    plt.show()
