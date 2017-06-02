"""
This file contains two functions to read in the weak lensing
and boost factor data.
"""
import numpy as np

def get_data_and_cov(datapath, covpath, lowcut = 0.2, highcut = 999):
    #lowcut is the lower cutoff, assumed to be 0.2 Mpc physical
    #highcut might not be implemented in this analysis
    R, ds, dse, dsx, dsxe = np.genfromtxt(datapath, unpack=True)
    cov = np.genfromtxt(covpath)
    indices = (R > lowcut)*(R < highcut)
    R   = R[indices]
    ds  = ds[indices]
    cov = cov[indices]
    cov = cov[:,indices]
    return R, ds, cov

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
