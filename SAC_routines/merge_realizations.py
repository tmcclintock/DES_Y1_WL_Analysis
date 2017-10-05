"""
Merge the realizations of DS into the covariance matrix.
"""
import numpy as np
import matplotlib.pyplot as plt
import helper_functions as HF
import clusterwl

path = "output_files/stack_realizations_z%d_l%d.txt"

cosmo_dict = HF.get_cosmo_dict()
h  = cosmo_dict['h']

zlenses = HF.get_all_zlenses()
pz_cals = np.loadtxt("../photoz_calibration/Y1_deltap1.txt")

N_realizations = 1000
N_Radii = 1000
Rp = np.logspace(-2, 2.4, N_Radii, base=10)
Nbins = 15

adss = np.zeros((N_realizations, Nbins))
amds  = np.zeros((Nbins))

for i in xrange(2,-1,-1):
    for j in xrange(6,5,-1):
        pz_cal = pz_cals[i, j]
        zlens = zlenses[i,j]
        binmin = 0.0323*(1+zlens)*h #Converted to comoving Mpc/h
        binmax = 30.0*(1+zlens)*h #Converted to comoving Mpc/h
        Redges = np.logspace(np.log(binmin), np.log(binmax), num=Nbins+1, base=np.e)
        Rbins = (Redges[:-1]+Redges[1:])/2.
        dss = np.loadtxt(path%(i,j))
        dsm = np.mean(dss, 0)
        clusterwl.averaging.average_profile_in_bins(Redges, Rp, dsm, amds)
        for r in range(N_realizations):
            clusterwl.averaging.average_profile_in_bins(Redges, Rp, dss[r], adss[r])
        C = np.zeros((Nbins, Nbins))
        #Note: deltasigmas are in Msun h/pc^2 comoving at this point
        for ii in range(Nbins):
            for jj in range(Nbins):
                Di = amds[ii] - adss[:, ii]
                Dj = amds[jj] - adss[:, jj]
                Di *= h*(1+zlens)**2 #Msun/pc physical
                Dj *= h*(1+zlens)**2 #Msun/pc physical
                C[ii,jj] = np.mean(Di*Dj)
        np.savetxt("output_files/tom_covariance_z%d_l%d.txt"%(i,j), C*pz_cal**2)
        print "done with z%d l%d"%(i,j)
