"""
Merge the realizations of DS into the covariance matrix.
"""
import numpy as np
import matplotlib.pyplot as plt
import helper_functions as HF
import clusterwl


cosmo_dict = HF.get_cosmo_dict()
h  = cosmo_dict['h']

zlenses = HF.get_all_zlenses()
pz_cals = np.loadtxt("../photoz_calibration/Y1_deltap1.txt")


N_Radii = 1000
Rp = np.logspace(-2, 2.4, N_Radii, base=10)
Nbins = 15

def do_merge(i, j, inpath, outpath):
    dss = np.loadtxt(inpath)
    dsm = np.mean(dss, 0) #mean deltasigma
    N_realizations = len(dss)

    adss = np.zeros((N_realizations, Nbins))
    amds  = np.zeros((Nbins)) #average of the mean deltasigma
    pz_cal = pz_cals[i, j] #photoz calibration
    m = 0.012 #shear calibration
    Am = pz_cal + m # 1+m+delta
    zlens = zlenses[i,j]
    binmin = 0.0323*(1+zlens)*h #Converted to comoving Mpc/h
    binmax = 30.0*(1+zlens)*h #Converted to comoving Mpc/h
    Redges = np.logspace(np.log(binmin), np.log(binmax), num=Nbins+1, base=np.e)
    Rbins = (Redges[:-1]+Redges[1:])/2.
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
    np.savetxt(outpath, C)#*Am**2)
    print "done with z%d l%d without Am applied saved to %s"%(i,j,outpath)
    return

if __name__ == "__main__":
    inpath = "output_files/stack_realizations_MLps%d_z%d_l%d.txt"
    outpath = "output_files/tom_covariance_MLps%d_z%d_l%d.txt"
    zi, lj = 0, 6
    for ps in [0, 10, 45, 60]:
        do_merge(zi, lj, inpath%(ps, zi, lj), outpath%(ps, zi, lj))
