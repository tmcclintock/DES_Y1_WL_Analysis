"""
Make individual components of the SAC for one bin
"""
import sys, os
import numpy as np
import helper_functions as HF
import clusterwl
import matplotlib.pyplot as plt

#Get concentration as a function of M and z 
concentration_spline = HF.get_concentration_spline()

#Get the cosmology dictionary
cosmo_dict = HF.get_cosmo_dict()
h  = cosmo_dict['h']
om = cosmo_dict['om']

N_realizations = 1000
N_Radii = 1000
Rp = np.logspace(-2, 2.4, N_Radii, base=10)
Nbins = 15

P_file_path = "/home/tmcclintock/Desktop/des_wl_work/DATA_FILES/y1_data_files/P_files/"
cluster_file_path = "/home/tmcclintock/Desktop/des_wl_work/DATA_FILES/y1_data_files/cluster_files/clusters_z%d_l%d.txt"

def component_realizations(zi, li, MLoff = False, MCoff = False, do_miscentering = True):
    if MLoff: ML_scatter = 0
    else: ML_scatter = 0.25 #percent
    if MCoff: MC_scatter = 0
    else: MC_scatter = 0.16 #dex scatter
    k = np.loadtxt(P_file_path+"k.txt")
    Plin = np.genfromtxt(P_file_path+"./plin_z%d_l%d.txt"%(zi, lj))
    Pnl  = np.genfromtxt(P_file_path+"/pnl_z%d_l%d.txt"%(zi, lj))
    zs, lams = np.loadtxt(cluster_file_path%(zi, lj)).T
    zlens = np.mean(zs)
    R     = np.logspace(-2, 3, N_Radii, base=10) #go higher than BAO
    xi_mm = clusterwl.xi.xi_mm_at_R(R, k, Pnl)
    R_perp = np.logspace(-2, 2.4, N_Radii, base=10)
    DeltaSigma_realizations = np.zeros((N_realizations, N_Radii))
    print "Starting realizations for z%d l%d"%(zi,lj)
    for real in range(N_realizations):
        M, conc, Rmis, ismis = HF.get_cluster_parameters(lams, zs, concentration_spline, ML_scatter=ML_scatter, MC_scatter=MC_scatter, do_miscentering=do_miscentering)
        N_kept = len(M)
        mean_DeltaSigma = np.zeros_like(R_perp)
        for cl in range(N_kept): #Loop over clusters
            xi_nfw = clusterwl.xi.xi_nfw_at_R(R, M[cl], conc[cl], om)
            bias = clusterwl.bias.bias_at_M(M[cl], k, Plin, om)
            xi_2halo = clusterwl.xi.xi_2halo(bias, xi_mm)
            xi_hm    = clusterwl.xi.xi_hm(xi_nfw, xi_2halo)
            Sigma    = clusterwl.deltasigma.Sigma_at_R(R_perp, R, xi_hm, M[cl], conc[cl], om)
            if not ismis[cl]: #isn't miscentered
                DeltaSigma = clusterwl.deltasigma.DeltaSigma_at_R(R_perp, R_perp, Sigma, M[cl], conc[cl], om)
            else: #is miscentered
                Sigma_single  = clusterwl.miscentering.Sigma_mis_single_at_R(R_perp, R_perp, Sigma, M[cl], conc[cl], om, Rmis[cl])
                DeltaSigma = clusterwl.miscentering.DeltaSigma_mis_at_R(R_perp, R_perp, Sigma_single)
            mean_DeltaSigma += DeltaSigma/N_kept
        DeltaSigma_realizations[real] += mean_DeltaSigma
    print "Made individual realizations for z%d l%d"%(zi,lj)
    return DeltaSigma_realizations

def merge_realizations(zi, lj, dss):
    adss = np.zeros((N_realizations, Nbins))
    amds  = np.zeros((Nbins))

    zlenses = HF.get_all_zlenses()
    zlens = zlenses[zi, lj]
    binmin = 0.0323*(1+zlens)*h #Converted to comoving Mpc/h
    binmax = 30.0*(1+zlens)*h #Converted to comoving Mpc/h
    Redges = np.logspace(np.log(binmin), np.log(binmax), num=Nbins+1, base=np.e)
    Rbins = (Redges[:-1]+Redges[1:])/2.
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
    return C


if __name__ == "__main__":
    zi, lj = 0, 6
    DSreal_ml = component_realizations(zi, lj, MLoff = False, MCoff = True, do_miscentering = False)
    np.savetxt("component_outfiles/MLonly_reals_z%d_l%d.txt"%(zi, lj), DSreal_ml)
    Cml = merge_realizations(zi, lj, DSreal_ml)
    np.savetxt("component_outfiles/cov_MLonly_z%d_l%d.txt"%(zi,lj), Cml)

    DSreal_mc = component_realizations(zi, lj, MLoff = True, MCoff = False, do_miscentering = False)
    np.savetxt("component_outfiles/MConly_reals_z%d_l%d.txt"%(zi,lj), DSreal_mc)
    Cmc = merge_realizations(zi, lj, DSreal_mc)
    np.savetxt("component_outfiles/cov_MConly_z%d_l%d.txt"%(zi,lj), Cmc)

    DSreal_mis = component_realizations(zi, lj, MLoff = True, MCoff = True, do_miscentering = True)
    np.savetxt("component_outfiles/actual_MISonly_reals_z%d_l%d.txt"%(zi,lj), DSreal_mis)
    Cmis = merge_realizations(zi, lj, DSreal_mis)
    np.savetxt("component_outfiles/actual_cov_MISonly_z%d_l%d.txt"%(zi,lj), Cmis)
