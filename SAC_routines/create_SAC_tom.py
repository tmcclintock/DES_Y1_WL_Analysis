"""
Create the part of the SAC that has M-c scatter, M-lambda scatter, and miscentering.
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

cluster_file_path = "/home/tmcclintock/Desktop/des_wl_work/Y1_work/data_files/cluster_files/clusters_z%d_l%d.txt"
for i in range(2, -1, -1): #z index 2, 1, 0
    for j in range(5, 4, -1): #lambda index 6 to 3, not doing 2,1,0
        #Start by getting xi_mm, which doesn't depend on mass
        k = np.loadtxt("./data_files/k.txt")
        Plin = np.genfromtxt("./data_files/plin_z%d_l%d.txt"%(i,j))
        Pnl  = np.genfromtxt("./data_files/pnl_z%d_l%d.txt"%(i,j))
        zs, lams = np.loadtxt(cluster_file_path%(i, j)).T
        zlens = np.mean(zs)
        R     = np.logspace(-2, 3, N_Radii, base=10) #go higher than BAO
        xi_mm = clusterwl.xi.xi_mm_at_R(R, k, Pnl)
        R_perp = np.logspace(-2, 2.4, N_Radii, base=10)

        DeltaSigma_realizations = np.zeros((N_realizations, N_Radii))
        for real in range(N_realizations):
            M, conc, Rmis, ismis = HF.get_cluster_parameters(lams, zs, concentration_spline)
            N_kept = len(M)
            mean_DeltaSigma = np.zeros_like(R_perp)
            for cl in range(N_kept): #Loop over clusters
                xi_nfw = clusterwl.xi.xi_nfw_at_R(R, M[cl], conc[cl], om)
                bias = clusterwl.bias.bias_at_M(M[cl], k, Plin, om)
                xi_2halo = clusterwl.xi.xi_2halo(bias, xi_mm)
                xi_hm    = clusterwl.xi.xi_hm(xi_nfw, xi_2halo)
                Sigma    = np.zeros_like(R_perp)
                DeltaSigma = np.zeros_like(R_perp)
                clusterwl.deltasigma.calc_Sigma_at_R(R_perp, R, xi_hm, M[cl], conc[cl], om, Sigma)
                if not ismis[cl]: #isn't miscentered
                    clusterwl.deltasigma.calc_DeltaSigma_at_R(R_perp, R_perp, Sigma, M[cl], conc[cl], om, DeltaSigma)
                else: #is miscentered
                    Sigma_single      = np.zeros_like(R_perp)
                    clusterwl.miscentering.calc_Sigma_mis_single_at_R(R_perp, R_perp, Sigma, M[cl], conc[cl], om, Rmis[cl], Sigma_single)
                    clusterwl.miscentering.calc_DeltaSigma_mis_at_R(R_perp, R_perp, Sigma_single, DeltaSigma)
                mean_DeltaSigma += DeltaSigma/N_kept
            DeltaSigma_realizations[real] += mean_DeltaSigma
        np.savetxt("output_files/stack_realizations_z%d_l%d.txt"%(i, j), DeltaSigma_realizations)
