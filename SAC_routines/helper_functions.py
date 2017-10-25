"""
These are helper functions used to assist in creating Tom's part of the SAC.
"""
import numpy as np
from scipy.interpolate import interp2d
from colossus.halo import concentration
from colossus.cosmology import cosmology

#Set the colossus cosmology
default_cos = {'flat':True,'H0':70.0,'Om0':0.3,'Ob0':0.05,'sigma8':0.8,'ns':0.96}
cosmology.addCosmology('fiducial', default_cos)
cosmology.setCosmology('fiducial')
h = default_cos['H0']/100.

def get_all_zlenses():
    cluster_file_path = "/home/tmcclintock/Desktop/des_wl_work/DATA_FILES/y1_data_files/cluster_files/clusters_z%d_l%d.txt"
    cluster_file_path = "/Users/tmcclintock/Data/DATA_FILES/y1_data_files/cluster_files/clusters_z%d_l%d.txt"
    zlens = np.zeros((3, 7))
    for i in xrange(2,-1,-1):
        for j in xrange(6,-1,-1):
            zs, lams = np.loadtxt(cluster_file_path%(i, j)).T
            zlens[i,j] = np.mean(zs)
    return zlens

def get_concentration_spline():
    Nm, Nz = 20, 20
    M = np.logspace(12,17,Nm,base=10)
    z = np.linspace(0.2,0.65,Nz)
    c_array = np.ones((Nz,Nm))
    for i in range(Nm):
        for j in range(Nz):
            c_array[j,i] = concentration.concentration(M[i],'200m',z=z[j],model='diemer15')
    return interp2d(M, z, c_array)

def get_cosmo_dict():
    h = default_cos['H0']/100.
    cosmo = {"h":h,"om":default_cos['Om0'],"ok":0.0}
    cosmo["ode"]=1.0-cosmo["om"]
    return cosmo

def get_cluster_parameters(lams, zs, c_spline, N_want=1000, ML_scatter=0.25, MC_scatter=0.16, do_miscentering=True):
    #print "ML precent scatter = ",ML_scatter
    N = len(lams)
    keep_inds = np.random.rand(N) < float(N_want)/N
    lams = lams[keep_inds]
    zs = zs[keep_inds]
    N = len(lams)
    #Masses by default have 25% scatter, units get changed to Msun/h
    Mp = h*10**14.371
    Masses = np.exp(np.log(Mp*(lams/30.0)**1.12)-0.5*ML_scatter**2 + ML_scatter*np.random.randn(N))
    #Draw concentrations with some amount of scatter from DK15, with 16% scatter using base10
    cs = np.array([c_spline(mi,zi) for mi,zi in zip(Masses,zs)])[:,0]
    #Note: MC_scatter is actually in dex, not in scatter of a lognormal
    conc = np.exp(np.random.randn(N)*MC_scatter*np.log(10) + np.log(cs)-0.5*(MC_scatter*np.log(10))**2)
    #Make draws for miscentering
    ismis = np.random.rand(N) < 0.32 #Y1 prior
    if not do_miscentering:
        ismis *= False
    tau = np.random.randn(N)*0.003 +0.153
    Rlam = (lams/100)**0.2 #Mpc/h
    Rmis = tau*Rlam #Mpc/h
    x, y = np.random.randn(N), np.random.randn(N)
    Rmis = np.sqrt(x**2+y**2)*Rmis
    return [Masses, conc, Rmis, ismis]
