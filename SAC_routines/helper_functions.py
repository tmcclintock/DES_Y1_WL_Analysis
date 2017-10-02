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

def get_all_zlenses():
    cluster_file_path = "/home/tmcclintock/Desktop/des_wl_work/DATA_FILES/y1_data_files/cluster_files/clusters_z%d_l%d.txt"
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

def get_cluster_parameters(lams, zs, c_spline, N_want=1000):
    N = len(lams)
    keep_inds = np.random.rand(N) < float(N_want)/N
    lams = lams[keep_inds]
    zs = zs[keep_inds]
    N = len(lams)
    #Draw log10M pivots and scatters, pivot richness of 30.0
    #Draw just from the SV M-lambda relation
    lMp = np.random.randn(N)*np.sqrt(0.04**2+0.022**2) + 14.371
    Fl = np.random.randn(N)*np.sqrt(0.20**2+0.06**2) + 1.12
    Masses = 10**(lMp + Fl*np.log10(lams/30.0))
    #Draw concentrations with some amount of scatter from DK15
    cs = np.array([c_spline(mi,zi) for mi,zi in zip(Masses,zs)])[:,0]
    conc = 10**(np.random.randn(N)*0.16 + np.log10(cs))
    #Make draws for miscentering
    ismis = np.random.rand(N) < 0.32 #Y1 prior
    tau = np.random.randn(N)*0.003 +0.153
    Rlam = (lams/100)**0.2 #Mpc/h
    Rmis = tau*Rlam #Mpc/h
    x, y = np.random.randn(N), np.random.randn(N)
    Rmis = np.sqrt(x**2+y**2)*Rmis
    return [Masses, conc, Rmis, ismis]
