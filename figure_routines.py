"""
This file contains the routines used to make figures
either used to diagnose the analysis so far or to make final
figures for the paper.
"""
import numpy as np
from get_all_data import *
from models import *
import os, sys
import matplotlib.pyplot as plt

#Set up the assumptions
cosmo = get_cosmo_default()
h = cosmo['h'] #Hubble constant
defaults = get_model_defaults(cosmo['h'])

def plot_DS_fit_one_bin(params, name, data):
    return 0

if __name__ == '__main__':
    #This specifies which analysis we are doing
    #Name options are full, fixed or Afixed
    name = "full" 
    bstatus  = "blinded" #blinded or unblinded

    #These are the basic paths to the data
    #They are completed when the redshift/richness bin is specified
    #as well as the blinding status
    base  = "/home/tmcclintock/Desktop/des_wl_work/Y1_work/data_files/"
    base2 = base+"%s_tamas_files/"%bstatus
    database     = base2+"full-mcal-raw_y1subtr_l%d_z%d_profile.dat"
    covbase      = base2+"full-mcal-raw_y1subtr_l%d_z%d_dst_cov.dat"
    boostbase    = base2+"full-mcal-raw_y1clust_l%d_z%d_pz_boost.dat"
    boostcovbase = "alsothis" #DOESN'T EXIST YET
    kpath        = "P_files/k.txt"
    Plinpath     = "P_files/plin_z%d_l%d.txt"
    Pnlpath      = "P_files/pnl_z%d_l%d.txt"
    
    #Output suffix to be appended on things
    basesuffix = bstatus+"_"+name+"_z%d_l%d"
    
    #Read in the redshifts and richnesses
    zs    = np.genfromtxt(base+"Y1_meanz.txt")
    lams  = np.genfromtxt(base+"Y1_meanl.txt")
    Rlams = 1.0*(lams/100.0)**0.2 #Mpc/h; richness radius

    bestfitbase = "bestfits/bf_%s.txt"%basesuffix
    chainbase   = "chains/chain_%s.txt"%basesuffix

    for i in xrange(0, 3): #z bins
        if i > 0: continue
        for j in xrange(0, 7): #lambda bins
            if j > 0: continue
            print "Working at z%d l%d for %s"%(i,j,name)
            #Read in everything
            z    = zs[i,j]
            lam  = lams[i,j]
            Rlam = Rlams[i,j]
            datapath     = database%(j,i)
            covpath      = covbase%(j,i)
            R, ds, icov, cov = get_data_and_icov(datapath, covpath)
            k    = np.genfromtxt(kpath)
            Plin = np.genfromtxt(Plinpath%(i,j))
            Pnl  = np.genfromtxt(Pnlpath%(i,j))
            ds_params = get_default_ds_params(z, h)
