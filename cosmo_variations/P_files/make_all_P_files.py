import numpy as np
from classy import Class

zs = np.loadtxt("Y1_meanz.txt")
print zs.shape

for H0 in [60.,65.,70.,75.,80.]:
    for Om in [0.2,0.25,0.3,0.35,0.40]:
        h = H0/100.
        Ob = 0.05
        Ocdm = Om - Ob
        A_s = 1.9735e-9 #Was set by hand. It gives s8=0.8 for h=.7,Om=.3
        params = {
            'output': 'mPk',
            "h":h,
            "A_s": A_s,
            #"sigma8":0.8,
            "n_s":0.96,
            "Omega_b":Ob,
            "Omega_cdm":Ocdm,
            'YHe':0.24755048455476272,#By hand, default value
            'P_k_max_1/Mpc':1000.,
            'z_max_pk':1.0,
            'non linear':'halofit'}
        cosmo = Class()
        cosmo.set(params)
        cosmo.compute()
        print "sigma8 is:", cosmo.sigma8()

        k = np.logspace(-5, 3, base=10, num=4000) #1/Mpc
        np.savetxt("k_H0-%d_Om-%.2f.txt"%(H0, Om), k/h)
        for i in range(3):
            for j in range(7):
                z = zs[i,j]
                Pnl  = np.array([cosmo.pk(ki, z) for ki in k]) 
                Plin = np.array([cosmo.pk_lin(ki, z) for ki in k])
                Pnl *= h**3
                Plin *= h**3
                np.savetxt("plin_z%d_l%d_H0-%d_Om-%.2f_As-%.3e.txt"%(i,j,H0,Om, A_s), Plin)
                np.savetxt("pnl_z%d_l%d_H0-%d_Om-%.2f_As-%.3e.txt"%(i,j,H0,Om, A_s), Pnl)
