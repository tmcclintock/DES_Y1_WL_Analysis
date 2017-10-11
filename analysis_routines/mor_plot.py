import numpy as np
from helper_functions import *
import matplotlib.pyplot as plt
useY1 = True
blinded = False
cosmo = get_cosmo_default()
h = cosmo['h'] #Hubble constant

def get_mass_and_var(i, j, from_chain=True):
    if from_chain: path = "not existing"
    else:
        lM = np.loadtxt("bestfits/bf_unblinded_y1_z%d_l%d.txt"%(i, j))[0]
        var = np.loadtxt("bestfits/bf_ihess_unblinded_y1_z%d_l%d.txt"%(i,j))[0,0]
        return lM, var
    return

if __name__=="__main__":
    usey1 = True
    zs, lams = get_zs_and_lams(usey1 = usey1)
    zs = zs[:, 3:]
    lams = lams[:, 3:]
    lMs = []
    var = []
    for i in range(3):
        lMi = []
        vari = []
        for j in range(7):
            if j <3: continue
            lMij, varij = get_mass_and_var(i, j, from_chain=False)
            lMi.append(lMij)
            vari.append(varij)
        lMs.append(lMi)
        var.append(vari)
    lMs = np.array(lMs) - np.log10(h)
    print lMs
    print np.sqrt(var)
    var = np.array(var) - np.log10(h**2)
    M = 10**lMs #Msun
    errM = np.sqrt(var)/np.log(10)*M
    print M
    print errM
    #np.savetxt("Mblind.txt", M)
    #np.savetxt("Mblinderr.txt", errM)
    print lams.shape, M.shape, errM.shape
    plt.errorbar(lams.flatten(), M.flatten(), errM.flatten(), ls='', marker='.')
    plt.yscale('log')
    plt.ylabel(r"$\mathcal{M}\ [{\rm M_\odot}]$")
    plt.xlabel(r"$\lambda$")
    plt.show()
