import numpy as np
import matplotlib.pyplot as plt
plt.rc("text", usetex=True)
plt.rc("font", size=12, family="serif")

y1zlabels = [r"$z\in[0.2;0.35)$", r"$z\in[0.35;0.5)$", r"$z\in[0.5;0.65)$"]
y1llabels = [r"$\lambda\in[5;10)$",r"$\lambda\in[10;14)$",r"$\lambda\in[14;20)$",
             r"$\lambda\in[20;30)$",r"$\lambda\in[30;45)$",r"$\lambda\in[45;60)$",
             r"$\lambda\in[60;\infty)$"]

#fullbase = "/home/tmcclintock/Desktop/des_wl_work" #susie
fullbase = "/Users/tmcclintock/Data" #laptop
#fullbase = "/calvin1/tmcclintock/DES_DATA_FILES" #calvin
y1base = fullbase+"/DATA_FILES/y1_data_files/"
y1base2 = y1base+"FINAL_FILES/"
y1database     = y1base2+"full-unblind-mcal-zmix_y1subtr_l%d_z%d_profile.dat"
y1JKcovbase      = y1base2+"full-unblind-mcal-zmix_y1subtr_l%d_z%d_dst_cov.dat"
y1SACcovbase     = y1base2+"SACs/SAC_z%d_l%d.txt"
y1boostbase    = y1base+"FINAL_FILES/full-mcal-zmix_y1clust_l%d_z%d_zpdf_boost.dat"
y1boostcovbase = y1base+"FINAL_FILES/full-mcal-zmix_y1clust_l%d_z%d_zpdf_boost_cov.dat"
y1zspath   = y1base+"Y1_meanz.txt"
y1lamspath = y1base+"Y1_meanl.txt"
y1numpath  = y1base+"Y1_number.txt"

def fix_errorbars(ds, err):
    """
    Find locations where the errorbars are larger than the measurement.
    Correct the lower part of the bar to be at 10^-2, which is below
    the lower limit of the plot.
    """
    bad = err>ds
    errout = np.vstack((err, err))
    errout[0,bad] = ds[bad]-1e-2
    return errout

def get_data(zi, lj, lowcut = 0.2, highcut = 999):
    datapath = y1database%(lj, zi)
    R, ds, dse, dsx, dsxe = np.genfromtxt(datapath, unpack=True)
    if zi == 0: highcut=21.5 #Just for z0 in SV
    inds = (R > lowcut)*(R < highcut)
    return R, ds, dse, inds

if __name__ == "__main__":
    zi, lj = 2, 3
    number = np.loadtxt(y1numpath)
    
    fig, axarr = plt.subplots(3, 7, sharex=True, sharey=True)

    for zi in range(3):
        for lj in range(7):
            Rdata, ds, dserr, good = get_data(zi, lj)
            bad = np.where(good==False)[0]
            dserr_fixed = fix_errorbars(ds, dserr)
            axarr[zi][lj].errorbar(Rdata[good], ds[good], dserr_fixed[:,good], c='k', marker='o', ls='', markersize=2, zorder=1)
            axarr[zi][lj].errorbar(Rdata[bad], ds[bad], dserr_fixed[:,bad], c='k', marker='o', mfc='w', markersize=2, ls='', zorder=1)
            axarr[zi][lj].set_ylim(0.1, 1e3)
            axarr[zi][lj].set_xlim(0.03, 30.)
            axarr[zi][lj].set_xscale('log')
            axarr[zi][lj].set_yscale('log')
            axarr[zi][lj].text(1, 3e2, y1zlabels[zi], fontsize=8)
            axarr[zi][lj].text(1, 1e2, y1llabels[lj], fontsize=8)
            axarr[zi][lj].text(.1, 1, r"$N_{\rm cl}$ = %d"%number[zi,lj], fontsize=8)
            if zi == 2 and lj == 3:axarr[zi][lj].set_xlabel(r"$R$ [Mpc]")
            if zi == 1 and lj == 0:axarr[zi][lj].set_ylabel(r"$\Delta\Sigma$ [M$_\odot$/pc$^2$]")
    plt.subplots_adjust(hspace=0, wspace=0)
    fig.set_size_inches(14, 6)
    plt.savefig("dataall.png", dpi=300)
    #plt.show()
