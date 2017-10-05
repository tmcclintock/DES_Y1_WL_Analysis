import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
plt.rc("text", usetex=True)

datapath = "sci_correction.dat"
z, r, re = np.genfromtxt(datapath, unpack=True)

rspl = interp.interp1d(z,r)# , kind="cubic")
respl = interp.interp1d(z,re)#, kind="cubic")


zy1 = np.loadtxt("/home/tmcclintock/Desktop/des_wl_work/DATA_FILES/y1_data_files/Y1_meanz.txt")

x = np.linspace(min(z), max(z), num=100)
rf = rspl(x)
ref = respl(x)

ry1 = rspl(zy1)
rey1 = respl(zy1)
dp1 = 1./ry1
dp1_err = rey1/ry1**2
np.savetxt("Y1_deltap1.txt", dp1)
np.savetxt("Y1_deltap1_var.txt", dp1_err**2)

plt.errorbar(z, r, re)
plt.plot(x, rf, c='k')
plt.fill_between(x, rf-ref, rf+ref, color='k', alpha=0.3)
plt.errorbar(zy1.flatten(), ry1.flatten(), rey1.flatten(), ls='', marker='.', c='r')
plt.ylabel(r"$(1+\delta)^{-1} = \Sigma_{crit,true}^{-1}/\Sigma_{crit,mof}^{-1}$")
plt.xlabel("redshift")
plt.show()
