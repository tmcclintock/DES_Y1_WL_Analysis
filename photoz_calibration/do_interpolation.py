import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
plt.rc("text", usetex=True)
plt.rc("font", size=18, family="serif")

datapath = "sci_correction.dat"
z, Ratio, Ratioe = np.genfromtxt(datapath, unpack=True)

Ratiospl = interp.interp1d(z,Ratio)# , kind="cubic")
Ratioespl = interp.interp1d(z,Ratioe)#, kind="cubic")


zy1 = np.loadtxt("Y1_meanz.txt")

x = np.linspace(min(z), max(z), num=100)
Ratiof = Ratiospl(x)
Ratioef = Ratioespl(x)

Ratioy1 = Ratiospl(zy1)
Ratioey1 = Ratioespl(zy1)

dp1 = 1./Ratioy1
dp1_err = Ratioey1/Ratioy1**2
print dp1_err[2,6]
dp1_err = Ratioey1 * dp1**2
print "err = ",dp1_err[2,6]
print "var = ",dp1_err[2,6]**2

print zy1[2, 6]
print dp1[2, 6], 1./dp1[2,6]
print Ratioey1[2, 6]

np.savetxt("Y1_deltap1.txt", dp1)
np.savetxt("Y1_deltap1_var.txt", dp1_err**2)

plt.errorbar(z, Ratio, Ratioe)
plt.plot(x, Ratiof, c='k')
plt.fill_between(x, Ratiof-Ratioef, Ratiof+Ratioef, color='k', alpha=0.3)
plt.errorbar(zy1.flatten(), Ratioy1.flatten(), Ratioey1.flatten(), ls='', marker='.', c='r')
plt.ylabel(r"$(1+\delta)^{-1} = \Sigma_{crit,true}^{-1}/\Sigma_{crit,mof}^{-1}$")
plt.xlabel("redshift")
plt.subplots_adjust(bottom=0.15, left=0.15)
#plt.show()
