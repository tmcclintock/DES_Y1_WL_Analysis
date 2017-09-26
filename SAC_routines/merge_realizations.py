"""
Merge the realizations of DS into the covariance matrix.
"""
import numpy as np
import matplotlib.pyplot as plt

path = "output_files/stack_realizations_z%d_l%d.txt"

N_Radii = 1000
R = np.logspace(-2, 2.4, N_Radii, base=10)

for i in xrange(2,-1,-1):
    for j in xrange(6,5,-1):
        dss = np.loadtxt(path%(i,j))

for i in range(len(dss)):
    plt.loglog(R, dss[i])
plt.show()
