"""
This file contains the blinding information, which get stored into a pickled file so that I can't easily read it. The form of the blinding is

B(lambda) = N(1, 0.05) * (lambda/30)^N(0, 0.2)

The random variables are saved once.
"""
import numpy as np
import pickle
import os

outpath = "blinding_file.p"

def make_random_variables():
    if os.path.isfile(outpath):
        return
    else:
        B0 = np.fabs(np.random.randn()*0.25 + 1.) #Amplitude
        alpha = np.random.randn()*0.2 # richness exponent 
        beta  = np.random.randn()*1.0 # 1+z exponent
        outarr = np.array([B0, alpha, beta])
        pickle.dump(outarr, open(outpath, "wb"))
    return

def get_blinding_variables():
    #To use this, multiply M by B0*(richness/30.0)^alpha*((1+z)/1.5)^beta
    B0, alpha, beta = pickle.load(open(outpath, "rb"))
    return B0, alpha, beta
    
if __name__ == "__main__":
    make_random_variables()
    b,a,be = get_blinding_variables()
    print be
