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
        B0 = np.fabs(np.random.randn()*0.15 + 1.)
        alpha = np.random.randn()*0.2
        outarr = np.array([B0, alpha])
        pickle.dump(outarr, open(outpath, "wb"))
    return

def get_blinding_variables():
    B0, alpha = pickle.load(open(outpath, "rb"))
    return B0, alpha
    
if __name__ == "__main__":
    make_random_variables()
