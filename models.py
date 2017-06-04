"""
This file contains interfaces for the boost factor model and the
DeltaSigma model.
"""
import numpy as np
import os, sys
sys.path.insert(0, "../Delta-Sigma/src/wrapper/")
import py_Delta_Sigma as pyDS

#Boost factor model
def get_boost_model(params, l, z, R):
    #Pivots are 30 richness, 0.5 redshift, 500 kpc physical radius
    b0,c,d,e = params
    return 1.0 - b0 * (l/30.0)**c * ((1.+z)/1.5)**d * (R/0.5)**e
