import numpy as np
import cluster_toolkit as ct

def swap(params, args):
    name = args['model_name']
    if name == 'nfw':
        B0, Rs = params
        return B0, Rs, 0.0
    if name == 'pl': #powerlaw
        return params

def get_boost_model(params, args):
    B0, Rs, alpha = params #Rs in Mpc physical
    name = args['model_name']
    Rb = args['Rb'] #Mpc physical
    if name == 'nfw':
        return ct.boostfactors.boost_nfw_at_R(Rb, B0, Rs)
    elif name == 'pl':
        return ct.boostfactors.boost_powerlaw_at_R(Rb, B0, Rs, alpha)
