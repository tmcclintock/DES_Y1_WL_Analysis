import numpy as np
from likelihood_functions import *
from helper_functions import *
import cluster_toolkit as ct
import os, sys

def test_call(args, bfpath=None, testbf=False):
    guess = get_model_start(model_name, args['lam'], args['h'])
    if testbf:
        print "Testing bestfit"
        guess = np.loadtxt(bfpath) #Has everything
        args['model_name'] = "full" #always use full here
    print "Test call: lnprob(start) = %.2e\n"%lnprob(guess, args)
    return

def find_best_fit(args, bestfitpath):
    z = args['z']
    lam = args['lam']
    h = args['h']
    model_name = args['model_name']
    guess = get_model_start(model_name, lam, h)
    import scipy.optimize as op
    nll = lambda *args: -lnprob(*args)
    print "Running best fit"
    result = op.minimize(nll, guess, args=(args,), tol=1e-2)
    print "Best fit being saved at :\n\t%s"%bestfitpath
    print "\tsuccess = %s"%result['success']
    print result
    np.savetxt(bestfitpath, result['x'])
    return 

def do_mcmc(args, bfpath, chainpath, likespath):
    nwalkers, nsteps = 32, 10000
    import emcee
    model_name = args['model_name']
    bfmodel = np.loadtxt(bfpath) #Has everything
    args['bf_defaults'] = bfmodel
    start = get_mcmc_start(bfmodel, model_name)
    ndim = len(start) #number of free parameters
    pos = [start + 1e-3*np.random.randn(ndim) for k in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(args,), threads=8)
    print "Starting MCMC, saving to %s"%chainpath
    sampler.run_mcmc(pos, nsteps)
    print "MCMC complete"
    if os.path.isfile(chainpath):
        np.savetxt(chainpath+".new", sampler.flatchain)
        np.savetxt(likespath+".new", sampler.flatlnprobability)
    else:
        np.savetxt(chainpath, sampler.flatchain)
        np.savetxt(likespath, sampler.flatlnprobability)
    return

if __name__ == '__main__':
    name = "Y1"
    model_name = "full" #Mc, full, Afixed, cfixed
    blinded = False
    cal = False
    useJK = False

    #Loop over bins
    zhi, zlo = 2, 1
    lhi, llo = 3, 2
    for i in xrange(zhi, zlo, -1):#z bins #only 2,1,0 for y1 and sv but 3,2,1,0 for cal
        for j in xrange(lhi, llo, -1): #lambda bins
            args = get_args(model_name, i, j, name, "SAC", blinded)
            paths = get_output_paths(model_name, i, j, name, "SAC", blinded)
            #paths, args = get_args_and_paths(name, i, j, model_name, blinded, cal, useJK)
            
            bfpath, chainpath, likespath = paths
            bfpath += ".orientation"
            chainpath += ".orientation"
            likespath += ".orientation"
            print "Working at z%d l%d for %s"%(i,j,name)
            print "\tMean z:%f\n\tMean lambda:%f"%(args['z'], args['lam'])
            print "Saving to:\n\t%s\n\t%s\n\t%s"%(bfpath, chainpath, likespath)

            print i,j, args['blinding_factor']
            #Flow control for whatever you want to do
            test_call(args)
            find_best_fit(args, bfpath)
            args["model_name"]=model_name #Reset this
            test_call(args, bfpath=bfpath, testbf=True)
            args["model_name"]=model_name #Reset this
            #do_mcmc(args, bfpath, chainpath, likespath)
