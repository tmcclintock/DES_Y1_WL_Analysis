import numpy as np
from likelihood_functions import *
from helper_functions import *
import cluster_toolkit as ct
import os, sys
from alter_args import *

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
    nwalkers, nsteps = 32, 4000
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
    name = "y1"
    model_name = "full" #Mc, full, Afixed, cfixed
    cal = False
    useJK = False

    #Loop over bins
    zhi, zlo = 2, 1
    lhi, llo = 6, 5
    for i in xrange(zhi, zlo, -1):#z bins #only 2,1,0 for y1 and sv but 3,2,1,0 for cal
        for j in xrange(lhi, llo, -1): #lambda bins
                    

            #LOOP OVER COSMOLOGICAL PARAMETERS HERE
            for H0 in [70.]:#[60.,65.,70.,75.,80.]:
                for Om in [0.2]:#,0.25,0.3,0.35,0.40]:
                    if Om == 0.3: continue #files not ready yet
                    paths, args = get_args_and_paths(name, i, j, H0, Om, model_name, cal, useJK)
                    args = update_args(args, i, j, H0, Om)
            
            bfpath, chainpath, likespath = paths
            print "Working at z%d l%d for %s"%(i,j,name)
            print "\tMean z:%f\n\tMean lambda:%f"%(args['z'], args['lam'])
            print "Saving to:\n\t%s\n\t%s\n\t%s"%(bfpath, chainpath, likespath)

            print i,j
            #Flow control for whatever you want to do
            test_call(args)
            find_best_fit(args, bfpath)
            #args["model_name"]=model_name #Reset this
            test_call(args, bfpath=bfpath, testbf=True)
            args["model_name"]=model_name #Reset this
            do_mcmc(args, bfpath, chainpath, likespath)
