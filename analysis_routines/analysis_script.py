import numpy as np
from likelihood_functions import *
from helper_functions import *
import models
import sys
import clusterwl #Used to get xi_mm(R) from P(k)
import blinding
Blinding_amp, lam_exp, z_exp = blinding.get_blinding_variables()
cal = True
zmap = np.array([0, 0, 1, 1]) #Maps zi to y1zi for the calibration
cosmo = get_cosmo_default(cal)
h = cosmo['h'] #Hubble constant

model_name = "full" #Mc, full, Afixed, cfixed

def test_call(args, bfpath=None, testbf=False):
    lam = args['lam']
    guess = get_model_start(model_name, lam, h)
    if testbf:
        print "Testing bestfit"
        guess = np.loadtxt(bfpath) #Has everything
        args['model_name'] = "full" #always use full here
    print "Test call: lnprob(start) = ", lnprob(guess, args)
    print ""
    return

def find_best_fit(args, bestfitpath, usey1):
    z = args['z']
    lam = args['lam']
    args['model_name'] = "full" #always use full here
    model_name = args['model_name']
    blinding_factor = args['blinding_factor']
    guess = get_model_start(model_name, lam, h)
    import scipy.optimize as op
    nll = lambda *args: -lnprob(*args)
    result = op.minimize(nll, guess, args=(args,), tol=1e-2)
    print "Best fit being saved at :\n%s"%bestfitpath
    print "\tsuccess = %s"%result['success']
    #outmodel = models.model_swap(result['x'], z, blinding_factor, model_name)
    np.savetxt(bestfitpath, result['x'])
    return 

def do_mcmc(args, bfpath, chainpath, likespath, usey1, new_chain=True):
    nwalkers, nsteps = 32, 3000
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
    if new_chain:
        np.savetxt(chainpath, sampler.flatchain)
        np.savetxt(likespath, sampler.flatlnprobability)
    else: #add to old chain
        chain = np.loadtxt(chainpath)
        likes = np.loadtxt(likespath)
        print chain.shape, sampler.flatchain.shape
        np.savetxt(chainpath, np.concatenate((chain, sampler.flatchain), axis=0))
        np.savetxt(likespath, np.concatenate((likes, sampler.flatlnprobability), axis=0))
    return 0

if __name__ == '__main__':
    usey1 = True
    blinded = True
    useJK = False
    zs, lams = get_zs_and_lams(usey1 = usey1, cal=cal)
    Rlams = (lams/100.0)**0.2 #Mpc/h; richness radius
    SCIs = get_sigma_crit_inverses(usey1)

    #This specifies which analysis we are doing
    #Name options are full, fixed, boostfixed or Afixed
    if usey1: 
        name = "y1"
        if useJK: covname = "JK"
        else: covname = "SAC"
    else:
        name = "sv"
        covname = "JK"
    if cal:
        name = "cal"
        covname = "SAC"
    basesuffix = name+"_"+covname+"_z%d_l%d"    
    bestfitbase = "bestfits/bf_%s.txt"%basesuffix
    chainbase   = "chains/chain_%s_%s.txt"%(model_name, basesuffix)
    likesbase   = "chains/likes_%s_%s.txt"%(model_name, basesuffix)

    import matplotlib.pyplot as plt
    #Loop over bins
    for i in xrange(3, -1, -1): #z bins #only 2,1,0 for y1 and sv but 3,2,1,0 for cal
        if i <3: continue
        for j in xrange(6, -1, -1): #lambda bins
            if j > 6 or j < 6: continue
            print "Working at z%d l%d for %s"%(i,j,name)
            z    = zs[i,j]
            lam  = lams[i,j]
            Rlam = Rlams[i,j]
            print "z = %.2f lam = %.2f"%(z, lam)
            if cal: sigma_crit_inv = SCIs[zmap[i],j]* h*(1+z)**2 #Convert to pc^2/hMsun comoving
            else: sigma_crit_inv = SCIs[i,j]* h*(1+z)**2 #Convert to pc^2/hMsun comoving
            k, Plin, Pnl = get_power_spectra(i, j, usey1, cal=cal)
            Rmodel = np.logspace(-2, 3, num=1000, base=10) 
            xi_mm = clusterwl.xi.xi_mm_at_R(Rmodel, k, Pnl)
            #Xi_mm MUST be evaluated to higher than BAO for correct accuracy

            #Note: convert Rlam to Mpc physical when we specificy the cuts
            Rdata, ds, icov, cov, inds = get_data_and_icov(i, j, usey1=usey1, useJK=useJK, cal=cal)

            if cal: Rb, Bp1, iBcov, Bcov = get_boost_data_and_cov(zmap[i], j, usey1=usey1)
            else: Rb, Bp1, iBcov, Bcov = get_boost_data_and_cov(i, j, usey1=usey1)
            bfpath    = bestfitbase%(i, j)
            chainpath = chainbase%(i, j)
            likespath = likesbase%(i, j)

            #Multiplicative prior
            if cal: Am_prior, Am_prior_var = get_Am_prior(zmap[i], j)
            else: Am_prior, Am_prior_var = get_Am_prior(i, j)

            #Group everything up for convenience
            Redges = get_Redges(usey1 = usey1) * h*(1+z) #Mpc/h comoving
            blinding_factor = 0
            if blinded: blinding_factor = np.log10(Blinding_amp) +  np.log10((lam/30.0)**lam_exp) + np.log10(((1+z)/1.5)**z_exp)
            if cal: blinding_factor*= 0
            
            args = {"z":z, "lam":lam, "Rlam":Rlam, "Rdata":Rdata, "ds":ds, "cov":cov, "icov":icov, "Rb":Rb, "Bp1":Bp1, "Bcov":Bcov, "iBcov":iBcov, "k":k, "Plin":Plin, "Pnl":Pnl, "Rmodel":Rmodel, "xi_mm":xi_mm, "Redges":Redges, "inds":inds, "Am_prior":Am_prior, "Am_prior_var":Am_prior_var, "sigma_crit_inv":sigma_crit_inv, "blinding_factor":blinding_factor, "model_name":model_name}

            #Flow control for whatever you want to do
            test_call(args)
            find_best_fit(args, bfpath, usey1)
            args["model_name"]=model_name #Reset this
            test_call(args, bfpath=bfpath, testbf=True)
            args["model_name"]=model_name #Reset this
            #do_mcmc(args, bfpath, chainpath, likespath, usey1)#, new_chain=False)
