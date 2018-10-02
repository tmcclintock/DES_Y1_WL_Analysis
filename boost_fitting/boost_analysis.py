import numpy as np
from likelihoods import *
from helper_functions import *
import cluster_toolkit as ct
import os, sys

def starts(name):
    if name == 'nfw':
        return [0.14, 1.5]
    elif name == 'pl':
        return [0.1, 1.0, -1.0]

def test_call(args):
    guess = starts(args['model_name'])
    print "Test call: lnprob(start) = %.e2\n"%lnprob(guess, args)
    return

def do_best_fit(args, bfpath):
    guess = starts(args['model_name'])
    import scipy.optimize as op
    nll = lambda *args: -lnprob(*args)
    print "Running best fit"
    result = op.minimize(nll, guess, args=(args,), method='Powell')
    print result
    print "Best fit saved at :\n\t%s"%bfpath
    print "\tSuccess = %s\n\t%s"%(result['success'],result['x'])
    #print result
    np.savetxt(bfpath, result['x'])
    return

def plot_bf(args, bfpath, show=False):
    import matplotlib.pyplot as plt
    import model as mod
    guess = np.loadtxt(bfpath)
    i, j, usey1 = args['zi'], args['lj'], args['usey1']
    Rb, Bp1, iBcov, Bcov = get_boost_data_and_cov(i, j, usey1=usey1, alldata=True)
    args['Rb'] = Rb
    B0, Rs, alpha = mod.swap(guess,args)
    boost = mod.get_boost_model(mod.swap(guess, args), args)
    plt.plot(Rb, boost, label="%s model"%args['model_name'])
    plt.errorbar(Rb, Bp1, np.sqrt(Bcov.diagonal()))
    plt.legend()
    plt.xscale('log')
    #plt.yscale('log')
    plt.title("z%d l%d"%(i,j))
    plt.gcf().savefig("figures/boost_%s_%s_z%d_l%d.png"%(args['name'], args['model_name'],i,j))
    #if show:
    plt.show()
    plt.clf()
    Rm = np.logspace(-1,np.log10(30), num=100)
    bout = ct.boostfactors.boost_nfw_at_R(Rm, B0, Rs)
    header = "R[Mpc; physical]; (1-f_{\rm cl})^{-1}"
    fmt = "%.3f %.4e"
    np.savetxt("tamas_files/boost_l%d_z%d.txt"%(j,i), np.array([Rm,bout]).T, header=header, fmt=fmt)

def do_mcmc(args, bfpath, chainpath, likespath):
    nwalkers, nsteps = 32, 3000
    import emcee
    bf = np.loadtxt(bfpath)
    ndim = len(bf)
    pos = [bf + 1e-3*np.random.randn(ndim) for k in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(args,), threads=4)
    print "Starting MCMC, saving to \n\t%s"%chainpath
    sampler.run_mcmc(pos, nsteps)
    print "MCMC complete"
    if os.path.isfile(chainpath):
        np.savetxt(chainpath+".new", sampler.flatchain)
        np.savetxt(likespath+".new", sampler.flatlnprobability)
    else:
        np.savetxt(chainpath, sampler.flatchain)
        np.savetxt(likespath, sampler.flatlnprobability)
    return

if __name__ == "__main__":
    name = 'y1'
    usey1 = True
    model_name = "nfw"
    #Model name can be nfw or pl
    
    if name != 'y1': #Doing SV
        usey1 = False
    bfbase = "bestfits/bf_boost_%s_%s"%(name,model_name)
    chainbase = "chains/chain_boost_%s_%s"%(name, model_name)
    likesbase = "chains/likes_boost_%s_%s"%(name, model_name)

    """
    bff = open("bestfit_params.txt", "w")
    bff.write("zi lj B0 Rs\n")
    for i in xrange(0,3):
        for j in xrange(3,7):
            bfpath = bfbase+"_z%d_l%d.txt"%(i, j)
            B0,Rs = np.loadtxt(bfpath)
            bff.write("%d %d %.3e %.3e\n"%(i,j,B0,Rs))
    bff.close()
    """
    #exit()
            
    Nz, Nl = 3, 7
    for i in xrange(2, 1, -1):
        for j in xrange(6, 5, -1):
            bfpath = bfbase+"_z%d_l%d.txt"%(i, j)
            chainpath = chainbase+"_z%d_l%d.txt"%(i, j)
            likespath = likesbase+"_z%d_l%d.txt"%(i, j)
            
            print "Working at z%d l%d"%(i, j)
            Rb, Bp1, iBcov, Bcov = get_boost_data_and_cov(i, j, usey1=usey1, alldata=False, diag_only=False)
            args = {'Rb':Rb, 'Bp1':Bp1, 'iBcov':iBcov, 'Bcov':Bcov, 'zi':i, 'lj':j, 'usey1':usey1, 'model_name':model_name, 'name':name}
            
            test_call(args)
            do_best_fit(args, bfpath)
            #plot_bf(args, bfpath, show=True)
            do_mcmc(args, bfpath, chainpath, likespath)
