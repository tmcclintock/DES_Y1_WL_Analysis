
def plot_boost_and_resid(params, args, i, j):
    Lm, c, tau, fmis, Am, B0, Rs = params
    Rdata = args['Rdata']
    cuts = args['cuts']
    cov = args['cov']
    z = args['z']
    lo,hi = cuts
    Rb = args['Rb']
    Bp1 = args['Bp1']
    Bcov = args['Bcov']
    Berr = np.sqrt(np.diag(Bcov))
    Rmodel, DSfull, DSc, DSm, boost, aDS = calc_DS_model(params, args)
    boost_Rb = clusterwl.boostfactors.boost_nfw_at_R(Rb, B0, Rs*h*(1+z))
    good = (lo<Rb)*(Rb<hi)
    bad  = (lo>Rb)+(Rb>hi)
    fig, axarr = plt.subplots(2, sharex=True)
    axarr[0].errorbar(Rb[good], Bp1[good], Berr[good], c='k', marker='o', ls='', markersize=3, zorder=1)
    axarr[0].errorbar(Rb[bad], Bp1[bad], Berr[bad], c='k', marker='o', ls='', markersize=3, zorder=1, mfc='w')
    axarr[0].plot(Rmodel, boost, c='r')
    axarr[0].axhline(1, ls='-', c='k')
    axarr[0].set_yticklabels([])
    axarr[0].get_yaxis().set_visible(False)
    axarr[0].set_ylim(.9, 1.8)
    axarr[0].set_ylabel(r"$(1-f_{\rm cl})^{-1}$")

    pd = (Bp1 - boost_Rb)#/(boost_Rb)
    pde = Berr/boost_Rb
    axarr[1].errorbar(Rb[good], pd[good], pde[good], c='k', marker='o', ls='', markersize=3, zorder=1)
    axarr[1].errorbar(Rb[bad], pd[bad], pde[bad], c='k', marker='o', mfc='w', markersize=3, ls='', zorder=1)
    axarr[1].axhline(0, ls='-', c='k')
    axarr[1].set_xscale('log')
    plt.xlim(0.1, 30.)
    axarr[1].set_ylabel(r"$\frac{(1-f_{\rm cl})^{-1}-\mathcal{B}}{\mathcal{B}}$")
    plt.gcf().savefig("figures/boostfactor_z%d_l%d.pdf"%(i,j))
    plt.show()

def plot_fourpanels(params, args, i, j):
    lM, c, tau, fmis, Am, B0, Rs, sigb = params
    Rdata = args['Rdata']
    cuts = args['cuts']
    cov = args['cov']
    z = args['z']
    lo,hi = cuts
    good = (lo<Rdata)*(Rdata<hi)
    bad  = (lo>Rdata)+(Rdata>hi)
    dserr = np.sqrt(np.diag(cov))
    dserr_fixed = fix_errorbars(ds, dserr)
    Rmodel, DSfull, DSc, DSm, boost, aDS = calc_DS_model(params, args)
    #Convert to Mpc physical
    Rmodel /= h*(1+z)
    DSfull *= h*(1+z)**2
    DSc *= h*(1+z)**2
    DSm *= h*(1+z)**2
    aDS *= h*(1+z)**2

    X = (ds- aDS)[good]
    cov = cov[good]
    cov = cov[:, good]
    chi2ds = np.dot(X, np.dot(np.linalg.inv(cov), X))
    Nds = len(X)

    gs = gridspec.GridSpec(3, 6)
    axarr = [plt.subplot(gs[0:2, 0:3]), plt.subplot(gs[0:2, 3:]), plt.subplot(gs[-1, 0:3]), plt.subplot(gs[-1, 3:])]
    axarr[0].errorbar(Rdata[good], ds[good], dserr_fixed[:,good], c='k', marker='o', ls='', markersize=3, zorder=1)
    axarr[0].errorbar(Rdata[bad], ds[bad], dserr_fixed[:,bad], c='k', marker='o', mfc='w', markersize=3, ls='', zorder=1)
    #First plot DeltaSigma
    axarr[0].loglog(Rmodel, DSfull, c='r', zorder=0)
    axarr[0].loglog(Rmodel, DSm, c='b', ls='--', zorder=-1)
    axarr[0].loglog(Rmodel, DSc, c='k', ls='-.', zorder=-3)
    axarr[0].loglog(Rmodel, DSfull*boost, c='g', ls='-', zorder=-2)
    axarr[0].set_ylabel(DSlabel)
    
    pd = (ds - aDS)/aDS
    pde = dserr/aDS
    axarr[2].errorbar(Rdata[good], pd[good], pde[good], c='k', marker='o', ls='', markersize=3, zorder=1)
    axarr[2].errorbar(Rdata[bad], pd[bad], pde[bad], c='k', marker='o', mfc='w', markersize=3, ls='', zorder=1)
    axarr[2].axhline(0, ls='-', c='k')
    
    Rb = args['Rb']
    Bp1 = args['Bp1']
    Bcov = args['Bcov']
    Berr = np.sqrt(np.diag(Bcov))
    boost_Rb = clusterwl.boostfactors.boost_nfw_at_R(Rb, B0, Rs*h*(1+z))
    good = (lo<Rb)*(Rb<hi)
    bad  = (lo>Rb)+(Rb>hi)
    axarr[1].errorbar(Rb[good], Bp1[good], Berr[good], c='k', marker='o', ls='', markersize=3, zorder=1)
    axarr[1].errorbar(Rb[bad], Bp1[bad], Berr[bad], c='k', marker='o', ls='', markersize=3, zorder=1, mfc='w')
    axarr[1].plot(Rmodel, boost, c='r')
    axarr[1].axhline(1, ls='-', c='k')
    axarr[1].set_yticklabels([])
    axarr[1].get_yaxis().set_visible(False)
    axarr[1].set_ylim(.9, 1.8)
    axtwin = axarr[1].twinx()
    axtwin.set_ylabel(r"$(1-f_{\rm cl})^{-1}$")
    axtwin.set_ylim(axarr[1].get_ylim())

    X = (Bp1 - boost_Rb)[good]
    Bcov = Bcov[good]
    Bcov = Bcov[:, good]
    chi2b = np.dot(X, np.dot(np.linalg.inv(Bcov), X))
    Nb = len(X)
    
    pd = (Bp1 - boost_Rb)/boost_Rb#(boost_Rb-1)
    pde = Berr/boost_Rb
    axarr[3].errorbar(Rb[good], pd[good], pde[good], c='k', marker='o', ls='', markersize=3, zorder=1)
    axarr[3].errorbar(Rb[bad], pd[bad], pde[bad], c='k', marker='o', mfc='w', markersize=3, ls='', zorder=1)
    axarr[3].axhline(0, ls='-', c='k')

    ylim = 1.2
    axarr[2].set_ylim(-ylim, ylim)
    ylim = 0.06
    axarr[3].set_ylim(-ylim, ylim)
    axarr[3].set_yticklabels([])
    axtwin2 = axarr[3].twinx()
    axtwin2.set_ylim(-ylim, ylim)
    axtwin2.set_ylabel(r"$\frac{(1-f_{\rm cl})^{-1}-\mathcal{B}}{\mathcal{B}}$")

    #axarr[2].set_ylabel(r"\% Diff")#, fontsize=14)
    #axarr[2].set_ylabel(r"${\rm \frac{Data-Model}{Model}}$")
    axarr[2].set_ylabel(r"${\rm \frac{\Delta\Sigma-\Delta\Sigma_{Model}}{\Delta\Sigma_{Model}}}$")
    axarr[2].set_xlabel(Rlabel)
    axarr[3].set_xlabel(Rlabel)    
    axarr[0].set_ylim(0.1, 1e3)
    for axinds in range(4):
        axarr[axinds].set_xscale('log')        
        axarr[axinds].set_xlim(0.1, 30.)
    axarr[0].set_xticklabels([])
    axarr[1].set_xticklabels([])
    axarr[3].set_xticks([1, 10])
    if usey1: zlabel, llabel = y1zlabels[i], y1llabels[j]
    else: zlabel, llabel = svzlabels[i], svllabels[j]
    labelfontsize=16
    axarr[1].text(.8, 1.65, zlabel, fontsize=labelfontsize)
    axarr[1].text(.8, 1.5,  llabel, fontsize=labelfontsize)
    axarr[1].text(.8, 1.35, r"$\chi^2_{\rm \mathcal{B}}=%.1f/%d$"%(chi2b, Nb), fontsize=labelfontsize)
    axarr[0].text(.2, .6, r"$\chi^2_{\Delta\Sigma}=%.1f/%d$"%(chi2ds, Nds), fontsize=labelfontsize)
    axarr[1].text(.8, 1.23, r"$\chi^2=%.1f/%d$"%(chi2ds+chi2b, Nds+Nb), fontsize=labelfontsize)

    plt.subplots_adjust(hspace=0.0, wspace=0.0, bottom=0.15, left=0.17, right=0.80)
    #plt.suptitle("%s %s"%(zlabel, llabel))
    plt.gcf().savefig("figures/fourpanel_z%d_l%d.pdf"%(i,j))
    plt.show()
    plt.clf()
    #plt.close()
