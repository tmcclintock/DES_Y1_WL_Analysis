import numpy as np

#fullbase = "/home/tmcclintock/Desktop/des_wl_work" #susie
fullbase = "/Users/tmcclintock/Data" #laptop
#fullbase = "/calvin1/tmcclintock/DES_DATA_FILES" #calvin

y1base = fullbase+"/DATA_FILES/y1_data_files/"
y1boostbase    = y1base+"FINAL_FILES/full-unblind-v2-mcal-zmix_y1clust_l%d_z%d_zpdf_boost.dat"
y1boostcovbase = y1base+"FINAL_FILES/full-unblind-v2-mcal-zmix_y1clust_l%d_z%d_zpdf_boost_cov.dat"

svbase = fullbase+"/DATA_FILES/sv_data_files/"
svboostbase = svbase+"SV_boost_factors.txt"
svboostcovbase = "need_z%d_l%d"

def get_boost_data_and_cov(zi, lj, lowcut = 0.2, highcut = 999, usey1=True, alldata=False, diag_only=False):
    if usey1:
        print "Y1 boost data"
        boostpath = y1boostbase%(lj, zi)
        bcovpath  = y1boostcovbase%(lj, zi)
        Bcov = np.loadtxt(bcovpath)
        Rb, Bp1, Be = np.genfromtxt(boostpath, unpack=True)
    else: #use_sv
        print "SV boost data"
        boostpath = svboostbase
        bcovpath  = svboostcovbase%(zi, lj) #doesn't exist
        if zi == 0: highcut = 21.5 #1 degree cut
        Bp1, Rb = np.genfromtxt(boostpath, unpack=True, skip_header=1)
        #SV didn't have boost errors. We construct them instead
        del2 = 10**-4.09 #Result from SV
        Bcov = np.diag(del2/Rb**2)
        Be = np.sqrt(Bcov.diagonal())
    #Cut out bad data, always make this cut
    Becut = Be > 1e-8 #errors go to 0 if the data is bad
    Bp1 = Bp1[Becut]
    Rb  = Rb[Becut]
    Be  = Be[Becut]
    Bcov = Bcov[Becut]
    Bcov = Bcov[:,Becut]
    if alldata:
        return Rb, Bp1, np.linalg.inv(Bcov), Bcov
    #Scale cut
    indices = (Rb > lowcut)*(Rb < highcut)
    Bp1 = Bp1[indices]
    Rb  = Rb[indices]
    Be  = Be[indices]
    Bcov = Bcov[indices]
    Bcov = Bcov[:,indices]
    Njk = 100.
    D = len(Rb)
    Bcov = Bcov*((Njk-1.)/(Njk-D-2)) #Hartlap correction
    if diag_only:
        Bcov = np.diag(Be**2)
    return Rb, Bp1, np.linalg.inv(Bcov), Bcov
