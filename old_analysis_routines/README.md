# DES_Y1_WL_Analysis
This contains the analysis code for the DES Y1 cluster WL analysis.

There are three relevant files in this directory:
* analysis_script.py: this is the "driver" of the analysis. It contains the maximum-likelihood and MCMC routines.
* get_all_data.py: this just reads in the DeltaSigma, covariances, and boost factor files that DES hasn't released yet but exist on my working computers.
* likelihood_functions.py: this contains the actually likelihoods used in the analysis. This includes the priors, the boost factor likelihood, and the DeltaSigma likelihood.
* figure_routines.py: this contains functions that create the plots we want to show. This file will show which routines correspond to which figures in the paper. This file will be written last and isn't close to complete yet.

There are four directories found in this one:
* P_files: contains the linear and non-linear power spectra at the mean redshift of each bin. Since the DES data isn't released yet, and one could get the redshifts from the power spectra given our fiducial cosmology, these files are not yet included in the public repository
* figures: contains figures. Some will go in the paper and some are just for diagnostic. Some will go in the repository and others will not.
* bestfits: contains the results of the maximum-likelihood analysis for the fits to each bin. These are not included in the public repository.
* chains: the full chains generated from emcee for each bin. These are not included in the public repository.