# DES_Y1_WL_Analysis
This contains the analysis code for the DES Y1 cluster WL analysis.

There are three relevant files in this directory:
* analysis_script.py: this is the "driver" of the analysis. It contains the maximum-likelihood and MCMC routines.
* get_all_data.py: this just reads in the DeltaSigma, covariances, and boost factor files that DES hasn't released yet but exist on my working computers.
* likelihood_functions.py: this contains the actually likelihoods used in the analysis. This includes the priors, the boost factor likelihood, and the DeltaSigma likelihood.
* figure_routines.py: this contains functions that create the plots we want to show. This hasn't been implemented yet.