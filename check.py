from src.gibbs.bayesian_lasso import lasso_hyperparameters
import numpy as np
d = np.load('data/processed/size_bm_25_arrays.npz')
hp = lasso_hyperparameters(d['R'], d['F'], d['F'].shape[2])
print(hp.lambda_r, hp.lambda_delta, hp.lambda_prior_rms)
