"""
src/likelihood/sur.py

Shared SUR (seemingly unrelated regressions) likelihood for the Bayesian
hierarchical asset-pricing model (Feng & He, 2020/2021). Imported identically
by all four model chats so the "same likelihood throughout" design decision
is enforced in code, not just in the write-up.

Two implementations are provided:
- NumPy versions (sur_predicted_returns, sur_residuals,
  sur_log_likelihood_numpy) -- used for diagnostics/evaluation across all
  four fitted models. The Gibbs models (Gaussian baseline, Bayesian LASSO)
  never call a log-likelihood function directly during sampling -- Feng &
  He's Gibbs updates are closed-form conjugate formulas (eq. 14-18) -- so
  these functions exist for post-hoc evaluation, not for the Gibbs update
  step itself.
- A PyTensor version (sur_log_likelihood_pytensor) -- used inside the
  Horseshoe / Regularised Horseshoe PyMC model definitions, since NUTS needs
  a differentiable log-density it can build a gradient through.

Likelihood structure
---------------------
Given R = FB + E with Omega = Cov(E) = Sigma (x) I_N (Feng & He, eq. 7, 12),
residuals are independent across time and correlated across assets within a
period. Rather than construct the full NT x NT Omega, the joint
log-likelihood factorises as a sum, over T independent time periods, of
N-dimensional multivariate normal log-densities with covariance Sigma. This
mirrors exactly how generate.py's generate_residuals() draws E, and is the
same dimension reduction Feng & He describe (N^2 T^2 -> N^2 parameters).
"""

from __future__ import annotations

import numpy as np

def sur_predicted_returns(F: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute fitted/predicted returns R_hat[i,t] = f_{i,t}' b_i, given
    predictors F (N,T,K) and coefficients b (N,K).

    This is the same signal-construction step as generate.py's
    compute_returns(), minus the residual term -- here b is a fitted or
    hypothesised estimate (e.g. a posterior mean or a single MCMC draw),
    not a known ground truth.
    """
    return np.einsum("itk,ik->it", F, b)


def sur_residuals(R: np.ndarray, F: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute residuals E[i,t] = R[i,t] - f_{i,t}' b_i for a given b.
    Shape (N,T), matching generate.py's E convention.
    """
    return R - sur_predicted_returns(F, b)

def sur_log_likelihood_numpy(R: np.ndarray, F: np.ndarray, b: np.ndarray,
                              Sigma: np.ndarray) -> float:
    """
    Log-likelihood of the SUR model, exploiting the Omega = Sigma (x) I_N
    block structure: log p(R | F, b, Sigma) = sum_t log N(e_t; 0, Sigma),
    where e_t = R[:,t] - F[:,t,:] @ b is the length-N residual vector at
    time t.

    R : (N,T), F : (N,T,K), b : (N,K), Sigma : (N,N) -> scalar log-likelihood
    """
    N, T = R.shape
    E = sur_residuals(R, F, b)  # (N, T)

    sign, logdet_Sigma = np.linalg.slogdet(Sigma)
    if sign <= 0:
        raise np.linalg.LinAlgError(
            "Sigma is not positive definite (slogdet sign <= 0)"
        )

    Sigma_inv = np.linalg.inv(Sigma)
    # quadratic form summed over all T periods: sum_t e_t' Sigma^{-1} e_t
    quad_form = np.einsum("it,ij,jt->", E, Sigma_inv, E)

    log_lik = (
        -0.5 * T * N * np.log(2 * np.pi)
        - 0.5 * T * logdet_Sigma
        - 0.5 * quad_form
    )
    return float(log_lik)

import pytensor.tensor as pt
import pytensor.tensor.nlinalg as ptnla


def sur_log_likelihood_pytensor(R, F, b, Sigma):
    """
    PyTensor version of sur_log_likelihood_numpy, for use inside NUTS models
    via pm.Potential. Same block-factorised math as the NumPy version, but
    built entirely from pytensor.tensor operations so PyMC/NUTS can
    differentiate through it -- e.g. w.r.t. Sigma (from its Inverse-Wishart
    prior) or b (from Horseshoe's local/global shrinkage scales).

    R : (N,T), F : (N,T,K), b : (N,K), Sigma : (N,N) -- pytensor tensors
    (numpy arrays / PyMC random variables are automatically wrapped)
    -> scalar pytensor expression (a graph node, not a number -- gets
    evaluated as part of the model's log-probability).
    """
    # predicted returns and residuals, via broadcasting instead of einsum
    # (avoids depending on pytensor's einsum support, which is newer/less
    # universally available across pytensor versions than basic ops):
    #   b[:, None, :] reshapes b from (N,K) to (N,1,K) so it broadcasts
    #   against F's (N,T,K) -- elementwise multiply, then sum over K.
    signal = (F * b[:, None, :]).sum(axis=-1)   # (N, T)
    E = R - signal                               # (N, T)

    N, T = R.shape

    Sigma_inv = ptnla.matrix_inverse(Sigma)
    logdet_Sigma = pt.log(ptnla.det(Sigma))

    # quad_form = sum_t e_t' Sigma_inv e_t, computed without ever forming a
    # (T,T) matrix: (Sigma_inv @ E) is (N,T); multiplying elementwise by E
    # and summing everything gives exactly sum_{i,j,t} E[i,t]*Sigma_inv[i,j]*E[j,t]
    quad_form = (E * pt.dot(Sigma_inv, E)).sum()

    log_lik = (
        -0.5 * T * N * np.log(2 * np.pi)
        - 0.5 * T * logdet_Sigma
        - 0.5 * quad_form
    )
    return log_lik