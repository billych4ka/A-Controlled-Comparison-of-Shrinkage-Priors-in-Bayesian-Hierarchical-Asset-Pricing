"""
src/priors/inverse_wishart.py

Custom, PyTensor-differentiable Inverse-Wishart log-density, needed because
PyMC has no native Inverse-Wishart distribution. Used by the Horseshoe and
Regularised Horseshoe models (NUTS) for Sigma's prior; the Gibbs-sampled
models (Gaussian baseline, Bayesian LASSO) don't need this file at all --
they use scipy.stats.invwishart directly, since Gibbs sampling doesn't need
a differentiable log-density, just the ability to draw from the conditional
posterior (Feng & He eq. 18).

Why this can't just be "the IW formula"
-----------------------------------------
NUTS requires every sampled quantity to be an *unconstrained* real number --
it has no mechanism for keeping a matrix symmetric and positive definite
(PD) at every step of sampling. Sigma itself can't be handed to NUTS
directly for this reason. The standard solution (used internally by Stan,
and by PyMC's own LKJCholeskyCov for correlation matrices) is to instead let
NUTS sample a Cholesky factor L (lower-triangular, positive diagonal), and
reconstruct Sigma = L @ L.T deterministically. Since this is a change of
variables -- NUTS is sampling in "L-space", not "Sigma-space" -- evaluating
p(Sigma) alone is not enough: the log-density needs an added Jacobian
correction term for the L -> Sigma transformation, or the sampler will
target the wrong distribution (silently -- it will still run without
erroring, just sample from an incorrect posterior).

This file provides:
1. inverse_wishart_logp(Sigma, nu, Psi) -- the plain IW log-density, as a
   function of an already-valid Sigma (no Jacobian; a building block).
2. packed_to_cholesky(packed_raw, dim) -- turns NUTS's raw unconstrained
   vector into a valid Cholesky factor L.
3. cholesky_jacobian_logdet(L_diag, dim) -- the Jacobian correction for the
   L -> Sigma change of variables (see Muirhead, "Aspects of Multivariate
   Statistics", or the Stan manual's chapter on covariance matrices, for the
   classical derivation this follows).
4. inverse_wishart_cholesky_logp(packed_raw, nu, Psi, dim) -- combines 1-3
   into the single function a NUTS model chat actually calls: returns both
   Sigma (for pm.Deterministic) and the corrected total log-density (for
   pm.Potential).
"""

from __future__ import annotations

import numpy as np
import pytensor.tensor as pt
import pytensor.tensor.nlinalg as ptnla
from scipy.special import multigammaln

def inverse_wishart_logp(Sigma, nu: float, Psi: np.ndarray):
    """
    Log-density of Sigma ~ Inverse-Wishart(nu, Psi), for an already-valid
    (symmetric, PD) Sigma. This is the textbook IW formula with no Jacobian
    correction -- a building block for inverse_wishart_cholesky_logp, not
    something called directly on a NUTS-sampled quantity.

    Sigma : (dim,dim) pytensor tensor (symmetric PD)
    nu    : degrees of freedom (fixed Python float, not sampled)
    Psi   : (dim,dim) numpy array, the scale matrix (fixed hyperparameter)
    -> scalar pytensor expression
    """
    dim = Psi.shape[0]

    # log-normalizing-constant: depends only on fixed hyperparameters (nu,
    # Psi), never on the sampled Sigma -- so this is plain Python/NumPy
    # arithmetic, computed once, not part of the differentiable graph.
    log_norm_const = (
        (nu / 2) * np.linalg.slogdet(Psi)[1]
        - (nu * dim / 2) * np.log(2)
        - multigammaln(nu / 2, dim)
    )

    Sigma_inv = ptnla.matrix_inverse(Sigma)
    logdet_Sigma = pt.log(ptnla.det(Sigma))

    # trace(Psi @ Sigma_inv): both Psi and Sigma_inv are symmetric, so
    # trace(AB) = sum(A * B) elementwise -- avoids relying on an uncertain
    # pytensor "trace" function name (same caution as the nlinalg lesson).
    trace_term = (Psi * Sigma_inv).sum()

    logp = (
        log_norm_const
        - ((nu + dim + 1) / 2) * logdet_Sigma
        - 0.5 * trace_term
    )
    return logp

