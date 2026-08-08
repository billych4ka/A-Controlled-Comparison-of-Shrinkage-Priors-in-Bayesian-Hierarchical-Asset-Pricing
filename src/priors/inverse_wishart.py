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

def packed_to_cholesky(packed_raw, dim: int):
    """
    Convert a vector of dim*(dim+1)/2 unconstrained real numbers (what NUTS
    actually samples) into a valid dim x dim lower-triangular Cholesky
    factor L with strictly positive diagonal entries.

    Off-diagonal entries of L map directly from packed_raw (any real number
    is already valid for an off-diagonal Cholesky entry). Diagonal entries
    are passed through exp(...) first, since a Cholesky factor's diagonal
    must be strictly positive -- exp() guarantees this for any real input.

    packed_raw : (dim*(dim+1)/2,) pytensor vector, NUTS's raw parameters
    -> (L, L_diag): L is (dim,dim) lower-triangular; L_diag is (dim,) the
       diagonal entries of L *before* returning, kept separate because
       cholesky_jacobian_logdet (next chunk) needs them directly.
    """
    L = pt.zeros((dim, dim))

    idx = 0
    diag_entries = []
    for i in range(dim):
        for j in range(i + 1):
            if i == j:
                # diagonal entry: exponentiate for positivity
                val = pt.exp(packed_raw[idx])
                diag_entries.append(val)
            else:
                # off-diagonal entry: unconstrained, used as-is
                val = packed_raw[idx]
            L = pt.set_subtensor(L[i, j], val)
            idx += 1

    L_diag = pt.stack(diag_entries)
    return L, L_diag

def cholesky_jacobian_logdet(L_diag, dim: int):
    """
    Log-Jacobian-determinant for the full change of variables from NUTS's
    unconstrained packed_raw vector to Sigma = L @ L.T.

    Two transformations are stacked here, each contributing its own
    Jacobian term:
      1. L -> Sigma = L @ L.T (the Cholesky map itself). Classical result
         (e.g. Muirhead, "Aspects of Multivariate Statistics"; also used
         internally by Stan's covariance-matrix machinery):
             |d Sigma / dL| = 2^dim * prod_i L_ii^(dim - i + 1)
         (product over i = 1, ..., dim, using 1-indexing)
      2. packed_raw's diagonal entries -> L's diagonal entries, via
         L_ii = exp(packed_raw_ii). Standard exponential-map Jacobian:
             |dL_ii / d(packed_raw_ii)| = L_ii
         (since d/dx[exp(x)] = exp(x) = L_ii itself)

    Combining (log of a product = sum of logs, and the two transformations
    compose multiplicatively):
        log|J| = dim*log(2) + sum_i (dim - i + 1)*log(L_ii) + sum_i log(L_ii)
               = dim*log(2) + sum_i (dim - i + 2)*log(L_ii)

    This must be ADDED to the log-density evaluated at Sigma(L), so that
    the total expression is the correct density for packed_raw -- NOT the
    density for Sigma alone (which is what inverse_wishart_logp computes on
    its own, and is not sufficient by itself once a change of variables is
    involved).

    L_diag : (dim,) pytensor vector, diagonal entries of L
    -> scalar pytensor expression
    """
    # i = 1, ..., dim (1-indexed, matching the classical formula above)
    i = pt.arange(1, dim + 1)
    exponents = dim - i + 2  # combines both Jacobian terms' exponents

    log_jacobian = dim * pt.log(2.0) + (exponents * pt.log(L_diag)).sum()
    return log_jacobian

def inverse_wishart_cholesky_logp(packed_raw, nu: float, Psi: np.ndarray):
    """
    The function a NUTS model chat actually calls. Combines packed_to_cholesky,
    inverse_wishart_logp, and cholesky_jacobian_logdet into the single total
    log-density for Sigma ~ IW(nu, Psi), correctly expressed in terms of
    NUTS's actual unconstrained sampled quantity (packed_raw).

    packed_raw : (dim*(dim+1)/2,) pytensor vector -- what NUTS samples
    nu, Psi    : fixed IW hyperparameters (Psi determines dim = Psi.shape[0])
    -> (Sigma, total_logp):
         Sigma      : (dim,dim) pytensor expression -- wrap in
                       pm.Deterministic("Sigma", Sigma) so it's saved and
                       usable downstream (e.g. by sur_log_likelihood_pytensor)
         total_logp : scalar pytensor expression -- pass to
                       pm.Potential("Sigma_prior", total_logp)

    Usage inside a PyMC model:
        dim = N  # number of assets
        n_params = dim * (dim + 1) // 2
        packed_raw = pm.Flat("Sigma_packed_raw", shape=n_params)
        # NOTE: must be pm.Flat, not pm.Normal or any other proper prior --
        # pm.Potential ADDS to the model's total logp, it doesn't replace
        # the declared variable's own prior. A proper prior here would
        # silently contaminate Sigma's effective distribution (confirmed
        # empirically: pm.Normal(0,1) here biased E[Sigma] by 8-11 MCMC
        # standard errors in testing; pm.Flat resolved it to <1.3 SE).

        Sigma, sigma_logp = inverse_wishart_cholesky_logp(packed_raw, nu_Sigma, Psi_Sigma)
        Sigma = pm.Deterministic("Sigma", Sigma)
        pm.Potential("Sigma_prior", sigma_logp)
    """
    dim = Psi.shape[0]

    L, L_diag = packed_to_cholesky(packed_raw, dim)
    Sigma = L @ L.T

    base_logp = inverse_wishart_logp(Sigma, nu, Psi)
    jacobian = cholesky_jacobian_logdet(L_diag, dim)

    total_logp = base_logp + jacobian
    return Sigma, total_logp
