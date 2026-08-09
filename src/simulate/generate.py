"""
src/simulate/generate.py

Simulated-data generator for the Bayesian hierarchical SUR asset-pricing model
(Feng & He, 2020/2021). Used to validate the Gibbs samplers and NUTS models
against known ground truth before touching real data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

DeviationSampler = Callable[[np.random.Generator, int, int, np.ndarray], np.ndarray]

@dataclass
class SimulatedSURData:
    """Container for a simulated dataset and the ground truth used to build it."""

    N: int                 # number of assets (portfolios)
    K: int                 # number of predictors (incl. intercept)
    T: int                 # number of time periods
    b_bar: np.ndarray      # (K,)     -- true pooled/average coefficient vector, shared across assets
    Delta_b: np.ndarray    # (K, K)   -- true covariance of asset deviations from b_bar
    Sigma: np.ndarray      # (N, N)   -- true residual covariance across assets (cross-sectional)
    theta: np.ndarray      # (N, K)   -- true deviation of each asset's coefficients from b_bar
    b: np.ndarray          # (N, K)   -- true asset-specific coefficients, b = b_bar + theta
    F: np.ndarray          # (N, T, K) -- simulated predictor data, F[i, t, :] = f_{i,t}
    E: np.ndarray          # (N, T)   -- simulated residuals (noise), E[i, t] = eps_{i,t+1}
    R: np.ndarray          # (N, T)   -- simulated observed returns, R[i, t] = r_{i,t+1}
    meta: dict = field(default_factory=dict)  # bookkeeping, e.g. which sampler/seed generated this run


def _random_pd_matrix(rng: np.random.Generator, dim: int, df_extra: int = 5,
                      scale: float = 1.0) -> np.ndarray:
    """
    Build a random positive-definite matrix via a Wishart draw, used only to
    fabricate plausible ground-truth Delta_b / Sigma when the caller doesn't
    supply their own. df_extra controls how well-conditioned the draw is
    (higher = closer to scale * I).
    """
    df = dim + df_extra
    A = rng.standard_normal((df, dim)) * np.sqrt(scale / df)
    return A.T @ A + 1e-6 * np.eye(dim)  # small jitter for numerical safety

def draw_deviations_gaussian(rng: np.random.Generator, N: int, K: int,
                              Delta_b: np.ndarray) -> np.ndarray:
    """
    Default deviation sampler: theta_i ~ N(0, Delta_b), iid across assets.
    This is the Gaussian-baseline generative process, and reproduces
    Feng & He's b_i ~ N(b_bar, Delta_b) once theta_i is added to b_bar.
    """
    return rng.multivariate_normal(mean=np.zeros(K), cov=Delta_b, size=N)

def draw_deviations_sparse(rng: np.random.Generator, N: int, K: int,
                            Delta_b: np.ndarray, sparsity: float = 0.7,
                            active_scale: float = 1.0) -> np.ndarray:
    """
    Alternative deviation sampler for later Horseshoe validation: each entry
    theta_ij is exactly zero with probability `sparsity`, and otherwise drawn
    from N(0, active_scale^2 * Delta_b_jj), i.e. genuinely sparse deviations
    rather than merely small ones. Kept in this module (rather than a
    Horseshoe-specific file) so `simulate_sur_data` can dispatch to it with
    no other code changes.
    """
    diag_scales = np.sqrt(np.diag(Delta_b)) * active_scale
    mask = rng.random((N, K)) > sparsity
    theta = rng.standard_normal((N, K)) * diag_scales[np.newaxis, :]
    return theta * mask

def generate_predictors(rng: np.random.Generator, N: int, T: int, K: int) -> np.ndarray:
    """
    Generate predictor data f_{i,t}. Random noise for now (mechanics-testing
    stage) -- shape (N, T, K). Column 0 is left as a genuine intercept
    (all ones) since every model formulation includes one.
    """
    F = rng.standard_normal((N, T, K))
    if K > 0:
        F[:, :, 0] = 1.0
    return F


def generate_residuals(rng: np.random.Generator, N: int, T: int,
                        Sigma: np.ndarray) -> np.ndarray:
    """
    Draw residuals E ~ N(0, Sigma (x) I_N): for each time period t, draw an
    N-length vector jointly from N(0, Sigma), giving residuals correlated
    across assets within a period and independent across time.

    Returns E with shape (N, T), i.e. E[:, t] ~ N(0, Sigma).
    """
    # rng.multivariate_normal(size=T) draws T iid vectors of length N,
    # each ~ N(0, Sigma) -- exactly the Omega = Sigma (x) I_N structure.
    E_T_by_N = rng.multivariate_normal(mean=np.zeros(N), cov=Sigma, size=T)  # (T, N)
    return E_T_by_N.T  # (N, T)

def compute_returns(F: np.ndarray, b: np.ndarray, E: np.ndarray) -> np.ndarray:
    """
    Compute observed excess returns r_{i,t+1} = f_{i,t}' b_i + eps_{i,t+1}.

    F : (N, T, K), b : (N, K), E : (N, T) -> R : (N, T)
    """
    # einsum contracts the K dimension per asset: R[i, t] = sum_k F[i,t,k]*b[i,k]
    signal = np.einsum("itk,ik->it", F, b)
    return signal + E


def simulate_sur_data(
    N: int,
    K: int,
    T: int,
    b_bar: Optional[np.ndarray] = None,
    Delta_b: Optional[np.ndarray] = None,
    Sigma: Optional[np.ndarray] = None,
    deviation_sampler: DeviationSampler = draw_deviations_gaussian,
    deviation_sampler_kwargs: Optional[dict] = None,
    seed: Optional[int] = None,
) -> SimulatedSURData:
    """
    Full generative process for the hierarchical SUR model.

    Parameters
    ----------
    N, K, T : dimensions (assets, predictors, time periods)
    b_bar   : (K,) true pooled coefficient vector. Randomly generated if None.
    Delta_b : (K,K) true deviation covariance. Randomly generated (PD) if None.
    Sigma   : (N,N) true residual covariance across assets. Randomly generated
              (PD) if None.
    deviation_sampler : callable(rng, N, K, Delta_b, **kwargs) -> (N,K) theta.
              Defaults to the Gaussian baseline; swap in
              `draw_deviations_sparse` (or a custom callable) to validate
              other priors (e.g. Horseshoe) against known-sparse ground truth.
    deviation_sampler_kwargs : extra kwargs forwarded to deviation_sampler.
    seed : for reproducibility.

    Returns
    -------
    SimulatedSURData with the ground truth and the simulated (F, R).
    """
    rng = np.random.default_rng(seed)
    deviation_sampler_kwargs = deviation_sampler_kwargs or {}

    if b_bar is None:
        b_bar = rng.standard_normal(K)
    if Delta_b is None:
        Delta_b = _random_pd_matrix(rng, K, df_extra=5, scale=0.5)
    if Sigma is None:
        Sigma = _random_pd_matrix(rng, N, df_extra=5, scale=1.0)

    theta = deviation_sampler(rng, N, K, Delta_b, **deviation_sampler_kwargs)
    b = b_bar[np.newaxis, :] + theta

    F = generate_predictors(rng, N, T, K)
    E = generate_residuals(rng, N, T, Sigma)
    R = compute_returns(F, b, E)

    return SimulatedSURData(
        N=N, K=K, T=T,
        b_bar=b_bar, Delta_b=Delta_b, Sigma=Sigma,
        theta=theta, b=b, F=F, E=E, R=R,
        meta={"deviation_sampler": deviation_sampler.__name__, "seed": seed},
    )

if __name__ == "__main__":

    from src.simulate.generate import simulate_sur_data

    data1 = simulate_sur_data(N=5, K=5, T=100, seed=42)
    data2 = simulate_sur_data(N=5, K=5, T=100, seed=42)

    print(np.allclose(data1.R, data2.R))       # expect True -- same seed, same data
    print(np.allclose(data1.Sigma, data2.Sigma))  # expect True

    data3 = simulate_sur_data(N=5, K=5, T=100, seed=1)
    print(np.allclose(data1.R, data3.R))       # expect False -- different seed
