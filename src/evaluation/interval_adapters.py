"""
src/evaluation/interval_adapters.py

fit_fn variants that return posterior DRAWS rather than a posterior mean, for
use with expanding_window_intervals.

These deliberately mirror the four production adapters (same priors, same
point-in-time hyperparameter recomputation, same seeds, same sampling budgets)
so that the intervals describe exactly the fits that produced the Chapter 5
point forecasts. The budgets are NOT enlarged; see the module docstring of
predictive_intervals.py for why the Section 4.4 argument about interval
endpoints does not bind on an aggregate coverage rate.

Kept in a separate module rather than added to the two backtest_adapter files
because those are imported by the production runners and have been used for
every result in the dissertation. Nothing here modifies them.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.gibbs.baseline_gaussian import (default_hyperparameters,
                                         rescaled_hyperparameters, run_gibbs)
from src.gibbs.bayesian_lasso import lasso_hyperparameters
from src.gibbs.bayesian_lasso import run_gibbs as lasso_run_gibbs
from src.nuts.horseshoe import horseshoe_hyperparameters
from src.nuts.horseshoe import run_nuts as run_nuts_hs
from src.nuts.regularised_horseshoe import reg_horseshoe_hyperparameters
from src.nuts.regularised_horseshoe import run_nuts as run_nuts_rhs


def _diag(Sigma_draws: np.ndarray) -> np.ndarray:
    """(n, N, N) -> (n, N), the per-asset residual variances.

    Only the diagonal is needed: the intervals are marginal per asset-month
    rather than joint across assets, so the cross-asset covariance does not
    enter."""
    return np.einsum("sii->si", np.asarray(Sigma_draws))


def gaussian_interval_fit_fn(prior: str = "rescaled", target_r2: float = 0.05,
                             n_draws: int = 500, n_burn: int = 100,
                             seed: int = 0, blocked: bool = True) -> Callable:
    """Mirrors gaussian_fit_fn; returns (B_draws, sigma2_draws)."""
    if prior not in ("feng_he", "rescaled"):
        raise ValueError(f"prior must be 'feng_he' or 'rescaled', got {prior!r}")

    def fit(R_train, F_train):
        K = F_train.shape[2]
        hp = (rescaled_hyperparameters(R_train, F_train, K, target_r2=target_r2)
              if prior == "rescaled" else default_hyperparameters(R_train, K))
        d = run_gibbs(R_train, F_train, hp, n_draws=n_draws, n_burn=n_burn,
                      seed=seed, blocked=blocked, n_track_offdiag=0,
                      progress_every=None)
        return d.B, _diag(d.Sigma)

    fit.settings = {"model": "baseline_gaussian", "prior": prior,
                    "target_r2": target_r2, "n_draws": n_draws,
                    "n_burn": n_burn, "seed": seed}
    return fit


def lasso_interval_fit_fn(target_r2: float = 0.05, n_draws: int = 1200,
                          n_burn: int = 700, seed: int = 0) -> Callable:
    """Mirrors lasso_fit_fn; returns (B_draws, sigma2_draws)."""

    def fit(R_train, F_train):
        K = F_train.shape[2]
        hp = lasso_hyperparameters(R_train, F_train, K, target_r2=target_r2)
        d = lasso_run_gibbs(R_train, F_train, hp, n_draws=n_draws,
                            n_burn=n_burn, seed=seed, progress_every=None)
        return d.B, _diag(d.Sigma)

    fit.settings = {"model": "bayesian_lasso", "target_r2": target_r2,
                    "n_draws": n_draws, "n_burn": n_burn, "seed": seed}
    return fit


def horseshoe_interval_fit_fn(p0: int = 23, target_r2: float = 0.05,
                              n_choice: str = "T", n_draws: int = 750,
                              n_tune: int = 500, seed: int = 0) -> Callable:
    """Mirrors horseshoe_fit_fn; returns (B_draws, sigma2_draws)."""

    def fit(R_train, F_train):
        K = F_train.shape[2]
        hp = horseshoe_hyperparameters(R_train, F_train, K, p0=p0,
                                       target_r2=target_r2, n_choice=n_choice)
        ch = run_nuts_hs(R_train, F_train, hp, n_draws=n_draws, n_tune=n_tune,
                         chains=1, cores=1, seed0=seed, progressbar=False)
        return ch[0].B, _diag(ch[0].Sigma)

    fit.settings = {"model": "horseshoe", "p0": p0, "target_r2": target_r2,
                    "n_draws": n_draws, "n_tune": n_tune, "seed": seed}
    return fit


def reg_horseshoe_interval_fit_fn(p0: int = 23, target_r2: float = 0.05,
                                  nu: float = 4.0, n_choice: str = "T",
                                  slab_scale: float | None = None,
                                  n_draws: int = 500, n_tune: int = 500,
                                  seed: int = 0) -> Callable:
    """Mirrors reg_horseshoe_fit_fn; returns (B_draws, sigma2_draws).

    Note the budget is 500/500, not the horseshoe's 750/500: each was
    validated against its own posterior rather than inherited."""

    def fit(R_train, F_train):
        K = F_train.shape[2]
        hp = reg_horseshoe_hyperparameters(R_train, F_train, K, p0=p0,
                                           target_r2=target_r2, nu=nu,
                                           n_choice=n_choice,
                                           slab_scale=slab_scale)
        ch = run_nuts_rhs(R_train, F_train, hp, n_draws=n_draws,
                          n_tune=n_tune, chains=1, cores=1, seed0=seed,
                          progressbar=False)
        return ch[0].B, _diag(ch[0].Sigma)

    fit.settings = {"model": "regularised_horseshoe", "p0": p0,
                    "target_r2": target_r2, "nu": nu, "n_draws": n_draws,
                    "n_tune": n_tune, "seed": seed}
    return fit

