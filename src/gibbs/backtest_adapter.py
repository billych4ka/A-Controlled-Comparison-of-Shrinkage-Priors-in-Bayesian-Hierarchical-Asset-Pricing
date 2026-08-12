"""
src/gibbs/backtest_adapter.py

Adapters that turn a Gibbs model into the `fit_fn` callable that
src/evaluation/backtest.py's expanding_window expects.

This file exists so that the evaluation code stays model-agnostic. Everything
model-specific -- which sampler, which prior, how hyperparameters are chosen
-- lives here, and expanding_window sees only

    fit_fn(R_train, F_train) -> b_hat  of shape (N, K)

Point-in-time hyperparameters
-----------------------------
The hyperparameters are recomputed inside every window from training data
only. Three quantities depend on data: V_Sigma = S_hat (the sample covariance
of excess returns), and, for the rescaled prior, Var(r) and trace(C), which
set the calibration scale. Using full-sample values would be look-ahead --
mild, since second moments move slowly, but real, and free to avoid.

A consequence worth stating in the write-up: under the rescaled prior the
hyperparameters differ slightly from window to window, so "the model" is a
procedure rather than one fixed specification. Under feng_he only V_Sigma
moves, since its other values are fixed constants. The four settings
therefore differ in how much of the prior is data-dependent.

Sampling budget
---------------
Backtest fits use far fewer sweeps than the full-sample production runs,
because the backtest uses only the POSTERIOR MEAN of b -- it never reports a
credible interval. An interval endpoint needs roughly 7x the draws of a mean
for equal precision (MCSE of the 2.5% quantile is 2.675/sqrt(ESS) against
1/sqrt(ESS) for the mean), so the production budget is sized for intervals
and the backtest budget is not. See validate_budget below, which checks the
reduced budget against the full one directly rather than assuming.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.gibbs.baseline_gaussian import (default_hyperparameters,
                                         rescaled_hyperparameters, run_gibbs)
from src.gibbs.bayesian_lasso import lasso_hyperparameters, pooled_residual_scale
from src.gibbs.bayesian_lasso import run_gibbs as lasso_run_gibbs


def gaussian_fit_fn(prior: str = "rescaled", target_r2: float = 0.05,
                    n_draws: int = 500, n_burn: int = 100, seed: int = 0,
                    blocked: bool = True) -> Callable:
    """
    Build a fit_fn for the Gaussian baseline.

    prior     : "feng_he" (their disclosed constants) or "rescaled"
    target_r2 : used only when prior == "rescaled"
    n_draws   : total sweeps per window fit, INCLUDING burn-in
    seed      : fixed, so the returned callable is deterministic given its
                inputs -- required by assert_no_lookahead, which would
                otherwise report sampling noise as look-ahead

    Returns a closure suitable for expanding_window's fit_fn argument.
    """
    if prior not in ("feng_he", "rescaled"):
        raise ValueError(f"prior must be 'feng_he' or 'rescaled', got {prior!r}")

    def fit(R_train: np.ndarray, F_train: np.ndarray) -> np.ndarray:
        K = F_train.shape[2]
        if prior == "rescaled":
            hp = rescaled_hyperparameters(R_train, F_train, K, target_r2=target_r2)
        else:
            hp = default_hyperparameters(R_train, K)

        draws = run_gibbs(R_train, F_train, hp, n_draws=n_draws, n_burn=n_burn,
                          seed=seed, blocked=blocked, n_track_offdiag=0,
                          progress_every=None)
        # posterior mean of the asset-specific coefficients, (N, K)
        return draws.B.mean(axis=0)

    fit.settings = {"prior": prior, "target_r2": target_r2 if prior == "rescaled" else None,
                    "n_draws": n_draws, "n_burn": n_burn, "seed": seed,
                    "blocked": blocked}
    return fit


def validate_budget(R: np.ndarray, F: np.ndarray, reference_B: np.ndarray,
                    reference_ess: float, prior: str = "rescaled",
                    target_r2: float = 0.05, n_draws: int = 500,
                    n_burn: int = 100, seed: int = 0) -> dict:
    """
    Check that the reduced backtest budget recovers the same posterior mean
    as a full production run.

    reference_B   : (N,K) posterior mean of B from the full run
    reference_ess : a representative effective sample size from that run,
                    used to put the comparison in Monte Carlo standard errors

    The comparison must be in MCSE units, not raw differences: two estimates
    of the same posterior mean differ by sampling error, and whether a gap of
    1e-5 is fine or damning depends entirely on how precise each estimate is.
    A maximum discrepancy under about 4 SE means the reduced budget is
    adequate for the backtest's purposes.
    """
    K = F.shape[2]
    if prior == "rescaled":
        hp = rescaled_hyperparameters(R, F, K, target_r2=target_r2)
    else:
        hp = default_hyperparameters(R, K)

    draws = run_gibbs(R, F, hp, n_draws=n_draws, n_burn=n_burn, seed=seed,
                      blocked=True, n_track_offdiag=0, progress_every=None)
    B_small = draws.B.mean(axis=0)
    n_kept = n_draws - n_burn

    # combined standard error of the two independent estimates
    sd = draws.B.std(axis=0)
    se = sd * np.sqrt(1.0 / n_kept + 1.0 / reference_ess)
    z = np.abs(B_small - reference_B) / np.where(se > 0, se, np.nan)

    return {"max_z": float(np.nanmax(z)),
            "median_z": float(np.nanmedian(z)),
            "frac_within_3": float(np.nanmean(z < 3)),
            "n_kept": n_kept,
            "corr_with_reference": float(np.corrcoef(B_small.ravel(),
                                                     reference_B.ravel())[0, 1])}

def lasso_fit_fn(target_r2: float = 0.05, lambda_multiplier: float = 1.0,
                 n_draws: int = 1200, n_burn: int = 700, seed: int = 0) -> Callable:
    """
    Build a fit_fn for the Bayesian LASSO.

    target_r2         : 0.05, fixed a priori. The LASSO runs ONE setting; the
                        four the baseline ran existed to test whether the
                        target_r2 arbitrariness reached performance, and this
                        model has no equivalent knob -- lambda is learned.
    lambda_multiplier : scales the Gamma hyperprior's centre. In-sample
                        sensitivity only; the backtest runs at 1.0.
    n_draws           : total sweeps per window fit, INCLUDING burn-in
    seed              : fixed, so the returned callable is deterministic given
                        its inputs -- required by assert_no_lookahead, which
                        would otherwise report sampling noise as look-ahead

    Budget
    ------
    600/200 rather than the baseline's 500/100. The extra draws are not for B,
    which mixes as well here as in the baseline (lag-1 autocorrelation 0.084,
    ESS 1271 of 1600, so ~500 sweeps clears ESS 400). They are for BURN-IN:
    lambda starts at its hyperprior centre and has to travel to wherever the
    data puts it, which on the full sample took roughly 200 sweeps. Retaining
    draws before lambda has settled would sample B under the wrong shrinkage
    level -- not a Monte Carlo error but a systematic one, repeated in every
    window.

    Early windows are the risk: at t=240 there are 96 residual degrees of
    freedom against 575 at the end, so lambda is less well identified and may
    settle more slowly. Verify with a single fit on R[:, :240] before
    committing to a full run.

    Returns a closure suitable for expanding_window's fit_fn argument.
    """
    def fit(R_train: np.ndarray, F_train: np.ndarray) -> np.ndarray:
        K = F_train.shape[2]
        hp = lasso_hyperparameters(R_train, F_train, K, target_r2=target_r2,
                                   lambda_prior_multiplier=lambda_multiplier)
        # pooled_residual_scale already solved the N least-squares problems
        # inside lasso_hyperparameters; hand the result to initialise_state
        # rather than solving them again in every one of the 40 windows
        _, B_ols = pooled_residual_scale(R_train, F_train)

        draws = lasso_run_gibbs(R_train, F_train, hp, n_draws=n_draws,
                                n_burn=n_burn, seed=seed, n_track_tau2=0,
                                B0=B_ols, progress_every=None)
        # posterior mean of the asset-specific coefficients, (N, K)
        return draws.B.mean(axis=0)

    fit.settings = {"model": "bayesian_lasso", "target_r2": target_r2,
                    "lambda_multiplier": lambda_multiplier,
                    "n_draws": n_draws, "n_burn": n_burn, "seed": seed,
                    "blocked": True}
    return fit


def validate_budget_lasso(R: np.ndarray, F: np.ndarray, reference_B: np.ndarray,
                          reference_ess: float, target_r2: float = 0.05,
                          n_draws: int = 600, n_burn: int = 200,
                          seed: int = 0) -> dict:
    """
    Check that the reduced backtest budget recovers the same posterior mean of
    B as a full production run. Mirrors validate_budget above.

    reference_B   : (N,K) posterior mean of B from the full run
    reference_ess : a representative effective sample size from that run

    Read the MEDIAN z, not the max: with 3,600 parameters, two perfectly
    agreeing estimates give a max |z| with median 3.73 and a 5th-95th range of
    3.34-4.34. A max under about 4 SE and a median near 1 is agreement.

    Note this validates B only. lambda's own ESS is far lower and is what sets
    the PRODUCTION budget, but the backtest never reports an interval on
    lambda -- it uses only the posterior mean of B.
    """
    K = F.shape[2]
    hp = lasso_hyperparameters(R, F, K, target_r2=target_r2)
    _, B_ols = pooled_residual_scale(R, F)

    draws = lasso_run_gibbs(R, F, hp, n_draws=n_draws, n_burn=n_burn, seed=seed,
                            n_track_tau2=0, B0=B_ols, progress_every=None)
    B_small = draws.B.mean(axis=0)
    n_kept = n_draws - n_burn

    sd = draws.B.std(axis=0)
    se = sd * np.sqrt(1.0 / n_kept + 1.0 / reference_ess)
    z = np.abs(B_small - reference_B) / np.where(se > 0, se, np.nan)

    return {"max_z": float(np.nanmax(z)),
            "median_z": float(np.nanmedian(z)),
            "frac_within_3": float(np.nanmean(z < 3)),
            "n_kept": n_kept,
            "lambda_mean": float(draws.lam.mean()),
            "corr_with_reference": float(np.corrcoef(B_small.ravel(),
                                                     reference_B.ravel())[0, 1])}