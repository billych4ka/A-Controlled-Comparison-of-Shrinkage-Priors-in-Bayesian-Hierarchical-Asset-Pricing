"""
src/nuts/backtest_adapter.py

Adapters that turn the NUTS models into the `fit_fn` callable that
src/evaluation/backtest.py's expanding_window expects.

Mirrors src/gibbs/backtest_adapter.py exactly in role and interface. The
evaluation code stays model-agnostic; everything model-specific -- which
sampler, which prior, how hyperparameters are chosen -- lives here, and
expanding_window sees only

    fit_fn(R_train, F_train) -> b_hat  of shape (N, K)

One adapter module per SAMPLER FAMILY, not per model: the regularised
horseshoe's fit_fn belongs alongside this one, not in a third file.

Point-in-time hyperparameters
-----------------------------
Recomputed inside every window from training data only, as for the Gibbs
models. For the horseshoe that is four quantities: V_Sigma = S_hat, Var(r)
and trace(C) (which set Delta_b_bar through the shared rescaling), and
sigma_pooled, which enters tau_0 directly. Using full-sample values would be
look-ahead -- mild, since second moments move slowly, but real and free to
avoid.

tau_0 therefore differs slightly window to window, so "the horseshoe" is a
PROCEDURE rather than one fixed specification -- the same statement already
required for the baseline's rescaled settings and the LASSO. p_0 = 23 and
target_r2 = 0.05 are fixed; what moves is the data-dependent scale they are
applied to.

Sampling budget
---------------
Backtest fits use far fewer draws than the production runs because the
backtest uses only the POSTERIOR MEAN of b -- it never reports a credible
interval. An interval endpoint needs roughly 7x the draws of a mean for equal
precision, so the production budget is sized for intervals and this one is
not.

The production run's own diagnostics say this can be aggressive. On the full
sample, B reached bulk ESS with a median of 5,591 of 6,000 across all 3,600
coefficients -- close to superefficient, as NUTS often is. B is the estimand
here; tau's much lower ESS (1,012) and Sigma's (158) set the PRODUCTION
budget and are irrelevant to a posterior mean of B.

WHAT THE TUNING PHASE HAS TO ACHIEVE, and why it cannot be cut to nothing.
tau starts at tau_0 and the data moves it to roughly 0.05 tau_0 -- a factor
of twenty. Retaining draws before tau has travelled would sample B under the
wrong shrinkage level: a systematic error repeated in every window, not a
Monte Carlo one. This is the same concern the LASSO's adapter records for
lambda, and it is why n_tune is not simply set to a token value.
Reassuringly, a 200-tune pilot on the full sample already had tau at
2.06e-05, so the journey is fast -- but early windows are the risk, and
validate_budget_horseshoe checks a window rather than assuming.

Early windows are thin: at t=240 with K=144 there are 96 residual degrees of
freedom against 575 at the end, so sigma_pooled is noisier, the likelihood is
less informative, and the prior has more say. That is also the direction that
makes the geometry harder, so per-fit cost may not fall as fast as the window
length suggests. Time the first window before committing to a full run.

cores=1 and chains=1 throughout. PyMC's multiprocessing killed a worker twice
in this project (an EOFError with no traceback at reduced dimensions, and a
production run that lost a process after an hour); every single-core run has
completed. Backtest fits are sequential by design, so the parallel path is
never needed here.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.nuts.horseshoe import horseshoe_hyperparameters, run_nuts

from src.nuts.regularised_horseshoe import (reg_horseshoe_hyperparameters,
                                            run_nuts as run_nuts_reg)


def horseshoe_fit_fn(p0: int = 23, target_r2: float = 0.05,
                     n_choice: str = "T", n_draws: int = 500,
                     n_tune: int = 500, seed: int = 0) -> Callable:
    """
    Build a fit_fn for the Horseshoe.

    p0        : 23, fixed a priori; NOT to be revised after seeing results
    target_r2 : 0.05, sets Delta_b_bar, shared with all four models
    n_choice  : "T" (adopted) or "NT" (disclosed robustness)
    n_draws   : RETAINED draws per window fit, EXCLUDING tuning -- the
                opposite of the Gibbs adapters' convention, where n_draws
                includes burn-in. PyMC's argument means retained, and
                silently redefining it would be worse than the inconsistency
    n_tune    : tuning iterations, discarded. Serves two purposes at once
                here: adapting the step size and mass matrix, AND letting tau
                travel from tau_0 to wherever the data puts it
    seed      : FIXED, so the returned callable is deterministic given its
                inputs -- required by assert_no_lookahead, which would
                otherwise report sampling noise as look-ahead. Every model's
                adapter must do the same

    Sampler settings (target_accept = 0.99, init = "adapt_diag",
    max_treedepth = 10) come from HorseshoeHyperparams, not from arguments
    here, so a backtest fit CANNOT silently differ from the production run.
    That is the uniform-setting decision enforced in code rather than by
    memory. target_accept = 0.99 was chosen by measurement: divergences fell
    16/200 -> 7/200 -> 3/200 across 0.90 / 0.95 / 0.99, and the posterior mean
    of b at 0.90 and 0.99 correlated at 0.9944 -- the same agreement the
    LASSO showed between random seeds -- so the uniform choice is
    conservative rather than necessary.

    Returns a closure suitable for expanding_window's fit_fn argument.
    """
    def fit(R_train: np.ndarray, F_train: np.ndarray) -> np.ndarray:
        K = F_train.shape[2]
        hp = horseshoe_hyperparameters(R_train, F_train, K, p0=p0,
                                       target_r2=target_r2, n_choice=n_choice)
        chains = run_nuts(R_train, F_train, hp, n_draws=n_draws,
                          n_tune=n_tune, chains=1, cores=1, seed0=seed,
                          progressbar=False)
        # posterior mean of the asset-specific coefficients, (N, K)
        return chains[0].B.mean(axis=0)

    fit.settings = {"model": "horseshoe", "p0": p0, "target_r2": target_r2,
                    "n_choice": n_choice, "n_draws": n_draws,
                    "n_tune": n_tune, "seed": seed, "target_accept": 0.99,
                    "init": "adapt_diag"}
    return fit


def validate_budget_horseshoe(R: np.ndarray, F: np.ndarray,
                              reference_B: np.ndarray, reference_ess: float,
                              p0: int = 23, target_r2: float = 0.05,
                              n_draws: int = 500, n_tune: int = 500,
                              seed: int = 0) -> dict:
    """
    Check that the reduced backtest budget recovers the same posterior mean of
    B as the full production run. Mirrors validate_budget in the Gibbs
    adapter, and must be run before a full backtest rather than after.

    reference_B   : (N,K) posterior mean of B from the production run,
                    pooled across chains
    reference_ess : a representative bulk ESS from that run. Use B's MEDIAN
                    (5,591 on size_bm_25), not its minimum: the minimum comes
                    from four stragglers of 3,600 and would understate the
                    reference's precision everywhere else

    READ THE MEDIAN z, NOT THE MAX. With 3,600 parameters, two perfectly
    agreeing estimates give a max |z| whose own median is 3.73, with a
    5th-95th range of 3.34-4.34 -- it is the maximum of 3,600 draws, so it is
    large by construction. A median near 1 and a max under about 4 is
    agreement. The baseline's accepted budget gave median 0.72 with 100%
    within 3 SE and correlation 0.9989.

    Note this validates B ONLY. tau's and Sigma's own ESS are far lower and
    are what set the production budget, but the backtest never reports an
    interval on either -- it uses only the posterior mean of B.
    """
    K = F.shape[2]
    hp = horseshoe_hyperparameters(R, F, K, p0=p0, target_r2=target_r2)
    chains = run_nuts(R, F, hp, n_draws=n_draws, n_tune=n_tune, chains=1,
                      cores=1, seed0=seed, progressbar=False)
    d = chains[0]
    B_hat = d.B.mean(axis=0)

    # combined Monte Carlo standard error of the two estimates
    ess_reduced = max(n_draws / 2.0, 1.0)      # conservative: assume ESS = n/2
    se = np.sqrt(d.B.var(axis=0, ddof=1) / ess_reduced
                 + d.B.var(axis=0, ddof=1) / max(reference_ess, 1.0))
    z = np.abs(B_hat - reference_B) / np.where(se > 0, se, np.nan)

    return {
        "median_z": float(np.nanmedian(z)),
        "max_z": float(np.nanmax(z)),
        "frac_within_3": float(np.nanmean(z < 3)),
        "correlation": float(np.corrcoef(B_hat.ravel(), reference_B.ravel())[0, 1]),
        "n_draws": n_draws, "n_tune": n_tune,
        "tau_mean": float(d.tau.mean()),
        "divergences": int(d.diverging.sum()),
        "tree_depth_mean": float(np.mean(d.tree_depth)),
        "seconds": float(d.meta.get("sampling_seconds_all_chains", np.nan)),
    }

# =========================================================================
# APPEND to src/nuts/backtest_adapter.py, below horseshoe_fit_fn and
# validate_budget_horseshoe. Add to the imports at the top of that file:
#
#     from src.nuts.regularised_horseshoe import (
#         reg_horseshoe_hyperparameters, run_nuts as run_nuts_reg)
#
# One adapter module per SAMPLER FAMILY, not per model -- the regularised
# horseshoe belongs here alongside the plain one, not in a third file.
# =========================================================================


def reg_horseshoe_fit_fn(p0: int = 23, target_r2: float = 0.05,
                         nu: float = 4.0, n_choice: str = "T",
                         slab_scale: float | None = None,
                         n_draws: int = 750, n_tune: int = 500,
                         seed: int = 0) -> Callable:
    """
    Build a fit_fn for the Regularised Horseshoe.

    Identical in role to horseshoe_fit_fn. Everything in that docstring about
    point-in-time hyperparameters applies unchanged -- Delta_b_bar, V_Sigma,
    sigma and tau_0 are recomputed inside every window from training data
    only, so "the model" is a PROCEDURE rather than one fixed specification.

    ONE ADDITION: slab_scale is also recomputed per window. It is derived from
    target_r2, Var(r) and the mean predictor variance, all of which move with
    the window, so passing a fixed value would freeze a quantity that should
    be point-in-time. Passing slab_scale explicitly (e.g. 2.0) overrides that
    and is only for the demonstration run showing P&V's illustrative default
    never binds.

    THE BUDGET IS INHERITED FROM THE PLAIN HORSESHOE'S 750/500 AND MUST BE
    VALIDATED, not assumed. Model 3's figure was validated against model 3's
    posterior, and this model's geometry is different in ways that could cut
    either way: the posterior is far better conditioned (tree depth 7 against
    9, step size 0.028 against 0.008, zero divergences against 0.9%), which
    argues for fewer draws; but c is a NEW global parameter and tau mixes
    somewhat worse (ESS 709 against 1,012), which argues for more.
    check_reg_backtest_budget.py settles it.

    WHAT THE TUNING PHASE HAS TO ACHIEVE HERE. tau starts at tau_0 and the
    data moves it to about 0.09 tau_0 -- a factor of eleven, against the plain
    horseshoe's twenty. c starts at slab_scale and the data pulls it to 0.69
    of prior E[c]. Retaining draws before either has travelled would sample B
    under the wrong shrinkage level: a systematic error repeated in all 40
    windows, not a Monte Carlo one.

    seed is FIXED so the returned callable is deterministic given its inputs,
    which assert_no_lookahead requires -- otherwise sampling noise is reported
    as look-ahead.

    Sampler settings (target_accept = 0.99, init = "adapt_diag",
    max_treedepth = 10) come from RegHorseshoeHyperparams, not from arguments
    here, so a backtest fit cannot silently differ from the production run.
    """
    def fit(R_train: np.ndarray, F_train: np.ndarray) -> np.ndarray:
        K = F_train.shape[2]
        hp = reg_horseshoe_hyperparameters(R_train, F_train, K, p0=p0,
                                           target_r2=target_r2, nu=nu,
                                           n_choice=n_choice,
                                           slab_scale=slab_scale)
        chains = run_nuts_reg(R_train, F_train, hp, n_draws=n_draws,
                              n_tune=n_tune, chains=1, cores=1, seed0=seed,
                              progressbar=False)
        return chains[0].B.mean(axis=0)

    fit.settings = {"model": "regularised_horseshoe", "p0": p0,
                    "target_r2": target_r2, "nu": nu, "n_choice": n_choice,
                    "slab_scale": slab_scale, "n_draws": n_draws,
                    "n_tune": n_tune, "seed": seed, "target_accept": 0.99,
                    "init": "adapt_diag"}
    return fit


def validate_budget_reg_horseshoe(R: np.ndarray, F: np.ndarray,
                                  reference_B: np.ndarray, reference_ess: float,
                                  p0: int = 23, target_r2: float = 0.05,
                                  n_draws: int = 750, n_tune: int = 500,
                                  seed: int = 0) -> dict:
    """
    Check that the reduced backtest budget recovers the same posterior mean of
    B as the full production run. Mirrors validate_budget_horseshoe, with two
    extra quantities returned because this model has two global parameters:
    tau AND c, and both must have travelled before retained draws are sampling
    B under the right shrinkage.

    reference_ess : use B's MEDIAN bulk ESS from the production run, not its
                    minimum -- the minimum comes from a handful of stragglers
                    and would understate the reference's precision everywhere
                    else.

    READ THE MEDIAN z, NOT THE MAX. With 3,600 parameters two perfectly
    agreeing estimates give a max |z| whose own median is 3.73. The plain
    horseshoe's accepted budget gave median 0.60, 99.8% within 3 SE,
    correlation 0.9990.
    """
    K = F.shape[2]
    hp = reg_horseshoe_hyperparameters(R, F, K, p0=p0, target_r2=target_r2)
    chains = run_nuts_reg(R, F, hp, n_draws=n_draws, n_tune=n_tune, chains=1,
                          cores=1, seed0=seed, progressbar=False)
    d = chains[0]
    B_hat = d.B.mean(axis=0)

    ess_reduced = max(n_draws / 2.0, 1.0)      # conservative: assume ESS = n/2
    se = np.sqrt(d.B.var(axis=0, ddof=1) / ess_reduced
                 + d.B.var(axis=0, ddof=1) / max(reference_ess, 1.0))
    z = np.abs(B_hat - reference_B) / np.where(se > 0, se, np.nan)
    ratio = d.lam_tilde / d.lam_local

    return {
        "median_z": float(np.nanmedian(z)),
        "max_z": float(np.nanmax(z)),
        "frac_within_3": float(np.nanmean(z < 3)),
        "correlation": float(np.corrcoef(B_hat.ravel(), reference_B.ravel())[0, 1]),
        "n_draws": n_draws, "n_tune": n_tune,
        "tau_mean": float(d.tau.mean()),
        "c_mean": float(d.c.mean()),
        "frac_binding": float((ratio < 0.99).mean()),
        "divergences": int(d.diverging.sum()),
        "tree_depth_mean": float(np.mean(d.tree_depth)),
        "seconds": float(d.meta.get("sampling_seconds_all_chains", np.nan)),
    }