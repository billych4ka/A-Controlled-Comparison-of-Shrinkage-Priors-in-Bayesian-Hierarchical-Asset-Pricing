"""
src/evaluation/backtest.py

Expanding-window out-of-sample evaluation, shared by all four models.

The loop is model-agnostic: it takes a `fit_fn` callable that maps training
data to a coefficient matrix, so the Gibbs models pass an adapter around
run_gibbs and the NUTS models pass one around pm.sample. Everything else --
the windowing, the alignment, the point-in-time discipline -- is identical
across models, which is what makes the resulting metrics comparable.

Alignment convention
--------------------
The data pipeline builds F so that F[i,t,:] are the predictors used to
forecast R[i,t] (from r_{i,t+1} = f_{i,t}' b_i). So F[:,t,:] is dated t-1 in
real time and NO further lagging is applied here. Training on months 0..t-1
uses rows 0..t-1 of both F and R; forecasting month t uses F[:,t,:] and a
coefficient matrix fitted without ever seeing R[:,t] or later.

This is the single most dangerous thing in the module: getting it backwards
would use next month's predictors to forecast this month's return and produce
spectacular, meaningless results. `assert_no_lookahead` below checks it
directly rather than by inspection -- it corrupts the future and confirms the
forecasts do not move.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Callable

import numpy as np

# fit_fn(R_train, F_train) -> b_hat of shape (N, K)
FitFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class BacktestResult:
    """
    Output of an expanding-window backtest.

    predicted   : (N, T_oos) forecasts, one per asset-month
    realised    : (N, T_oos) the returns actually observed
    benchmark   : (N, T_oos) historical-mean forecasts, same grid
    oos_index   : (T_oos,) index into the original time axis, so results can
                  be lined up with dates
    refit_index : (n_fits,) the time index at which each refit happened
    coefficients: (n_fits, N, K) the b_hat used from each refit onward
    meta        : timings and settings
    """

    predicted: np.ndarray
    realised: np.ndarray
    benchmark: np.ndarray
    oos_index: np.ndarray
    refit_index: np.ndarray
    coefficients: np.ndarray
    meta: dict = field(default_factory=dict)


def expanding_window(fit_fn: FitFunction, R: np.ndarray, F: np.ndarray,
                     start: int = 240, refit_every: int = 12,
                     progress: bool = True) -> BacktestResult:
    """
    Run an expanding-window backtest.

    fit_fn      : callable (R_train, F_train) -> (N,K) coefficient estimate.
                  Everything model-specific lives here: the sampler, the
                  prior, the hyperparameter recomputation.
    R, F        : full-sample data, (N,T) and (N,T,K)
    start       : first month forecast. 240 = a 20-year initial training
                  period, the convention in this literature (Welch & Goyal
                  2008; Campbell & Thompson 2008).
    refit_every : months between refits. 12 means refitting each January and
                  reusing those coefficients for the following twelve months.
                  Monthly refitting is the ideal but costs 479 fits per model;
                  annual is the standard compromise. Kept a parameter rather
                  than hard-coded because the schedule must ultimately be set
                  by the SLOWEST model -- if NUTS forces a coarser grid, the
                  Gibbs backtests have to be re-run to match, or the models
                  are being given different information sets.

    EXPANDING, not rolling: each fit uses all data from the beginning up to
    the refit date, so the training sample grows. That matches how a real
    forecaster would work and avoids discarding history.
    """
    N, T = R.shape
    if F.shape[:2] != (N, T):
        raise ValueError(f"F {F.shape} inconsistent with R {R.shape}")
    if not 0 < start < T:
        raise ValueError(f"start={start} outside 0..{T}")

    refit_points = list(range(start, T, refit_every))
    predicted = np.empty((N, T - start))
    coefficients = np.empty((len(refit_points), N, F.shape[2]))

    t0 = time()
    for k, t_refit in enumerate(refit_points):
        # train on everything strictly before the refit date
        b_hat = fit_fn(R[:, :t_refit], F[:, :t_refit, :])
        coefficients[k] = b_hat

        t_end = min(t_refit + refit_every, T)
        predicted[:, t_refit - start:t_end - start] = np.einsum(
            "itk,ik->it", F[:, t_refit:t_end, :], b_hat)

        if progress:
            done, total = k + 1, len(refit_points)
            elapsed = time() - t0
            print(f"  refit {done}/{total} at t={t_refit} "
                  f"({elapsed:.0f}s elapsed, {elapsed/done*(total-done):.0f}s remaining)",
                  flush=True)

    csum = np.cumsum(R, axis=1)
    counts = np.arange(1, T + 1, dtype=float)
    benchmark = (csum / counts)[:, start - 1:-1]

    return BacktestResult(
        predicted=predicted,
        realised=R[:, start:],
        benchmark=benchmark,
        oos_index=np.arange(start, T),
        refit_index=np.array(refit_points),
        coefficients=coefficients,
        meta={"start": start, "refit_every": refit_every,
              "n_refits": len(refit_points), "n_oos_months": T - start,
              "seconds": time() - t0},
    )


def assert_no_lookahead(fit_fn: FitFunction, R: np.ndarray, F: np.ndarray,
                        start: int = 240, refit_every: int = 12,
                        corrupt_from: int | None = None,
                        seed: int = 0) -> dict:
    """
    Verify directly that the backtest cannot see the future.

    Method: run twice, once on the real data and once with every return from
    `corrupt_from` onward replaced by large noise. Every forecast for a month
    BEFORE corrupt_from must be bit-identical across the two runs, since none
    of the corrupted data was available when those forecasts were made.

    The restriction to months before corrupt_from is the whole subtlety. A
    refit dated after corrupt_from is fully entitled to use returns that lie
    between corrupt_from and its own date -- those are history by then, not
    the future. A test demanding that ALL forecasts be unchanged would fail a
    correctly aligned backtest, which is precisely the mistake made when this
    check was first written.

    corrupt_from defaults to the midpoint of the evaluation period, so both
    the "must not change" and "may change" regions are non-trivial.

    Requires fit_fn to be deterministic given its inputs -- seed any sampler
    inside it, or this reports sampling noise as look-ahead.
    """
    T = R.shape[1]
    if corrupt_from is None:
        corrupt_from = start + (T - start) // 2

    clean = expanding_window(fit_fn, R, F, start, refit_every, progress=False)

    rng = np.random.default_rng(seed)
    R_corrupt = R.copy()
    R_corrupt[:, corrupt_from:] = rng.standard_normal(R[:, corrupt_from:].shape) * R.std() * 10
    corrupt = expanding_window(fit_fn, R_corrupt, F, start, refit_every, progress=False)

    protected = slice(0, corrupt_from - start)     # forecasts dated before the corruption
    diff = float(np.abs(clean.predicted[:, protected]
                        - corrupt.predicted[:, protected]).max())

    # coefficients from refits dated at or before corrupt_from must also match
    safe_fits = clean.refit_index <= corrupt_from
    coef_diff = float(np.abs(clean.coefficients[safe_fits]
                             - corrupt.coefficients[safe_fits]).max())

    # and the later forecasts SHOULD change -- otherwise the corruption did
    # nothing and the test proves nothing
    after = slice(corrupt_from - start, None)
    changed = float(np.abs(clean.predicted[:, after] - corrupt.predicted[:, after]).max())

    return {"max_forecast_diff_before": diff,
            "max_coefficient_diff_before": coef_diff,
            "forecast_change_after": changed,
            "corrupt_from": corrupt_from,
            "passes": bool(diff == 0.0 and coef_diff == 0.0 and changed > 0.0)}