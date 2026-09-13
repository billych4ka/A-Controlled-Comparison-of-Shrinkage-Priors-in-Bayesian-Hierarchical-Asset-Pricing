"""
src/evaluation/predictive_intervals.py

Out-of-sample predictive-interval coverage for the four models.

WHY A SEPARATE MODULE. expanding_window in src/evaluation/backtest.py is
shared by all four models, has been verified end to end by
assert_no_lookahead, and produced every number in Chapter 5. Modifying its
fit_fn contract this late would put those results at risk for no benefit. The
loop below therefore duplicates its windowing and alignment EXACTLY (same
start, same refit grid, same F[:,t,:] convention, same expanding training
slice) and adds only the interval computation. The duplication is
deliberate: the two functions must agree on point forecasts by construction.
No automated test enforces that agreement; it rests on the line-by-line
correspondence of the two loops.

WHAT IS COMPUTED, and why both.

  COEFFICIENT-ONLY intervals use only parameter uncertainty, quantiles of
  f'b^(s) across posterior draws s. This is the construction in Feng and He's
  Eq. (21), which computes the covariance of PREDICTED returns across draws.

  FULL PREDICTIVE intervals add the residual shock, quantiles of
  f'b^(s) + e^(s) with e^(s) ~ N(0, Sigma_ii^(s)) drawn per posterior draw.
  This is the predictive distribution of the RETURN rather than of its
  conditional mean.

The two differ by roughly an order of magnitude here. The median posterior
standard deviation of a single coefficient is 4.28e-04, so coefficient
uncertainty contributes on the order of 5e-03 to a forecast, against a
residual standard deviation of 5.57e-02, about eleven times larger. A
coefficient-only 95% interval should therefore cover far less than 95% of
realised returns. Reporting both quantifies how much of predictive
uncertainty is estimation risk in this setting, which is the same finding as
the loss attribution of Section 5.4 expressed in a different currency.

MONTE CARLO PRECISION OF THE ENDPOINTS. Section 4.4 notes that an interval
endpoint needs roughly seven times the draws of a posterior mean for equal
precision, which is why the backtest budgets are sized for means. That
argument does not bind here. At ESS 500 the MCSE of a 2.5% quantile is
2.675/sqrt(ESS) = 0.120 posterior standard deviations, or 6.1% of the
interval half-width. Averaged over 479 months and 25 assets the resulting
noise is symmetric and does not shift an aggregate coverage rate. The
existing backtest budgets are therefore retained unchanged, so the intervals
are computed from exactly the fits that produced the Chapter 5 forecasts.

SCOPE. Primary universe only. The exercise requires re-running the backtests
with draws retained, roughly 33 hours across the four models, and repeating
it on the robustness universes would add a replication rather than a finding.
Unlike the Sharpe comparison, where the claim is that a direction repeats
across sorts, coverage is a property of the intervals themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Callable

import numpy as np

IntervalFitFunction = Callable[[np.ndarray, np.ndarray], tuple]

LEVELS = (0.025, 0.975)


@dataclass
class IntervalResult:
    """
    predicted        : (N, T_oos) posterior-mean forecasts, for cross-checking
                       against the Chapter 5 backtest
    realised         : (N, T_oos)
    lo_coef, hi_coef : (N, T_oos) coefficient-uncertainty-only interval
    lo_full, hi_full : (N, T_oos) full predictive interval
    oos_index        : (T_oos,)
    refit_index      : (n_fits,)
    """

    predicted: np.ndarray
    realised: np.ndarray
    lo_coef: np.ndarray
    hi_coef: np.ndarray
    lo_full: np.ndarray
    hi_full: np.ndarray
    oos_index: np.ndarray
    refit_index: np.ndarray
    meta: dict = field(default_factory=dict)

    def coverage(self) -> dict:
        """Empirical coverage and mean width of both constructions."""
        cov_c = (self.realised >= self.lo_coef) & (self.realised <= self.hi_coef)
        cov_f = (self.realised >= self.lo_full) & (self.realised <= self.hi_full)
        n = cov_c.size
        return {
            "n_asset_months": int(n),
            "coverage_coef_only": float(cov_c.mean()),
            "coverage_full": float(cov_f.mean()),
            "binomial_se": float(np.sqrt(0.95 * 0.05 / n)),
            "mean_width_coef_only": float((self.hi_coef - self.lo_coef).mean()),
            "mean_width_full": float((self.hi_full - self.lo_full).mean()),
            "width_ratio": float((self.hi_full - self.lo_full).mean()
                                 / (self.hi_coef - self.lo_coef).mean()),
        }


def expanding_window_intervals(fit_fn: IntervalFitFunction,
                               R: np.ndarray, F: np.ndarray,
                               start: int = 240, refit_every: int = 12,
                               seed: int = 0,
                               progress: bool = True) -> IntervalResult:
    """
    Mirror of expanding_window that additionally records predictive intervals.

    The windowing is IDENTICAL by construction: same refit grid, same
    expanding training slice R[:, :t_refit], same forecast convention
    F[:, t, :] for month t with no further lagging. Any divergence would make
    the intervals describe different fits from the Chapter 5 point forecasts.
    This is maintained by construction, not by an automated check; see the
    module docstring.

    seed : for the residual draws entering the full predictive interval. Fixed
           so the result is reproducible; it does not affect the fits.
    """
    N, T = R.shape
    if F.shape[:2] != (N, T):
        raise ValueError(f"F {F.shape} inconsistent with R {R.shape}")
    if not 0 < start < T:
        raise ValueError(f"start={start} outside 0..{T}")

    rng = np.random.default_rng(seed)
    refit_points = list(range(start, T, refit_every))
    n_oos = T - start
    predicted = np.empty((N, n_oos))
    lo_c = np.empty((N, n_oos)); hi_c = np.empty((N, n_oos))
    lo_f = np.empty((N, n_oos)); hi_f = np.empty((N, n_oos))

    t0 = time()
    for k, t_refit in enumerate(refit_points):
        B_draws, sig2_draws = fit_fn(R[:, :t_refit], F[:, :t_refit, :])
        B_draws = np.asarray(B_draws)
        sig2_draws = np.asarray(sig2_draws)
        if B_draws.ndim != 3 or B_draws.shape[1:] != (N, F.shape[2]):
            raise ValueError(f"B_draws has shape {B_draws.shape}, expected "
                             f"(n_draws, {N}, {F.shape[2]})")
        if sig2_draws.shape != (B_draws.shape[0], N):
            raise ValueError(f"sigma2_draws has shape {sig2_draws.shape}, "
                             f"expected {(B_draws.shape[0], N)}")

        b_hat = B_draws.mean(axis=0)
        t_end = min(t_refit + refit_every, T)

        for t in range(t_refit, t_end):
            f_t = F[:, t, :]
            mu = np.einsum("ik,sik->si", f_t, B_draws)
            col = t - start
            predicted[:, col] = f_t @ b_hat if b_hat.ndim == 1 else \
                np.einsum("ik,ik->i", f_t, b_hat)
            lo_c[:, col], hi_c[:, col] = np.quantile(mu, LEVELS, axis=0)

            eps = rng.standard_normal(mu.shape) * np.sqrt(sig2_draws)
            lo_f[:, col], hi_f[:, col] = np.quantile(mu + eps, LEVELS, axis=0)

        if progress:
            done, total = k + 1, len(refit_points)
            el = time() - t0
            print(f"  refit {done}/{total} at t={t_refit} "
                  f"({el:.0f}s elapsed, {el/done*(total-done):.0f}s remaining)",
                  flush=True)

    return IntervalResult(
        predicted=predicted, realised=R[:, start:],
        lo_coef=lo_c, hi_coef=hi_c, lo_full=lo_f, hi_full=hi_f,
        oos_index=np.arange(start, T),
        refit_index=np.array(refit_points),
        meta={"start": start, "refit_every": refit_every,
              "n_refits": len(refit_points), "n_oos_months": n_oos,
              "levels": LEVELS, "seed": seed, "seconds": time() - t0},
    )
