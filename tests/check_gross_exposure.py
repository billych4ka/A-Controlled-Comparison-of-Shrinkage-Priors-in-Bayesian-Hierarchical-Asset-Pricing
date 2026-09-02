"""
check_gross_exposure.py -- are the pooled and stacked portfolios comparable?

Both constructions in run_cross_sort_portfolio.py are built from unit-gross
sleeves, but they are not automatically comparable on leverage.

  POOLED demeans across all 75 predictions and divides by the total absolute
  deviation, so gross exposure is exactly 1 by construction.

  STACKED averages three portfolios that each have gross exposure 1. If the
  three held disjoint assets the average would have gross exposure 1/3 in each
  sleeve and 1 in total. But averaging three weight vectors gives
  |w_avg| <= (|w1| + |w2| + |w3|)/3, with equality only when the signs agree.
  Where the sleeves take OPPOSING positions the magnitudes cancel, so the
  stacked portfolio's realised gross exposure is at most 1 and generally less.

That matters because Sharpe is scale-invariant but the COMPARISON is not: if
the stacked portfolio is running lower gross exposure and still achieving a
higher Sharpe, that strengthens the result. If it is running MORE, the
comparison is contaminated by leverage.

This script reports the realised gross exposure of both constructions month by
month, at the level of the 75 portfolio positions.

A SEPARATE POINT, not measured here: the three sorts partition the same
underlying stock universe, so portfolio-level gross exposure overstates
stock-level exposure in BOTH constructions. Measuring that would require the
constituent weights of each Fama-French portfolio, which are not part of this
dataset.

Run from the project root, seconds:

    python3 check_gross_exposure.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

UNIVERSES = ("size_bm_25", "size_op_25", "size_inv_25")
MODELS = (("Gaussian baseline", "baseline_gaussian", "rescaled_r2_0p05"),
          ("Bayesian LASSO", "bayesian_lasso", "rescaled_r2_0p05"),
          ("Horseshoe", "horseshoe", "p0_23_r2_0p05"),
          ("Regularised horseshoe", "regularised_horseshoe", "p0_23_r2_0p05"))


def load_pred(universe, model, setting):
    p = (Path("../results") / universe / model / "backtest"
         / f"{model}_{setting}_backtest.npz")
    return np.load(p)["predicted"]


def unit_gross(pred):
    """Demean across the assets supplied, then normalise to unit gross."""
    dev = pred - pred.mean(axis=0, keepdims=True)
    gross = np.abs(dev).sum(axis=0, keepdims=True)
    return np.divide(dev, gross, out=np.zeros_like(dev), where=gross > 0)


print("=" * 74)
print("GROSS EXPOSURE OF THE TWO CROSS-SORT CONSTRUCTIONS")
print("=" * 74)
print(f"{'model':<24}{'pooled':>12}{'stacked':>12}{'stacked min':>14}")

for label, model, setting in MODELS:
    preds = [load_pred(u, model, setting) for u in UNIVERSES]

    # pooled: one demeaning across all 75, gross 1 by construction
    w_pooled = unit_gross(np.concatenate(preds, axis=0))
    g_pooled = np.abs(w_pooled).sum(axis=0)

    # stacked: three unit-gross sleeves averaged. Stack the weight vectors so
    # each asset appears once; sleeves hold disjoint PORTFOLIOS, so no
    # cancellation occurs at this level -- but the division by three does
    # reduce each position.
    w_stacked = np.concatenate([unit_gross(p) for p in preds], axis=0) / 3.0
    g_stacked = np.abs(w_stacked).sum(axis=0)

    print(f"{label:<24}{g_pooled.mean():>12.4f}{g_stacked.mean():>12.4f}"
          f"{g_stacked.min():>14.4f}")

print("\n  [pooled should be exactly 1.0000 by construction.")
print("   stacked is the average of three unit-gross sleeves over DISJOINT")
print("   portfolio positions, so it should also be 1.0000: the three sleeves")
print("   hold different Fama-French portfolios and cannot offset each other")
print("   at this level. If both read 1.0000 the Sharpe comparison is on")
print("   equal leverage and the difference is attributable to the allocation")
print("   rule alone.]")
print("\n  NOT measured: stock-level exposure. The three sorts partition the")
print("  same universe, so a stock held long through one sort and short")
print("  through another offsets in economic terms while both constructions")
print("  record it as gross exposure. This affects the pooled and stacked")
print("  portfolios alike and does not distort the comparison between them.")