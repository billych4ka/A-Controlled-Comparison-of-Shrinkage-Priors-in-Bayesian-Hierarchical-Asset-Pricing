"""
run_cross_sort_portfolio.py

Pools the out-of-sample forecasts from all three portfolio universes into a
single long-short allocation, and compares it against the three within-sort
portfolios reported in Section 5.3.

WHY. The evaluation in Section 4.6 demeans predicted returns within a universe
and normalises to unit gross exposure. The strategy is therefore always fully
invested in relative bets inside one sort, and takes positions even in months
when that sort contains little useful cross-sectional information. Section 6.3
records this as a limitation. Pooling the 75 predictions each month tests
whether the horseshoe family's economic ordering survives a wider opportunity
set.

NO REFITTING. This uses the predictions already saved by the four backtest
runners: the same posterior means, the same 479 forecast months, the same
refit dates. Only the portfolio rule changes, so any difference is attributable
to the allocation and not to the models.

TWO POOLING RULES, and the distinction matters.

  WITHIN-SORT DEMEANING then combining gives three market-neutral sleeves
  stacked at equal weight. Capital cannot move between sorts, so this mostly
  averages the existing results and is reported only as a reference point.

  POOLED DEMEANING across all 75 predictions each month is the version that
  actually tests the limitation: a portfolio short of attractive positions in
  one sort can allocate to another. This is the headline construction.

WHAT THIS CANNOT SHOW. The three sorts partition the SAME underlying US stock
universe, so a given stock appears in all three. The 75 assets are therefore
far from 75 independent bets, and a pooled long-short can hold offsetting
positions: long a stock through its small/high-BM portfolio and short it
through its small/weak-OP portfolio. Realised stock-level exposure is smaller
than the portfolio-level gross exposure suggests, which is a reason the pooled
Sharpe might not improve even if the diversification argument is sound.

INTERPRETATION. The standard error of an annualised Sharpe over 479 months is
0.158. If both horseshoes again land above both Gibbs models, that is a fourth
replication of the same DIRECTION, not a detected effect.

Run from the project root, seconds:

    python3 run_cross_sort_portfolio.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.evaluation.metrics import (certainty_equivalent, out_of_sample_r2,
                                    portfolio_returns, sharpe_ratio)

UNIVERSES = ("size_bm_25", "size_op_25", "size_inv_25")
MODELS = (("Gaussian baseline", "baseline_gaussian", "rescaled_r2_0p05"),
          ("Bayesian LASSO", "bayesian_lasso", "rescaled_r2_0p05"),
          ("Horseshoe", "horseshoe", "p0_23_r2_0p05"),
          ("Regularised horseshoe", "regularised_horseshoe", "p0_23_r2_0p05"))
SHARPE_SE = 0.158


def load(universe: str, model: str, setting: str):
    p = (Path("results") / universe / model / "backtest"
         / f"{model}_{setting}_backtest.npz")
    if not p.exists():
        raise FileNotFoundError(p)
    d = np.load(p)
    return d["predicted"], d["realised"], d["benchmark"], d["oos_index"]


def pooled_weights(pred: np.ndarray) -> np.ndarray:
    """Demean across ALL assets present each month, then normalise to unit
    gross exposure. Identical in form to the within-sort rule of Section 4.6,
    applied to the pooled cross-section.

    pred is (n_assets, n_months). Returns weights of the same shape, with each
    column summing to zero and its absolute values summing to one.
    """
    dev = pred - pred.mean(axis=0, keepdims=True)
    gross = np.abs(dev).sum(axis=0, keepdims=True)
    return np.divide(dev, gross, out=np.zeros_like(dev), where=gross > 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results/cross_sort")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 82)
    print("CROSS-SORT POOLED PORTFOLIO")
    print("75 predictions pooled monthly from three universes; no refitting")
    print("=" * 82)

    results = {}
    for label, model, setting in MODELS:
        P, R, Bm, idx = [], [], [], None
        for u in UNIVERSES:
            p, r, b, oi = load(u, model, setting)
            if idx is None:
                idx = oi
            elif not np.array_equal(idx, oi):
                raise ValueError(f"{model}: forecast months differ across "
                                 f"universes; pooling would misalign them")
            P.append(p); R.append(r); Bm.append(b)

        pred = np.concatenate(P, axis=0)
        real = np.concatenate(R, axis=0)
        bench = np.concatenate(Bm, axis=0)
        n_assets, n_months = pred.shape

        w = pooled_weights(pred)
        r_pooled = (w * real).sum(axis=0)

        sleeves = [portfolio_returns(r, p) for p, r
                   in zip(P, R)]
        r_stacked = np.mean(np.vstack(sleeves), axis=0)

        r2 = out_of_sample_r2(real, pred, bench)

        results[label] = {
            "pooled_sharpe": sharpe_ratio(r_pooled),
            "pooled_ce": certainty_equivalent(r_pooled),
            "stacked_sharpe": sharpe_ratio(r_stacked),
            "stacked_ce": certainty_equivalent(r_stacked),
            "pooled_r2": r2,
            "n_assets": int(n_assets), "n_months": int(n_months),
        }
        if r2 > 0.05:
            print(f"  *** {label}: pooled OOS R^2 {r2:+.4f} exceeds 0.05. "
                  f"Assume an alignment bug until proven otherwise. ***")

    print(f"\n{'model':<24}{'BM':>10}{'OP':>10}{'Inv':>10}"
          f"{'stacked':>10}{'POOLED':>10}")
    print("  Sharpe")
    within = {}
    for label, model, setting in MODELS:
        row = []
        for u in UNIVERSES:
            p, r, _, _ = load(u, model, setting)
            row.append(sharpe_ratio(portfolio_returns(r, p)))
        within[label] = row
        print(f"  {label:<22}" + "".join(f"{v:>10.3f}" for v in row)
              + f"{results[label]['stacked_sharpe']:>10.3f}"
              + f"{results[label]['pooled_sharpe']:>10.3f}")

    print("  Certainty equivalent")
    for label, model, setting in MODELS:
        row = []
        for u in UNIVERSES:
            p, r, _, _ = load(u, model, setting)
            row.append(certainty_equivalent(portfolio_returns(r, p)))
        print(f"  {label:<22}" + "".join(f"{v:>10.4f}" for v in row)
              + f"{results[label]['stacked_ce']:>10.4f}"
              + f"{results[label]['pooled_ce']:>10.4f}")

    print("\n" + "=" * 82)
    hs = [results[l]["pooled_sharpe"] for l in
          ("Horseshoe", "Regularised horseshoe")]
    gb = [results[l]["pooled_sharpe"] for l in
          ("Gaussian baseline", "Bayesian LASSO")]
    print(f"pooled Sharpe: horseshoes {hs[0]:+.3f}, {hs[1]:+.3f}   "
          f"Gibbs models {gb[0]:+.3f}, {gb[1]:+.3f}")
    if min(hs) > max(gb):
        print("-> the family ordering survives pooling. Report as a FOURTH")
        print("   replication of the same direction, not as a detected effect:")
        print(f"   the SE of an annualised Sharpe over {n_months} months is "
              f"{SHARPE_SE:.3f}.")
    else:
        print("-> the family ordering does NOT survive pooling. That is a")
        print("   result, not a failure: it would indicate the within-sort")
        print("   advantage does not extend to a combined opportunity set.")

    best_within = {l: max(v) for l, v in within.items()}
    print("\npooled versus the best single sort, per model:")
    for label in within:
        d = results[label]["pooled_sharpe"] - best_within[label]
        print(f"   {label:<24}{d:+.3f}")
    print("   [the three sorts partition the SAME stock universe, so the 75")
    print("    assets are far from 75 independent bets and a pooled long-short")
    print("    can hold offsetting stock-level positions. A pooled Sharpe that")
    print("    does not exceed the best single sort is therefore consistent")
    print("    with the diversification argument rather than against it.]")

    with open(outdir / "cross_sort_results.json", "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\nsaved -> {outdir / 'cross_sort_results.json'}")


if __name__ == "__main__":
    main()
