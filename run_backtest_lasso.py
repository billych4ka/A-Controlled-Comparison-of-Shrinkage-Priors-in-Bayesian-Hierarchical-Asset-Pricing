"""
run_backtest_lasso.py

Expanding-window out-of-sample backtest for the Bayesian LASSO. Saves
predictions and per-window coefficients so the metrics can be recomputed later
without re-running the sampling.

ONE SETTING, where the Gaussian baseline ran four. The baseline's four existed
to test whether the arbitrariness of target_r2 reached performance -- which it
did not, though it did move which predictors looked important in sample. The
LASSO has no equivalent knob: lambda is learned from the data in every window.
Its only comparable choice is the Gamma hyperprior's centre, and that washes
out (the data contributes NK = 3,600 pseudo-observations against r = 1), so it
is checked in sample rather than backtested.

Everything else matches the baseline exactly, because it must: same evaluation
start, same refit frequency, same portfolio rule, same metrics. Refit frequency
in particular defines the information set each model is given, so it cannot
differ across models even though sweep budgets can.

Per-window budget
-----------------
1,200 sweeps with 700 burn-in, against the baseline's 500/100. The extra is
not for B, which mixes just as well here (lag-1 autocorrelation 0.084). It is
for lambda, which does not converge to a point within a 240-month window at
all: over 1,200 sweeps it wandered between roughly 455 and 740 with no trend.

That turns out not to matter, which is why the budget is 1,200 and not 3,000.
The backtest uses only the posterior MEAN of B, and B is drawn given lambda,
so averaging 500 retained draws averages over lambda's uncertainty rather than
conditioning on one value. Verified directly: three fits of the same window
under different seeds gave lambda 551, 556, 555 and posterior-mean B
correlating at 0.9955 and 0.9944, with a maximum difference of 0.29 of B's own
standard deviation.

Worth reporting: lambda settles near 553 in the first window against 419 on the
full sample. The LASSO learns a tighter shrinkage level when it has less data
and loosens as history accumulates. The baseline cannot do this -- its prior
scale is fixed by target_r2 regardless of window length.

Usage:
    python3 run_backtest_lasso.py
    python3 run_backtest_lasso.py --refit-every 60 --draws 800
    python3 run_backtest_lasso.py --skip-lookahead     # if already verified
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import time

import numpy as np

from src.evaluation.backtest import assert_no_lookahead, expanding_window
from src.evaluation.metrics import (certainty_equivalent, clark_west, diebold_mariano,
                                    out_of_sample_r2, portfolio_returns, r2_by_asset,
                                    sharpe_ratio)
from src.gibbs.backtest_adapter import lasso_fit_fn

MODEL = "bayesian_lasso"
BASELINE = "baseline_gaussian"
SETTING = "rescaled_r2_0p05"


def load_universe(universe: str):
    path = Path("data/processed") / f"{universe}_arrays.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    d = np.load(path, allow_pickle=True)
    return d["F"], d["R"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="size_bm_25")
    ap.add_argument("--start", type=int, default=240,
                    help="first month forecast; 240 = 20-year initial training window")
    ap.add_argument("--refit-every", type=int, default=12,
                    help="MUST match every other model -- it defines the information "
                         "set, not just the cost")
    ap.add_argument("--draws", type=int, default=1200,
                    help="sweeps per window fit, including burn-in")
    ap.add_argument("--burn", type=int, default=700)
    ap.add_argument("--target-r2", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-lookahead", action="store_true",
                    help="skip the look-ahead verification (it costs two extra "
                         "backtests at reduced settings)")
    args = ap.parse_args()

    F, R = load_universe(args.universe)
    N, T, K = F.shape
    outdir = Path("results") / args.universe / MODEL / "backtest"
    outdir.mkdir(parents=True, exist_ok=True)

    n_fits = len(range(args.start, T, args.refit_every))
    print(f"Bayesian LASSO backtest | {args.universe} | N={N} T={T} K={K}")
    print(f"  train from month 0, first forecast month {args.start}, "
          f"{T - args.start} out-of-sample months")
    print(f"  refit every {args.refit_every} months -> {n_fits} fits")
    print(f"  {args.draws} sweeps per fit ({args.burn} burn-in), 1 chain")
    print(f"  target_r2={args.target_r2} (fixed a priori); lambda learned per window\n")

    fit_fn = lasso_fit_fn(target_r2=args.target_r2, n_draws=args.draws,
                          n_burn=args.burn, seed=args.seed)

    # ---- look-ahead verification ------------------------------------------
    # Corrupt every return from the midpoint onward and confirm that no forecast
    # dated BEFORE the corruption moves. This is the one bug class that would
    # produce spectacular, meaningless results, and it cannot be checked by
    # inspection -- the alignment convention (F[i,t,:] forecasts R[i,t], so
    # F[:,t,:] is dated t-1) is easy to get backwards and looks fine either way.
    if not args.skip_lookahead:
        print("--- look-ahead verification (reduced settings) ---", flush=True)
        t0 = time()
        quick = lasso_fit_fn(target_r2=args.target_r2, n_draws=200, n_burn=100,
                             seed=args.seed)
        la = assert_no_lookahead(quick, R, F, start=args.start,
                                 refit_every=args.refit_every * 5)
        print(f"    forecasts before corruption: max diff {la['max_forecast_diff_before']:.3e}"
              f"   (must be exactly 0)")
        print(f"    coefficients before:         max diff {la['max_coefficient_diff_before']:.3e}"
              f"   (must be exactly 0)")
        print(f"    forecasts after corruption:  max change {la['forecast_change_after']:.3e}"
              f"   (must be > 0, or the test proves nothing)")
        print(f"    PASSES: {la['passes']}   ({time()-t0:.0f}s)\n", flush=True)
        if not la["passes"]:
            raise SystemExit("look-ahead test FAILED -- do not trust any results "
                             "from this configuration")

    # ---- the backtest -----------------------------------------------------
    t0 = time()
    bt = expanding_window(fit_fn, R, F, start=args.start,
                          refit_every=args.refit_every, progress=True)

    np.savez_compressed(
        outdir / f"{MODEL}_{SETTING}_backtest.npz",
        predicted=bt.predicted, realised=bt.realised, benchmark=bt.benchmark,
        oos_index=bt.oos_index, refit_index=bt.refit_index,
        coefficients=bt.coefficients)
    meta = dict(bt.meta)
    meta.update(fit_fn.settings)
    meta["label"] = SETTING
    with open(outdir / f"{MODEL}_{SETTING}_backtest_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2, default=str)

    # ---- metrics ----------------------------------------------------------
    pr = portfolio_returns(bt.realised, bt.predicted)
    cw = clark_west(bt.realised, bt.predicted, bt.benchmark)
    by_asset = r2_by_asset(bt.realised, bt.predicted, bt.benchmark)
    result = {
        "r2": out_of_sample_r2(bt.realised, bt.predicted, bt.benchmark),
        "r2_median_asset": float(np.nanmedian(by_asset)),
        "r2_assets_positive": int(np.nansum(by_asset > 0)),
        "sharpe": sharpe_ratio(pr),
        "ce": certainty_equivalent(pr),
        "cw_stat": cw["statistic"], "cw_p": cw["p_value"],
        "pred_vol_ratio": float(bt.predicted.std() / bt.realised.std()),
        "seconds": time() - t0,
    }

    print("\n" + "=" * 78)
    print(f"RESULTS  ({result['seconds']:.0f}s)")
    print("=" * 78)
    print(f"   OOS R^2                {result['r2']:+.4f}")
    print(f"   median per-asset R^2   {result['r2_median_asset']:+.4f}"
          f"   ({result['r2_assets_positive']}/{N} assets positive)")
    print(f"   Sharpe (annualised)    {result['sharpe']:+.3f}")
    print(f"   certainty equivalent   {result['ce']:+.4f}")
    print(f"   Clark-West             {result['cw_stat']:+.2f}  (p {result['cw_p']:.4f}, "
          f"one-sided)")
    print(f"   prediction volatility  {result['pred_vol_ratio']:.3f} of realised")

    # ---- against the baseline --------------------------------------------
    base_path = (Path("results") / args.universe / BASELINE / "backtest"
                 / f"{BASELINE}_{SETTING}_backtest.npz")
    if base_path.exists():
        z = np.load(base_path)
        print("\n" + "=" * 78)
        print(f"VERSUS THE GAUSSIAN BASELINE ({SETTING})")
        print("=" * 78)
        b_r2 = out_of_sample_r2(z["realised"], z["predicted"], z["benchmark"])
        b_pr = portfolio_returns(z["realised"], z["predicted"])
        print(f"   {'':<24s} {'LASSO':>10s} {'baseline':>10s}")
        print(f"   {'OOS R^2':<24s} {result['r2']:>+10.4f} {b_r2:>+10.4f}")
        print(f"   {'Sharpe':<24s} {result['sharpe']:>+10.3f} "
              f"{sharpe_ratio(b_pr):>+10.3f}")
        print(f"   {'certainty equivalent':<24s} {result['ce']:>+10.4f} "
              f"{certainty_equivalent(b_pr):>+10.4f}")
        print(f"   {'prediction volatility':<24s} {result['pred_vol_ratio']:>10.3f} "
              f"{z['predicted'].std()/z['realised'].std():>10.3f}")

        dm = diebold_mariano(bt.realised, bt.predicted, z["predicted"])
        result["dm_vs_baseline"] = dm["statistic"]
        result["dm_p"] = dm["p_value"]
        print(f"\n   Diebold-Mariano: {dm['statistic']:+.2f} (p {dm['p_value']:.4f}, "
              f"two-sided)")
        print("   [NEGATIVE favours the LASSO. DM is the right test here because the")
        print("    two models are NON-NESTED. Clark-West above is for the nested")
        print("    comparison against the historical mean -- applying DM there gives")
        print("    a mean statistic of +5.38 under the null and declares the")
        print("    benchmark the winner 99.7% of the time.]")

        # where the loss sits: common (market-wide) vs cross-sectional
        print("\n   LOSS DECOMPOSITION   (share of benchmark squared error)")
        print(f"   {'':<14s} {'var common':>11s} {'var cross':>10s} "
              f"{'cov common':>11s} {'cov cross':>10s}")
        for lbl, pred, real, bench in [
                ("LASSO", bt.predicted, bt.realised, bt.benchmark),
                ("baseline", z["predicted"], z["realised"], z["benchmark"])]:
            e, d = real - bench, pred - bench
            sse = (e ** 2).sum()
            ec, dc = e.mean(0), d.mean(0)
            ex, dx = e - ec, d - dc
            print(f"   {lbl:<14s} {N*(dc**2).sum()/sse:>+11.4f} "
                  f"{(dx**2).sum()/sse:>+10.4f} {-2*N*(ec*dc).sum()/sse:>+11.4f} "
                  f"{-2*(ex*dx).sum()/sse:>+10.4f}")
        print("   [the baseline's out-of-sample loss was 98.8% attributable to b_bar,")
        print("    whose prior is shared by all four models. The LASSO acts on theta,")
        print("    which carried 1.2%. Similar totals are therefore what the DESIGN")
        print("    predicts, not a null result -- but the decomposition shows whether")
        print("    the composition of the loss changed even when the total did not.]")

    print("\n   OOS R^2 is against the expanding historical mean. Realistic monthly")
    print("   values are 0.005-0.01 and often negative; anything above ~0.05 should")
    print("   be assumed an alignment bug. Sharpe and CE are annualised, from a")
    print("   zero-cost long-short portfolio with unit gross exposure.")

    with open(outdir / f"{MODEL}_backtest_summary.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\n   saved to {outdir}")


if __name__ == "__main__":
    main()

