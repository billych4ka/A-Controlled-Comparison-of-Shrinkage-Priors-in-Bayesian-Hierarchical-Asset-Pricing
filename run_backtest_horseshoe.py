"""
run_backtest_horseshoe.py

Expanding-window out-of-sample backtest for the Horseshoe. Saves predictions
and per-window coefficients so the metrics can be recomputed later without
re-running ten hours of sampling.

Mirrors run_backtest_gaussian.py and run_backtest_lasso.py exactly in
settings, so the three are comparable: start month 240, refit every 12 months,
hyperparameters recomputed per window from training data only, proportional
long-short portfolio, the same three metrics.

Per-window budget: 750 retained draws with 500 tuning iterations, chosen by
check_backtest_budget.py against the full four-chain, 1,500-draw posterior:

    draws/tune   median z   max z   <3 SE     corr   tau/ref   min/fit
    300/300          0.42    3.70   100.0%   0.9985     1.036      12.9
    500/500          0.56    4.34    99.8%   0.9985     1.038      15.5
    750/500          0.60    3.84    99.8%   0.9990     1.019      21.1
    1000/1000        0.52    4.76    99.8%   0.9993     0.997      34.8

750/500 is the smallest budget clearing every pre-declared criterion (median
z below 1, all within 4 SE, correlation above 0.999, tau within 20% of the
reference). 300/300 and 500/500 sit at 0.9985, just under the correlation
line, and 300/300 additionally saturates tree depth at 9.98, which would
truncate trajectories in all 40 windows. 1000/1000 buys 0.0003 of correlation
for 65% more time. Do not switch to a cheaper budget after seeing that it also
passes -- that is selection on outcomes, the same genre as revising target_r2.

tau reaching 1.019x its production value at this budget matters more than the
z statistics: tau starts at tau_0 and the data moves it to about 0.05 tau_0,
so draws retained before it has travelled would sample B under the wrong
shrinkage level -- a systematic error repeated in every window.

THE GUARDRAIL: a monthly OOS R^2 above ~0.05 should be assumed an alignment
bug until proven otherwise. Realistic values are 0.005-0.01, often negative;
the baseline and the LASSO both came in near -0.049 on this universe.

Usage:
    caffeinate -i python3 run_backtest_horseshoe.py
    python3 run_backtest_horseshoe.py --skip-lookahead     # only if already verified
    python3 run_backtest_horseshoe.py --universe size_op_25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import time

import numpy as np

from src.evaluation.backtest import assert_no_lookahead, expanding_window
from src.evaluation.metrics import (certainty_equivalent, clark_west,
                                    diebold_mariano, out_of_sample_r2,
                                    portfolio_returns, r2_by_asset, sharpe_ratio)
from src.nuts.backtest_adapter import horseshoe_fit_fn

MODEL = "horseshoe"
BASELINE, LASSO = "baseline_gaussian", "bayesian_lasso"
COMPARISON_SETTING = "rescaled_r2_0p05"      # the comparison of record


def load_universe(universe: str):
    path = Path("data/processed") / f"{universe}_arrays.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    d = np.load(path, allow_pickle=True)
    return d["F"], d["R"]


def load_comparison(universe: str, model: str):
    p = (Path("results") / universe / model / "backtest"
         / f"{model}_{COMPARISON_SETTING}_backtest.npz")
    return np.load(p) if p.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="size_bm_25")
    ap.add_argument("--start", type=int, default=240,
                    help="first month forecast; 240 = 20-year initial training window")
    ap.add_argument("--refit-every", type=int, default=12,
                    help="MUST match the other models: the refit schedule defines "
                         "the information set")
    ap.add_argument("--draws", type=int, default=750,
                    help="RETAINED draws per window fit, excluding tuning")
    ap.add_argument("--tune", type=int, default=500)
    ap.add_argument("--p0", type=int, default=23)
    ap.add_argument("--target-r2", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-lookahead", action="store_true")
    ap.add_argument("--lookahead-draws", type=int, default=100)
    ap.add_argument("--lookahead-tune", type=int, default=200)
    ap.add_argument("--lookahead-refit-every", type=int, default=120,
                    help="coarse, because the check runs TWO full backtests")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    F, R = load_universe(args.universe)
    N, T, K = F.shape
    tag = f"p0_{args.p0}_r2_{args.target_r2:g}".replace(".", "p")
    outdir = Path(args.outdir or f"results/{args.universe}/{MODEL}/backtest")
    outdir.mkdir(parents=True, exist_ok=True)

    n_fits = len(range(args.start, T, args.refit_every))
    print(f"Horseshoe backtest | {args.universe} | N={N} T={T} K={K}")
    print(f"  train from month 0, first forecast month {args.start}, "
          f"{T - args.start} out-of-sample months")
    print(f"  refit every {args.refit_every} months -> {n_fits} fits")
    print(f"  {args.draws} retained draws per fit ({args.tune} tune), 1 chain")
    print(f"  p0={args.p0} (fixed a priori), target_r2={args.target_r2}; "
          f"tau_0 and tau learned per window")

    fit_fn = horseshoe_fit_fn(p0=args.p0, target_r2=args.target_r2,
                              n_draws=args.draws, n_tune=args.tune,
                              seed=args.seed)

    # ---- look-ahead verification -------------------------------------------
    # Runs TWO full backtests, so it uses a reduced budget and a coarse refit
    # grid. It is not optional: the alignment convention (F[:,t,:] forecasts
    # R[:,t], no further lagging) is the single most dangerous thing in the
    # evaluation code, and getting it backwards would produce spectacular,
    # meaningless results rather than an error.
    if not args.skip_lookahead:
        print("\n--- look-ahead verification (reduced settings) ---")
        t0 = time()
        check_fn = horseshoe_fit_fn(p0=args.p0, target_r2=args.target_r2,
                                    n_draws=args.lookahead_draws,
                                    n_tune=args.lookahead_tune, seed=args.seed)
        la = assert_no_lookahead(check_fn, R, F, start=args.start,
                                 refit_every=args.lookahead_refit_every)
        print(f"    forecasts before corruption: max diff "
              f"{la['max_forecast_diff_before']:.3e}   (must be exactly 0)")
        print(f"    coefficients before:         max diff "
              f"{la['max_coefficient_diff_before']:.3e}   (must be exactly 0)")
        print(f"    forecasts after corruption:  max change "
              f"{la['forecast_change_after']:.3e}   (must be > 0, or the test "
              f"proves nothing)")
        print(f"    PASSES: {la['passes']}   ({time()-t0:.0f}s)")
        if not la["passes"]:
            raise SystemExit("look-ahead check FAILED -- do not run the backtest")

    # ---- the backtest -------------------------------------------------------
    print()
    t0 = time()
    bt = expanding_window(fit_fn, R, F, start=args.start,
                          refit_every=args.refit_every, progress=True)

    np.savez_compressed(outdir / f"{MODEL}_{tag}_backtest.npz",
                        predicted=bt.predicted, realised=bt.realised,
                        benchmark=bt.benchmark, oos_index=bt.oos_index,
                        refit_index=bt.refit_index, coefficients=bt.coefficients)
    meta = dict(bt.meta)
    meta.update(fit_fn.settings)
    meta["label"] = tag
    meta["universe"] = args.universe
    with open(outdir / f"{MODEL}_{tag}_backtest_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2, default=str)

    # ---- metrics ------------------------------------------------------------
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
    print(f"   Clark-West             {result['cw_stat']:+.2f}  "
          f"(p {result['cw_p']:.4f}, one-sided)")
    print(f"   prediction volatility  {result['pred_vol_ratio']:.3f} of realised")

    if result["r2"] > 0.05:
        print("\n   *** OOS R^2 above 0.05. Assume an ALIGNMENT BUG until proven")
        print("   otherwise -- realistic monthly values are 0.005-0.01 and often")
        print("   negative. Check the look-ahead result and the F/R alignment")
        print("   before reporting anything. ***")

    # ---- against the other two models ---------------------------------------
    for label, model in (("baseline", BASELINE), ("LASSO", LASSO)):
        z = load_comparison(args.universe, model)
        if z is None:
            print(f"\n   ({model} backtest not found, skipping comparison)")
            continue
        o_r2 = out_of_sample_r2(z["realised"], z["predicted"], z["benchmark"])
        o_pr = portfolio_returns(z["realised"], z["predicted"])
        dm = diebold_mariano(bt.realised, bt.predicted, z["predicted"])
        result[f"dm_vs_{model}"] = dm["statistic"]
        result[f"dm_p_vs_{model}"] = dm["p_value"]

        print("\n" + "=" * 78)
        print(f"VERSUS THE {label.upper()} ({COMPARISON_SETTING})")
        print("=" * 78)
        print(f"   {'':<24s} {'horseshoe':>11s} {label:>11s}")
        print(f"   {'OOS R^2':<24s} {result['r2']:>+11.4f} {o_r2:>+11.4f}")
        print(f"   {'Sharpe':<24s} {result['sharpe']:>+11.3f} "
              f"{sharpe_ratio(o_pr):>+11.3f}")
        print(f"   {'certainty equivalent':<24s} {result['ce']:>+11.4f} "
              f"{certainty_equivalent(o_pr):>+11.4f}")
        print(f"   {'prediction volatility':<24s} {result['pred_vol_ratio']:>11.3f} "
              f"{z['predicted'].std()/z['realised'].std():>11.3f}")
        print(f"\n   Diebold-Mariano: {dm['statistic']:+.2f} "
              f"(p {dm['p_value']:.4f}, two-sided)")
        print(f"   [NEGATIVE favours the horseshoe. Non-nested comparison, so DM")
        print(f"    is correct here; NEVER use DM against the historical mean --")
        print(f"    under the nested null it gives a mean statistic of +5.38 and")
        print(f"    declares the benchmark the winner 99.7% of the time.]")

    with open(outdir / f"{MODEL}_{tag}_backtest_result.json", "w") as fh:
        json.dump(result, fh, indent=2, default=str)

    print("\n" + "=" * 78)
    print("   The SE of an annualised Sharpe over 479 months is 0.158, so a")
    print("   difference of a few hundredths is a DIRECTION, not a detected")
    print("   effect. Report it as consistent across universes if it is, not as")
    print("   significance.")
    print("\n   PRE-REGISTERED: R^2 differences smaller than the baseline's own")
    print("   between-setting spread, because b_bar carries 98.8% of the loss and")
    print("   its prior is fixed across all four models. Larger differences")
    print("   predicted in Sharpe and CE, which see only the cross-sectional")
    print("   component where theta acts.")


if __name__ == "__main__":
    main()