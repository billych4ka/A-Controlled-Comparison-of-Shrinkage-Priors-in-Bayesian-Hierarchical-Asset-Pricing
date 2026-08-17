"""
run_backtest_regularised_horseshoe.py

Expanding-window out-of-sample backtest for the Regularised Horseshoe. Saves
predictions and per-window coefficients so the metrics can be recomputed later
without re-running the sampling.

Mirrors run_backtest_gaussian.py, run_backtest_lasso.py and
run_backtest_horseshoe.py exactly in settings, so all four are comparable:
start month 240, refit every 12 months, hyperparameters recomputed per window
from training data only, proportional long-short portfolio, the same three
metrics. THE REFIT SCHEDULE DEFINES THE INFORMATION SET and must not differ
between models.

Per-window budget: 500 retained draws with 500 tuning iterations, chosen by
check_reg_backtest_budget.py against the full four-chain, 1,500-draw posterior:

    draws/tune   med z   max z    corr   tau/ref  c/ref  bind%  min/fit
    300/300       0.53    3.10  0.9983     1.006  0.972  12.9%      3.5
    500/500       0.48    2.58  0.9990     0.968  1.020  12.0%      4.9
    750/500       0.53    2.89  0.9991     0.971  1.021  12.1%      5.5
    1000/1000     0.55    3.16  0.9994     0.978  1.024  12.0%      8.5

500/500 is the smallest budget clearing every pre-declared criterion. THE
PLAIN HORSESHOE REQUIRED 750/500 UNDER THE IDENTICAL PROCEDURE -- each budget
was validated against its own posterior rather than inherited, and this
model's better-conditioned geometry needs fewer draws for the same precision.
That difference is worth a sentence in the write-up: it is the check doing its
job rather than a number carried over.

TWO global parameters had to have travelled, not one: tau from tau_0 to
0.091 tau_0, and c from slab_scale to 0.69 of prior E[c]. Both land within 3%
of their production values at this budget. The binding fraction at 12.0%
against production's 12.6% confirms the backtest fits the same model the
production run described -- a short budget binding at a materially different
rate would not.

Cost: ~4.9 min at the longest window, and tree depth is FLAT at 7.00 across
both window lengths, so unlike the plain horseshoe -- whose cost grew faster
than linearly in T as depth climbed from 8 to 9 -- this should stay near 3
hours rather than drifting upward.

THE GUARDRAIL: a monthly OOS R^2 above ~0.05 should be assumed an alignment
bug until proven otherwise. Realistic values are 0.005-0.01, often negative;
the plain horseshoe gave -0.0455 on this universe, the LASSO -0.0494, the
baseline -0.0493.

THE STOPPING RULE, fixed in advance: if the Sharpe difference against the
plain horseshoe is below one standard error (0.158 over 479 months), the
primary universe is reported alone, with the binding diagnostic and the b_bar
agreement (correlation +0.9991, top-20 overlap 1.00 against a within-model
Monte Carlo ceiling of +0.999 and 0.95) as the explanation. Above it, OP and
Inv are run.

Usage:
    caffeinate -i python3 -u run_backtest_regularised_horseshoe.py 2>&1 \\
        | tee -a results/size_bm_25/regularised_horseshoe/backtest_log.txt
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
from src.nuts.backtest_adapter import reg_horseshoe_fit_fn

MODEL = "regularised_horseshoe"
COMPARISONS = (("horseshoe", "horseshoe", "p0_23_r2_0p05"),
               ("baseline", "baseline_gaussian", "rescaled_r2_0p05"),
               ("LASSO", "bayesian_lasso", "rescaled_r2_0p05"))
SHARPE_SE = 0.158       # annualised, over 479 months


def load_universe(universe: str):
    path = Path("data/processed") / f"{universe}_arrays.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    d = np.load(path, allow_pickle=True)
    return d["F"], d["R"]


def load_comparison(universe: str, model: str, setting: str):
    p = (Path("results") / universe / model / "backtest"
         / f"{model}_{setting}_backtest.npz")
    return np.load(p) if p.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="size_bm_25")
    ap.add_argument("--start", type=int, default=240)
    ap.add_argument("--refit-every", type=int, default=12,
                    help="MUST match the other models: the refit schedule "
                         "defines the information set")
    ap.add_argument("--draws", type=int, default=500,
                    help="RETAINED draws per window fit, excluding tuning")
    ap.add_argument("--tune", type=int, default=500)
    ap.add_argument("--p0", type=int, default=23)
    ap.add_argument("--target-r2", type=float, default=0.05)
    ap.add_argument("--nu", type=float, default=4.0)
    ap.add_argument("--slab-scale", type=float, default=None,
                    help="None recomputes s per window from training data, "
                         "which is correct; a fixed value is look-ahead")
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
    print(f"Regularised Horseshoe backtest | {args.universe} | N={N} T={T} K={K}")
    print(f"  first forecast month {args.start}, {T-args.start} out-of-sample months")
    print(f"  refit every {args.refit_every} months -> {n_fits} fits")
    print(f"  {args.draws} retained draws per fit ({args.tune} tune), 1 chain")
    print(f"  p0={args.p0}, nu={args.nu:g}; tau_0, s, tau and c all learned "
          f"per window")

    fit_fn = reg_horseshoe_fit_fn(p0=args.p0, target_r2=args.target_r2,
                                  nu=args.nu, slab_scale=args.slab_scale,
                                  n_draws=args.draws, n_tune=args.tune,
                                  seed=args.seed)

    # ---- look-ahead verification --------------------------------------------
    # Runs TWO full backtests, hence a reduced budget and a coarse refit grid.
    # Not optional: the alignment convention is the single most dangerous thing
    # in the evaluation code, and getting it backwards produces spectacular,
    # meaningless results rather than an error.
    if not args.skip_lookahead:
        print("\n--- look-ahead verification (reduced settings) ---")
        t0 = time()
        check_fn = reg_horseshoe_fit_fn(p0=args.p0, target_r2=args.target_r2,
                                        nu=args.nu, n_draws=args.lookahead_draws,
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

    # ---- the backtest --------------------------------------------------------
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

    # ---- metrics -------------------------------------------------------------
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
        print("   negative. Check the look-ahead result and the F/R alignment. ***")

    # ---- against the other three models --------------------------------------
    hs_sharpe = None
    for label, model, setting in COMPARISONS:
        z = load_comparison(args.universe, model, setting)
        if z is None:
            print(f"\n   ({model} backtest not found, skipping comparison)")
            continue
        o_r2 = out_of_sample_r2(z["realised"], z["predicted"], z["benchmark"])
        o_pr = portfolio_returns(z["realised"], z["predicted"])
        o_sharpe = sharpe_ratio(o_pr)
        dm = diebold_mariano(bt.realised, bt.predicted, z["predicted"])
        result[f"dm_vs_{model}"] = dm["statistic"]
        result[f"dm_p_vs_{model}"] = dm["p_value"]
        if model == "horseshoe":
            hs_sharpe = o_sharpe

        print("\n" + "=" * 78)
        print(f"VERSUS THE {label.upper()} ({setting})")
        print("=" * 78)
        print(f"   {'':<24s} {'reg. HS':>11s} {label:>11s}")
        print(f"   {'OOS R^2':<24s} {result['r2']:>+11.4f} {o_r2:>+11.4f}")
        print(f"   {'Sharpe':<24s} {result['sharpe']:>+11.3f} {o_sharpe:>+11.3f}")
        print(f"   {'certainty equivalent':<24s} {result['ce']:>+11.4f} "
              f"{certainty_equivalent(o_pr):>+11.4f}")
        print(f"   {'prediction volatility':<24s} {result['pred_vol_ratio']:>11.3f} "
              f"{z['predicted'].std()/z['realised'].std():>11.3f}")
        print(f"\n   Diebold-Mariano: {dm['statistic']:+.2f} "
              f"(p {dm['p_value']:.4f}, two-sided)")
        print("   [NEGATIVE favours the regularised horseshoe. Non-nested, so DM")
        print("    is correct; NEVER use DM against the historical mean.]")

    # ---- THE STOPPING RULE ---------------------------------------------------
    if hs_sharpe is not None:
        gap = abs(result["sharpe"] - hs_sharpe)
        print("\n" + "=" * 78)
        print("THE PRE-REGISTERED STOPPING RULE")
        print("=" * 78)
        print(f"   Sharpe: reg. HS {result['sharpe']:+.3f}, horseshoe "
              f"{hs_sharpe:+.3f}, gap {gap:.3f}")
        print(f"   threshold: one Sharpe SE over 479 months = {SHARPE_SE:.3f}")
        if gap < SHARPE_SE:
            print("   -> BELOW THRESHOLD. Report the primary universe alone, with")
            print("      the binding diagnostic (12.4% of local scales) and the")
            print("      b_bar agreement (correlation +0.9991, top-20 overlap 1.00")
            print("      against a within-model ceiling of +0.999 and 0.95) as the")
            print("      explanation. Replicating a PREDICTED null across three")
            print("      sorts adds little; the informative quantity is the")
            print("      binding fraction, which is measured directly.")
        else:
            print("   -> ABOVE THRESHOLD. Run size_op_25 and size_inv_25 to")
            print("      establish whether the direction is consistent across")
            print("      sorts. ~3 hours each.")
        print("\n   NOTE: a gap below threshold is 'no difference detected at the")
        print("   available power', NOT 'the two models are equivalent'. A genuine")
        print("   difference of moderate size could fall below 0.158 by chance.")

    with open(outdir / f"{MODEL}_{tag}_backtest_result.json", "w") as fh:
        json.dump(result, fh, indent=2, default=str)

    print("\n   PRE-REGISTERED: R^2 indistinguishable from the plain horseshoe --")
    print("   the slab touches only theta, and theta carries ~1% of out-of-sample")
    print("   loss. Any difference appears in Sharpe or CE, where demeaned")
    print("   portfolio weights let one outsized coefficient dominate a month's")
    print("   cross-sectional ranking, and where the slab caps exactly that.")


if __name__ == "__main__":
    main()

