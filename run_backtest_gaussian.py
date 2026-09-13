"""
run_backtest_gaussian.py

Expanding-window out-of-sample backtest for the Gaussian baseline, across all
four prior settings. Saves predictions and per-window coefficients so the
metrics can be recomputed later without re-running seven hours of sampling.

Before the backtest, assert_no_lookahead corrupts every return from the
midpoint of the evaluation period onward and confirms that no forecast dated
before the corruption moves, exactly as in the other three backtest runners.

Usage:
    python run_backtest_gaussian.py
    python run_backtest_gaussian.py --settings rescaled_r2_0p05
    python run_backtest_gaussian.py --refit-every 60 --draws 300
    python run_backtest_gaussian.py --skip-lookahead     (if already verified)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import time

import numpy as np

from src.evaluation.backtest import assert_no_lookahead, expanding_window
from src.evaluation.metrics import (certainty_equivalent, clark_west, out_of_sample_r2,
                                    portfolio_returns, r2_by_asset, sharpe_ratio)
from src.gibbs.backtest_adapter import gaussian_fit_fn

MODEL = "baseline_gaussian"

SETTINGS = {
    "feng_he": ("feng_he", None),
    "rescaled_r2_0p05": ("rescaled", 0.05),
    "rescaled_r2_0p01": ("rescaled", 0.01),
    "rescaled_r2_0p1": ("rescaled", 0.10),
}


def load_universe(universe: str):
    path = Path("data/processed") / f"{universe}_arrays.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    d = np.load(path, allow_pickle=True)
    return d["F"], d["R"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="size_bm_25")
    ap.add_argument("--settings", nargs="*", default=list(SETTINGS),
                    help="subset of settings to run")
    ap.add_argument("--start", type=int, default=240,
                    help="first month forecast; 240 = 20-year initial training window")
    ap.add_argument("--refit-every", type=int, default=12)
    ap.add_argument("--draws", type=int, default=500,
                    help="sweeps per window fit, including burn-in")
    ap.add_argument("--burn", type=int, default=100)
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
    print(f"Gaussian baseline backtest | {args.universe} | N={N} T={T} K={K}")
    print(f"  train from month 0, first forecast month {args.start}, "
          f"{T - args.start} out-of-sample months")
    print(f"  refit every {args.refit_every} months -> {n_fits} fits per setting")
    print(f"  {args.draws} sweeps per fit ({args.burn} burn-in), 1 chain")
    print(f"  settings: {', '.join(args.settings)}\n")

    if not args.skip_lookahead:
        print("--- look-ahead verification (reduced settings) ---", flush=True)
        t0 = time()
        quick = gaussian_fit_fn(prior="rescaled", target_r2=0.05, n_draws=200,
                                n_burn=100, seed=args.seed)
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
            raise SystemExit("look-ahead test FAILED; do not trust any results "
                             "from this configuration")

    results = {}
    t_all = time()
    for label in args.settings:
        if label not in SETTINGS:
            raise SystemExit(f"unknown setting {label!r}; choose from {list(SETTINGS)}")
        prior, target_r2 = SETTINGS[label]

        print(f"--- {label} ---", flush=True)
        fit_fn = gaussian_fit_fn(prior=prior, target_r2=target_r2 or 0.05,
                                 n_draws=args.draws, n_burn=args.burn, seed=args.seed)
        t0 = time()
        bt = expanding_window(fit_fn, R, F, start=args.start,
                              refit_every=args.refit_every, progress=True)

        np.savez_compressed(
            outdir / f"{MODEL}_{label}_backtest.npz",
            predicted=bt.predicted, realised=bt.realised, benchmark=bt.benchmark,
            oos_index=bt.oos_index, refit_index=bt.refit_index,
            coefficients=bt.coefficients)
        meta = dict(bt.meta); meta.update(fit_fn.settings); meta["label"] = label
        with open(outdir / f"{MODEL}_{label}_backtest_meta.json", "w") as fh:
            json.dump(meta, fh, indent=2, default=str)

        pr = portfolio_returns(bt.realised, bt.predicted)
        cw = clark_west(bt.realised, bt.predicted, bt.benchmark)
        results[label] = {
            "r2": out_of_sample_r2(bt.realised, bt.predicted, bt.benchmark),
            "r2_median_asset": float(np.nanmedian(r2_by_asset(bt.realised, bt.predicted,
                                                             bt.benchmark))),
            "r2_assets_positive": int(np.nansum(r2_by_asset(bt.realised, bt.predicted,
                                                           bt.benchmark) > 0)),
            "sharpe": sharpe_ratio(pr), "ce": certainty_equivalent(pr),
            "cw_stat": cw["statistic"], "cw_p": cw["p_value"],
            "seconds": time() - t0,
        }
        print(f"    done in {time()-t0:.0f}s\n", flush=True)

    print("=" * 78)
    print(f"RESULTS  ({time()-t_all:.0f}s total)")
    print("=" * 78)
    print(f"   {'setting':<20s} {'OOS R2':>9s} {'med asset':>10s} {'n>0':>5s} "
          f"{'Sharpe':>8s} {'CE':>9s} {'CW stat':>9s} {'CW p':>8s}")
    for label, v in results.items():
        print(f"   {label:<20s} {v['r2']:>+9.4f} {v['r2_median_asset']:>+10.4f} "
              f"{v['r2_assets_positive']:>3d}/{N:<2d} {v['sharpe']:>+8.2f} "
              f"{v['ce']:>+9.4f} {v['cw_stat']:>+9.2f} {v['cw_p']:>8.4f}")
    print("\n   OOS R2 vs the expanding historical mean. Realistic monthly values are")
    print("   0.005-0.01 and often negative; anything above ~0.05 should be assumed")
    print("   an alignment bug. Clark-West is one-sided (nested): p < 0.05 means the")
    print("   model beats the historical mean. Sharpe and CE are annualised, from a")
    print("   zero-cost long-short portfolio with unit gross exposure.")

    with open(outdir / f"{MODEL}_backtest_summary.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n   saved to {outdir}")


if __name__ == "__main__":
    main()
