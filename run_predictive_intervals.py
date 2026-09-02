"""
run_predictive_intervals.py

Re-runs the expanding-window backtest for one model with posterior draws
retained, and reports out-of-sample predictive-interval coverage under two
constructions: coefficient uncertainty only, as in Feng and He's Eq. (21), and
the full predictive interval including the residual shock.

Primary universe only. See src/evaluation/predictive_intervals.py for why the
existing sampling budgets are retained and why coverage is not replicated
across the robustness universes.

THE CHECK THAT MATTERS. This module duplicates expanding_window's loop rather
than modifying it, so the two must be shown to agree. The script compares its
posterior-mean forecasts against the saved Chapter 5 backtest and reports the
maximum absolute difference. Gibbs models with a fixed seed should agree to
floating-point exactness; NUTS models will differ by sampling noise, since a
single chain re-run reproduces the same posterior but not the same draws.

Usage, one model at a time so the runs can be sequenced overnight:

    python3 run_predictive_intervals.py --model baseline_gaussian
    python3 run_predictive_intervals.py --model bayesian_lasso
    python3 run_predictive_intervals.py --model regularised_horseshoe
    python3 run_predictive_intervals.py --model horseshoe        # ~20 h

Expect roughly 4, 4, 6 and 20 hours respectively.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.evaluation.interval_adapters import (gaussian_interval_fit_fn,
                                              horseshoe_interval_fit_fn,
                                              lasso_interval_fit_fn,
                                              reg_horseshoe_interval_fit_fn)
from src.evaluation.predictive_intervals import expanding_window_intervals

BUILDERS = {
    "baseline_gaussian": (gaussian_interval_fit_fn, "rescaled_r2_0p05"),
    "bayesian_lasso": (lasso_interval_fit_fn, "rescaled_r2_0p05"),
    "horseshoe": (horseshoe_interval_fit_fn, "p0_23_r2_0p05"),
    "regularised_horseshoe": (reg_horseshoe_interval_fit_fn, "p0_23_r2_0p05"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(BUILDERS))
    ap.add_argument("--universe", default="size_bm_25")
    ap.add_argument("--start", type=int, default=240)
    ap.add_argument("--refit-every", type=int, default=12)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    d = np.load(Path("data/processed") / f"{args.universe}_arrays.npz",
                allow_pickle=True)
    F, R = d["F"], d["R"]
    builder, setting = BUILDERS[args.model]
    fit_fn = builder()

    outdir = Path(args.outdir or f"results/{args.universe}/{args.model}/intervals")
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Predictive intervals | {args.model} | {args.universe}")
    print(f"  settings: {fit_fn.settings}")
    print(f"  {len(range(args.start, R.shape[1], args.refit_every))} refits, "
          f"draws retained per window\n")

    res = expanding_window_intervals(fit_fn, R, F, start=args.start,
                                     refit_every=args.refit_every)

    # ---- agreement with the Chapter 5 backtest -----------------------------
    ref = (Path("results") / args.universe / args.model / "backtest"
           / f"{args.model}_{setting}_backtest.npz")
    if ref.exists():
        z = np.load(ref)
        diff = float(np.abs(res.predicted - z["predicted"]).max())
        rel = diff / float(np.abs(z["predicted"]).mean())
        print(f"\nagreement with the Chapter 5 backtest: max |diff| {diff:.3e} "
              f"({rel:.2%} of the mean absolute forecast)")
        print("   [Gibbs models share a fixed seed and should agree essentially")
        print("    exactly; NUTS models re-run a single chain and will differ by")
        print("    sampling noise, which is the same magnitude as the seed-to-seed")
        print("    variation already documented in Section 4.4.]")
    else:
        diff = rel = float("nan")
        print(f"\n(no saved backtest at {ref}, skipping the agreement check)")

    # ---- coverage ----------------------------------------------------------
    c = res.coverage()
    print("\n" + "=" * 74)
    print(f"PREDICTIVE INTERVAL COVERAGE | {args.model}")
    print("=" * 74)
    print(f"   {c['n_asset_months']:,} asset-months, nominal 95%")
    print(f"   coefficient uncertainty only : {100*c['coverage_coef_only']:6.2f}%"
          f"   mean width {c['mean_width_coef_only']:.4f}")
    print(f"   full predictive              : {100*c['coverage_full']:6.2f}%"
          f"   mean width {c['mean_width_full']:.4f}")
    print(f"   width ratio (full / coef)    : {c['width_ratio']:.1f}x")
    print(f"\n   binomial SE at nominal coverage: {100*c['binomial_se']:.2f}%")
    print("   [a LOWER BOUND on the true standard error: asset-months are not")
    print("    independent, being serially dependent within a refit year and")
    print("    cross-sectionally dependent through Sigma. Quote it as such.]")

    out = {**c, "model": args.model, "universe": args.universe,
           "settings": fit_fn.settings, "meta": res.meta,
           "max_diff_vs_backtest": diff, "rel_diff_vs_backtest": rel}
    with open(outdir / f"{args.model}_{setting}_coverage.json", "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    np.savez_compressed(
        outdir / f"{args.model}_{setting}_intervals.npz",
        predicted=res.predicted, realised=res.realised,
        lo_coef=res.lo_coef, hi_coef=res.hi_coef,
        lo_full=res.lo_full, hi_full=res.hi_full,
        oos_index=res.oos_index, refit_index=res.refit_index)
    print(f"\nsaved -> {outdir}")


if __name__ == "__main__":
    main()