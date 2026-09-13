"""
diagnose_baseline_gaussian.py

Reads every saved Gaussian-baseline run for one universe and prints the
diagnostics the write-up needs:

  1. convergence: R-hat and ESS per setting, per parameter block
  2. sampler check: blocked vs sequential agreement at the feng_he prior
  3. shrinkage: how far each rescaled setting moves from feng_he
  4. rank stability: do the settings agree on WHICH predictors matter
  5. top predictors per setting, with credible intervals

Usage:
    python diagnose_baseline_gaussian.py
    python diagnose_baseline_gaussian.py --universe size_bm_100 --top 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.gibbs.baseline_gaussian import GibbsDraws
from src.diagnostics.convergence import (chains_agree, credible_interval, diagnose,
                                         load_chains, posterior_mean, rank_stability,
                                         shrinkage_ratio, stable_core,
                                         within_setting_rank_stability)

MODEL = "baseline_gaussian"
SETTINGS = ["feng_he", "rescaled_r2_0p05", "rescaled_r2_0p01", "rescaled_r2_0p1"]
SEQUENTIAL = "feng_he_sequential"


def predictor_names(universe: str, K: int) -> list[str]:
    for name in (f"{universe}_metadata.json", f"{universe}.json"):
        path = Path("data/processed") / name
        if path.exists():
            try:
                cols = json.load(open(path)).get("predictor_columns")
                if cols and len(cols) == K:
                    return cols
            except (json.JSONDecodeError, OSError):
                pass
    return [f"predictor {j}" for j in range(K)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="size_bm_25")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--fields", nargs="*",
                    default=["b_bar", "Sigma", "Delta_b_diag", "Delta_b_off"],
                    help="B is omitted by default: 3,600 parameters, slow")
    args = ap.parse_args()

    loaded, missing = {}, []
    for s in SETTINGS + [SEQUENTIAL]:
        try:
            loaded[s] = load_chains(args.universe, MODEL, s, GibbsDraws)
        except FileNotFoundError:
            missing.append(s)
    if missing:
        print(f"(not found, skipping: {', '.join(missing)})\n")
    if not loaded:
        raise SystemExit("no runs found; check --universe and the results layout")

    K = loaded[next(iter(loaded))][0].b_bar.shape[1]
    names = predictor_names(args.universe, K)

    print("=" * 78)
    print("1. CONVERGENCE   (threshold R-hat < 1.01, Vehtari et al. 2021)")
    print("=" * 78)
    for s, chains in loaded.items():
        meta = chains[0].meta
        print(f"\n{s}   ({len(chains)} chains x {chains[0].b_bar.shape[0]} kept draws, "
              f"blocked={meta.get('blocked', 'unrecorded')})")
        for f in args.fields:
            try:
                print("   " + diagnose(chains, f).summary())
            except (ValueError, AttributeError) as e:
                print(f"   {f:<14s} skipped ({e})")

    if "feng_he" in loaded and SEQUENTIAL in loaded:
        print("\n" + "=" * 78)
        print("2. SAMPLER CHECK  blocked vs sequential, same prior, same data")
        print("=" * 78)
        for f in ["b_bar", "Sigma"]:
            r = chains_agree(loaded["feng_he"], loaded[SEQUENTIAL], f)
            print(f"   {f:<8s} max {r['max_z']:.2f} SE   median {r['median_z']:.2f} SE   "
                  f"{100*r['frac_within_3']:.1f}% within 3 SE   ({r['n_params']} params)")
        print("   [both samplers target the same posterior; max under ~4 SE is agreement]")

    if "feng_he" in loaded:
        print("\n" + "=" * 78)
        print("3. SHRINKAGE relative to feng_he")
        print("=" * 78)
        print(f"   {'setting':<20s} {'norm ratio':>11s} {'median|b| ratio':>16s} {'correlation':>12s}")
        for s in SETTINGS[1:]:
            if s in loaded:
                r = shrinkage_ratio(loaded[s], loaded["feng_he"])
                print(f"   {s:<20s} {r['norm_ratio']:>11.4f} "
                      f"{r['median_abs_ratio']:>16.4f} {r['correlation']:>12.3f}")
        print("   [correlation near 1 = proportional shrinkage; well below 1 = reordering]")

        rescaled = {s: loaded[s] for s in SETTINGS[1:] if s in loaded}
        if len(rescaled) > 1:
            print("\n" + "=" * 78)
            print(f"4. RANK STABILITY across R^2 targets   (top-{args.top})")
            print("=" * 78)
            print("   (a) WITHIN each setting: the ceiling set by Monte Carlo error alone")
            for s, ch in rescaled.items():
                w = within_setting_rank_stability(ch, top=args.top, n_reps=40)
                print(f"       {s:<18s} spearman {w['spearman_median']:+.3f} "
                      f"(p10 {w['spearman_p10']:+.3f})   "
                      f"overlap {w[f'top{args.top}_overlap_median']:.2f}")
            print("   (b) BETWEEN settings: only meaningful relative to (a)")
            for (a, b), v in rank_stability(rescaled, top=args.top).items():
                print(f"       {a:<18s} vs {b:<18s} spearman {v['spearman']:+.3f}   "
                      f"overlap {v[f'top{args.top}_overlap']:.2f}")
            sc = stable_core(rescaled, top=args.top)
            print(f"   (c) STABLE CORE: in the top-{args.top} of all {sc['n_settings']} settings:")
            for j in sc["core"]:
                print(f"       {names[j]}")
            print("   [if (b) is close to (a), the R^2 choice is within Monte Carlo noise]")

    print("\n" + "=" * 78)
    print(f"5. TOP {args.top} PREDICTORS BY |posterior mean b_bar|")
    print("=" * 78)
    for s in SETTINGS:
        if s not in loaded:
            continue
        m = posterior_mean(loaded[s], "b_bar")
        lo, hi = credible_interval(loaded[s], "b_bar")
        excl = int(np.sum(lo * hi > 0))
        print(f"\n{s}   ({excl} of {K} intervals exclude zero)")
        for j in np.argsort(-np.abs(m))[:args.top]:
            star = "*" if lo[j] * hi[j] > 0 else " "
            print(f"   {star} {names[j]:<24s} {m[j]:+.5f}  [{lo[j]:+.5f}, {hi[j]:+.5f}]")
    print("\n   NOTE: intervals are not significance tests. A tighter prior produces")
    print("   narrower intervals AND smaller point estimates, so counting stars across")
    print("   settings compares priors, not evidence. Selection needs out-of-sample")
    print("   performance (src/evaluation/metrics.py), not these intervals.")


if __name__ == "__main__":
    main()
