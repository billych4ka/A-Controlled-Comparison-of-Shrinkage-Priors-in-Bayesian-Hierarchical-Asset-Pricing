"""
diagnose_bayesian_lasso.py

Reads every saved Bayesian LASSO run for one universe and prints the
diagnostics the write-up needs:

  1. convergence     -- R-hat and ESS per parameter block
  2. lambda          -- the learned global shrinkage level, and what it implies
  3. tau^2           -- where the local shrinkage actually lands
  4. comparison      -- LASSO vs the Gaussian baseline's rescaled_r2_0p05
  5. top predictors  -- with credible intervals, and the caveat about them

Mirrors diagnose_baseline_gaussian.py, with three differences that follow from
the model: there is no Delta_b (its Gibbs step is replaced by the tau^2
update, not supplemented), lambda is a sampled scalar rather than a fixed
hyperparameter, and tau^2 is stored as running means plus a tracked subset
rather than in full.

The baseline's section 2 (blocked vs sequential agreement) has no counterpart:
no sequential sampler is built for this model, and the blocked draw is instead
validated against a brute-force NT x NK stacked system. Its section 4
(between-setting rank stability) is reduced to the within-setting control,
since the LASSO runs one setting -- lambda being learned rather than chosen is
the point of the model.

Usage:
    python3 diagnose_bayesian_lasso.py
    python3 diagnose_bayesian_lasso.py --universe size_bm_100 --top 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.diagnostics.convergence import (chains_agree, credible_interval, diagnose,
                                         load_chains, posterior_mean, shrinkage_ratio,
                                         within_setting_rank_stability)
from src.gibbs.baseline_gaussian import GibbsDraws
from src.gibbs.bayesian_lasso import LassoDraws

MODEL = "bayesian_lasso"
BASELINE = "baseline_gaussian"
SETTING = "rescaled_r2_0p05"
# lambda values that reproduce each baseline setting's deviation prior sd,
# computed as sqrt(2)*s/sd_target at the full-sample s = 0.0557
LAMBDA_EQUIVALENTS = [(425, "target_r2 0.10"), (601, "target_r2 0.05"),
                      (1345, "target_r2 0.01")]


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
    ap.add_argument("--setting", default=SETTING)
    ap.add_argument("--top", type=int, default=20,
                    help="how many predictors to list; 8 proved too narrow "
                         "across 144 predictors in the baseline")
    ap.add_argument("--fields", nargs="*",
                    default=["b_bar", "Sigma", "lam", "tau2_track"],
                    help="B is omitted by default: 3,600 parameters, slow")
    args = ap.parse_args()

    chains = load_chains(args.universe, MODEL, args.setting, LassoDraws)
    K = chains[0].b_bar.shape[1]
    N = chains[0].Sigma.shape[1]
    names = predictor_names(args.universe, K)
    meta = chains[0].meta
    s = float(meta.get("s", np.nan))

    print("=" * 78)
    print(f"BAYESIAN LASSO | {args.universe} | {args.setting}")
    print("=" * 78)
    print(f"   {len(chains)} chains x {chains[0].b_bar.shape[0]} kept draws"
          f"   (n_draws={meta.get('n_draws')}, n_burn={meta.get('n_burn')})")
    print(f"   s={s:.4f}   sd_target={float(meta.get('sd_target', np.nan)):.4e}"
          f"   lambda hyperprior r={meta.get('lambda_r')}, "
          f"delta={float(meta.get('lambda_delta', np.nan)):.4e}")

    # ---- 1. convergence ---------------------------------------------------
    print("\n" + "=" * 78)
    print("1. CONVERGENCE   (threshold R-hat < 1.01, Vehtari et al. 2021)")
    print("=" * 78)
    for f in args.fields:
        try:
            print("   " + diagnose(chains, f).summary())
        except (ValueError, AttributeError) as e:
            print(f"   {f:<14s} skipped ({e})")
    print("\n   [lambda is expected to be the slowest-mixing parameter by a wide")
    print("    margin. Collapsing tau^2 out of its update (sample_lambda_collapsed)")
    print("    removed a 1,500-sweep drift and cut its lag-1 autocorrelation from")
    print("    0.973 to 0.93, but it stays coupled to theta through sum|theta_ij|.")
    print("    The sweep budget is sized for lambda; B and b_bar reach ESS 400 in")
    print("    about 500 sweeps.]")
    print("   [tau2_track is a fixed random SUBSAMPLE of 200 of the 3,600 auxiliary")
    print("    scales, chosen with a seed independent of the run seed. A strict")
    print("    reading of Vehtari et al. checks every sampled parameter; this is a")
    print("    subsample and should be described as one.]")

    # ---- 2. lambda --------------------------------------------------------
    print("\n" + "=" * 78)
    print("2. LAMBDA -- THE LEARNED SHRINKAGE LEVEL")
    print("=" * 78)
    lam = np.concatenate([c.lam for c in chains])
    lo, hi = np.percentile(lam, [2.5, 97.5])
    print(f"   posterior mean {lam.mean():.1f}   sd {lam.std():.1f}   "
          f"95% CI [{lo:.1f}, {hi:.1f}]")
    print(f"   per chain: " + "  ".join(f"{c.lam.mean():.1f}" for c in chains))
    print("   [chains agreeing here is the multimodality check. Park & Casella's")
    print("    unimodality guarantee was forfeited when the plug-in scale replaced")
    print("    their sampled sigma^2, so this is the evidence that nothing went")
    print("    wrong as a result.]")
    print(f"\n   implied theta prior sd = sqrt(2)*s/lambda = "
          f"{np.sqrt(2)*s/lam.mean():.4e}")
    print(f"   {'baseline equivalent':<24s} {'lambda':>8s}")
    for lv, lbl in LAMBDA_EQUIVALENTS:
        marker = " <--" if abs(lam.mean() - lv) < 100 else ""
        print(f"   {lbl:<24s} {lv:>8d}{marker}")
    print("   [higher lambda = MORE shrinkage. This is the pre-registered test:")
    print("    whether the data, choosing for itself, asks for more or less")
    print("    shrinkage than target_r2 = 0.05 imposes.]")

    # ---- 3. tau^2 ---------------------------------------------------------
    print("\n" + "=" * 78)
    print("3. TAU^2 -- WHERE THE LOCAL SHRINKAGE LANDS")
    print("=" * 78)
    prec = np.mean([c.tau2_inv_mean for c in chains], axis=0)      # (N,K)
    print("   E[1/tau_ij^2] is the prior PRECISION, so larger = shrunk harder.")
    q = np.percentile(prec.ravel(), [1, 25, 50, 75, 99])
    print(f"   across all {prec.size} coefficients: p1 {q[0]:.3e}  p25 {q[1]:.3e}  "
          f"median {q[2]:.3e}  p75 {q[3]:.3e}  p99 {q[4]:.3e}")
    print(f"   spread p99/p1 = {q[4]/q[0]:.0f}x")
    print("   [this spread is the local adaptivity the Gaussian baseline lacks")
    print("    entirely: its Delta_b is isotropic and ~97.5% prior-pinned, so one")
    print("    shrinkage level applies to all 3,600 deviations at once.]")

    is_int = np.array(["_x_" in c for c in names])
    idx = np.arange(K)
    groups = [("intercept", idx == 0),
              ("macro main effects", (~is_int) & (idx >= 1) & (idx <= 15)),
              ("characteristic main", (~is_int) & (idx >= 16) & (idx <= 23)),
              ("interactions", is_int)]
    print(f"\n   {'group':<22s} {'n':>4s} {'mean E[1/tau2]':>16s} {'vs overall':>11s}")
    overall = prec.mean()
    for nm, mask in groups:
        if mask.sum():
            g = prec[:, mask].mean()
            print(f"   {nm:<22s} {int(mask.sum()):>4d} {g:>16.4e} {g/overall:>10.2f}x")
    print("   [pre-registered prediction: the macro block would be shrunk HARDER")
    print("    than the characteristic block, since the baseline's anti-predictive")
    print("    content was 91-112% in the common component. Note the prediction is")
    print("    about b_bar, which has no tau^2 -- so a null here is coherent, not a")
    print("    failure. Compare the group spread against the 1-99 percentile spread")
    print("    above: if groups differ by ~1.3x while coefficients differ by 1000x,")
    print("    the adaptivity is real but not along this split.]")

    # least and most shrunk individual coefficients
    flat = prec.mean(axis=0)              # average across assets, per predictor
    print(f"\n   {'least-shrunk predictors':<28s} {'E[1/tau2]':>12s}     "
          f"{'most-shrunk predictors':<28s} {'E[1/tau2]':>12s}")
    lo_j, hi_j = np.argsort(flat)[:10], np.argsort(-flat)[:10]
    for a, b in zip(lo_j, hi_j):
        print(f"   {names[a]:<28s} {flat[a]:>12.3e}     "
              f"{names[b]:<28s} {flat[b]:>12.3e}")
    print("   [averaged over the 25 assets, so this is per-PREDICTOR shrinkage.")
    print("    tau^2 is per (asset, predictor), so a predictor can be free for one")
    print("    asset and shrunk for another -- that is the point of the local layer,")
    print("    and it is invisible in this table.]")

    # ---- 4. versus the baseline ------------------------------------------
    print("\n" + "=" * 78)
    print("4. VERSUS THE GAUSSIAN BASELINE (rescaled_r2_0p05)")
    print("=" * 78)
    try:
        base = load_chains(args.universe, BASELINE, SETTING, GibbsDraws)
    except FileNotFoundError:
        base = None
        print("   (baseline chains not found, skipping)")
    if base is not None:
        mb_l = posterior_mean(chains, "b_bar")
        mb_b = posterior_mean(base, "b_bar")
        # same helper the baseline's diagnose script uses, so the three numbers
        # are computed identically across models rather than reimplemented here
        sr = shrinkage_ratio(chains, base, "b_bar")
        print(f"   norm ratio ||lasso|| / ||baseline||: {sr['norm_ratio']:.4f}")
        print(f"   median |b_bar| ratio:                {sr['median_abs_ratio']:.4f}")
        print(f"   correlation of posterior mean b_bar: {sr['correlation']:.4f}")
        print("   [correlation near 1 = the Laplace prior rescaled the baseline's")
        print("    answer; well below 1 = it reordered which predictors matter. The")
        print("    baseline's own rescaling gave 0.586 against feng_he.]")
        ag = chains_agree(chains, base, "b_bar")
        print(f"   b_bar agreement: max {ag['max_z']:.2f} SE, median "
              f"{ag['median_z']:.2f} SE, {100*ag['frac_within_3']:.0f}% within 3 SE")
        print("   [these are DIFFERENT models, so disagreement is the finding, not")
        print("    an error. The agreement statistic is reported because a value")
        print("    near zero would mean the Laplace prior changed nothing.]")
        top_l = set(np.argsort(-np.abs(mb_l))[:args.top])
        top_b = set(np.argsort(-np.abs(mb_b))[:args.top])
        shared = sorted(top_l & top_b, key=lambda j: -abs(mb_l[j]))
        print(f"\n   in the top-{args.top} of BOTH models ({len(shared)} of {args.top}):")
        for j in shared:
            print(f"      {names[j]:<24s} lasso {mb_l[j]:+.5f}   baseline {mb_b[j]:+.5f}")
        print("   [the baseline's prior-robust core was SMB_x_lag1,")
        print("    RMW_x_roll_mean12, RMW_x_mom_12_1 -- all interactions. Two models")
        print("    with different priors agreeing is stronger evidence than either")
        print("    alone, and bears directly on the p_0 = 23 rewrite.]")

    # ---- 5. top predictors -----------------------------------------------
    print("\n" + "=" * 78)
    print(f"5. TOP {args.top} PREDICTORS BY |posterior mean b_bar|")
    print("=" * 78)
    m = posterior_mean(chains, "b_bar")
    lo_i, hi_i = credible_interval(chains, "b_bar")
    excl = int(np.sum(lo_i * hi_i > 0))
    print(f"   ({excl} of {K} intervals exclude zero)")
    for j in np.argsort(-np.abs(m))[:args.top]:
        star = "*" if lo_i[j] * hi_i[j] > 0 else " "
        print(f"   {star} {names[j]:<24s} {m[j]:+.5f}  [{lo_i[j]:+.5f}, {hi_i[j]:+.5f}]")

    w = within_setting_rank_stability(chains, top=args.top, n_reps=40)
    print(f"\n   within-setting rank stability (Monte Carlo ceiling): spearman "
          f"{w['spearman_median']:+.3f}, top-{args.top} overlap "
          f"{w[f'top{args.top}_overlap_median']:.2f}")
    print("   [this is the ceiling any between-model comparison can reach. A")
    print("    between-model overlap close to this number means the two models")
    print("    agree as well as one model agrees with itself.]")

    print("\n   NOTE: intervals are not significance tests. A tighter prior produces")
    print("   narrower intervals AND smaller point estimates, so counting stars")
    print("   across models compares priors, not evidence. The baseline gave 0, 11,")
    print("   8 and 17 across four settings -- non-monotonic in prior strength.")
    print("   Predictor selection comes from out-of-sample performance.")


if __name__ == "__main__":
    main()

