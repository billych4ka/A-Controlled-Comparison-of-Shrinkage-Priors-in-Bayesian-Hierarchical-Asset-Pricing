"""
diagnose_regularised_horseshoe.py

Reads every saved Regularised Horseshoe run for one universe and prints what
the write-up needs. Mirrors diagnose_horseshoe.py section for section, with
two additions specific to this model: the slab's own convergence and interval,
and the BINDING diagnostic measured from the draws.

THE QUESTION THIS SCRIPT EXISTS TO SETTLE. The production run gave tau =
3.6064e-05 against the plain horseshoe's 2.1019e-05 -- 1.72x larger, all four
chains agreeing to within 3%. The tempting reading is that the slab relieves
tau: with the tails bounded, tau no longer has to collapse to control them.
That is a real mechanism and the binding fraction supports it.

BUT matched-dimension calibration found tau's recovery correlation at -0.036
for this model against +0.996 for the plain horseshoe. Where the slab binds,
tau * lambda~ -> c, so the coefficient scale becomes c REGARDLESS of tau and
the global scale stops being identified through those coefficients. Coverage
stayed honest (0.980) because the intervals widen correctly -- but tau was not
being recovered.

At 61% binding, which is where that calibration sat. Production binds at
12.4%, so most local scales still carry information about tau. WHETHER TAU IS
IDENTIFIED AT THE PRODUCTION OPERATING POINT IS AN EMPIRICAL QUESTION, and
tau's ESS is the answer: weak identification shows as poor mixing. Model 3
gave ESS 1,012 and R-hat 1.0015.

  If tau's ESS is comparable -> tau is identified here, the calibration
  finding is about that simulated regime, and the 1.72x difference can be
  stated as substantive.

  If it collapses -> the difference cannot be read as "the slab changes the
  global scale", and section 5.6 reports c, the binding fraction and kappa as
  the interpretable quantities with tau caveated.

Do not skip this check and quote the tau ratio.

Usage:
    python3 diagnose_regularised_horseshoe.py
    python3 diagnose_regularised_horseshoe.py --settings p0_23_r2_0p05 p0_23_r2_0p05_s2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.diagnostics.convergence import (credible_interval, diagnose, load_chains,
                                         posterior_mean, rank_stability,
                                         within_setting_rank_stability)
from src.gibbs.baseline_gaussian import GibbsDraws
from src.gibbs.bayesian_lasso import LassoDraws
from src.nuts.horseshoe import HorseshoeDraws
from src.nuts.regularised_horseshoe import (RegHorseshoeDraws, binding_summary,
                                            ebfmi, reg_horseshoe_hyperparameters,
                                            shrinkage_factors)

MODEL = "regularised_horseshoe"
DEFAULT_SETTING = "p0_23_r2_0p05"
HORSESHOE, HS_SETTING = "horseshoe", "p0_23_r2_0p05"
BASELINE, BL_SETTING = "baseline_gaussian", "rescaled_r2_0p05"
LASSO, LA_SETTING = "bayesian_lasso", "rescaled_r2_0p05"

# the plain horseshoe's figures on size_bm_25, for direct comparison
HS_TAU, HS_TAU_ESS, HS_TAU_RHAT = 2.1019e-05, 1012.0, 1.0015


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
    ap.add_argument("--settings", nargs="*", default=[DEFAULT_SETTING])
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--fields", nargs="*",
                    default=["b_bar", "tau", "c", "Sigma", "lam_local", "lam_tilde"],
                    help="B is omitted by default: 3,600 parameters")
    args = ap.parse_args()

    loaded = {}
    for s in args.settings:
        try:
            loaded[s] = load_chains(args.universe, MODEL, s, RegHorseshoeDraws)
        except FileNotFoundError:
            print(f"(not found, skipping: {s})")
    if not loaded:
        raise SystemExit("no runs found")

    primary = args.settings[0] if args.settings[0] in loaded else next(iter(loaded))
    chains = loaded[primary]
    K = chains[0].b_bar.shape[1]
    N = chains[0].Sigma.shape[1]
    names = predictor_names(args.universe, K)
    meta = chains[0].meta
    d = np.load(Path("data/processed") / f"{args.universe}_arrays.npz",
                allow_pickle=True)
    F, R = d["F"], d["R"]
    hp = reg_horseshoe_hyperparameters(
        R, F, K, p0=int(meta.get("p0", 23)),
        target_r2=float(meta.get("target_r2", 0.05)),
        nu=float(meta.get("slab_df", 4.0)),
        n_choice=str(meta.get("n_choice", "T")),
        slab_scale=float(meta["slab_scale"]) if "slab_scale" in meta else None)

    print("=" * 78)
    print(f"REGULARISED HORSESHOE | {args.universe} | {primary}")
    print("=" * 78)
    print(f"   {len(chains)} chains x {chains[0].b_bar.shape[0]} retained draws "
          f"({meta.get('n_tune')} tune)")
    print(f"   tau_0={hp.tau_0:.4e}  s={hp.slab_scale:.4e}  E[c]={hp.slab_sd:.4e}")
    print(f"   target_accept={meta.get('target_accept')}  "
          f"max_treedepth={meta.get('max_treedepth')}")

    # ---- 1. convergence ----------------------------------------------------
    print("\n" + "=" * 78)
    print("1. CONVERGENCE   (R-hat < 1.01; ESS target 400 for posterior means)")
    print("=" * 78)
    tau_ess = tau_rhat = np.nan
    for field in args.fields:
        if not hasattr(chains[0], field):
            continue
        dg = diagnose(chains, field)
        print(f"   {dg.summary()}")
        if field == "tau":
            tau_ess = float(np.nanmin(getattr(dg, "ess_bulk", np.array([np.nan]))))
            tau_rhat = float(np.nanmax(getattr(dg, "rhat", np.array([np.nan]))))

    print("\n" + "-" * 78)
    print("   TAU'S IDENTIFICATION -- the question this script exists to settle")
    print("-" * 78)
    print(f"   this model : ESS {tau_ess:.0f}, R-hat {tau_rhat:.4f}")
    print(f"   horseshoe  : ESS {HS_TAU_ESS:.0f}, R-hat {HS_TAU_RHAT:.4f}")
    if np.isfinite(tau_ess):
        ratio = tau_ess / HS_TAU_ESS
        print(f"   ratio {ratio:.2f}")
        if ratio > 0.5:
            print("   -> tau mixes comparably. It IS identified at this operating")
            print("      point, the calibration's -0.036 recovery correlation")
            print("      belongs to that 61%-binding regime, and the 1.72x")
            print("      difference from the plain horseshoe can be stated as")
            print("      substantive -- with the mechanism (the slab relieves tau)")
            print("      supported by the binding fraction below.")
        else:
            print("   -> tau mixes MUCH worse. Weak identification is the likely")
            print("      cause: where the slab binds, tau*lam~ -> c and the")
            print("      coefficient scale is c regardless of tau. Report c, the")
            print("      binding fraction and kappa as the interpretable")
            print("      quantities, and CAVEAT any tau comparison.")

    # ---- 2. NUTS -----------------------------------------------------------
    print("\n" + "=" * 78)
    print("2. NUTS DIAGNOSTICS, per chain")
    print("=" * 78)
    print(f"   {'chain':<7}{'seed':>5}{'div':>7}{'div %':>8}{'depth':>8}"
          f"{'sat %':>8}{'step':>10}{'E-BFMI':>9}{'bind %':>9}")
    n_draws = chains[0].b_bar.shape[0]
    max_depth = int(meta.get("max_treedepth", 10))
    for k, c in enumerate(chains):
        div = int(c.diverging.sum())
        sat = float(np.mean(c.tree_depth >= max_depth))
        bind = float((c.lam_tilde / c.lam_local < 0.99).mean())
        print(f"   {k:<7d}{c.meta.get('seed', k):>5}{div:>7d}"
              f"{100*div/n_draws:>7.1f}%{np.mean(c.tree_depth):>8.2f}"
              f"{100*sat:>7.0f}%{c.step_size[-1]:>10.4f}{ebfmi(c.energy):>9.2f}"
              f"{100*bind:>8.1f}%")
    tot = sum(int(c.diverging.sum()) for c in chains)
    print(f"   {'total':<7}{'':>5}{tot:>7d}{100*tot/(len(chains)*n_draws):>7.1f}%")
    print("\n   [the plain horseshoe on this universe: 57 of 6,000 (0.9%), tree")
    print("    depth 9.00, step size 0.006-0.011, 3h12m. P&V report the")
    print("    regularised version as no slower and often faster BECAUSE the")
    print("    posterior behaves better -- a bounded prior means milder worst-case")
    print("    curvature, so the step size can rise and trees shorten.]")

    # ---- 3. the slab -------------------------------------------------------
    print("\n" + "=" * 78)
    print("3. THE SLAB   -- what the regularised horseshoe adds")
    print("=" * 78)
    c_all = np.concatenate([c.c for c in chains])
    clo, chi = np.percentile(c_all, [2.5, 97.5])
    print(f"   c        {c_all.mean():.4e}  [{clo:.4e}, {chi:.4e}]")
    print(f"   prior E[c] {hp.slab_sd:.4e}  ->  ratio {c_all.mean()/hp.slab_sd:.4f}")
    print(f"   = {c_all.mean()/hp.sd_target:.1f} x sd_target, "
          f"{c_all.mean()/hp.sigma_pooled:.4f} sigma")

    bs = binding_summary(chains[0], hp)
    print(f"\n   BINDING (chain 0): lam~/lam < 0.99 for "
          f"{100*bs['frac_binding']:.2f}% of local scales,")
    print(f"      {bs['n_binding_per_draw']:.0f} of {N*K} per draw; "
          f"{100*bs['frac_halved']:.2f}% halved or more")
    print(f"      median lam~/lam {bs['median_ratio']:.4f}, min {bs['min_ratio']:.4f}")
    print(f"      c/tau {bs['c_over_tau']:.1f}, so the slab bites at lam above that")
    print(f"      prior prediction: {100*bs['prior_frac_at_tau0']:.2f}% at tau_0, "
          f"{100*bs['prior_frac_at_posterior_tau']:.2f}% at this tau")
    if bs["frac_binding"] < 1e-3:
        print("\n   *** THE SLAB IS NOT BINDING: this model has collapsed into the")
        print("   plain horseshoe and any difference between them is noise. ***")

    tau_all = np.concatenate([c.tau for c in chains])
    tlo, thi = np.percentile(tau_all, [2.5, 97.5])
    print(f"\n   tau      {tau_all.mean():.4e}  [{tlo:.4e}, {thi:.4e}]"
          f"   ratio to tau_0 {tau_all.mean()/hp.tau_0:.4f}")
    print(f"   horseshoe {HS_TAU:.4e} (ratio 0.0533)  ->  this model is "
          f"{tau_all.mean()/HS_TAU:.2f}x larger")
    print("   [mechanism, IF tau is identified above: with the tails bounded by")
    print("    the slab, tau no longer has to collapse to control them. That also")
    print("    explains why binding stays near its prior value of 12.91% rather")
    print("    than falling to the 0.70% predicted by holding tau at the plain")
    print("    horseshoe's posterior -- tau is not fixed, it is what the slab")
    print("    relieves.]")

    # ---- 4. kappa across all four models -----------------------------------
    print("\n" + "=" * 78)
    print("4. SHRINKAGE FACTORS kappa   -- the axis on which all four compare")
    print("=" * 78)
    kap = np.concatenate([shrinkage_factors(c, F, hp) for c in chains], axis=0)
    print(f"   reg. horseshoe : mean {kap.mean():.4f}   "
          f"[{kap.min():.4f}, {kap.max():.4f}]   "
          f"-> {(1-kap.mean())*K:.2f} free per asset")
    try:
        from src.nuts.horseshoe import horseshoe_hyperparameters
        from src.nuts.horseshoe import shrinkage_factors as hs_kappa
        hs = load_chains(args.universe, HORSESHOE, HS_SETTING, HorseshoeDraws)
        hp_hs = horseshoe_hyperparameters(R, F, K, p0=hp.p0)
        khs = np.concatenate([hs_kappa(c, F, hp_hs) for c in hs], axis=0)
        print(f"   horseshoe      : mean {khs.mean():.4f}   "
              f"[{khs.min():.4f}, {khs.max():.4f}]   "
              f"-> {(1-khs.mean())*K:.2f} free per asset")
        print("\n   [read the two together: if the regularised model has a HIGHER")
        print("    floor but a LOWER mean, it shrinks less on average while")
        print("    refusing to leave any coefficient as free as the plain")
        print("    horseshoe permits. That is exactly what a slab does, measured")
        print("    on the one commensurable axis.]")
    except (FileNotFoundError, AttributeError, KeyError) as exc:
        print(f"   horseshoe      : not comparable ({type(exc).__name__})")

    # ---- 5. cross-model agreement ------------------------------------------
    print("\n" + "=" * 78)
    print(f"5. CROSS-MODEL AGREEMENT ON b_bar   (top-{args.top})")
    print("=" * 78)
    w = within_setting_rank_stability(chains, top=args.top, n_reps=40)
    print(f"   within-model ceiling (Monte Carlo alone): spearman "
          f"{w['spearman_median']:+.3f}, overlap "
          f"{w[f'top{args.top}_overlap_median']:.2f}")
    m_rh = posterior_mean(chains, "b_bar")
    others = {}
    for label, model, setting, cls in (
            ("horseshoe", HORSESHOE, HS_SETTING, HorseshoeDraws),
            ("baseline", BASELINE, BL_SETTING, GibbsDraws),
            ("LASSO", LASSO, LA_SETTING, LassoDraws)):
        try:
            others[label] = load_chains(args.universe, model, setting, cls)
        except FileNotFoundError:
            print(f"   {label}: not found")
    for label, ch in others.items():
        m_o = posterior_mean(ch, "b_bar")
        corr = float(np.corrcoef(m_rh, m_o)[0, 1])
        ov = len(set(np.argsort(-np.abs(m_rh))[:args.top])
                 & set(np.argsort(-np.abs(m_o))[:args.top])) / args.top
        print(f"   reg. HS vs {label:<10s} correlation {corr:+.4f}   "
              f"top-{args.top} overlap {ov:.2f}")
    if others:
        allm = {"reg_horseshoe": chains, **others}
        print(f"\n   pairwise Spearman on |b_bar| across ALL FOUR models:")
        for (a, b), v in rank_stability(allm, top=args.top).items():
            print(f"      {a:<14s} vs {b:<14s} spearman {v['spearman']:+.3f}   "
                  f"overlap {v[f'top{args.top}_overlap']:.2f}")

    # ---- 6. top predictors --------------------------------------------------
    print("\n" + "=" * 78)
    print(f"6. TOP {args.top} PREDICTORS BY |posterior mean b_bar|")
    print("=" * 78)
    lo_i, hi_i = credible_interval(chains, "b_bar")
    kap_mean = kap.mean(axis=0)
    print(f"   ({int(np.sum(lo_i * hi_i > 0))} of {K} intervals exclude zero)")
    for j in np.argsort(-np.abs(m_rh))[:args.top]:
        star = "*" if lo_i[j] * hi_i[j] > 0 else " "
        print(f"   {star} {names[j]:<24s} {m_rh[j]:+.5f}  "
              f"[{lo_i[j]:+.5f}, {hi_i[j]:+.5f}]   kappa {kap_mean[:, j].mean():.4f}")
    print("\n   NOTE: intervals are not significance tests. Counting stars across")
    print("   models compares priors, not evidence -- the baseline gave 0, 11, 8")
    print("   and 17 across four settings, non-monotonic in prior strength.")


if __name__ == "__main__":
    main()