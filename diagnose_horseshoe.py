"""
diagnose_horseshoe.py

Reads every saved Horseshoe run for one universe and prints what the write-up
needs:

  1. convergence      -- R-hat and ESS, including ALL 3,600 local scales
  2. NUTS diagnostics -- divergences, tree-depth saturation, E-BFMI, step size,
                         per chain
  3. tau against tau_0 -- the headline: how far the data moves the global scale
  4. shrinkage factors kappa -- the ONE axis on which all four models compare
  5. cross-model agreement on b_bar, against the baseline and the LASSO
  6. top predictors with credible intervals

Section 1 differs from the LASSO's in an important way. The LASSO had to
summarise tau^2 to running means plus 200 tracked traces, so its convergence
diagnostics cover a SUBSAMPLE. The horseshoe stores all 3,600 local scales in
full, so R-hat and ESS are reported for every one. That is a stricter standard
than the LASSO is held to -- a difference in how the models are CHECKED, not in
how they are specified, and it should be stated rather than left for a reader
to notice that one model reports 3,600 R-hats and the other 200.

Section 4 is the section the whole design has been pointing at. tau and the
LASSO's lambda are not commensurable -- one is a global scale multiplying
heavy-tailed local scales, the other the rate of an exponential on
per-coefficient variances. But every model in the design implies a shrinkage
factor

    kappa_ij = 1 / (1 + n sigma^-2 v_ij s_j^2)

where v_ij is whatever per-coefficient prior variance that model puts on
theta_ij: Delta_b_jj for the Gaussian baseline (the same for every asset,
because it has no local layer at all), s^2 tau^2_ij for the LASSO, and
tau^2 lambda^2_ij for the horseshoe. kappa near 1 is complete shrinkage toward
the common b_bar; near 0 is a coefficient left free.

Usage:
    python3 diagnose_horseshoe.py
    python3 diagnose_horseshoe.py --universe size_op_25 --top 20
    python3 diagnose_horseshoe.py --settings p0_23_r2_0p05 p0_12_r2_0p05
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
from src.nuts.horseshoe import (HorseshoeDraws, ebfmi, horseshoe_hyperparameters,
                                implied_m_eff, shrinkage_factors)

MODEL = "horseshoe"
DEFAULT_SETTING = "p0_23_r2_0p05"
BASELINE, BASELINE_SETTING = "baseline_gaussian", "rescaled_r2_0p05"
LASSO, LASSO_SETTING = "bayesian_lasso", "rescaled_r2_0p05"


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


def load_universe(universe: str):
    d = np.load(Path("data/processed") / f"{universe}_arrays.npz", allow_pickle=True)
    return d["F"], d["R"]


def kappa_from_variance(v: np.ndarray, F: np.ndarray, sigma: float) -> np.ndarray:
    """kappa = 1/(1 + n sigma^-2 v s_j^2), with s_j the root MEAN SQUARE of
    predictor j (not its standard deviation -- the intercept column has mean 1
    and sd exactly 0, and would otherwise be assigned no information)."""
    N, T, K = F.shape
    s = np.sqrt((F ** 2).mean(axis=1))
    return 1.0 / (1.0 + (T / sigma ** 2) * v * s ** 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="size_bm_25")
    ap.add_argument("--settings", nargs="*", default=[DEFAULT_SETTING])
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--fields", nargs="*",
                    default=["b_bar", "tau", "Sigma", "lam_local"],
                    help="B is omitted by default: 3,600 parameters, slow")
    args = ap.parse_args()

    loaded, missing = {}, []
    for s in args.settings:
        try:
            loaded[s] = load_chains(args.universe, MODEL, s, HorseshoeDraws)
        except FileNotFoundError:
            missing.append(s)
    if missing:
        print(f"(not found, skipping: {', '.join(missing)})\n")
    if not loaded:
        raise SystemExit("no runs found -- check --universe and the results layout")

    primary = args.settings[0] if args.settings[0] in loaded else next(iter(loaded))
    chains = loaded[primary]
    K = chains[0].b_bar.shape[1]
    N = chains[0].Sigma.shape[1]
    names = predictor_names(args.universe, K)
    meta = chains[0].meta
    F, R = load_universe(args.universe)
    hp = horseshoe_hyperparameters(R, F, K, p0=int(meta.get("p0", 23)),
                                   target_r2=float(meta.get("target_r2", 0.05)),
                                   n_choice=str(meta.get("n_choice", "T")))

    print("=" * 78)
    print(f"HORSESHOE | {args.universe} | {primary}")
    print("=" * 78)
    print(f"   {len(chains)} chains x {chains[0].b_bar.shape[0]} retained draws "
          f"({meta.get('n_tune')} tune)")
    print(f"   tau_0={hp.tau_0:.4e}  (p0={hp.p0} per asset, n={hp.n_choice})   "
          f"sigma={hp.sigma_pooled:.4f}")
    print(f"   target_accept={meta.get('target_accept')}  "
          f"init={meta.get('init')}  max_treedepth={meta.get('max_treedepth')}")

    # ---- 1. convergence ---------------------------------------------------
    print("\n" + "=" * 78)
    print("1. CONVERGENCE   (R-hat < 1.01, Vehtari et al. 2021 rank-normalised")
    print("   folded split-R-hat; ESS target 400 for posterior means)")
    print("=" * 78)
    for field in args.fields:
        if not hasattr(chains[0], field):
            continue
        d = diagnose(chains, field)
        print(f"   {d.summary()}")
    print(f"\n   NOTE: lam_local covers all {N*K} local scales, not a subsample.")
    print("   The LASSO's tau^2 diagnostics cover 200 tracked traces, because")
    print("   Gibbs needed far more draws for the same ESS and the full array")
    print("   could not be stored. The horseshoe is therefore held to a")
    print("   STRICTER diagnostic -- a difference in checking, not in design.")

    # ---- 2. NUTS diagnostics ----------------------------------------------
    print("\n" + "=" * 78)
    print("2. NUTS DIAGNOSTICS, per chain")
    print("=" * 78)
    print(f"   {'chain':<7}{'seed':>5}{'div':>7}{'div %':>8}{'depth':>8}"
          f"{'sat %':>8}{'step':>10}{'E-BFMI':>9}")
    n_draws = chains[0].b_bar.shape[0]
    max_depth = int(meta.get("max_treedepth", 10))
    for k, c in enumerate(chains):
        div = int(c.diverging.sum())
        sat = float(np.mean(c.tree_depth >= max_depth))
        print(f"   {k:<7d}{c.meta.get('seed', k):>5}{div:>7d}"
              f"{100*div/n_draws:>7.1f}%{np.mean(c.tree_depth):>8.2f}"
              f"{100*sat:>7.0f}%{c.step_size[-1]:>10.4f}{ebfmi(c.energy):>9.2f}")
    tot = sum(int(c.diverging.sum()) for c in chains)
    print(f"   {'total':<7}{'':>5}{tot:>7d}{100*tot/(len(chains)*n_draws):>7.1f}%")
    worst = max(float(np.mean(c.tree_depth >= max_depth)) for c in chains)
    if worst > 0.5:
        print(f"\n   *** a chain saturates tree depth on {worst:.0%} of draws.")
        print("   Trajectories are being truncated, which is a bias risk, not just")
        print("   a cost. Check that chain's b_bar and tau against the others")
        print("   before reporting -- if they agree, report the saturation with")
        print("   the evidence that it did not matter. ***")
    print("\n   [P&V report 1-30% divergences for the ORIGINAL horseshoe across")
    print("    four real datasets, calling it a lot and warning that biased")
    print("    inference is a concern. This rate is citable behaviour, not a")
    print("    fault -- and their recommended remedy is the regularised")
    print("    horseshoe, which is model 4.]")

    # ---- 3. tau ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("3. THE GLOBAL SCALE   tau against its calibrated tau_0")
    print("=" * 78)
    tau_pooled = np.concatenate([c.tau for c in chains])
    lo, hi = np.percentile(tau_pooled, [2.5, 97.5])
    d_tau = diagnose(chains, "tau")
    print(f"   tau     {tau_pooled.mean():.4e}  [{lo:.4e}, {hi:.4e}]")
    print(f"   tau_0   {hp.tau_0:.4e}   ->  ratio {tau_pooled.mean()/hp.tau_0:.4f}")
    print(f"   tau_0 inside the 95% interval: "
          f"{'YES' if lo <= hp.tau_0 <= hi else 'NO'}")
    mcse = tau_pooled.std() / np.sqrt(max(getattr(d_tau, 'ess_bulk', np.array([1.0])).min(), 1))
    print(f"   distance of tau_0 from the posterior mean: "
          f"{(hp.tau_0 - tau_pooled.mean())/tau_pooled.std():.1f} posterior sd")
    print(f"\n   implied active coefficients per asset (P&V Eq. 3.8):")
    print(f"      at tau_0 (the prior's centre)      {implied_m_eff(hp, F):.2f} of {K}")
    print(f"      at the posterior mean tau          "
          f"{implied_m_eff(hp, F, tau=tau_pooled.mean()):.2f}")
    print(f"      across the 95% interval            "
          f"{implied_m_eff(hp, F, tau=lo):.2f} to {implied_m_eff(hp, F, tau=hi):.2f}")
    print("\n   [The LASSO's lambda moved ~30% from its calibrated value. tau moves")
    print("    by a factor of ~20, in the opposite direction -- and that is NOT a")
    print("    contradiction: the LASSO's single scale sets one Laplace width for")
    print("    every coefficient, while the horseshoe buys sparsity by shrinking")
    print("    the bulk and letting the Cauchy tails carry the exceptions.]")

    # ---- 4. shrinkage factors ---------------------------------------------
    print("\n" + "=" * 78)
    print("4. SHRINKAGE FACTORS kappa   -- the axis on which all models compare")
    print("=" * 78)
    kap = np.concatenate([shrinkage_factors(c, F, hp) for c in chains], axis=0)
    kap_mean = kap.mean(axis=0)
    print(f"   horseshoe : mean {kap.mean():.4f}   "
          f"[{kap.min():.4f}, {kap.max():.4f}]   "
          f"-> {(1-kap.mean())*K:.2f} effectively-free per asset")
    for q in (1, 5, 25, 50):
        print(f"      {q:2d}th percentile of kappa across coefficients: "
              f"{np.percentile(kap_mean, q):.4f}")

    try:
        bl = load_chains(args.universe, BASELINE, BASELINE_SETTING, GibbsDraws)
        v_bl = posterior_mean(bl, "Delta_b_diag")
        k_bl = kappa_from_variance(np.tile(v_bl, (N, 1)), F, hp.sigma_pooled)
        print(f"   baseline  : mean {k_bl.mean():.4f}   "
              f"[{k_bl.min():.4f}, {k_bl.max():.4f}]   "
              f"-> {(1-k_bl.mean())*K:.2f} effectively-free per asset")
        print("               (varies only with the predictor scale s_j: the")
        print("                baseline has NO local layer, so every asset gets")
        print("                the same shrinkage on the same predictor)")
    except (FileNotFoundError, AttributeError, KeyError) as exc:
        print(f"   baseline  : not comparable ({type(exc).__name__})")

    try:
        la = load_chains(args.universe, LASSO, LASSO_SETTING, LassoDraws)
        s_la = float(la[0].meta.get("s", np.nan))
        v_la = s_la ** 2 * la[0].tau2_mean
        k_la = kappa_from_variance(v_la, F, hp.sigma_pooled)
        print(f"   LASSO     : mean {k_la.mean():.4f}   "
              f"[{k_la.min():.4f}, {k_la.max():.4f}]   "
              f"-> {(1-k_la.mean())*K:.2f} effectively-free per asset")
    except (FileNotFoundError, AttributeError, KeyError) as exc:
        print(f"   LASSO     : not comparable ({type(exc).__name__})")

    print("\n   [kappa near 1 = shrunk to the common b_bar; near 0 = left free.")
    print("    The baseline's spread comes ENTIRELY from the predictor scales;")
    print("    the two shrinkage models add a local layer beneath a global one.")
    print("    This is the sharper claim the design isolates: the baseline has")
    print("    no local adaptivity at all.]")

    # ---- 5. cross-model agreement on b_bar --------------------------------
    print("\n" + "=" * 78)
    print(f"5. CROSS-MODEL AGREEMENT ON b_bar   (top-{args.top})")
    print("=" * 78)
    w = within_setting_rank_stability(chains, top=args.top, n_reps=40)
    print(f"   within-horseshoe ceiling (Monte Carlo alone): spearman "
          f"{w['spearman_median']:+.3f}, overlap "
          f"{w[f'top{args.top}_overlap_median']:.2f}")
    m_hs = posterior_mean(chains, "b_bar")
    others = {}
    for label, model, setting, cls in (("baseline", BASELINE, BASELINE_SETTING, GibbsDraws),
                                       ("LASSO", LASSO, LASSO_SETTING, LassoDraws)):
        try:
            others[label] = load_chains(args.universe, model, setting, cls)
        except FileNotFoundError:
            print(f"   {label}: not found, skipping")
    for label, ch in others.items():
        m_o = posterior_mean(ch, "b_bar")
        corr = float(np.corrcoef(m_hs, m_o)[0, 1])
        ov = len(set(np.argsort(-np.abs(m_hs))[:args.top])
                 & set(np.argsort(-np.abs(m_o))[:args.top])) / args.top
        print(f"   horseshoe vs {label:<9s} correlation {corr:+.4f}   "
              f"top-{args.top} overlap {ov:.2f}")
    if others:
        allm = {"horseshoe": chains, **others}
        print(f"\n   pairwise Spearman on |b_bar|:")
        for (a, b), v in rank_stability(allm, top=args.top).items():
            print(f"      {a:<10s} vs {b:<10s} spearman {v['spearman']:+.3f}   "
                  f"overlap {v[f'top{args.top}_overlap']:.2f}")
        print("\n   [Compare the between-model overlap against the within-horseshoe")
        print("    ceiling above. An overlap close to it means the models agree as")
        print("    well as one model agrees with itself -- three prior families")
        print("    across two sampling paradigms is far stronger evidence about")
        print("    the data than any one of them alone.]")

    # ---- 6. top predictors -------------------------------------------------
    print("\n" + "=" * 78)
    print(f"6. TOP {args.top} PREDICTORS BY |posterior mean b_bar|")
    print("=" * 78)
    lo_i, hi_i = credible_interval(chains, "b_bar")
    excl = int(np.sum(lo_i * hi_i > 0))
    print(f"   ({excl} of {K} intervals exclude zero)")
    for j in np.argsort(-np.abs(m_hs))[:args.top]:
        star = "*" if lo_i[j] * hi_i[j] > 0 else " "
        print(f"   {star} {names[j]:<24s} {m_hs[j]:+.5f}  "
              f"[{lo_i[j]:+.5f}, {hi_i[j]:+.5f}]   kappa {kap_mean[:, j].mean():.4f}")

    print("\n   NOTE: intervals are not significance tests. A tighter prior gives")
    print("   narrower intervals AND smaller point estimates, so counting stars")
    print("   across models compares priors, not evidence -- the baseline gave")
    print("   0, 11, 8 and 17 across four settings, non-monotonic in prior")
    print("   strength. Predictor selection comes from out-of-sample performance.")


if __name__ == "__main__":
    main()

