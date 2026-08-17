"""
run_regularised_horseshoe.py

Production runner for the Regularised Horseshoe. Mirrors run_horseshoe.py in
structure and filename convention, so convergence.load_chains finds the output
unchanged and the two models' logs read alike.

WHAT THIS RUN HAS TO ANSWER, beyond the usual posterior summaries. Three
numbers, and they are printed together at the end:

  1. THE BINDING FRACTION on real data. The slab modifies the horseshoe only
     where tau^2 lambda^2 approaches c^2. Predicted at 12.91% from the prior
     at tau_0, but only ~0.70% if evaluated at the plain horseshoe's posterior
     tau; a 60-draw pilot gave 12.4%; the matched-dimension calibration gave
     61%. The production figure is the one that matters, and it is what the
     scope decision rests on: if the slab binds for almost nothing, this model
     IS the plain horseshoe and any difference between them is noise.

  2. TAU's ESS AND R-HAT, read against the plain horseshoe's 1,012 and 1.0015.
     Calibration at matched dimensions found tau's recovery correlation at
     -0.036 for this model against +0.996 for the plain horseshoe -- where the
     slab binds, tau*lambda~ -> c, so the coefficient scale is c REGARDLESS of
     tau and the global scale stops being identified through those
     coefficients. Coverage stayed honest (0.980) because the intervals widen
     correctly, but tau is not being recovered. Whether that persists at
     production's much lower binding fraction is the open question, and poor
     mixing is how weak identification usually shows.

  3. c, the slab width, which is a sampled parameter here and needs its own
     R-hat and interval.

The consequence for reporting: for the plain horseshoe, tau is a headline
result -- it lands at 5% of tau_0 with the interval excluding it. For this
model the interpretable quantities are c, the binding fraction, and the
shrinkage factors kappa, with tau reported and caveated. Do not read a
difference between the two models' tau as "the slab changes the global scale"
without first checking that tau is identified here.

cores DEFAULTS TO 1, unlike run_horseshoe.py's 4. PyMC's multiprocessing
killed a worker twice in this project, including a production run that lost a
process after 70 minutes with no traceback. At the measured 0.354 s/iteration
a four-chain sequential run costs about an hour, against three for the plain
horseshoe, so the parallel path buys little and has already cost a night.

Usage:
    caffeinate -i python3 -u run_regularised_horseshoe.py 2>&1 \
        | tee -a results/size_bm_25/regularised_horseshoe/run_log.txt

    python3 run_regularised_horseshoe.py --slab-scale 2.0   # P&V's default;
        # EVIDENCE, not a mistake -- it should bind for ~0.02 of 3,600
        # coefficients, demonstrating that applied verbatim the regularised
        # horseshoe IS the plain horseshoe at this data's scale
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import time

import numpy as np

from src.gibbs.io import load_draws, save_draws
from src.nuts.regularised_horseshoe import (RegHorseshoeDraws, binding_summary,
                                            reg_horseshoe_hyperparameters,
                                            run_nuts, shrinkage_factors)

MODEL = "regularised_horseshoe"


def load_predictor_names(universe: str, K: int) -> list[str]:
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
    path = Path("data/processed") / f"{universe}_arrays.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    d = np.load(path, allow_pickle=True)
    return d["F"], d["R"], d["asset_names"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="size_bm_25")
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--cores", type=int, default=1,
                    help="1 by default: PyMC's multiprocessing has killed a "
                         "worker twice in this project")
    ap.add_argument("--draws", type=int, default=1500,
                    help="RETAINED draws per chain, excluding tuning")
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--seed0", type=int, default=0,
                    help="chain k uses seed0 + k, and so does its FILENAME")
    ap.add_argument("--p0", type=int, default=23,
                    help="fixed a priori; NOT to be revised after seeing results")
    ap.add_argument("--target-r2", type=float, default=0.05)
    ap.add_argument("--nu", type=float, default=4.0,
                    help="slab degrees of freedom; dimensionless, so it "
                         "transports where s does not")
    ap.add_argument("--slab-scale", type=float, default=None,
                    help="s. Default derives it from p0. Pass 2.0 for P&V's "
                         "illustrative value, which should bind for ~nothing")
    ap.add_argument("--n-choice", choices=["T", "NT"], default="T")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    F, R, _ = load_universe(args.universe)
    N, T, K = F.shape
    hp = reg_horseshoe_hyperparameters(R, F, K, p0=args.p0,
                                       target_r2=args.target_r2, nu=args.nu,
                                       n_choice=args.n_choice,
                                       slab_scale=args.slab_scale)

    # no dots in the tag: Path.with_suffix("") once parsed "..._r2_0.05_chain0"
    # as stem "..._r2_0" plus suffix ".05_chain0"
    tag = f"p0_{args.p0}_r2_{args.target_r2:g}".replace(".", "p")
    if args.slab_scale is not None:
        tag += f"_s{args.slab_scale:g}".replace(".", "p")
    if args.n_choice != "T":
        tag += f"_n{args.n_choice}"

    outdir = Path(args.outdir or f"results/{args.universe}/{MODEL}")
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Regularised Horseshoe | {args.universe} | N={N} T={T} K={K}")
    print(f"  setting={tag}")
    print(f"  Delta_b_bar={hp.Delta_b_bar[0,0]:.4e}  (shared with all four models)")
    print(f"  sigma={hp.sigma_pooled:.4f}  sd_target={hp.sd_target:.4e}")
    print(f"  tau_0={hp.tau_0:.4e}  (p0={hp.p0} per asset, n={hp.n_choice})"
          f"   -- IDENTICAL to the plain horseshoe")
    print(f"  slab: s={hp.slab_scale:.4e}, nu={hp.slab_df:g}, "
          f"E[c]={hp.slab_sd:.4e} = {hp.slab_sd/hp.sd_target:.1f} x sd_target")
    print(f"  prior binding P(lam > c/tau): {100*hp.binding_fraction():.2f}% at "
          f"tau_0, {100*hp.binding_fraction(2.1019e-05):.2f}% at the plain "
          f"horseshoe's posterior tau")
    print(f"  nu_Sigma={hp.nu_Sigma:.0f}  V_Sigma=S_hat")
    print(f"  target_accept={hp.target_accept}  init={hp.init}  "
          f"max_treedepth={hp.max_treedepth}")
    print(f"  {args.chains} chains x {args.draws} retained draws "
          f"({args.tune} tune) on {args.cores} core(s)\n")

    t_all = time()
    chains = run_nuts(R, F, hp, n_draws=args.draws, n_tune=args.tune,
                      chains=args.chains, cores=args.cores, seed0=args.seed0)

    for k, draws in enumerate(chains):
        draws.meta["setting"] = tag
        draws.meta["target_r2"] = args.target_r2
        draws.meta["universe"] = args.universe
        # seed0 + k in the FILENAME, not k: run_bayesian_lasso.py uses the loop
        # index, so --seed0 4 silently overwrote chains 0-3
        out = save_draws(draws, outdir / f"{MODEL}_{tag}_chain{args.seed0 + k}")
        print(f"  chain {args.seed0 + k} (seed {draws.meta['seed']}): "
              f"tau {draws.tau.mean():.4e}, c {draws.c.mean():.4e}, "
              f"{draws.meta['divergences']} div, "
              f"depth {draws.meta['tree_depth_mean']:.2f}, "
              f"bind {100*draws.meta['frac_binding_0p99']:.1f}%, "
              f"{out.stat().st_size/1e6:.0f} MB -> {out}")

    print(f"\ntotal {time()-t_all:.0f}s")

    # ---- a first look ------------------------------------------------------
    d0 = load_draws(outdir / f"{MODEL}_{tag}_chain{args.seed0}", RegHorseshoeDraws)
    names = load_predictor_names(args.universe, K)
    tau_all = np.concatenate([c.tau for c in chains])
    c_all = np.concatenate([c.c for c in chains])

    lo, hi = np.percentile(tau_all, [2.5, 97.5])
    print(f"\ntau (all chains): {tau_all.mean():.4e}  [{lo:.4e}, {hi:.4e}]")
    print(f"   against tau_0 {hp.tau_0:.4e}  ->  ratio {tau_all.mean()/hp.tau_0:.4f}")
    print(f"   tau_0 inside the interval: {'YES' if lo <= hp.tau_0 <= hi else 'NO'}")
    print(f"   the plain horseshoe gave 2.1019e-05 (ratio 0.0533) on this universe")
    print("   [READ WITH CARE: matched-dimension calibration found tau's recovery")
    print("    correlation at -0.036 for this model against +0.996 for the plain")
    print("    horseshoe. Where the slab binds, tau*lam~ -> c, so the coefficient")
    print("    scale is c REGARDLESS of tau. Check tau's ESS in diagnose before")
    print("    reading any difference from the plain horseshoe as substantive.]")

    clo, chi = np.percentile(c_all, [2.5, 97.5])
    print(f"\nc  (all chains): {c_all.mean():.4e}  [{clo:.4e}, {chi:.4e}]")
    print(f"   prior E[c] {hp.slab_sd:.4e}  ->  ratio {c_all.mean()/hp.slab_sd:.4f}")
    print(f"   c/tau = {c_all.mean()/tau_all.mean():.1f}, so the slab bites at "
          f"lam above that")

    bs = binding_summary(d0, hp)
    print(f"\nSLAB BINDING (chain {args.seed0}) -- the diagnostic this model exists for")
    print(f"   lam~/lam < 0.99 for {100*bs['frac_binding']:.2f}% of local scales, "
          f"{bs['n_binding_per_draw']:.0f} of {N*K} per draw")
    print(f"   halved or more: {100*bs['frac_halved']:.2f}%   "
          f"median lam~/lam {bs['median_ratio']:.4f}   min {bs['min_ratio']:.4f}")
    print(f"   prior prediction at this tau: "
          f"{100*bs['prior_frac_at_posterior_tau']:.2f}%   at tau_0: "
          f"{100*bs['prior_frac_at_tau0']:.2f}%")
    if bs["frac_binding"] < 1e-3:
        print("   *** THE SLAB IS NOT BINDING. This model has collapsed into the")
        print("   plain horseshoe and any difference between them is noise. ***")

    kappa = shrinkage_factors(d0, F, hp)
    print(f"\nshrinkage factors kappa (chain {args.seed0}): mean {kappa.mean():.4f}, "
          f"range [{kappa.min():.4f}, {kappa.max():.4f}]")
    print(f"   effectively-free coefficients per asset: {(1-kappa.mean())*K:.2f}")
    print("   [the plain horseshoe gave mean 0.9885, range [0.0001, 1.0000],")
    print("    1.66 free per asset. kappa is the ONLY axis on which all four")
    print("    models are commensurable.]")

    m = d0.b_bar.mean(axis=0)
    print(f"\nposterior mean b_bar, 5 largest by |value| (chain {args.seed0}):")
    for j in np.argsort(-np.abs(m))[:5]:
        blo, bhi = np.percentile(d0.b_bar[:, j], [2.5, 97.5])
        star = "*" if blo * bhi > 0 else " "
        print(f"   {star} {names[j]:<24s} {m[j]:+.5f}  [{blo:+.5f}, {bhi:+.5f}]")
    print("   (* = interval excludes zero. Counting stars compares priors, not")
    print("    evidence, and is non-monotonic in prior strength.)")

    tot_div = sum(c.meta["divergences"] for c in chains)
    bfmi = " ".join(f"{c.meta['ebfmi']:.2f}" for c in chains)
    print(f"\ndivergences {tot_div} of {args.chains * args.draws} "
          f"({100*tot_div/(args.chains*args.draws):.1f}%)  |  E-BFMI {bfmi}")
    print("   [the plain horseshoe gave 0.9% on this universe. P&V report the")
    print("    regularised version as substantially more robust, so a lower rate")
    print("    here replicates their Sec. 4.2 claim on this data.]")
    sat = max(c.meta["tree_depth_saturating"] for c in chains)
    if sat > 0.0:
        print(f"  *** tree depth saturating on up to {sat:.0%} of draws: "
              f"trajectories truncated, and the backtest cost estimate no "
              f"longer holds. ***")


if __name__ == "__main__":
    main()