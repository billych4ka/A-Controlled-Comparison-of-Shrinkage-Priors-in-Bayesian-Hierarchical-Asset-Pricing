"""
run_horseshoe.py

Production runner for the Horseshoe. Draws all chains in a single pm.sample
call so PyMC can run them in parallel across cores, then saves each to
results/<universe>/horseshoe/ under the same filename convention every other
model uses, so convergence.load_chains finds them unchanged.

Multiple chains are the point: R-hat compares within-chain to between-chain
variance and is undefined for a single chain. They matter more here than for
the Gibbs models, because tau is by far the slowest-mixing parameter and a
single chain would give no way to tell a converged tau from a stuck one.

Draw budget
-----------
--draws is RETAINED draws per chain, EXCLUDING tuning. This is the opposite of
the Gibbs runners, where --draws includes burn-in. PyMC's argument means
retained and silently redefining it would be worse than the inconsistency, but
it does mean the thesis's budget table is comparing different units and should
report achieved ESS alongside.

The default 1,500 with 1,000 tune is set by TAU, not by B. Measured on the
full sample:

    b_bar        ESS ~282 per 200 draws (superefficient, as NUTS often is)
    tau          ESS ~14  per 200 draws  -> ~5,700 retained draws for ESS 400

4 chains x 1,500 clears tau's target with margin. Everything else is far past
its own by then.

Sampler settings come from HorseshoeHyperparams, not from arguments here, so a
backtest fit cannot silently differ from the production run: target_accept =
0.99, init = "adapt_diag", max_treedepth = 10. NEVER "jitter+adapt_diag" --
its U(-1,1) jitter is ~1,300 prior standard deviations on b_bar and drove the
step size to 9.1e-22 in an early run, a sampler that never moved while
reporting a fast fit.

Usage:
    python3 run_horseshoe.py                                # defaults
    python3 run_horseshoe.py --chains 4 --draws 1500 --tune 1000
    python3 run_horseshoe.py --p0 12                        # sensitivity
    python3 run_horseshoe.py --n-choice NT                  # sensitivity
    python3 run_horseshoe.py --universe size_op_25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import time

import numpy as np

from src.gibbs.io import load_draws, save_draws
from src.nuts.horseshoe import (HorseshoeDraws, ebfmi, horseshoe_hyperparameters,
                                implied_m_eff, run_nuts, shrinkage_factors)

MODEL = "horseshoe"


def load_predictor_names(universe: str, K: int) -> list[str]:
    """Predictor names from the pipeline's metadata sidecar, so output reads
    'RMW_x_mom_12_1' rather than 'predictor 127'. Falls back to indices: a
    missing sidecar should make the report less readable, not stop the run."""
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
        raise FileNotFoundError(
            f"{path} not found. Available: "
            f"{sorted(p.name for p in Path('data/processed').glob('*_arrays.npz'))}"
        )
    d = np.load(path, allow_pickle=True)
    return d["F"], d["R"], d["asset_names"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="size_bm_25")
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--draws", type=int, default=1500,
                    help="RETAINED draws per chain, excluding tuning; sized for "
                         "tau's ESS, not b_bar's")
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--seed0", type=int, default=0,
                    help="first chain's seed; chain k uses seed0 + k, and so "
                         "does its FILENAME")
    ap.add_argument("--p0", type=int, default=23,
                    help="fixed a priori; NOT to be revised after seeing "
                         "results. 12 is the pre-declared sensitivity")
    ap.add_argument("--n-choice", choices=["T", "NT"], default="T",
                    help="which sample size enters tau_0; NT is the disclosed "
                         "robustness setting")
    ap.add_argument("--target-r2", type=float, default=0.05,
                    help="sets Delta_b_bar, shared with all four models; fixed "
                         "a priori")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    F, R, asset_names = load_universe(args.universe)
    N, T, K = F.shape
    hp = horseshoe_hyperparameters(R, F, K, p0=args.p0,
                                   target_r2=args.target_r2,
                                   n_choice=args.n_choice)

    # the setting tag must never contain a dot: Path.with_suffix("") once
    # parsed "..._r2_0.05_chain0" as stem "..._r2_0" plus suffix
    # ".05_chain0", which would have silently overwritten three of four chains
    tag = f"p0_{args.p0}_r2_{args.target_r2:g}".replace(".", "p")
    if args.n_choice != "T":
        tag += f"_n{args.n_choice}"

    outdir = Path(args.outdir or f"results/{args.universe}/{MODEL}")
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Horseshoe | {args.universe} | N={N} T={T} K={K}")
    print(f"  setting={tag}")
    print(f"  Delta_b_bar={hp.Delta_b_bar[0,0]:.4e}  (shared with all four models)")
    print(f"  sigma={hp.sigma_pooled:.4f} (pooled, df-corrected)  "
          f"sd_target={hp.sd_target:.4e}")
    print(f"  tau_0={hp.tau_0:.4e}  (p0={hp.p0} per asset, n={hp.n_choice})"
          f"   tau_0/sd_target={hp.tau0_over_sd_target:.3f}")
    print(f"  prior implied active per asset = {implied_m_eff(hp, F):.2f} of {K}"
          f"   ({implied_m_eff(hp, F) * N:.0f} of {N*K} in total)")
    print(f"  nu_Sigma={hp.nu_Sigma:.0f}  V_Sigma=S_hat")
    print(f"  target_accept={hp.target_accept}  init={hp.init}  "
          f"max_treedepth={hp.max_treedepth}")
    print(f"  {args.chains} chains x {args.draws} retained draws "
          f"({args.tune} tune) on {args.cores} cores\n")

    t_all = time()
    chains = run_nuts(R, F, hp, n_draws=args.draws, n_tune=args.tune,
                      chains=args.chains, cores=args.cores, seed0=args.seed0)

    for k, draws in enumerate(chains):
        draws.meta["setting"] = tag
        draws.meta["target_r2"] = args.target_r2
        draws.meta["universe"] = args.universe
        # seed0 + k in the FILENAME, not k. run_bayesian_lasso.py uses the loop
        # index, so --seed0 4 silently overwrote chains 0-3.
        out = save_draws(draws, outdir / f"{MODEL}_{tag}_chain{args.seed0 + k}")
        print(f"  chain {args.seed0 + k} (seed {draws.meta['seed']}): "
              f"tau {draws.tau.mean():.4e}, {draws.meta['divergences']} div, "
              f"depth {draws.meta['tree_depth_mean']:.2f}, "
              f"E-BFMI {draws.meta['ebfmi']:.2f}, "
              f"{out.stat().st_size/1e6:.0f} MB -> {out}")

    print(f"\ntotal {time()-t_all:.0f}s")

    # ---- a first look, so an hour-long run is not opaque until diagnose runs --
    d0 = load_draws(outdir / f"{MODEL}_{tag}_chain{args.seed0}", HorseshoeDraws)
    names = load_predictor_names(args.universe, K)
    tau_all = np.concatenate([c.tau for c in chains])

    lo, hi = np.percentile(tau_all, [2.5, 97.5])
    print(f"\ntau (all chains): {tau_all.mean():.4e}  [{lo:.4e}, {hi:.4e}]")
    print(f"   against tau_0 {hp.tau_0:.4e}  ->  ratio {tau_all.mean()/hp.tau_0:.4f}")
    print(f"   tau_0 inside the interval: {'YES' if lo <= hp.tau_0 <= hi else 'NO'}")
    print(f"   implied active per asset at the posterior mean tau: "
          f"{implied_m_eff(hp, F, tau=tau_all.mean()):.2f}"
          f"   (prior centred on {implied_m_eff(hp, F):.2f})")

    kappa = shrinkage_factors(d0, F, hp)
    print(f"\nshrinkage factors kappa (chain {args.seed0}): mean {kappa.mean():.4f}, "
          f"range [{kappa.min():.4f}, {kappa.max():.4f}]")
    print(f"   implied effectively-free coefficients per asset: "
          f"{(1 - kappa.mean()) * K:.2f}")
    print("   [kappa near 1 = shrunk to the common b_bar; near 0 = left free.")
    print("    THIS is the axis on which the four models are comparable --")
    print("    tau and the LASSO's lambda are not commensurable.]")

    m = d0.b_bar.mean(axis=0)
    print(f"\nposterior mean b_bar, 5 largest by |value| (chain {args.seed0}):")
    for j in np.argsort(-np.abs(m))[:5]:
        blo, bhi = np.percentile(d0.b_bar[:, j], [2.5, 97.5])
        star = "*" if blo * bhi > 0 else " "
        print(f"   {star} {names[j]:<24s} {m[j]:+.5f}  [{blo:+.5f}, {bhi:+.5f}]")
    print("   (* = 95% credible interval excludes zero. Interval counts compare")
    print("    priors, not evidence, and are non-monotonic in prior strength.)")

    tot_div = sum(c.meta["divergences"] for c in chains)
    bfmi_str = " ".join(f"{c.meta['ebfmi']:.2f}" for c in chains)
    print(f"\ndivergences {tot_div} of {args.chains * args.draws} "
          f"({100*tot_div/(args.chains*args.draws):.1f}%)  |  E-BFMI {bfmi_str}")
    sat = max(c.meta["tree_depth_saturating"] for c in chains)
    if sat > 0.0:
        print(f"  *** tree depth saturating on up to {sat:.0%} of draws: "
              f"trajectories are being truncated, and the cost estimate for the "
              f"backtest no longer holds. ***")
    print("  [P&V report 1-30% divergences for the plain horseshoe on their own")
    print("   datasets. Report the rate; it is documented behaviour, not a fault.]")


if __name__ == "__main__":
    main()