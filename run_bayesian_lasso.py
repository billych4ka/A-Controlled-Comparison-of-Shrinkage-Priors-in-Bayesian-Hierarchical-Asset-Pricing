"""
run_bayesian_lasso.py

Production runner for the Bayesian LASSO. Runs several independent chains
(different seeds, same data and hyperparameters) and saves each to
results/<universe>/bayesian_lasso/.

Multiple chains are the point: R-hat compares within-chain to between-chain
variance and is undefined for a single chain. They also address the one risk
accepted when the plug-in scale was chosen over Park & Casella's sampled
sigma^2 -- the loss of their formal unimodality guarantee. Multimodality would
show up as chains disagreeing, which is exactly what R-hat measures.

Sweep budget
------------
The default 4,000 sweeps with 1,000 burn-in is set by LAMBDA, not by B.
Measured on the full sample over 1,600 post-burn-in draws:

    ||b_bar||   lag-1 autocorrelation 0.085,  ESS 1271  -> 503 sweeps for ESS 400
    ||B||                             0.084,  ESS 1271  -> 504
    Sigma[0,0]                       -0.016,  ESS 1600  -> 400
    lambda                            0.934,  ESS   52  -> ~12,300

lambda remains the slowest-mixing parameter even after its update was
collapsed over tau^2 (see sample_lambda_collapsed): collapsing removed a
1,500-sweep drift and improved the autocorrelation from 0.973, but lambda
stays coupled to theta through sum|theta_ij|. Four chains x 3,000 kept draws
gives lambda an ESS near 390. Everything else is far past its target by then.

Usage:
    python run_bayesian_lasso.py                          # defaults
    python run_bayesian_lasso.py --chains 4 --draws 4000 --burn 1000
    python run_bayesian_lasso.py --lambda-multiplier 0.1  # hyperprior sensitivity
    python run_bayesian_lasso.py --universe size_bm_100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import time

import numpy as np

from src.gibbs.bayesian_lasso import (LassoDraws, lasso_hyperparameters,
                                      lasso_prior_implied_r2, run_gibbs)
from src.gibbs.io import load_draws, save_draws

MODEL = "bayesian_lasso"


def load_predictor_names(universe: str, K: int) -> list[str]:
    """
    Predictor column names from the data pipeline's metadata sidecar, so
    output reads "DY_x_lag3" rather than "predictor 27". Falls back to
    indices if the file isn't where we expect -- a missing sidecar should
    make the report less readable, not stop an hour-long run.
    """
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
    """Load a prepared dataset from data/processed/<universe>_arrays.npz."""
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
    ap.add_argument("--draws", type=int, default=4000,
                    help="total sweeps per chain; sized for lambda's ESS, not B's")
    ap.add_argument("--burn", type=int, default=1000)
    ap.add_argument("--seed0", type=int, default=0,
                    help="first chain's seed; chain k uses seed0 + k")
    ap.add_argument("--target-r2", type=float, default=0.05,
                    help="fixed a priori; NOT to be revised after seeing results")
    ap.add_argument("--lambda-multiplier", type=float, default=1.0,
                    help="scales the Gamma hyperprior's centre; 0.1 and 10 are the "
                         "sensitivity settings, which should barely move the posterior")
    ap.add_argument("--track-tau2", type=int, default=200,
                    help="how many tau^2 entries get full traces (the rest are "
                         "summarised by running means, which are exact)")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    F, R, asset_names = load_universe(args.universe)
    N, T, K = F.shape
    hp = lasso_hyperparameters(R, F, K, target_r2=args.target_r2,
                               lambda_prior_multiplier=args.lambda_multiplier)

    tag = f"rescaled_r2_{args.target_r2:g}".replace(".", "p")
    if args.lambda_multiplier != 1.0:
        # the setting tag must never contain a dot: Path.with_suffix("") once
        # parsed "..._r2_0.05_chain0" as stem "..._r2_0" + suffix ".05_chain0",
        # which would have silently overwritten three of four chains
        tag += f"_lam{args.lambda_multiplier:g}".replace(".", "p")

    outdir = Path(args.outdir or f"results/{args.universe}/{MODEL}")
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Bayesian LASSO | {args.universe} | N={N} T={T} K={K}")
    print(f"  setting={tag}")
    print(f"  Delta_b_bar={hp.Delta_b_bar[0,0]:.4e}  (shared with all four models)")
    print(f"  s={hp.s:.4f} (pooled, df-corrected)  sd_target={hp.sd_target:.4e}")
    print(f"  lambda hyperprior: r={hp.lambda_r:g}, delta={hp.lambda_delta:.4e}"
          f"  -> prior rms lambda {hp.lambda_prior_rms:.1f}")
    print(f"  nu_Sigma={hp.nu_Sigma:.0f}  V_Sigma=S_hat  |  "
          f"prior implied R^2 = {lasso_prior_implied_r2(hp, R, F):.4g}")
    print(f"  {args.chains} chains x {args.draws} sweeps ({args.burn} burn-in)\n")

    t_all = time()
    for k in range(args.chains):
        seed = args.seed0 + k
        t0 = time()
        draws = run_gibbs(R, F, hp, n_draws=args.draws, n_burn=args.burn,
                          seed=seed, n_track_tau2=args.track_tau2,
                          progress_every=max(args.draws // 10, 1))
        draws.meta["setting"] = tag
        draws.meta["target_r2"] = args.target_r2
        draws.meta["lambda_multiplier"] = args.lambda_multiplier
        draws.meta["universe"] = args.universe
        # seed0 + k in the FILENAME, not k, so --seed0 4 cannot overwrite
        # chains 0-3. With the default seed0 = 0 the names are unchanged.
        out = save_draws(draws, outdir / f"{MODEL}_{tag}_chain{seed}")
        print(f"  chain {seed} (seed {seed}): {time()-t0:.0f}s, "
              f"lambda mean {draws.lam.mean():.1f}, "
              f"{out.stat().st_size/1e6:.0f} MB -> {out}")

    print(f"\ntotal {time()-t_all:.0f}s")

    # ---- a first look, so an hour-long run is not opaque until diagnose runs --
    d0 = load_draws(outdir / f"{MODEL}_{tag}_chain{args.seed0}", LassoDraws)
    names = load_predictor_names(args.universe, K)

    print(f"\nlambda (chain {args.seed0}): {d0.lam.mean():.1f} +- {d0.lam.std():.1f}"
          f"   [prior rms {hp.lambda_prior_rms:.1f}]")
    print(f"  implied theta prior sd {np.sqrt(2)*hp.s/d0.lam.mean():.4e}"
          f"   vs the baseline's sd_target {hp.sd_target:.4e}")
    print("  [baseline-equivalent scales: lambda 425 ~ target_r2 0.10,")
    print("   601 ~ 0.05, 1345 ~ 0.01. Higher lambda = more shrinkage.]")

    m = d0.b_bar.mean(axis=0)
    print(f"\nposterior mean b_bar, 5 largest by |value| (chain {args.seed0}):")
    for j in np.argsort(-np.abs(m))[:5]:
        lo, hi = np.percentile(d0.b_bar[:, j], [2.5, 97.5])
        star = "*" if lo * hi > 0 else " "
        print(f"   {star} {names[j]:<22s} {m[j]:+.5f}  [{lo:+.5f}, {hi:+.5f}]")
    print("   (* = 95% credible interval excludes zero; NOT a significance test --")
    print("    a tighter prior narrows intervals and shrinks point estimates at the")
    print("    same time, so counting stars compares priors, not evidence)")

    pred = np.einsum("itk,ik->it", F, d0.B.mean(axis=0))
    print(f"\nprediction volatility: sd(fitted)/sd(realised) = {pred.std()/R.std():.3f}")
    print("   [baseline in-sample: feng_he 0.58, r2_0p05 0.12, r2_0p01 0.06]")

    print(f"\nRun diagnose_bayesian_lasso.py for R-hat, ESS and the full picture.")


if __name__ == "__main__":
    main()

