"""
run_baseline_gaussian.py

Production runner for the Gaussian baseline. Runs several independent
chains (different seeds, same data and hyperparameters) and saves each to
results/<universe>/.

Multiple chains are the point: R-hat compares within-chain to between-chain
variance and is undefined for a single chain. Each chain is independent, so
this is also the natural place to parallelise later if runtimes grow.

Usage:
    python run_baseline_gaussian.py                       # defaults
    python run_baseline_gaussian.py --chains 4 --draws 3000 --burn 1000
    python run_baseline_gaussian.py --universe size_bm_100
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import time

import numpy as np

from src.gibbs.baseline_gaussian import (default_hyperparameters, prior_implied_r2,
                                         rescaled_hyperparameters, run_gibbs)
from src.gibbs.io import save_draws


def load_predictor_names(universe: str, K: int) -> list[str]:
    """
    Predictor column names from the data pipeline's metadata sidecar, so
    output reads "DY_x_lag3" rather than "predictor 27". Falls back to
    indices if the file isn't where we expect -- a missing sidecar should
    make the report less readable, not stop a 45-minute run.
    """
    import json
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
    ap.add_argument("--draws", type=int, default=3000,
                    help="total sweeps per chain (Feng & He use 3000)")
    ap.add_argument("--burn", type=int, default=1000,
                    help="sweeps discarded per chain (Feng & He use 1000)")
    ap.add_argument("--seed0", type=int, default=0,
                    help="first chain's seed; chain k uses seed0 + k")
    ap.add_argument("--prior", choices=["feng_he", "rescaled"], default="feng_he",
                    help="feng_he = their disclosed values verbatim; "
                         "rescaled = same structure, calibrated to our predictor scale")
    ap.add_argument("--target-r2", type=float, default=0.05,
                    help="prior implied R^2, used only when --prior rescaled")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    F, R, asset_names = load_universe(args.universe)
    N, T, K = F.shape
    if args.prior == "rescaled":
        hp = rescaled_hyperparameters(R, F, K, target_r2=args.target_r2)
        tag = f"rescaled_r2_{args.target_r2:g}".replace(".", "p")
    else:
        hp = default_hyperparameters(R, K)
        tag = "feng_he"

    outdir = Path(args.outdir or f"results/{args.universe}/baseline_gaussian")
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Gaussian baseline | {args.universe} | N={N} T={T} K={K}")
    print(f"  prior={tag}  nu_b={hp.nu_b:.0f}  V_b={hp.V_b[0,0]:.3e}  "
          f"Delta_b_bar={hp.Delta_b_bar[0,0]:.3e}")
    print(f"  nu_Sigma={hp.nu_Sigma:.0f}  V_Sigma=S_hat  |  "
          f"prior implied R^2 = {prior_implied_r2(hp, R, F):.4g}")
    print(f"  {args.chains} chains x {args.draws} sweeps ({args.burn} burn-in)\n")

    t_all = time()
    for k in range(args.chains):
        seed = args.seed0 + k
        t0 = time()
        draws = run_gibbs(R, F, hp, n_draws=args.draws, n_burn=args.burn,
                          seed=seed, progress_every=max(args.draws // 10, 1))
        draws.meta["prior"] = tag
        draws.meta["target_r2"] = args.target_r2 if args.prior == "rescaled" else None
        # seed0 + k in the FILENAME, not k, so --seed0 4 cannot overwrite
        # chains 0-3. With the default seed0 = 0 the names are unchanged.
        out = save_draws(draws, outdir / f"baseline_gaussian_{tag}_chain{seed}")
        print(f"  chain {seed} (seed {seed}): {time()-t0:.0f}s, "
              f"{out.stat().st_size/1e6:.0f} MB -> {out}")

    print(f"\ntotal {time()-t_all:.0f}s")
    print(f"posterior mean b_bar, 5 largest by |value| (chain {args.seed0}):")
    from src.gibbs.io import load_draws
    from src.gibbs.baseline_gaussian import GibbsDraws
    d0 = load_draws(outdir / f"baseline_gaussian_{tag}_chain{args.seed0}", GibbsDraws)
    names = load_predictor_names(args.universe, K)
    m = d0.b_bar.mean(axis=0)
    for j in np.argsort(-np.abs(m))[:5]:
        lo, hi = np.percentile(d0.b_bar[:, j], [2.5, 97.5])
        star = "*" if lo * hi > 0 else " "
        print(f"   {star} {names[j]:<22s} {m[j]:+.5f}  [{lo:+.5f}, {hi:+.5f}]")
    print("   (* = 95% credible interval excludes zero)")


if __name__ == "__main__":
    main()