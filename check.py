"""
check_chunk3.py -- verify run_nuts, HorseshoeDraws and the io round-trip.

Deliberately a SHORT run (default 60 draws, 200 tune, 2 chains, ~3 minutes).
This chunk is plumbing: shapes, seeds, filenames, round-tripping and the
kappa identity. Whether the posterior is CORRECT is checked by chunk 2's
log-density test and later by calibration; whether it has CONVERGED is a
production-run question. Sampling long here would test neither and cost 40
minutes.

What each check is for:

  shapes and dtypes      -- that the split by chain took the right axis. An
                            off-by-one on the chain axis gives arrays of the
                            right shape containing the wrong chain, which no
                            shape assertion catches, so the seeds are checked
                            separately and the chains are required to DIFFER.
  chains differ          -- two chains with different seeds must not be
                            identical. If run_nuts ever collapsed to writing
                            one chain four times, R-hat would read 1.0000 and
                            certify perfect convergence. That failure is
                            silent and catastrophic, and it is exactly the
                            shape of the filename collision that once wrote
                            four chains to one file.
  z is recoverable       -- (B - b_bar)/(lam_local*tau) must reproduce a
                            standard normal-ish quantity, confirming the claim
                            that dropping z loses nothing
  kappa                  -- shrinkage_factors must lie in (0,1) and must match
                            an independent NumPy computation
  io round-trip          -- save_draws/load_draws must return every field
                            bit-identically, including the sample statistics
  filenames              -- the exact pattern load_chains globs, and chain
                            indices taken from seed0 + k, not the loop index
  load_chains            -- the real function, on the real files

Run from the project root:

    python3 check_chunk3.py

Expected: 15 checks, all PASS. Writes to results/_chunk3_scratch/ and deletes
it afterwards; nothing touches results/size_bm_25/.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from src.diagnostics.convergence import load_chains
from src.gibbs.io import load_draws, save_draws
from src.nuts.horseshoe import (HorseshoeDraws, ebfmi, horseshoe_hyperparameters,
                                run_nuts, shrinkage_factors)

N_DRAWS, N_TUNE, CHAINS, SEED0 = 60, 200, 2, 3
n_pass = n_fail = 0


def ok(label: str, condition: bool, detail: str = "") -> None:
    global n_pass, n_fail
    n_pass, n_fail = n_pass + bool(condition), n_fail + (not condition)
    print(f"  [{'PASS' if condition else 'FAIL'}] {label:<52s} {detail}")


d = np.load(Path("data/processed/size_bm_25_arrays.npz"), allow_pickle=True)
F, R = d["F"], d["R"]
N, T, K = F.shape
hp = horseshoe_hyperparameters(R, F, K)

print(f"size_bm_25: N={N} T={T} K={K}   tau_0={hp.tau_0:.6e}")
print(f"short run: {CHAINS} chains x {N_DRAWS} draws ({N_TUNE} tune), seed0={SEED0}\n")
chains = run_nuts(R, F, hp, n_draws=N_DRAWS, n_tune=N_TUNE, chains=CHAINS,
                  cores=min(CHAINS, 2), seed0=SEED0, progressbar=True)

print("\nshapes and structure")
ok("one HorseshoeDraws per chain", len(chains) == CHAINS, f"{len(chains)}")
c0 = chains[0]
shapes = {"B": (N_DRAWS, N, K), "b_bar": (N_DRAWS, K), "lam_local": (N_DRAWS, N, K),
          "tau": (N_DRAWS,), "Sigma": (N_DRAWS, N, N), "diverging": (N_DRAWS,),
          "step_size": (N_DRAWS,), "energy": (N_DRAWS,)}
bad = {k: getattr(c0, k).shape for k, v in shapes.items() if getattr(c0, k).shape != v}
ok("every field has the expected shape", not bad, str(bad) if bad else "")
ok("all finite", all(np.all(np.isfinite(getattr(c0, k))) for k in shapes),
   "" if True else "")

print("\nthe chains are genuinely different chains")
ok("seeds are seed0 + k", [c.meta["seed"] for c in chains] == [SEED0 + k for k in range(CHAINS)],
   str([c.meta["seed"] for c in chains]))
same = np.array_equal(chains[0].B, chains[1].B)
ok("chain 0 and chain 1 are NOT identical", not same,
   "identical -- R-hat would read 1.0000 on one chain copied" if same else "")
d0, d1 = chains[0].b_bar.mean(0), chains[1].b_bar.mean(0)
ok("but they agree on b_bar to within sampling error",
   float(np.corrcoef(d0, d1)[0, 1]) > 0.9, f"corr {np.corrcoef(d0, d1)[0, 1]:.4f}")

print("\nwhat is not stored is exactly recoverable")
z_rec = (c0.B - c0.b_bar[:, None, :]) / (c0.lam_local * c0.tau[:, None, None])
ok("z = (B - b_bar)/(lam*tau) is finite and O(1)",
   np.all(np.isfinite(z_rec)) and abs(float(z_rec.std()) - 1.0) < 0.6,
   f"sd {z_rec.std():.3f}, |z| max {np.abs(z_rec).max():.1f}")

kappa = shrinkage_factors(c0, F, hp)
s = np.sqrt((F ** 2).mean(axis=1))
a = (np.sqrt(T) / hp.sigma_pooled) * c0.tau[:, None, None] * c0.lam_local * s
kappa_ref = 1.0 / (1.0 + a ** 2)
ok("kappa matches an independent computation",
   float(np.abs(kappa - kappa_ref).max()) < 1e-15, f"max diff {np.abs(kappa-kappa_ref).max():.1e}")
# 0 < kappa <= 1. kappa = 1 is the legitimate limit as a -> 0, not an error;
# the first version of this check demanded kappa < 1 strictly and failed on
# the intercept, which turned out to be a real bug in shrinkage_factors --
# F.std gives the intercept column a scale of 0 where the root mean square
# gives 1. The assertion was wrong AND it caught something.
ok("kappa lies in (0, 1]", bool(kappa.min() > 0 and kappa.max() <= 1.0),
   f"[{kappa.min():.4f}, {kappa.max():.4f}], mean {kappa.mean():.4f}")
# With F.std the intercept's kappa is EXACTLY 1 for every draw and asset, so
# any summary catches it. The max is the wrong statistic for correct code:
# lambda_i0 is occasionally tiny, which sends that draw's kappa to 1
# legitimately. What must not happen is kappa being pinned at 1 throughout.
ok("the intercept is not assigned zero information",
   bool(kappa[:, :, 0].mean() < 0.999 and not np.allclose(kappa[:, :, 0], 1.0)),
   f"intercept kappa mean {kappa[:, :, 0].mean():.4f}, "
   f"min {kappa[:, :, 0].min():.4f} (F.std would give exactly 1 everywhere)")

print("\nio round-trip and the filename convention")
root = Path("results/_chunk3_scratch")
folder = root / "size_bm_25" / "horseshoe"
folder.mkdir(parents=True, exist_ok=True)
tag = f"p0_{hp.p0}_r2_0p05"
for k, c in enumerate(chains):
    save_draws(c, folder / f"horseshoe_{tag}_chain{SEED0 + k}")   # seed0 + k, NOT k

names = sorted(p.name for p in folder.glob("*.npz"))
ok("filenames use seed0 + k", names == [f"horseshoe_{tag}_chain{SEED0+k}.npz"
                                        for k in range(CHAINS)], str(names))
ok("no dots in the setting tag", "." not in tag, tag)

back = load_draws(folder / f"horseshoe_{tag}_chain{SEED0}", HorseshoeDraws)
# np.array_equal, not subtraction: `diverging` is a boolean array and NumPy
# refuses the `-` operator on booleans
FIELDS = ("B", "b_bar", "lam_local", "tau", "Sigma", "diverging",
          "tree_depth", "n_steps", "step_size", "energy")
mismatched = [f for f in FIELDS
              if not np.array_equal(np.asarray(getattr(back, f)),
                                    np.asarray(getattr(c0, f)))]
ok("every field round-trips bit-identically", not mismatched,
   f"mismatched: {mismatched}" if mismatched else f"{len(FIELDS)} fields")
ok("meta survives the round-trip",
   back.meta.get("seed") == SEED0 and back.meta.get("p0") == hp.p0,
   f"seed {back.meta.get('seed')}, p0 {back.meta.get('p0')}, "
   f"n_choice {back.meta.get('n_choice')!r}")

loaded = load_chains("size_bm_25", "horseshoe", tag, HorseshoeDraws, results_root=root)
ok("load_chains finds them, in seed order", len(loaded) == CHAINS
   and [c.meta["seed"] for c in loaded] == [SEED0 + k for k in range(CHAINS)],
   f"{len(loaded)} chains")

ok("E-BFMI is computed and sane", all(0.0 < c.meta["ebfmi"] < 3.0 for c in chains),
   " ".join(f"{c.meta['ebfmi']:.2f}" for c in chains))

shutil.rmtree(root)
print(f"\n(removed {root})")
print(f"\n{n_pass} passed, {n_fail} failed")
raise SystemExit(1 if n_fail else 0)