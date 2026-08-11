"""
check_chunk5.py -- temporary check for the assembled Gibbs loop in
src/gibbs/bayesian_lasso.py.

Sections 1-3 are quick. Section 4 runs a single long chain on real data to
answer the open question from the brief's section 7: does the theta/tau^2 data
augmentation mix? Budget roughly 0.26 s per sweep.

Run from the project root:
    python check_chunk5.py                 # 1,500 sweeps for the mixing check
    python check_chunk5.py --sweeps 400    # quicker, less reliable
"""
import argparse
import time

import numpy as np

from src.gibbs.bayesian_lasso import lasso_hyperparameters, run_gibbs

ap = argparse.ArgumentParser()
ap.add_argument("--sweeps", type=int, default=1500)
args = ap.parse_args()

d = np.load('data/processed/size_bm_25_arrays.npz', allow_pickle=True)
F, R = d['F'], d['R']
N, T, K = F.shape
hp = lasso_hyperparameters(R, F, K)
print(f"N={N} T={T} K={K}   s={hp.s:.4f}   lambda_cal={hp.lambda_prior_mean:.1f}")

# ---- 1. shapes, storage, reproducibility --------------------------------
print("\n1. STRUCTURE")
dr = run_gibbs(R, F, hp, n_draws=30, n_burn=10, seed=0, store_tau2_full=True)
print(f"   B {dr.B.shape}   b_bar {dr.b_bar.shape}   Sigma {dr.Sigma.shape}   "
      f"lam {dr.lam.shape}")
print(f"   tau2_mean {dr.tau2_mean.shape}   tau2_inv_mean {dr.tau2_inv_mean.shape}   "
      f"tau2_track {dr.tau2_track.shape}")
print(f"   burn-in discarded: {dr.B.shape[0]} kept of 30 with n_burn=10  "
      f"-> {dr.B.shape[0] == 20}")

# running means must be EXACT, not approximate
print("\n2. RUNNING MEANS ARE EXACT")
err_m = np.abs(dr.tau2_mean - dr.tau2.mean(axis=0)).max()
err_i = np.abs(dr.tau2_inv_mean - (1.0 / dr.tau2).mean(axis=0)).max()
rel_m = err_m / dr.tau2_mean.mean()
print(f"   tau2_mean     vs mean of stored draws: max abs {err_m:.3e} "
      f"({rel_m:.1e} relative)")
print(f"   tau2_inv_mean vs mean of stored draws: max abs {err_i:.3e}")
ti = dr.tau2_track_idx
tr_ok = np.abs(dr.tau2_track - dr.tau2[:, ti[:, 0], ti[:, 1]]).max()
print(f"   tau2_track matches the stored draws at those indices: {tr_ok:.3e}")
print(f"   E[1/tau2] != 1/E[tau2]: ratio {(dr.tau2_inv_mean * dr.tau2_mean).mean():.3f} "
      f"(would be 1.000 if they were reciprocal)")

d2 = run_gibbs(R, F, hp, n_draws=30, n_burn=10, seed=0)
print(f"   same seed reproduces B exactly: {np.array_equal(dr.B, d2.B)}")
d3 = run_gibbs(R, F, hp, n_draws=30, n_burn=10, seed=1)
print(f"   different seed differs:         {not np.array_equal(dr.B, d3.B)}")

# ---- 3. tracked subset is fixed independently of the run seed -----------
print("\n3. TRACKED SUBSET IS SEED-INDEPENDENT")
print(f"   seed 0 vs seed 1 track the same coefficients: "
      f"{np.array_equal(dr.tau2_track_idx, d3.tau2_track_idx)}")
print(f"   first five tracked (asset, predictor): "
      f"{[tuple(int(v) for v in x) for x in dr.tau2_track_idx[:5]]}")

# ---- 4. MIXING ON REAL DATA --------------------------------------------
print(f"\n4. MIXING  ({args.sweeps} sweeps, no burn-in discarded)")
t0 = time.time()
m = run_gibbs(R, F, hp, n_draws=args.sweeps, n_burn=0, seed=1,
              progress_every=max(args.sweeps // 5, 1))
el = time.time() - t0
n = args.sweeps
print(f"   {el:.0f}s total, {el/n:.3f} s/sweep")


def ac1(x):
    x = np.asarray(x, float) - np.mean(x)
    return float((x[1:] @ x[:-1]) / (x @ x))


def ess_1d(x):
    """Geyer initial-positive-sequence ESS for a single chain."""
    x = np.asarray(x, float) - np.mean(x)
    n_ = len(x)
    v = x @ x / n_
    if v <= 0:
        return float(n_)
    rho, k = [], 1
    while k < n_ - 1:
        r = (x[k:] @ x[:-k]) / (n_ * v)
        rho.append(r)
        if k % 2 == 0 and len(rho) >= 2 and rho[-1] + rho[-2] < 0:
            break
        k += 1
    tau = 1.0 + 2.0 * sum(rho[:-2]) if len(rho) > 2 else 1.0
    return float(n_ / max(tau, 1.0))


half = n // 2
nb = np.linalg.norm(m.b_bar, axis=1)
nB = np.linalg.norm(m.B.reshape(n, -1), axis=1)
st = m.tau2_track.sum(axis=1)
print(f"\n   {'quantity':<20s} {'lag-1 AC':>10s} {'2nd half':>10s} {'ESS (2nd half)':>15s} "
      f"{'sweeps for ESS 400':>20s}")
for nm, x in [("||b_bar||", nb), ("||B||", nB), ("lambda", m.lam),
              ("sum tracked tau2", st), ("Sigma[0,0]", m.Sigma[:, 0, 0])]:
    e = ess_1d(x[half:])
    need = half / e * 400 if e > 0 else np.inf
    print(f"   {nm:<20s} {ac1(x):>10.3f} {ac1(x[half:]):>10.3f} {e:>15.0f} {need:>20.0f}")

print("\n   HAS lambda SETTLED?  means by fifth of the run:")
fifth = n // 5
seg = [m.lam[i*fifth:(i+1)*fifth].mean() for i in range(5)]
print("      " + "  ".join(f"{v:.1f}" for v in seg))
print(f"      last fifth: mean {m.lam[-fifth:].mean():.1f}  sd {m.lam[-fifth:].std():.1f}"
      f"   [lambda_cal = {hp.lambda_prior_mean:.1f}]")
print("   [if the five means still trend in one direction, burn-in is too short and")
print("    the autocorrelation above is inflated by the drift]")

print("\n   WHAT lambda IMPLIES  (baseline-equivalent prior scale)")
lam_hat = m.lam[half:].mean()
print(f"      posterior lambda ~ {lam_hat:.0f}  ->  theta prior sd "
      f"{np.sqrt(2)*hp.s/lam_hat:.3e}")
print(f"      baseline equivalents: lambda 425 = target_r2 0.10,  601 = 0.05,  "
      f"1345 = 0.01")
print("      [higher lambda = more shrinkage. This is the pre-registered test:")
print("       does the data ask for more shrinkage than target_r2=0.05 implies?]")

print("\n   PREDICTION VOLATILITY  (compare to the baseline's 58% / 12% / 6%)")
B_hat = m.B[half:].mean(axis=0)
pred = np.einsum("itk,ik->it", F, B_hat)
print(f"      sd(predicted)/sd(realised) = {pred.std()/R.std():.3f}")