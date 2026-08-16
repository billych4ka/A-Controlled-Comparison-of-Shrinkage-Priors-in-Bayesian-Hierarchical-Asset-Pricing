"""
check_sparsity_recovery.py -- validation step 6.

Does the horseshoe shrink coefficients that are GENUINELY ZERO harder than
coefficients that are merely small? This is the test the Gaussian baseline
cannot be run through at all -- not that it would fail, but that it has no
per-coefficient quantity to measure.

THE METRIC, and why the obvious alternatives are rejected.

    rmse of posterior-mean theta at the positions where the truth is exactly
    zero, divided by the mean |theta| among the non-zero positions.

Reported against a DENSE CONTROL in which no coefficient is zero, measured at
the same NUMBER of smallest-|truth| positions. The control is the whole point:
a model that simply shrinks everything scores well on the sparse case alone.

NOT separation statistics. AUC and the precision ratio were used for this in
the LASSO's validation and are recorded there as NOT DIAGNOSTIC: AUC scored
0.984 on sparse data and 0.936 on the dense control, and the precision ratio
flipped ordering across seeds (12.2 vs 8.3 one run, 8.2 vs 14.4 the next).
They test only whether the local scales are ORDERED like |theta|, which every
shrinkage prior satisfies. Absolute recovery against a dense control is the
test; separation is not.

LASSO benchmark: 0.084 sparse against 0.214 dense.

n_null IS COUNTED, NOT ASSUMED -- this was a bug in the first version and it
mattered. draw_deviations_sparse's sparsity fraction is APPROXIMATE: asked for
0.7 it produced 196 exact zeros of 300 (0.653), and its own verification
recorded 0.695 against a 0.7 target. Scoring a hardcoded round(0.7*N*K) = 210
positions as null therefore fed 14 genuinely active coefficients, of true
magnitude ~0.30, into the null rmse -- enough on their own to exceed the whole
reported figure, and biased UPWARD, which is the direction that would have
made the horseshoe look worse than the LASSO. The sparse case now counts exact
zeros per dataset; the dense control uses the mean of those counts so the two
are compared at matched position counts.

FOUR DESIGN CHOICES, each deliberate.

 1. DIMENSIONS N=10, K=30, T=300, matching calibration run 2. NOTE: the
    LASSO's benchmark was produced at dimensions not recorded in the coding
    notes, so 0.084 and the figure produced here may not be directly
    comparable. Re-running the LASSO's test at these dimensions is minutes of
    Gibbs compute and is the clean fix.

 2. MATCHED TOTAL SIGNAL. Sparse has ~30% of coefficients active; dense has
    100%. At the same per-coefficient scale the dense control would carry 3.3x
    the total signal, and the comparison would confound sparsity with signal
    strength. The dense scale is therefore divided so total squared signal
    matches, leaving CONCENTRATION as the only difference.

 3. p_0 FOR THE FITTED MODEL is set from the PRODUCTION RATIO, p_0/K =
    23/144, giving p_0 = 5 of 30 -- deliberately understating the true 9
    active. Setting p_0 to the truth would tell the model the answer;
    understating is the conservative direction and mirrors how the real prior
    relates to reality.

 4. draw_deviations_sparse USES ONLY THE DIAGONAL of Delta_b. The horseshoe
    has no mechanism to represent correlation between theta_ij -- each has its
    own independent local scale -- so correlated sparse truth would test
    recovery of structure the model cannot represent even in principle. This
    is recorded as a methodological choice in the notes and carries over.

The dense control uses draw_deviations_gaussian, which is the BASELINE's
generative process. That is correct here: the control's job is to be a
non-sparse world, not to match any particular model's prior.

PRE-REGISTERED, before running: the horseshoe will achieve a LOWER
rmse-where-zero than the LASSO's 0.084, because heavy-tailed local scales
shrink genuinely null coefficients harder than an exponential tail while
leaving active ones free. A comparable or worse figure would say the LASSO's
single global scale is sufficient for this problem -- a substantive finding
about the design, not a null result.

Run from the project root. About 5.3 minutes per fit measured, so ~1.8 hours
for the default 10 sparse + 10 dense:

    python3 check_sparsity_recovery.py --datasets 2        # time it first
    caffeinate -i python3 check_sparsity_recovery.py
"""
from __future__ import annotations

import argparse
from time import time

import numpy as np

from src.nuts.horseshoe import HorseshoeHyperparams, run_nuts
from src.simulate.generate import (draw_deviations_gaussian,
                                   draw_deviations_sparse, simulate_sur_data)

SPARSITY = 0.7          # requested fraction of theta_ij exactly zero (approximate)
ACTIVE_SE = 5.0         # active coefficients, in standard errors
P0_RATIO = 23.0 / 144.0  # the production ratio, applied to K


def fit_hyperparameters(N, K, T, sd_bbar=2.0) -> HorseshoeHyperparams:
    """The prior the model is GIVEN. p_0 from the production ratio, so the
    model is not told the true active count. sigma = 1 because the simulated
    residual scale is 1 (V_Sigma = (nu-N-1) I = I)."""
    p0 = max(1, int(round(P0_RATIO * K)))
    return HorseshoeHyperparams(
        b_bar_bar=np.zeros(K),
        Delta_b_bar=sd_bbar ** 2 * np.eye(K),
        nu_Sigma=float(N + 2),
        V_Sigma=np.eye(N),
        tau_0=(p0 / (K - p0)) * 1.0 / np.sqrt(T),
        p0=p0, n_choice="T",
        sigma_pooled=1.0, sd_target=1.0 / np.sqrt(T),
        max_treedepth=12,
    )


def make_dataset(N, K, T, sparse: bool, seed: int):
    """Sparse truth, or a dense control carrying the SAME total squared signal.

    active variance v_s over a fraction (1-SPARSITY) of coefficients; the dense
    control uses v_d = (1-SPARSITY) * v_s over all of them, so sum(theta^2)
    matches in expectation and only the CONCENTRATION differs.
    """
    se = 1.0 / np.sqrt(T)
    v_active = (ACTIVE_SE * se) ** 2
    frac_active = 1.0 - SPARSITY

    if sparse:
        Delta_b = v_active * np.eye(K)
        sampler = lambda rng, N_, K_, D: draw_deviations_sparse(
            rng, N_, K_, D, sparsity=SPARSITY, active_scale=1.0)
    else:
        Delta_b = (frac_active * v_active) * np.eye(K)
        sampler = draw_deviations_gaussian

    return simulate_sur_data(N=N, K=K, T=T, Delta_b=Delta_b, Sigma=np.eye(N),
                             deviation_sampler=sampler, seed=seed)


def score(theta_hat: np.ndarray, theta_true: np.ndarray, n_null: int | None = None):
    """rmse at the n_null smallest-|truth| positions, as a fraction of the mean
    |truth| among the rest.

    n_null=None counts EXACT zeros, which is what the sparse generator actually
    produced. Passing a fixed count (as the dense control does) forces the two
    cases to be compared at the same number of positions. Never assume the
    requested sparsity fraction: it is approximate, and over-counting nulls
    feeds active coefficients into the rmse and biases it upward.
    """
    flat_t, flat_h = theta_true.ravel(), theta_hat.ravel()
    if n_null is None:
        n_null = int(np.sum(flat_t == 0.0))
    order = np.argsort(np.abs(flat_t))
    null_idx, active_idx = order[:n_null], order[n_null:]
    rmse = float(np.sqrt(np.mean((flat_h[null_idx] - flat_t[null_idx]) ** 2)))
    scale = float(np.mean(np.abs(flat_t[active_idx])))
    return rmse, scale, rmse / scale, n_null


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=int, default=10)
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--tune", type=int, default=800)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--cores", type=int, default=1)
    args = ap.parse_args()

    N, K, T = args.N, args.K, args.T
    hp = fit_hyperparameters(N, K, T)

    print(f"{'='*76}")
    print(f"SPARSITY RECOVERY  N={N} K={K} T={T} | {args.datasets} sparse + "
          f"{args.datasets} dense")
    print(f"requested sparsity {SPARSITY:.0%} (APPROXIMATE -- exact zeros are "
          f"counted per dataset)")
    print(f"active coefficients at {ACTIVE_SE:.0f} SE; dense control rescaled "
          f"to matched total signal")
    print(f"model is given p_0 = {hp.p0} of {K} (production ratio 23/144), "
          f"tau_0 = {hp.tau_0:.4e}")
    print(f"true active per asset ~= {int((1-SPARSITY)*K)}, so p_0 UNDERSTATES it")
    print(f"{'='*76}\n")

    def run_block(label, sparse, n_null_fixed=None):
        ratios, rmses, scales, nulls = [], [], [], []
        t0 = time()
        for d in range(args.datasets):
            data = make_dataset(N, K, T, sparse, seed=7000 + 100 * sparse + d)
            out = run_nuts(data.R, data.F, hp, n_draws=args.draws,
                           n_tune=args.tune, chains=args.chains,
                           cores=args.cores, seed0=20_000 + 10 * d,
                           progressbar=False)
            B = np.concatenate([c.B for c in out], axis=0)
            bb = np.concatenate([c.b_bar for c in out], axis=0)
            theta_hat = (B - bb[:, None, :]).mean(axis=0)
            rmse, scale, ratio, n_null = score(theta_hat, data.theta, n_null_fixed)
            ratios.append(ratio); rmses.append(rmse); scales.append(scale)
            nulls.append(n_null)
            exact = int(np.sum(data.theta == 0.0))
            print(f"  {label:<14s} {d:2d}: rmse {rmse:.4f}  mean|active| "
                  f"{scale:.4f}  ratio {ratio:.4f}   "
                  f"({exact} exact zeros, {n_null} scored as null)", flush=True)
        print(f"  {label}: {(time()-t0)/60:.1f} min\n")
        return (np.array(ratios), np.array(rmses), np.array(scales),
                np.array(nulls))

    sparse_res = run_block("sparse", True)
    n_null_matched = int(round(sparse_res[3].mean()))
    print(f"  dense control scored at {n_null_matched} null positions, "
          f"matching the sparse mean\n")
    dense_res = run_block("dense control", False, n_null_fixed=n_null_matched)

    print(f"{'='*76}\n{'':<38s}{'sparse truth':>16s}{'dense control':>16s}")
    for name, idx in (("rmse where truth is zero/small", 1),
                      ("mean |non-zero|", 2)):
        s, dch = sparse_res[idx], dense_res[idx]
        print(f"  {name:<36s}{s.mean():>12.4f}{dch.mean():>16.4f}")
    s, dch = sparse_res[0], dense_res[0]
    print(f"  {'as fraction of mean |non-zero|':<36s}{s.mean():>12.4f}"
          f"{dch.mean():>16.4f}")
    print(f"  {'(SE over datasets)':<36s}{s.std(ddof=1)/np.sqrt(len(s)):>12.4f}"
          f"{dch.std(ddof=1)/np.sqrt(len(dch)):>16.4f}")

    gap = (dch.mean() - s.mean()) / np.sqrt(s.var(ddof=1)/len(s)
                                            + dch.var(ddof=1)/len(dch))
    print(f"\n  separation: dense - sparse = {dch.mean()-s.mean():+.4f}, "
          f"{gap:+.1f} SE")
    print(f"  LASSO benchmark: 0.084 sparse, 0.214 dense -- dimensions NOT "
          f"recorded in the\n  coding notes, so NOT necessarily comparable. "
          f"Re-run the LASSO's test at\n  N={N} K={K} T={T} before placing the "
          f"two side by side.")
    print("\n  No AUC, no precision ratio: those scored 0.984 sparse and 0.936 "
          "dense for the\n  LASSO, and the precision ratio flipped ordering "
          "across seeds. They test only\n  whether local scales are ORDERED "
          "like |theta|, which every shrinkage prior\n  satisfies.")


if __name__ == "__main__":
    main()


