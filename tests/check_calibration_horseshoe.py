"""
check_calibration_horseshoe.py -- validation step 5.

Simulate a parameter set from the horseshoe's OWN prior, generate returns from
it, fit, and ask whether the truth falls inside the 95% credible intervals.
About 95% should. This is the only end-to-end test in the sequence:
an independent NumPy reference proved the log-density correct at one point
during development; this proves the whole machine correct on average.

Reproduces Table 4.6, the horseshoe column and the note on the three excluded
datasets, and Appendix A.6.4, the reduced dimensions N=6, K=5, T=200 at
TAU0_MULT=0.01. With --full-scale it reproduces Appendix A.6.7, the abandoned
production-dimension runs. Supports Section 4.5 on prior-generative
calibration.

EVERY quantity is drawn from the prior the sampler conditions on -- b_bar from
N(0, Delta_b_bar), tau from C+(0, tau_0), lambda from C+(0,1), z from N(0,1),
Sigma from IW(nu_Sigma, V_Sigma). simulate_sur_data is deliberately NOT used:
it generates Delta_b and Sigma from _random_pd_matrix and b_bar from a
standard normal, none of which match the priors being fitted. Bayesian
coverage is nominal only when the truth comes from the prior you condition on,
and the baseline's first calibration measured prior/truth mismatch rather than
sampler error because of exactly this -- reporting Delta_b_diag at +6.1 SE,
which vanished to +1.3 once the truth was drawn correctly. That is the
strongest instance yet of verification code failing the way the code under
test does: it ran, produced plausible numbers, and answered a subtly different
question from the one asked.

REPORTED AS PER-DATASET COVERAGE, mean +/- SE ACROSS DATASETS. The pooled
binomial p-value is NOT reported. Intervals within a dataset are not
independent -- 30 theta intervals from one chain are not 30 independent trials
-- and the LASSO's re-run demonstrated the consequence: its pooled p for theta
swung from 0.002 to 0.048 between two runs differing by less than their own
Monte Carlo error, while the per-dataset test barely moved. A statistic that
unstable under a change that should not matter is not measuring what it
claims. Pooled counts are still printed, because they are the natural
denominator, but the mean +/- SE column is the one to read.

TWO MODES, for two different purposes.

  --match-dims   N=6, K=5, T=200, 100 datasets. MATCHES the dimensions and
                 dataset count of the baseline's and the LASSO's calibrations
                 exactly, so the three can go in one harmonised Chapter 4
                 table where a given number of standard errors means the same
                 thing in every row. Fast: the model has 87 coordinates here
                 rather than 7,670.

  default        N=10, K=30, T=300, 20 datasets. Coverage where the geometry
                 actually exists. At K=5 the horseshoe has 30 local scales and
                 tau is barely identified, so the matched-dimension run is
                 comparable but not by itself meaningful for this model.

  --full-scale   3 datasets at N=25, K=144, T=719. NOT a coverage rate --
                 three datasets is far too few. Checks that tau's intervals
                 and the divergence rate behave the same way where the
                 b_bar/theta ridge and tau's identification bite.

WHY THE SIMULATION REGIME IS MILDER THAN PRODUCTION. The first design set the
simulation's tau_0 = sigma/sqrt(T), intending deviations of about one standard
error. But tau ~ C+(0, tau_0) and lambda ~ C+(0,1) compound: seed 0 drew
tau = 0.28 with max lambda = 613, giving a true deviation of 144 -- roughly
2,500 standard errors. Coverage came out b 0.960 but b_bar 0.400, theta 0.533,
Sigma 0.510, with tree depth pinned at max on every draw: the b_bar/theta
ridge, with trajectories truncated before they could traverse it. Not a bug --
every field moved monotonically toward 0.95 when the budget went up tenfold,
which a wrong log-density does not do. TAU0_MULT = 0.01 standard errors gives
a median largest deviation of 2.4 SE instead. Simulation and fit use identical
hyperparameters, so coverage remains exactly valid; only the regime is chosen
to be one the sampler can reach. What is NOT tested is the extreme tail, and
that belongs in the write-up: it is the same fact the prior predictive reports
from the other direction, that the Cauchy tails put 60% of prior mass at
implausible R^2 values.

The tail does not disappear -- the Cauchy's shape is scale-invariant, so about
7% of datasets still draw a deviation past 50 SE. Those hit the CONVERGENCE
GATE and are excluded with their tau and R-hat printed. Excluding on
convergence biases the rate toward easy datasets, so the exclusions belong in
the thesis beside the rate, not in a footnote.

cores = 1 BY DEFAULT: PyMC's multiprocessing killed a worker at these
dimensions (EOFError, child dead with no traceback). A workaround, not a
diagnosis.

USAGE -- time one dataset before committing to a hundred:

    python3 check_calibration_horseshoe.py --match-dims --datasets 2
    caffeinate -i python3 check_calibration_horseshoe.py --match-dims
    caffeinate -i python3 check_calibration_horseshoe.py --datasets 20 --draws 400 --tune 800
    caffeinate -i python3 check_calibration_horseshoe.py --full-scale
"""
from __future__ import annotations

import argparse
from time import time

import numpy as np
from scipy.stats import invwishart

from src.nuts.horseshoe import HorseshoeHyperparams, run_nuts

TAU0_MULT = 0.01        # simulation tau_0, in standard errors. See docstring.
RHAT_MAX = 1.05
DEPTH_SATURATION = 0.5
FIELDS = ("b_bar", "theta", "b", "Sigma", "tau")


def simulation_hyperparameters(N, K, T, sd_bbar=2.0, tau0_mult=TAU0_MULT
                               ) -> HorseshoeHyperparams:
    """
    Hyperparameters for the simulated regime, constructed directly rather than
    derived from data. V_Sigma = (nu-N-1) I = I, so E[Sigma] = I and the
    residual scale is 1. max_treedepth is 12 because this geometry is harder
    than the real one, where 9 suffices.

    p0 is set to -1 and is UNUSED: tau_0 is set directly from TAU0_MULT rather
    than derived through P&V Eq. (3.12), so any p0 recorded here would be
    inconsistent with it. (Solving p0/(K-p0) = 0.01 would give p0 ~ 0.3 -- the
    fittable regime is a sparser world than production assumes.) Nothing in
    this script reads p0; it is carried only so the dataclass is complete.
    """
    return HorseshoeHyperparams(
        b_bar_bar=np.zeros(K),
        Delta_b_bar=sd_bbar ** 2 * np.eye(K),
        nu_Sigma=float(N + 2),
        V_Sigma=np.eye(N),
        # tau0_mult must SHRINK as N*K grows. The largest of n half-Cauchy
        # draws grows roughly linearly in n, so at full scale (3,600 local
        # scales) the same multiplier that gives a median largest deviation of
        # 2.1 SE at N=10 K=30 gives 29.9 SE, with a 90th percentile of 330 --
        # datasets no achievable budget can fit. 0.0008 = 0.01 * 300/3600
        # restores the reduced-scale difficulty profile exactly.
        tau_0=tau0_mult / np.sqrt(T),
        p0=-1, n_choice="T",
        sigma_pooled=1.0, sd_target=1.0 / np.sqrt(T),
        max_treedepth=12,
    )


def simulate_from_prior(hp, N, K, T, seed):
    rng = np.random.default_rng(seed)
    sd_bbar = float(np.sqrt(hp.Delta_b_bar[0, 0]))
    b_bar = sd_bbar * rng.standard_normal(K)
    tau = hp.tau_0 * abs(rng.standard_cauchy())
    lam = np.abs(rng.standard_cauchy((N, K)))
    z = rng.standard_normal((N, K))
    theta = z * lam * tau
    b = b_bar[None, :] + theta
    Sigma = np.atleast_2d(invwishart.rvs(df=hp.nu_Sigma, scale=hp.V_Sigma,
                                         random_state=rng))
    F = rng.standard_normal((N, T, K))
    F[:, :, 0] = 1.0
    E = rng.multivariate_normal(np.zeros(N), Sigma, size=T).T
    R = np.einsum("itk,ik->it", F, b) + E
    return {"b_bar": b_bar, "tau": tau, "lam": lam, "theta": theta, "b": b,
            "Sigma": Sigma, "F": F, "R": R}


def covered(draws, truth, lo=2.5, hi=97.5):
    a, c = np.percentile(draws, lo, axis=0), np.percentile(draws, hi, axis=0)
    return (truth >= a) & (truth <= c)


def run_stage(label, N, K, T, n_datasets, n_draws, n_tune, chains, cores, seed0,
              tau0_mult=TAU0_MULT):
    import arviz as az

    hp0 = simulation_hyperparameters(N, K, T, tau0_mult=tau0_mult)
    se = 1.0 / np.sqrt(T)
    print(f"\n{'='*76}\n{label}")
    print(f"N={N} K={K} T={T} | {n_datasets} datasets | {chains} chains x "
          f"{n_draws} draws ({n_tune} tune)")
    print(f"simulation tau_0 = {hp0.tau_0:.3e} = {TAU0_MULT} SE (1 SE = {se:.4f}) | "
          f"truth drawn from the model's own prior\n{'='*76}")

    pooled = {f: [0, 0] for f in FIELDS}
    per_dataset = {f: [] for f in FIELDS}        # THE column of record
    ranks = {f: [] for f in ("b_bar", "theta")}
    taus, excluded = [], []
    t_all = time()

    for d in range(n_datasets):
        seed = seed0 + d
        hp = simulation_hyperparameters(N, K, T, tau0_mult=tau0_mult)
        truth = simulate_from_prior(hp, N, K, T, seed)
        t0 = time()
        out = run_nuts(truth["R"], truth["F"], hp, n_draws=n_draws, n_tune=n_tune,
                       chains=chains, cores=cores, seed0=10_000 + 10 * seed,
                       progressbar=False)
        dt = time() - t0

        P = {f: np.concatenate([getattr(c, f) for c in out], axis=0)
             for f in ("b_bar", "tau", "Sigma", "B")}
        div = sum(int(c.diverging.sum()) for c in out)
        depth_sat = float(np.mean(np.concatenate([c.tree_depth for c in out])
                                  >= hp.max_treedepth))
        rhat = float(max(np.nanmax(az.rhat(np.stack([c.b_bar for c in out]))),
                         np.nanmax(az.rhat(np.stack([c.tau for c in out]))))) \
            if chains > 1 else np.nan

        max_dev = float(np.abs(truth["theta"]).max() / se)
        note = f"tau {truth['tau']/hp.tau_0:5.1f}x, max|th| {max_dev:6.1f} SE"

        if (chains > 1 and rhat > RHAT_MAX) or depth_sat > DEPTH_SATURATION:
            excluded.append((seed, truth["tau"] / hp.tau_0, max_dev, rhat, depth_sat))
            print(f"  {d:3d}: EXCLUDED  R-hat {rhat:.3f}, depth sat {depth_sat:.0%}"
                  f" | {note} | {dt/60:.1f} min", flush=True)
            continue

        pairs = {"b_bar": (P["b_bar"], truth["b_bar"]),
                 "theta": (P["B"] - P["b_bar"][:, None, :], truth["theta"]),
                 "b": (P["B"], truth["b"]),
                 "Sigma": (P["Sigma"], truth["Sigma"]),
                 "tau": (P["tau"], np.array(truth["tau"]))}
        for f, (dr, tr) in pairs.items():
            c = covered(dr, tr)
            pooled[f][0] += int(np.sum(c)); pooled[f][1] += int(np.size(c))
            per_dataset[f].append(float(np.mean(c)))     # cannot be recovered later
        for f in ranks:
            dr, tr = pairs[f]
            ranks[f].append(float((dr < tr).mean(axis=0).mean()))
        taus.append((truth["tau"], float(P["tau"].mean())))
        print(f"  {d:3d}: b_bar {np.mean(covered(P['b_bar'], truth['b_bar'])):.3f}"
              f"  b {np.mean(covered(P['B'], truth['b'])):.3f}"
              f"  tau {'in ' if covered(P['tau'], np.array(truth['tau'])) else 'OUT'}"
              f"  R-hat {rhat:.3f}  {div} div | {note} | {dt/60:.1f} min", flush=True)

    n_used = n_datasets - len(excluded)
    print(f"\n  {(time()-t_all)/60:.1f} min | {n_used} of {n_datasets} datasets used, "
          f"{len(excluded)} excluded by the convergence gate")
    if excluded:
        print("  EXCLUDED -- these bias the rate toward easy datasets; report them:")
        for s, tr, m, r, ds in excluded:
            print(f"    seed {s}: tau {tr:.1f}x tau_0, max|theta| {m:.0f} SE, "
                  f"R-hat {r:.3f}, depth saturating {ds:.0%}")
    if n_used == 0:
        print("\n  no datasets passed the gate -- nothing to report")
        return

    print(f"\n  {'field':<8s} {'pooled':>14s} {'per-dataset mean':>18s} {'z vs 0.95':>10s}")
    print(f"  {'':8s} {'(denominator)':>14s} {'(THE column)':>18s}")
    for f in FIELDS:
        k, n = pooled[f]
        r = np.array(per_dataset[f])
        if n == 0:
            continue
        if len(r) > 1:
            sem = r.std(ddof=1) / np.sqrt(len(r))
            z = (r.mean() - 0.95) / sem if sem > 0 else np.nan
            print(f"  {f:<8s} {k:>6d} /{n:<7d} {r.mean():>10.3f} +/- {sem:.3f} "
                  f"{z:>+10.1f}")
        else:
            print(f"  {f:<8s} {k:>6d} /{n:<7d} {r.mean():>10.3f}        - "
                  f"{'-':>10s}")
    print("\n  No binomial p-value: intervals within a dataset are not independent,")
    print("  and pooling them is unstable -- the LASSO's pooled p for theta swung")
    print("  from 0.002 to 0.048 between runs differing by less than Monte Carlo")
    print("  error, while its per-dataset test barely moved.")

    if n_used >= 3:
        print("\n  rank uniformity, per dataset (mean rank ~0.5)")
        for f, r in ranks.items():
            r = np.array(r)
            sem = r.std(ddof=1) / np.sqrt(len(r))
            print(f"  {f:<8s} {r.mean():.3f} +/- {sem:.3f}   z = {(r.mean()-0.5)/sem:+.2f}")
        tt = np.array(taus)
        print(f"\n  tau: correlation(true, posterior mean) = "
              f"{np.corrcoef(tt[:,0], tt[:,1])[0,1]:+.3f} over {len(tt)} datasets")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match-dims", action="store_true",
                    help="N=6 K=5 T=200, matching the baseline's and LASSO's design")
    ap.add_argument("--full-scale", action="store_true")
    ap.add_argument("--datasets", type=int, default=None)
    ap.add_argument("--draws", type=int, default=None)
    ap.add_argument("--tune", type=int, default=None)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--cores", type=int, default=1)
    args = ap.parse_args()

    if args.cores > 1:
        print("WARNING: cores > 1 killed a worker at these dimensions (EOFError).")

    if args.full_scale:
        run_stage("FULL SCALE -- confirmation only, NOT a coverage rate",
                  25, 144, 719, args.datasets or 3, args.draws or 500,
                  args.tune or 1000, args.chains, args.cores, seed0=500,
                  tau0_mult=0.0008)
    elif args.match_dims:
        run_stage("MATCHED DIMENSIONS -- for the harmonised Chapter 4 table",
                  6, 5, 200, args.datasets or 100, args.draws or 750,
                  args.tune or 1000, args.chains, args.cores, seed0=1000)
    else:
        run_stage("REDUCED SCALE -- coverage where the geometry exists",
                  10, 30, 300, args.datasets or 20, args.draws or 400,
                  args.tune or 800, args.chains, args.cores, seed0=0)


if __name__ == "__main__":
    main()

