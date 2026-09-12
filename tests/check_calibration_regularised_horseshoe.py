"""
check_calibration_regularised_horseshoe.py -- validation step 5 for model 4.

Simulate a parameter set from the regularised horseshoe's OWN prior, generate
returns, fit, and ask whether the truth falls inside the 95% credible
intervals. About 95% should.

Reproduces Table 4.6, the regularised horseshoe column. Supports Section 4.5
on prior-generative calibration.

MATCHED DIMENSIONS ONLY: N=6, K=5, T=200, 100 datasets -- identical to the
design used for the Gaussian baseline, the Bayesian LASSO and the plain
horseshoe, so all four rows of the Chapter 4 table are directly comparable and
a given number of standard errors means the same thing in every row.

WHY THIS IS RUN AT ALL, given most of the machinery is shared and already
verified. Two reasons.

  A gap in a four-row table is a question. "The other three were calibrated,
  the fourth was not" invites an answer, and "it shares most of its machinery"
  is weaker than a row of numbers.

  More substantively: the pilot suggests the slab CHANGES TAU. With the slab
  controlling the tails, tau no longer has to be driven down to control them,
  so c and tau are coupled and model 4's posterior geometry is not simply
  model 3's with a capped tail -- it is a different geometry. Calibration
  tests whether the sampler produces honest intervals in THAT geometry, which
  model 3's calibration cannot speak to.

WHAT IS DELIBERATELY NOT REPEATED, and why. The N=10/K=30 run and the
full-scale attempt tested behaviour driven by the plain horseshoe's UNBOUNDED
tails -- the regime where draws from the prior produce coefficients thousands
of standard errors from zero and no achievable budget converges. The slab
bounds exactly that, so those runs would be testing a pathology this model
does not have. Sparsity recovery is also skipped: it would report that the
regularised horseshoe recovers sparse structure much as the plain one does,
at two hours, and nothing in the argument depends on it.

DESIGN, carried over unchanged from the horseshoe's calibration:

  Truth drawn from THIS model's own prior -- b_bar ~ N(0, Delta_b_bar),
  tau ~ C+(0, tau_0), lambda ~ C+(0,1), caux ~ InvGamma(nu/2, nu/2),
  c = s sqrt(caux), lambda~ from P&V Eq. (2.8), z ~ N(0,1),
  Sigma ~ IW(nu_Sigma, V_Sigma). simulate_sur_data is NOT used: it generates
  Delta_b and Sigma from _random_pd_matrix and b_bar from a standard normal,
  none of which match the priors being fitted, and the baseline's first
  calibration measured prior/truth mismatch rather than sampler error because
  of exactly that.

  TAU0_MULT = 0.01 standard errors. Coverage is correct by construction for
  ANY prior provided the same prior is fitted, but POWER depends on the
  regime: at the production tau_0 the simulated deviations would sit far
  below what T observations resolve, every interval would be enormous, and a
  mildly wrong sampler would pass. This places a coefficient with lambda = 1
  at about one standard error.

  THE SLAB SCALE IS SCALED TO MATCH. s is set so that E[c] sits at the same
  multiple of the coefficient scale as in production -- 14.7 x sd_target --
  rather than at its production value, which would be enormous relative to
  this regime's coefficients and would never bind. A calibration in which the
  slab never binds would be calibrating the plain horseshoe.

  Per-dataset coverage mean +/- SE across datasets is THE column. No binomial
  p-value: intervals within a dataset are not independent, and the pooled
  statistic is unstable -- the LASSO's pooled p for theta swung from 0.002 to
  0.048 between runs differing by less than their own Monte Carlo error.

  A convergence gate excludes datasets with R-hat > 1.05 or tree depth
  saturating on more than half the draws, and the exclusions are REPORTED,
  because excluding on convergence biases the rate toward easy datasets.
  Expect far fewer than the horseshoe's 3 of 100: the slab bounds the draws
  that made them unfittable.

  cores=1: PyMC's multiprocessing killed a worker twice in this project.

Run from the project root. At the pilot's 0.354 s/iteration this should take
roughly 30-40 minutes, against 101 for the plain horseshoe:

    python3 check_calibration_regularised_horseshoe.py --datasets 2   # time it first
    caffeinate -i python3 check_calibration_regularised_horseshoe.py
"""
from __future__ import annotations

import argparse
from time import time

import numpy as np
from scipy.stats import invgamma, invwishart

from src.nuts.regularised_horseshoe import (RegHorseshoeHyperparams,
                                            lambda_tilde, run_nuts)

TAU0_MULT = 0.01        # simulation tau_0, in standard errors
# E[c] / tau_0, matching production: 1.918988e-03 / 3.945463e-04.
# THE BINDING FRACTION DEPENDS ON c/tau, NOT c/sd_target. An earlier version
# scaled the slab to sd_target, but this regime's tau_0 is 0.01 SE against
# production's 3.01 x sd_target, so that left c/tau_0 at 1,470 instead of 4.86
# and the slab bound for 0.04% of coefficients -- calibrating the plain
# horseshoe under another name. Caught by the binding guard.

SLAB_OVER_TAU0 = 4.8638
RHAT_MAX = 1.05
DEPTH_SATURATION = 0.5
NU = 4.0
FIELDS = ("b_bar", "theta", "b", "Sigma", "tau", "c")


def simulation_hyperparameters(N, K, T, sd_bbar=2.0) -> RegHorseshoeHyperparams:
    """Constructed directly rather than derived from data. V_Sigma =
    (nu-N-1) I = I, so E[Sigma] = I and the residual scale is 1.

    The slab is scaled to sit at the SAME multiple of the coefficient scale as
    in production. Using the production s here would put E[c] far above
    anything this regime's coefficients reach, the slab would never bind, and
    the run would calibrate the plain horseshoe under another name.

    p0 = -1 and is UNUSED: tau_0 is set directly from TAU0_MULT rather than
    derived through P&V Eq. (3.12), so any p0 recorded here would be
    inconsistent with it.
    """
    se = 1.0 / np.sqrt(T)
    slab_sd = SLAB_OVER_TAU0 * TAU0_MULT * se   # E[c], so c/tau_0 matches production
    return RegHorseshoeHyperparams(
        b_bar_bar=np.zeros(K),
        Delta_b_bar=sd_bbar ** 2 * np.eye(K),
        nu_Sigma=float(N + 2),
        V_Sigma=np.eye(N),
        tau_0=TAU0_MULT * se,
        slab_scale=float(slab_sd / np.sqrt(NU / (NU - 2.0))),
        slab_df=NU,
        p0=-1, n_choice="T",
        sigma_pooled=1.0, sd_target=se,
        max_treedepth=12,
    )


def simulate_from_prior(hp, N, K, T, seed):
    rng = np.random.default_rng(seed)
    sd_bbar = float(np.sqrt(hp.Delta_b_bar[0, 0]))
    b_bar = sd_bbar * rng.standard_normal(K)
    tau = hp.tau_0 * abs(rng.standard_cauchy())
    lam = np.abs(rng.standard_cauchy((N, K)))
    caux = invgamma.rvs(a=hp.slab_df / 2.0, scale=hp.slab_df / 2.0,
                        random_state=rng)
    c = hp.slab_scale * np.sqrt(caux)
    lam_t = lambda_tilde(lam, tau, c)
    z = rng.standard_normal((N, K))
    theta = z * lam_t * tau
    b = b_bar[None, :] + theta
    Sigma = np.atleast_2d(invwishart.rvs(df=hp.nu_Sigma, scale=hp.V_Sigma,
                                         random_state=rng))
    F = rng.standard_normal((N, T, K))
    F[:, :, 0] = 1.0
    E = rng.multivariate_normal(np.zeros(N), Sigma, size=T).T
    R = np.einsum("itk,ik->it", F, b) + E
    return {"b_bar": b_bar, "tau": tau, "c": c, "lam": lam, "lam_tilde": lam_t,
            "theta": theta, "b": b, "Sigma": Sigma, "F": F, "R": R}


def covered(draws, truth, lo=2.5, hi=97.5):
    a, b = np.percentile(draws, lo, axis=0), np.percentile(draws, hi, axis=0)
    return (truth >= a) & (truth <= b)


def main() -> None:
    import arviz as az

    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=int, default=100)
    ap.add_argument("--N", type=int, default=6)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--draws", type=int, default=750)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--cores", type=int, default=1)
    ap.add_argument("--seed0", type=int, default=2000)
    args = ap.parse_args()

    N, K, T = args.N, args.K, args.T
    hp0 = simulation_hyperparameters(N, K, T)
    se = 1.0 / np.sqrt(T)

    print("=" * 78)
    print(f"REGULARISED HORSESHOE CALIBRATION -- matched dimensions")
    print(f"N={N} K={K} T={T} | {args.datasets} datasets | {args.chains} chains "
          f"x {args.draws} draws ({args.tune} tune)")
    print(f"tau_0 = {hp0.tau_0:.4e} = {TAU0_MULT} SE   "
          f"E[c] = {hp0.slab_sd:.4e} = {SLAB_OVER_TAU0} x tau_0 "
          f"(matching production, where E[c]/tau_0 = 4.864)")
    print(f"prior binding at tau_0: {100*hp0.binding_fraction():.2f}%")
    print(f"truth drawn from the model's own prior, INCLUDING the slab")
    print("=" * 78 + "\n")

    pooled = {f: [0, 0] for f in FIELDS}
    per_dataset = {f: [] for f in FIELDS}
    excluded, taus, cs, binds = [], [], [], []
    t_all = time()

    for d in range(args.datasets):
        seed = args.seed0 + d
        hp = simulation_hyperparameters(N, K, T)
        truth = simulate_from_prior(hp, N, K, T, seed)
        out = run_nuts(truth["R"], truth["F"], hp, n_draws=args.draws,
                       n_tune=args.tune, chains=args.chains, cores=args.cores,
                       seed0=30_000 + 10 * seed, progressbar=False)

        P = {f: np.concatenate([getattr(c, f) for c in out], axis=0)
             for f in ("b_bar", "tau", "c", "Sigma", "B", "lam_tilde", "lam_local")}
        div = sum(int(c.diverging.sum()) for c in out)
        depth_sat = float(np.mean(np.concatenate([c.tree_depth for c in out])
                                  >= hp.max_treedepth))
        rhat = float(max(np.nanmax(az.rhat(np.stack([c.b_bar for c in out]))),
                         np.nanmax(az.rhat(np.stack([c.tau for c in out]))),
                         np.nanmax(az.rhat(np.stack([c.c for c in out]))))) \
            if args.chains > 1 else np.nan
        bind = float((P["lam_tilde"] / P["lam_local"] < 0.99).mean())

        if (args.chains > 1 and rhat > RHAT_MAX) or depth_sat > DEPTH_SATURATION:
            excluded.append((seed, truth["tau"] / hp.tau_0, rhat, depth_sat))
            print(f"  {d:3d}: EXCLUDED  R-hat {rhat:.3f}, depth sat "
                  f"{depth_sat:.0%}", flush=True)
            continue

        pairs = {"b_bar": (P["b_bar"], truth["b_bar"]),
                 "theta": (P["B"] - P["b_bar"][:, None, :], truth["theta"]),
                 "b": (P["B"], truth["b"]),
                 "Sigma": (P["Sigma"], truth["Sigma"]),
                 "tau": (P["tau"], np.array(truth["tau"])),
                 "c": (P["c"], np.array(truth["c"]))}
        for f, (dr, tr) in pairs.items():
            cv = covered(dr, tr)
            pooled[f][0] += int(np.sum(cv)); pooled[f][1] += int(np.size(cv))
            per_dataset[f].append(float(np.mean(cv)))
        taus.append((truth["tau"], float(P["tau"].mean())))
        cs.append((truth["c"], float(P["c"].mean())))
        binds.append(bind)
        print(f"  {d:3d}: b_bar {np.mean(covered(P['b_bar'], truth['b_bar'])):.3f}"
              f"  tau {'in ' if covered(P['tau'], np.array(truth['tau'])) else 'OUT'}"
              f"  c {'in ' if covered(P['c'], np.array(truth['c'])) else 'OUT'}"
              f"  R-hat {rhat:.3f}  {div} div  bind {100*bind:.1f}%", flush=True)

    n_used = args.datasets - len(excluded)
    print(f"\n  {(time()-t_all)/60:.1f} min | {n_used} of {args.datasets} used, "
          f"{len(excluded)} excluded")
    if excluded:
        print("  EXCLUDED -- these bias the rate toward easy datasets; report them:")
        for s, tr, r, ds in excluded:
            print(f"    seed {s}: tau {tr:.1f}x tau_0, R-hat {r:.3f}, "
                  f"depth saturating {ds:.0%}")
    if n_used == 0:
        return

    print(f"\n  {'field':<8s} {'pooled':>14s} {'per-dataset mean':>18s} {'z vs 0.95':>10s}")
    for f in FIELDS:
        k, n = pooled[f]
        r = np.array(per_dataset[f])
        if n == 0 or len(r) < 2:
            continue
        sem = r.std(ddof=1) / np.sqrt(len(r))
        z = (r.mean() - 0.95) / sem if sem > 0 else np.nan
        print(f"  {f:<8s} {k:>6d} /{n:<7d} {r.mean():>10.3f} +/- {sem:.3f} {z:>+10.1f}")
    print("\n  No binomial p-value: intervals within a dataset are not independent.")

    tt, cc = np.array(taus), np.array(cs)
    print(f"\n  tau recovery: correlation(true, posterior mean) = "
          f"{np.corrcoef(tt[:,0], tt[:,1])[0,1]:+.3f}")
    print(f"  c   recovery: correlation(true, posterior mean) = "
          f"{np.corrcoef(cc[:,0], cc[:,1])[0,1]:+.3f}")
    print(f"  slab binding across datasets: mean {100*np.mean(binds):.1f}%, "
          f"range {100*np.min(binds):.1f}-{100*np.max(binds):.1f}%")
    print("  [if binding is near zero the slab never engaged and this run")
    print("   calibrated the plain horseshoe under another name]")


if __name__ == "__main__":
    main()