"""
tests/check_calibration_bayesian_lasso.py

Calibration of the Bayesian LASSO sampler.

Reproduces Table 4.6, the Bayesian LASSO column, and Appendix A.7.4, the Sigma
mismatch that moved coverage by less than Monte Carlo error. Supports Section
4.5 on prior-generative calibration, and Appendix A.7.5's second example, the
rank-uniformity chi^2 that returned p = 0.004 on clustered ranks.

The truth is drawn from the MODEL'S OWN PRIOR. Bayesian coverage is nominal
only when the data-generating truth comes from the prior being conditioned on,
so a deviation is then unambiguous evidence of a sampler bug. Draw the truth
from anywhere else and the theorem does not apply: a reading of 93.7% could be
the sampler or could be the mismatch, and there is no way to tell.

Two mismatches were present in the original version of this test and both are
corrected here:

  theta:    generate.py's draw_deviations_gaussian is the BASELINE's
            generative process. Fitting a Laplace prior to Gaussian truth would
            show under-coverage that is correct model behaviour, not a bug.
            theta is therefore drawn from Laplace(0, s/lambda) by inverse
            transform, with lambda from the model's own Gamma hyperprior.
            (This was correct in the original version.)
  Sigma:    was drawn from generate.py's _random_pd_matrix rather than from
            IW(nu_Sigma, V_Sigma). This was the one quantity that slipped
            through. Correcting it moved no row by more than its own Monte
            Carlo error at 20 datasets, which is itself informative: Sigma does
            not enter theta's prior under the plug-in scale, so a wrong Sigma
            perturbs coverage only through the likelihood. The equivalent
            mismatch in the baseline, where all four quantities were wrong,
            produced a +6.1 SE artefact on Delta_b_diag.

WEAK BUT PROPER hyperparameters throughout. At the production prior
(Delta_b_bar = 5.7e-07) the posterior for b_bar is pinned near its prior mean
whatever the data says, so intervals would cover trivially and the test would
confirm the prior rather than the code. Delta_b_bar = 4.0 here is weak but
proper: coverage of b_bar is undefined if b_bar is not actually drawn from
the prior the model uses.

REPORT THE PER-DATASET TEST, NOT THE POOLED BINOMIAL. Intervals within one
dataset share b_bar, Sigma and lambda, so pooling 30 of them as independent
manufactures significance. In testing, theta's pooled binomial p swung from
0.002 to 0.048 under a change that moved the per-dataset mean by 0.4 SE. The
binomial column is not reported here for that reason.

100 datasets at N=6, K=5, T=120 takes about 3 minutes.

Run from the project root:  python3 tests/check_calibration_bayesian_lasso.py
"""
import time

import numpy as np
from scipy.stats import invwishart

from src.gibbs.bayesian_lasso import LassoHyperparams, run_gibbs
from src.simulate.generate import (compute_returns, generate_predictors,
                                   generate_residuals)

N, K, T, REPS = 6, 5, 120, 100
DRAWS, BURN = 3000, 1000

weak = LassoHyperparams(
    b_bar_bar=np.zeros(K),
    Delta_b_bar=np.eye(K) * 4.0,
    nu_Sigma=N + 2.0,
    V_Sigma=np.eye(N) * 1.0,
    s=1.0,
    lambda_r=1.0,
    lambda_delta=0.04,
    sd_target=np.nan,
)

iu = np.triu_indices(N)
res = {k: [] for k in ["b_bar", "theta", "B", "Sigma", "lambda"]}

t0 = time.time()
for r in range(REPS):
    rng = np.random.default_rng(9000 + r)

    lam = np.sqrt(rng.gamma(shape=weak.lambda_r, scale=1.0 / weak.lambda_delta))
    b_bar = rng.multivariate_normal(weak.b_bar_bar, weak.Delta_b_bar)
    u = rng.random((N, K)) - 0.5
    theta = -(weak.s / lam) * np.sign(u) * np.log(1.0 - 2.0 * np.abs(u))
    b = b_bar[None, :] + theta
    Sigma = invwishart.rvs(df=weak.nu_Sigma, scale=weak.V_Sigma, random_state=rng)

    F = generate_predictors(rng, N, T, K)
    R = compute_returns(F, b, generate_residuals(rng, N, T, Sigma))

    fit = run_gibbs(R, F, weak, n_draws=DRAWS, n_burn=BURN, seed=r)
    theta_draws = fit.B - fit.b_bar[:, None, :]

    def covers(draws, truth):
        lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
        return ((lo <= truth) & (truth <= hi)).ravel()

    res["b_bar"].append(covers(fit.b_bar, b_bar))
    res["theta"].append(covers(theta_draws, theta))
    res["B"].append(covers(fit.B, b))
    res["Sigma"].append(covers(fit.Sigma[:, iu[0], iu[1]], Sigma[iu]))
    lo, hi = np.percentile(fit.lam, [2.5, 97.5])
    res["lambda"].append(np.array([lo <= lam <= hi]))

    if (r + 1) % 20 == 0:
        print(f"   {r+1}/{REPS} datasets ({time.time()-t0:.0f}s)", flush=True)

print(f"\nBAYESIAN LASSO | truth from the model's own prior | "
      f"N={N} K={K} T={T}, {REPS} datasets, {time.time()-t0:.0f}s\n")
print(f"{'parameter':<14s} {'coverage':>11s} {'pct':>7s} "
      f"{'per-dataset mean +/- se':>28s}")
for k in res:
    hits = sum(int(x.sum()) for x in res[k])
    n = sum(x.size for x in res[k])
    per_ds = np.array([x.mean() for x in res[k]])
    se = per_ds.std(ddof=1) / np.sqrt(len(per_ds))
    z = (per_ds.mean() - 0.95) / se
    print(f"{k:<14s} {hits:>5d}/{n:<5d} {100*hits/n:>6.1f}%      "
          f"{per_ds.mean():.3f} +/- {se:.3f}  ({z:+.1f} se)")

print("\n   [The per-dataset column is the honest test. Intervals within one")
print("    dataset are not independent (they share b_bar, Sigma and lambda),")
print("    so a pooled binomial test treats clustered data as independent and")
print("    manufactures significance.]")
print("   [lambda is expected to be the weakest row. It is the slowest-mixing")
print("    parameter in this sampler (R-hat 1.018, bulk ESS 384 on real data at")
print("    24,000 draws), and an interval estimated from an autocorrelated chain")
print("    is slightly too narrow, which under-covers. Same cause, two symptoms.]")
