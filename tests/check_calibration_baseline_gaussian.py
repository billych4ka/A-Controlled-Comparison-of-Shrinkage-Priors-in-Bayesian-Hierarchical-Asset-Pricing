"""
Calibration of the Gaussian baseline sampler.

Reproduces Table 4.6, the Gaussian baseline column, and Appendix A.6.4's
first paragraph: the initial mismatched generic SUR generator placed
Delta_theta 6.1 standard errors from nominal, falling to 1.8 once the truth
was drawn from the fitted prior. Supports Section 4.5 on prior-generative
calibration.

The truth is drawn from the MODEL'S OWN PRIOR, not from simulate_sur_data.
Bayesian coverage is nominal only when the data-generating truth comes from
the prior being conditioned on; simulate_sur_data draws Delta_b and Sigma from
_random_pd_matrix and b_bar from a standard normal, none of which match the
priors used in fitting. Under that mismatch this test reported B at 93.7%
(-2.8 se) and Delta_b_diag at 98.6% (+6.1 se) -- both artefacts, both gone
once the truth is drawn correctly.

Weak hyperparameters throughout: at the production prior a coverage test would
confirm the prior rather than the code.
"""
import numpy as np, time
from scipy.stats import invwishart, binomtest
from src.gibbs.baseline_gaussian import run_gibbs, GaussianBaselineHyperparams
from src.simulate.generate import generate_predictors, generate_residuals, compute_returns

N, K, T, REPS = 6, 5, 200, 100
hp = GaussianBaselineHyperparams(np.zeros(K), np.eye(K)*1.0,
                                 K+2.0, np.eye(K), N+2.0, np.eye(N))
iu = np.triu_indices(N)                       # unique Sigma entries only
res = {k: [] for k in ["b_bar", "theta", "B", "Sigma", "Delta_b_diag"]}

t0 = time.time()
for r in range(REPS):
    rng = np.random.default_rng(9000 + r)
    Delta_b = invwishart.rvs(df=hp.nu_b, scale=hp.V_b, random_state=rng)
    Sigma   = invwishart.rvs(df=hp.nu_Sigma, scale=hp.V_Sigma, random_state=rng)
    b_bar   = rng.multivariate_normal(hp.b_bar_bar, hp.Delta_b_bar)
    theta   = rng.multivariate_normal(np.zeros(K), Delta_b, size=N)
    b       = b_bar[None, :] + theta
    F = generate_predictors(rng, N, T, K)
    R = compute_returns(F, b, generate_residuals(rng, N, T, Sigma))

    dr = run_gibbs(R, F, hp, n_draws=1500, n_burn=500, seed=r)
    th = dr.B - dr.b_bar[:, None, :]
    for key, d, t in [("b_bar", dr.b_bar, b_bar), ("theta", th, theta), ("B", dr.B, b),
                      ("Sigma", dr.Sigma[:, iu[0], iu[1]], Sigma[iu]),
                      ("Delta_b_diag", dr.Delta_b_diag, np.diag(Delta_b))]:
        lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
        res[key].append((t >= lo) & (t <= hi))

print("truth drawn from the model's own prior | N=%d K=%d T=%d, %d datasets, %.0fs\n"
      % (N, K, T, REPS, time.time()-t0))
print("%-14s %11s %7s %10s %26s" % ("parameter", "coverage", "pct", "binom p",
                                    "per-dataset mean +/- se"))
for k in res:
    h = sum(int(x.sum()) for x in res[k]); n = sum(x.size for x in res[k])
    v = np.array([x.mean() for x in res[k]]); se = v.std(ddof=1)/np.sqrt(len(v))
    print("%-14s %5d/%-5d %7.1f%% %10.3f      %.3f +/- %.3f  (%+.1f se)"
          % (k, h, n, 100*h/n, binomtest(h, n, 0.95).pvalue, v.mean(), se,
             (v.mean()-0.95)/se))
print("\nbinomial p pools intervals that are NOT independent within a dataset "
      "(30 per dataset from one chain). The per-dataset test on the right is "
      "the honest one -- prefer it.")