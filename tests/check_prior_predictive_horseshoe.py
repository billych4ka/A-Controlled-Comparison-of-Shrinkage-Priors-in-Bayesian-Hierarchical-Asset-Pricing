"""
check_prior_predictive_horseshoe.py -- what does tau ~ C+(0, tau_0) actually imply?

Validation step 3. NOTE THE FRAMING: this is not a correctness check. That
was done by check_chunk2.py, whose total-log-density test against an
independent NumPy reference establishes that the prior block is what it
claims to be. This script answers a CALIBRATION question -- what the prior
implies about sparsity and about explained variance -- which is precisely
what Piironen & Vehtari recommend doing when a model is more complicated
than the linear regression their closed form was derived for:

    "we recommend a pragmatic approach of drawing from the prior for
    different values of tau and studying the effect on the sparsity"

Because the prior on theta involves no data, the draws are taken in NumPy
rather than through pm.sample_prior_predictive. Two reasons: Sigma's prior is
a pm.Potential on a pm.Flat variable, which prior-predictive sampling cannot
draw from; and Sigma is irrelevant to what is being asked, since the question
concerns theta's marginal behaviour. The distributions drawn here are the
ones build_model declares, and check_chunk2 is what ties the two together.

THREE QUESTIONS, and the third is the uncomfortable one.

  1. Sparsity. Does the prior on m_eff sit where p_0 says it should? The
     analytic value at tau = tau_0 exactly is 30.04 per asset (not 23 -- the
     closed form assumes unit-variance predictors and this project's
     interaction columns have a root mean square near 1.4). Under the full
     prior, with tau itself half-Cauchy, m_eff should be centred near that
     with a wide spread in both directions.

  2. Coefficient scale. The median |theta| should be comparable to the scale
     the other two models are calibrated to, sd_target = 1.309e-04.

  3. Implied prior R^2. The baseline and the LASSO are both calibrated to
     exactly 0.05. The horseshoe cannot be, because a Cauchy-tailed prior has
     no variance to calibrate -- so the implied R^2 has no mean, and its
     median and upper quantiles are large. This is the same diagnostic that
     condemned Feng & He's disclosed hyperparameters (implied R^2 ~ 8,756),
     and it must be reported for this model too rather than applied only
     where it is convenient. The difference is in the reason: Feng & He's
     prior is mis-SCALED at this standardisation, whereas the horseshoe's is
     correctly scaled at the median and heavy-TAILED in the aggregate, which
     is what the prior is for. The regularised horseshoe's slab is the
     principled cap, and this is a third independent argument for fitting it.

    DO NOT USE THIS OUTPUT TO REVISE tau_0. p_0 = 23 is fixed a priori on the
    same discipline as target_r2 = 0.05, the posterior tau lands at about 5%
    of tau_0 regardless, and adjusting a hyperparameter because a prior
    summary looked uncomfortable is selection on outcomes.

Run from the project root:

    python3 check_prior_predictive_horseshoe.py

Runs in a few seconds. Record the printed table before deleting.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.nuts.horseshoe import horseshoe_hyperparameters, implied_m_eff

M = 20_000            # prior draws
M_R2 = 4_000          # subset used for the R^2 quadratic form (144x144 per draw)
SEED = 0

d = np.load(Path("../data/processed/size_bm_25_arrays.npz"), allow_pickle=True)
F, R = d["F"], d["R"]
N, T, K = F.shape
hp = horseshoe_hyperparameters(R, F, K)

rng = np.random.default_rng(SEED)
rms = np.sqrt((F ** 2).mean(axis=1))          # (N,K), the scale entering a_j
C = np.cov(F.reshape(-1, K).T)                # predictor covariance, as in prior_implied_r2
var_r = float(R.var())
sd_bbar = float(np.sqrt(hp.Delta_b_bar[0, 0]))

# --- draws from the prior. One asset's worth of local scales: theta_ij are
#     exchangeable across assets given tau, so the marginal behaviour of one
#     asset's 144 deviations is what m_eff and the implied R^2 are about.
tau = hp.tau_0 * np.abs(rng.standard_cauchy(M))
lam = np.abs(rng.standard_cauchy((M, K)))
z = rng.standard_normal((M, K))
theta = z * lam * tau[:, None]
b_bar = sd_bbar * rng.standard_normal((M, K))
b = b_bar + theta

print(f"size_bm_25: N={N} T={T} K={K}")
print(f"tau_0 = {hp.tau_0:.6e}   p0 = {hp.p0}   n_choice = {hp.n_choice!r}")
print(f"{M:,} prior draws, seed {SEED}\n")

# --- 1. sparsity
a = tau[:, None] / hp.sigma_pooled * np.sqrt(T) * rms[0]
m_eff = (a / (1.0 + a)).sum(axis=1)
analytic = implied_m_eff(hp, F)
print("1. implied effective model size, per asset (of K = 144)")
for q in (5, 25, 50, 75, 95, 99):
    print(f"     p{q:<3d}{np.percentile(m_eff, q):9.2f}")
print(f"     analytic at tau = tau_0 exactly           {analytic:9.2f}")
print(f"     P(m_eff > K/2) {np.mean(m_eff > K / 2):.3f}     "
      f"P(m_eff < 5) {np.mean(m_eff < 5):.3f}")
med = np.median(m_eff)
print(f"     median vs analytic: {med:.2f} vs {analytic:.2f} -- these should be close;"
      f" a large gap means the prior draws and implied_m_eff disagree")

# --- 2. coefficient scale
print("\n2. coefficient scale")
print(f"     median |theta|                            {np.median(np.abs(theta)):.4e}")
print(f"     sd_target (baseline / LASSO calibration)  {hp.sd_target:.4e}")
print(f"     ratio                                     "
      f"{np.median(np.abs(theta)) / hp.sd_target:.3f}")
print("     comparable at the MEDIAN is the point: the horseshoe differs from the")
print("     other models in its tails, not in its central scale.")

# --- 3. implied prior R^2
sub = slice(0, M_R2)
r2 = np.einsum("mj,jk,mk->m", b[sub], C, b[sub]) / var_r
print(f"\n3. implied prior R^2   ({M_R2:,} draws; baseline and LASSO are exactly 0.0500)")
for q in (5, 25, 50, 75, 90, 95, 99):
    print(f"     p{q:<3d}{np.percentile(r2, q):14.4f}")
print(f"     P(R^2 > 1) = {np.mean(r2 > 1):.4f}")
print(f"     the MEAN is {r2.mean():.3g} and is not a summary of anything: a")
print("     Cauchy-tailed prior has no second moment, so the sample mean is")
print("     determined by the largest draw and does not converge.")

# the mechanism, so the number is explicable rather than alarming
j_max = np.abs(theta[sub]).argmax(axis=1)
biggest = np.abs(theta[sub][np.arange(M_R2), j_max])
print(f"\n     mechanism: the largest |theta| in a draw has median {np.median(biggest):.4e},")
print(f"     which alone contributes a median R^2 of "
      f"{np.median(biggest ** 2 * np.diag(C)[j_max]) / var_r:.3f}.")
print("     One tail coefficient accounts for most of the implied R^2.")

print("\n     DISCLOSE, DO NOT REVISE. See the module docstring.")