"""
check_lasso_conditionals.py

Verifies each Bayesian LASSO conditional against a brute-force evaluation of
the model's OWN log joint density, evaluated on a grid. A conditional is
correct only if (log joint - claimed kernel) is CONSTANT in the parameter
being varied; the spread of that difference is the reported agreement.

Reproduces Appendix A.6.1, first paragraph. Was check_chunk2.py (local scale)
and check_chunk3.py (both global-scale updates) during development; merged
here because they test the same layer.

Recorded results:
    tau^2, GIG(p=1/2)                   1.7e-13 over 500 points
    lambda | tau^2, Gamma               2.2e-11 over 500 points
    lambda | theta, collapsed           7.8e-14 over 6,000 points

The 2.2e-11 is the largest of the four and belongs to the NON-COLLAPSED
update, which the sampler does not use. Its calculation carries an exponential
term of order exp(NK log lambda) inherited from the Gamma kernel at shape
3,601, so the absolute spread is larger for reasons of magnitude rather than
weaker verification. The collapsed update actually used agrees to 7.8e-14.

Run from the project root:  python3 tests/check_lasso_conditionals.py
"""
import numpy as np
from scipy.stats import gamma as gamma_dist

from src.gibbs.bayesian_lasso import (lasso_hyperparameters, sample_lambda,
                                      sample_lambda_collapsed, sample_tau2)

d = np.load("data/processed/size_bm_25_arrays.npz", allow_pickle=True)
F, R = d["F"], d["R"]
N, T, K = F.shape
hp = lasso_hyperparameters(R, F, K)
s, lam_cal = hp.s, hp.lambda_prior_mean
rng = np.random.default_rng(0)
NK = N * K

print(f"N={N} K={K} T={T}   s={s:.4f}   lambda_cal={lam_cal:.1f}   "
      f"r={hp.lambda_r:g}   delta={hp.lambda_delta:.4e}\n")

# ---- 1. tau^2: GIG(p=1/2), i.e. 1/tau^2 inverse Gaussian -----------------
# theta_ij ~ N(0, s^2 tau^2) and tau^2 ~ Exp(lambda^2/2), so as a function of
# tau^2 the joint is (tau^2)^(-1/2) exp(-theta^2/(2 s^2 tau^2) - lam^2 tau^2/2).
th0 = 1.7e-4
# the grid must cover tau^2's tail: truncating at 4e-5 reports a spurious
# 8-sigma disagreement in the moment check below
grid = np.linspace(1e-9, 4e-3, 200_000)
joint = (-0.5 * np.log(s ** 2 * grid) - th0 ** 2 / (2 * s ** 2 * grid)
         - lam_cal ** 2 * grid / 2)
kernel = -0.5 * np.log(grid) - th0 ** 2 / (2 * s ** 2 * grid) - lam_cal ** 2 * grid / 2
print("1. tau^2  |  GIG(p=1/2)")
print(f"   log joint - kernel, spread: {(joint - kernel).ptp():.3e}")

w = np.exp(joint - joint.max()); w /= w.sum()
draws = sample_tau2(np.full(2_000_000, th0), s, lam_cal, rng)
mcse = draws.std() / np.sqrt(2e6)
print(f"   E[tau^2]: sampler {draws.mean():.6e}   numerical integration "
      f"{(w*grid).sum():.6e}   z = {(draws.mean()-(w*grid).sum())/mcse:+.2f}")

# ---- 2. lambda | tau^2: Gamma(r + NK, delta + sum(tau^2)/2) --------------
# NOT used by the sampler; retained as an independent confirmation that the
# collapsed update below targets the same distribution.
tau2 = rng.exponential(scale=2 / lam_cal ** 2, size=(N, K))
g2 = np.linspace(1e5, 8e5, 500)                       # values of lambda^2
joint2 = (NK * np.log(g2 / 2) - g2 * tau2.sum() / 2
          + (hp.lambda_r - 1) * np.log(g2) - hp.lambda_delta * g2)
shape = hp.lambda_r + NK
rate = hp.lambda_delta + 0.5 * tau2.sum()
claimed2 = gamma_dist.logpdf(g2, a=shape, scale=1.0 / rate)
print("\n2. lambda | tau^2  |  Gamma(r+NK, delta+sum(tau^2)/2)   [NOT USED]")
print(f"   log joint - kernel, spread: {(joint2 - claimed2).ptp():.3e}")
print(f"   shape = r + NK = {shape:.0f}; relative sd of lambda = "
      f"{1/(2*np.sqrt(shape)):.4f}")

# ---- 3. lambda | theta, collapsed over tau^2  ---------------------------
# The scale mixture integrates to Laplace exactly, so tau^2 can be
# marginalised out: p(lambda|theta) prop lambda^(NK+2r-1) exp(-lambda S1/s
# - delta lambda^2), with S1 = sum|theta_ij|.
Nc, Kc = 4, 3
theta = rng.standard_normal((Nc, Kc)) * 1e-3
S1 = np.abs(theta).sum()
g3 = np.linspace(50, 4000, 6000)
joint3 = (theta.size * np.log(g3 / (2 * s)) - g3 * S1 / s
          + (hp.lambda_r - 1) * np.log(g3 ** 2) - hp.lambda_delta * g3 ** 2
          + np.log(2 * g3))
claimed3 = ((theta.size + 2 * hp.lambda_r - 1) * np.log(g3)
            - g3 * S1 / s - hp.lambda_delta * g3 ** 2)
print("\n3. lambda | theta, collapsed  |  THE UPDATE THE SAMPLER USES")
print(f"   log joint - kernel, spread: {(joint3 - claimed3).ptp():.3e}")

# the scale mixture really is Laplace: integrate tau^2 out numerically
print("\n   scale mixture check: int N(theta;0,s^2 t) Exp(t; lam^2/2) dt = Laplace")
lg = np.linspace(-22, 5, 400_000)
t2 = np.exp(lg)
for th in (1e-4, 5e-4, 2e-3):
    integ = ((1 / np.sqrt(2 * np.pi * s ** 2 * t2))
             * np.exp(-th ** 2 / (2 * s ** 2 * t2))
             * (lam_cal ** 2 / 2) * np.exp(-lam_cal ** 2 * t2 / 2) * t2)
    num = np.trapezoid(integ, lg)
    lap = (lam_cal / (2 * s)) * np.exp(-lam_cal * abs(th) / s)
    print(f"      theta={th:.0e}:  marginal/Laplace = {num/lap:.8f}")

# ---- 4. recovery across the reachable range of lambda -------------------
print("\n4. COLLAPSED DRAW vs NUMERICAL INTEGRATION OF ITS TARGET")
print(f"   {'lambda true':>12s} {'sampled':>10s} {'exact':>10s} {'z':>7s}")
for lt in (200.0, 417.0, 601.4, 1345.0, 4000.0):
    u = rng.random((N, K)) - 0.5
    th = -(s / lt) * np.sign(u) * np.log(1 - 2 * np.abs(u))   # Laplace
    dr = np.array([sample_lambda_collapsed(th, s, hp, rng) for _ in range(3000)])
    a = th.size + 2 * hp.lambda_r - 1
    rt = np.abs(th).sum() / s
    gg = np.linspace(dr.mean() * 0.9, dr.mean() * 1.1, 300_000)
    lg2 = (a - 1) * np.log(gg) - rt * gg - hp.lambda_delta * gg ** 2
    ww = np.exp(lg2 - lg2.max()); ww /= ww.sum()
    exact = (ww * gg).sum()
    z = (dr.mean() - exact) / (dr.std() / np.sqrt(3000))
    print(f"   {lt:>12.1f} {dr.mean():>10.2f} {exact:>10.2f} {z:>7.2f}")
print("   [the 20x range matters: the rejection step's acceptance probability")
print("    falls with lambda, and an earlier version of this function raised")
print("    RuntimeError at lambda = 1345 while working correctly at 450]")