"""
check_lasso_blocked_sur.py

Verifies the Bayesian LASSO's blocked (B, b_bar) draw. This is the ONLY check
on that update: no sequential sampler is built for this model, so there is no
second implementation to agree with.

NOT a duplicate of tests/check_blocked_sur_kronecker.py, which checks the
BASELINE's update. The precision matrices differ: the baseline shares one
dense Delta_b across assets, whereas the LASSO has a per-asset DIAGONAL
1/(s^2 tau_ij^2), a different coupling block, and a (b_bar, b_bar) block that
SUMS per-asset precisions rather than multiplying one by N.

Reproduces Appendix A.7.1 (second paragraph) and A.7.2 (first paragraph).

Recorded results (N=5, K=4, T=60, seed 11):
    likelihood precision vs brute force     2.274e-13
    likelihood location vs brute force      5.684e-14
    log joint vs Gaussian implied by (P, rhs)   2.046e-12
    60,000 draws: mean max |z| 1.34, median 0.60
    60,000 draws: covariance corr 0.999922, max discrepancy 1.76 MCSE
    copy-paste trap silent in 111 of 200 random tau^2 draws

Run from the project root:  python3 tests/check_lasso_blocked_sur.py
"""
import numpy as np
from scipy.linalg import block_diag

from src.gibbs.baseline_gaussian import _spd_inverse, precompute_cross_products
from src.gibbs.bayesian_lasso import sample_B_and_b_bar

rng = np.random.default_rng(11)
N, K, T = 5, 4, 60
F = rng.standard_normal((N, T, K))
R = rng.standard_normal((N, T)) * 0.5
A = rng.standard_normal((N + 4, N)); Sigma = A.T @ A / (N + 4)
tau2 = rng.exponential(scale=1.0, size=(N, K))
s = 0.6
b_bar_bar = rng.standard_normal(K) * 0.1
Delta_b_bar = np.diag(rng.uniform(0.4, 1.2, K))

G, Fr = precompute_cross_products(F, R)
NK, M = N * K, N * K + K
Sinv, Dbb_inv = _spd_inverse(Sigma), _spd_inverse(Delta_b_bar)
D_inv = 1.0 / (s * s * tau2)

P = np.zeros((M, M))
blk = G * Sinv[:, None, :, None]
ii, kk = np.arange(N)[:, None], np.arange(K)[None, :]
blk[ii, kk, ii, kk] += D_inv
P[:NK, :NK] = blk.reshape(NK, NK)
rows, cols = np.arange(NK), NK + np.tile(np.arange(K), N)
P[rows, cols] = -D_inv.ravel()
P[cols, rows] = -D_inv.ravel()
P[NK:, NK:] = np.diag(D_inv.sum(axis=0)) + Dbb_inv
rhs = np.empty(M)
rhs[:NK] = np.einsum("ij,ijk->ik", Sinv, Fr, optimize=True).reshape(NK)
rhs[NK:] = Dbb_inv @ b_bar_bar

print("1. BRUTE-FORCE NT x NK STACKED SYSTEM")
X = block_diag(*[F[i] for i in range(N)])
y = R.ravel()
Omega_inv = np.kron(Sinv, np.eye(T))
lik_prec = X.T @ Omega_inv @ X
lik_loc = X.T @ Omega_inv @ y
prior_prec = np.diag(D_inv.ravel())
print(f"   likelihood precision  max |diff| "
      f"{np.abs(P[:NK, :NK] - prior_prec - lik_prec).max():.3e}")
print(f"   likelihood location   max |diff| {np.abs(rhs[:NK] - lik_loc).max():.3e}")
print(f"   (X is {X.shape}, Omega^-1 is {Omega_inv.shape}, built with no")
print("    Kronecker shortcuts, so this tests Sigma^-1[i,j] * f_i'f_j by a")
print("    genuinely different route)")

print("\n2. LOG JOINT vs THE GAUSSIAN IMPLIED BY (P, rhs)")


def log_joint(B, b_bar):
    E = R - np.einsum("itk,ik->it", F, B)
    lp = -0.5 * np.trace(Sinv @ (E @ E.T))
    th = B - b_bar[None, :]
    lp += (-0.5 * th ** 2 / (s ** 2 * tau2)).sum()
    dd = b_bar - b_bar_bar
    lp += -0.5 * dd @ Dbb_inv @ dd
    return lp


resid = []
for _ in range(80):
    x = rng.standard_normal(M) * 0.5
    resid.append(log_joint(x[:NK].reshape(N, K), x[NK:])
                 - (-0.5 * x @ P @ x + x @ rhs))
print(f"   spread over 80 points: {np.ptp(resid):.3e}")
print(f"   P symmetric {np.allclose(P, P.T)}   min eigenvalue "
      f"{np.linalg.eigvalsh(P).min():.4e}")

print("\n3. SAMPLING DISTRIBUTION vs P^-1 rhs AND P^-1")
n_draw = 60_000
r2 = np.random.default_rng(5)
draws = np.empty((n_draw, M))
for m in range(n_draw):
    Bd, bd = sample_B_and_b_bar(r2, G, Fr, Sigma, tau2, s, b_bar_bar, Delta_b_bar)
    draws[m] = np.concatenate([Bd.ravel(), bd])
Pinv = _spd_inverse(P)
target = Pinv @ rhs
emp, emp_cov = draws.mean(0), np.cov(draws.T)
z = (emp - target) / np.sqrt(np.diag(Pinv) / n_draw)
print(f"   mean:  max |z| {np.abs(z).max():.2f}   median |z| "
      f"{np.median(np.abs(z)):.2f}   ({M} parameters)")
print(f"   covariance: max discrepancy "
      f"{np.abs(emp_cov-Pinv).max()/np.sqrt(np.abs(Pinv).max()**2/n_draw):.2f} MCSE"
      f"   corr(vec) {np.corrcoef(emp_cov.ravel(), Pinv.ravel())[0,1]:.6f}")

print("\n4. THE COPY-PASTE TRAP")
silent, shifts = 0, []
r3 = np.random.default_rng(21)
n_trial = 200
for _ in range(n_trial):
    t2t = r3.exponential(scale=1.0, size=(N, K))
    Dt = 1.0 / (s * s * t2t)
    Pt = np.zeros((M, M))
    b2 = G * Sinv[:, None, :, None]
    b2[ii, kk, ii, kk] += Dt
    Pt[:NK, :NK] = b2.reshape(NK, NK)
    Pt[rows, cols] = -Dt.ravel(); Pt[cols, rows] = -Dt.ravel()
    Pt_ok = Pt.copy()
    Pt_ok[NK:, NK:] = np.diag(Dt.sum(axis=0)) + Dbb_inv
    Pt[NK:, NK:] = N * np.diag(Dt[0]) + Dbb_inv
    if np.linalg.eigvalsh(Pt).min() <= 0:
        continue
    silent += 1
    sd_b = np.sqrt(np.diag(_spd_inverse(Pt_ok))[NK:])
    shifts.append(np.abs((_spd_inverse(Pt) @ rhs
                          - _spd_inverse(Pt_ok) @ rhs)[NK:] / sd_b).max())
print(f"   'N * D_1^-1' instead of 'sum_i D_i^-1' in the b_bar block,")
print(f"   over {n_trial} random draws of tau^2:")
print(f"      still positive definite, so the Cholesky SUCCEEDS and the error")
print(f"      is silent: {silent}/{n_trial}")
if shifts:
    shifts = np.array(shifts)
    print(f"      when silent, b_bar's posterior mean shifts by median "
          f"{np.median(shifts):.1f}, max {shifts.max():.1f} posterior sd")
print("   [the Cholesky catches it only sometimes, and which times depends on")
print("    the tau^2 draw, so a run can proceed for thousands of sweeps against")
print("    the wrong target and then raise, or never raise at all]")
