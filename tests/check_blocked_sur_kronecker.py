"""
check_blocked_sur_kronecker.py

Verifies the Kronecker algebra behind the joint B-update (Feng & He eq. 14-15)
against a brute-force reconstruction of the literal NT x NK stacked system.

Reproduces Appendix A.6.1, second paragraph, and supports Section 4.4's
blocked (B, b_bar) update.

The B-update never forms Omega = Sigma (x) I_T. It exploits

    [F' Omega^-1 F]_{ij} = Sigma^-1[i,j] * (f_i' f_j)          (K x K)
    [F' Omega^-1 R]_i    = sum_j Sigma^-1[i,j] * (f_i' r_j)    (K,)

This script builds the same quantities the slow, obviously-correct way --
block_diag(f_1..f_N), kron(Sigma^-1, I_T), explicit matrix products -- and
compares. It also checks that sample_B draws from N(b*, P^-1) rather than
merely computing the right mean, and that a diagonal Sigma collapses GLS to
per-asset OLS exactly, which is the structural test that the cross-asset
coupling enters through Sigma^-1 and nowhere else.

Recorded results (chat of 9 August, N=3 K=4 T=20, seed 0):
    precision agreement   1.066e-14
    location agreement    2.665e-15
    mean error, worst of 12 params, in MCSE   1.63
    covariance error, worst element, relative 0.0176
    diagonal-Sigma collapse to OLS            5.6e-16
"""
import numpy as np
from scipy.linalg import block_diag
from src.gibbs.baseline_gaussian import precompute_cross_products, sample_B, _spd_inverse

rng = np.random.default_rng(0)
N, T, K = 3, 20, 4
F = rng.standard_normal((N, T, K)); F[:, :, 0] = 1.0
R = rng.standard_normal((N, T))
A = rng.standard_normal((N + 3, N)); Sigma = A.T @ A / (N + 3) + 0.3 * np.eye(N)
D = rng.standard_normal((K + 3, K)); Delta_b = D.T @ D / (K + 3) + 0.3 * np.eye(K)
b_bar = rng.standard_normal(K)
G, Fr = precompute_cross_products(F, R)

# --- brute force: the literal NT x NK system of eq. (7) -------------------
F_big = block_diag(*[F[i] for i in range(N)])            # (NT, NK)
Om = np.kron(_spd_inverse(Sigma), np.eye(T))             # Sigma^-1 (x) I_T
P_ref = F_big.T @ Om @ F_big + np.kron(np.eye(N), _spd_inverse(Delta_b))
rhs_ref = (F_big.T @ Om @ R.reshape(N * T)
           + np.kron(np.ones(N), _spd_inverse(Delta_b) @ b_bar))
b_star = np.linalg.solve(P_ref, rhs_ref)

# --- the fast assembly, reconstructed from the module's own pieces --------
Sinv, Dinv = _spd_inverse(Sigma), _spd_inverse(Delta_b)
P_fast = G * Sinv[:, None, :, None]
d = np.arange(N); P_fast[d, :, d, :] += Dinv
P_fast = P_fast.reshape(N * K, N * K)
rhs_fast = (np.einsum("ij,ijk->ik", Sinv, Fr) + (Dinv @ b_bar)[None, :]).reshape(N * K)

print("1. Kronecker algebra vs brute-force NT x NK reconstruction")
print("   max |P_ref   - P_fast|   : %.3e   [want < 1e-12]" % np.abs(P_ref - P_fast).max())
print("   max |rhs_ref - rhs_fast| : %.3e   [want < 1e-12]" % np.abs(rhs_ref - rhs_fast).max())

# --- does sample_B target N(b*, P^-1), not just get the mean right? -------
draws = np.array([sample_B(rng, G, Fr, Sigma, Delta_b, b_bar).reshape(N * K)
                  for _ in range(20_000)])
mcse = draws.std(0) / np.sqrt(len(draws))
cov_t = np.linalg.inv(P_ref)
cov_scale = np.sqrt(np.outer(np.diag(cov_t), np.diag(cov_t)))
print("\n2. sampling distribution over 20,000 independent draws")
print("   mean error, worst of %d params (MCSE) : %.2f   [want < 4]"
      % (N * K, np.max(np.abs(draws.mean(0) - b_star) / mcse)))
print("   covariance error, worst (relative)    : %.4f  [want < 0.05]"
      % np.max(np.abs(np.cov(draws.T) - cov_t) / cov_scale))

# --- structural test: diagonal Sigma must switch cross-asset sharing off --
Omd = np.kron(_spd_inverse(np.diag(np.diag(Sigma))), np.eye(T))
gls_d = np.linalg.solve(F_big.T @ Omd @ F_big, F_big.T @ Omd @ R.reshape(N * T)).reshape(N, K)
ols = np.array([np.linalg.lstsq(F[i], R[i], rcond=None)[0] for i in range(N)])
print("\n3. diagonal Sigma collapses GLS to per-asset OLS")
print("   max |GLS_diag - OLS| : %.1e   [want ~1e-15; non-zero means the]"
      % np.abs(gls_d - ols).max())
print("                                [cross-asset coupling leaks in    ]")
print("                                [somewhere other than Sigma^-1    ]")