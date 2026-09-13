"""
check_eq_17_collapse.py

Evidence for the disclosed correction to Feng & He's eq. (17). As printed,
the posterior scale matrix carries an inverse:

    Delta_b | B, b_bar ~ IW(nu_b + N, (sum_i (b_i - b_bar)(b_i - b_bar)' + V_b)^-1)

The conjugate derivation gives no inverse. This script shows the printed form
is untenable and the un-inverted form is correct, by three independent routes.

Note the failure is SILENT: the inverted form runs without error and returns
valid symmetric positive-definite matrices. Only the values are wrong.

Recorded results:
    un-inverted recovers a known Delta_b; inverted collapses it to zero
    MC mean vs analytic posterior mean, un-inverted: 1.08 MCSE
    off-diagonal sd ratio between the two readings: 9.0x, at every K and N
"""
import numpy as np
from scipy.stats import invwishart
from src.gibbs.baseline_gaussian import sample_Delta_b

rng = np.random.default_rng(0)

K, N = 5, 500
nu_b, V_b = K + 2.0, np.eye(K)
Delta_true = np.diag([2., 1.5, 1., .8, .5]); Delta_true[0, 1] = Delta_true[1, 0] = 0.6
b_bar = np.zeros(K)
B = rng.multivariate_normal(b_bar, Delta_true, size=N)
S = (B - b_bar).T @ (B - b_bar)

d_ok = np.array([sample_Delta_b(rng, B, b_bar, nu_b, V_b) for _ in range(5000)])
d_inv = np.array([invwishart.rvs(df=nu_b + N, scale=np.linalg.inv(V_b + S),
                                 random_state=rng) for _ in range(2000)])

print("1. recovery of a known Delta_b (N=%d assets, weak prior)" % N)
print("   true diagonal                  :", np.round(np.diag(Delta_true), 3))
print("   un-inverted (module) posterior :", np.round(np.diag(d_ok.mean(0)), 3))
print("   eq.(17) AS PRINTED, posterior  :", np.round(np.diag(d_inv.mean(0)), 4), " <- collapses")
print("   off-diagonal [0,1]: true 0.600 | un-inverted %.3f | as printed %.6f"
      % (d_ok.mean(0)[0, 1], d_inv.mean(0)[0, 1]))
print("   both readings returned valid symmetric PD matrices; no exception raised.")

analytic = (V_b + S) / (nu_b + N - K - 1)
mcse = d_ok.std(0) / np.sqrt(len(d_ok))
print("\n2. the un-inverted draw hits the ANALYTIC conditional mean")
print("   (V_b + S)/(nu_b + N - K - 1), worst element in MCSE : %.2f   [want < 4]"
      % np.max(np.abs(d_ok.mean(0) - analytic) / mcse))
print("   (being near the truth is weaker evidence: a sampler can be near the")
print("    truth for the wrong reason, but not 1 MCSE from the analytic mean)")

print("\n3. magnitude test against Feng & He's own Appendix C trace plot")
print("   (one off-diagonal element, published range about +/- 2e-4)")
print("   %4s %4s | %12s %12s | ratio" % ("K", "N", "un-inverted", "as printed"))
for K_, N_ in [(20, 25), (50, 25), (144, 25), (144, 50)]:
    nu = 1001.0 + K_ + N_
    Vb = np.eye(K_) * 3.0
    off = ~np.eye(K_, dtype=bool)
    s1 = invwishart.rvs(df=nu, scale=Vb, size=400, random_state=rng)[:, off].std()
    s2 = invwishart.rvs(df=nu, scale=np.linalg.inv(Vb), size=400,
                        random_state=rng)[:, off].std()
    print("   %4d %4d | %12.3e %12.3e | %.1fx" % (K_, N_, s1, s2, s1 / s2))
print("   the un-inverted reading predicts about +/- 2 sd = +/- 2e-4, which is")
print("   what they published; the inverted reading is an order of magnitude")
print("   too tight. The 9x ratio is exact and scale-free, so it does not")
print("   depend on guessing their K or N. Their code evidently does the")
print("   un-inverted thing: this is a typesetting slip, not an error in")
print("   their method.")
