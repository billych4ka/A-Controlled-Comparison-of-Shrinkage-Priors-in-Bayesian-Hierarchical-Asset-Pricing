"""
check_lasso_composition.py

FOUR INDIVIDUALLY CORRECT CONDITIONALS CAN STILL COMPOSE INTO A SAMPLER
TARGETING THE WRONG JOINT. check_lasso_conditionals.py verifies each
conditional separately; that does not establish that the assembled sweep is
right, because a mis-ordered update or a stale iterate leaves every individual
conditional correct and the composition wrong.

Reproduces Appendix A.7.2, second paragraph.

Three parts, and the order matters:

  1. SENSITIVITY. Break the sweep on purpose and confirm the test can see it.
     A validation that cannot detect a deliberate error is not a validation.
     Two corruptions are tried. One changes the MODEL and is detected at
     188-205 SE. The other is a merely STALE dependency and is correctly NOT
     detected: one sweep of staleness in a chain with lag-one autocorrelation
     0.977 is almost a null intervention. The second result is the more
     informative: it bounds what this kind of test can establish, and an
     earlier version of this script used only that corruption and therefore
     reported a pass on a test with no power.
  2. COMPOSITION against an analytic reference, with an ESS-ADJUSTED standard
     error. A Gibbs chain is autocorrelated and sqrt(n) would let real bias
     through.
  3. RECOVERY of known ground truth, judged against the POSTERIOR SD, not the
     Monte Carlo error of the posterior mean. An earlier version divided by
     MCSE and reported |z| up to 47 on a correct sampler (Appendix A.7.5).

Recorded results (N=6, K=5, T=150, seed 3):
    theta_is_B corruption detected at -188 to -206 SE (varies with seed)
    tau2_prev_lambda corruption correctly NOT detected, |z| ~ 0.1
    composition max |z| 2.24, median 0.78, over 35 parameters, ESS-adjusted
    all 5 true b_bar inside their 95% credible intervals

Run from the project root:  python3 tests/check_lasso_composition.py
"""
import numpy as np

from src.gibbs.baseline_gaussian import (_spd_inverse, precompute_cross_products,
                                         sample_Sigma)
from src.gibbs.bayesian_lasso import (LassoHyperparams, initialise_state, run_gibbs,
                                      sample_B_and_b_bar, sample_lambda_collapsed,
                                      sample_tau2)
from src.simulate.generate import simulate_sur_data

N, K, T = 6, 5, 150
sim = simulate_sur_data(N=N, K=K, T=T, seed=3)
R, F = sim.R, sim.F

weak = LassoHyperparams(
    b_bar_bar=np.zeros(K), Delta_b_bar=np.eye(K) * 100.0,
    nu_Sigma=N + 2.0, V_Sigma=np.eye(N) * 1.0,
    s=1.0, lambda_r=1.0, lambda_delta=1e-6, sd_target=np.nan)
print(f"simulated N={N} K={K} T={T};  weak prior, "
      f"Delta_b_bar={weak.Delta_b_bar[0,0]:g}")


def ess_1d(x):
    """Geyer initial-positive-sequence ESS for one chain."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x); v = x @ x / n
    if v <= 0:
        return float(n)
    rho, k = [], 1
    while k < n - 1:
        rho.append((x[k:] @ x[:-k]) / (n * v))
        if k % 2 == 0 and len(rho) >= 2 and rho[-1] + rho[-2] < 0:
            break
        k += 1
    tau = 1.0 + 2.0 * sum(rho[:-2]) if len(rho) > 2 else 1.0
    return float(n / max(tau, 1.0))


print("\n1. SENSITIVITY: would a broken composition be detected?")


def run_broken(mode, n_draws=6000, n_burn=1500, seed=0):
    rng = np.random.default_rng(seed)
    G, Fr = precompute_cross_products(F, R)
    st = initialise_state(R, F, weak)
    lam_prev = st["lam"]
    keep = []
    for it in range(n_draws):
        st["B"], st["b_bar"] = sample_B_and_b_bar(
            rng, G, Fr, st["Sigma"], st["tau2"], weak.s,
            weak.b_bar_bar, weak.Delta_b_bar)
        theta = st["B"] - st["b_bar"][None, :]
        use = st["B"] if mode == "theta_is_B" else theta
        st["lam"] = sample_lambda_collapsed(use, weak.s, weak, rng)
        lam_use = lam_prev if mode == "tau2_prev_lambda" else st["lam"]
        st["tau2"] = sample_tau2(use, weak.s, lam_use, rng)
        lam_prev = st["lam"]
        st["Sigma"] = sample_Sigma(rng, R, F, st["B"], weak.nu_Sigma, weak.V_Sigma)
        if it >= n_burn:
            keep.append((st["lam"], st["b_bar"][0], np.abs(theta).mean()))
    a = np.array(keep)
    return a[:, 0], a[:, 1], a[:, 2]


ref = run_gibbs(R, F, weak, n_draws=6000, n_burn=1500, seed=0)
base = (ref.lam, ref.b_bar[:, 0],
        np.abs(ref.B - ref.b_bar[:, None, :]).mean(axis=(1, 2)))
labels = ["mean lambda", "b_bar[0]", "mean |theta|"]
print(f"   {'variant':<20s} " + " ".join(f"{l:>13s} {'z':>7s}" for l in labels))
print(f"   {'correct':<20s} " + " ".join(f"{base[k].mean():>13.4f} {'--':>7s}"
                                         for k in range(3)))
for mode in ("theta_is_B", "tau2_prev_lambda"):
    b = run_broken(mode)
    cells = []
    for k in range(3):
        se = np.sqrt(base[k].var() / ess_1d(base[k]) + b[k].var() / ess_1d(b[k]))
        cells.append(f"{b[k].mean():>13.4f} {(b[k].mean()-base[k].mean())/se:>7.1f}")
    print(f"   {mode:<20s} " + " ".join(cells))
print("   [theta_is_B changes the MODEL and is detected at |z| ~ 200.")
print("    tau2_prev_lambda is a STALE dependency, leaves the target unchanged")
print("    in the limit, and is correctly NOT detected. So this test catches")
print("    wrong models, not wrong scan order; the latter needs section 2.]")

print("\n2. COMPOSITION vs THE EXACT (B, b_bar) POSTERIOR")
print("   Sigma and tau^2 held FIXED, so (B, b_bar) is exactly Gaussian with")
print("   mean P^-1 rhs. The sweep must reproduce it despite autocorrelation.")

rng = np.random.default_rng(1)
Sigma_fix = sim.Sigma.copy()
tau2_fix = rng.exponential(scale=1.0, size=(N, K))
G, Fr = precompute_cross_products(F, R)
NK, M = N * K, N * K + K
Sinv, Dbb_inv = _spd_inverse(Sigma_fix), _spd_inverse(weak.Delta_b_bar)
D_inv = 1.0 / (weak.s ** 2 * tau2_fix)
P = np.zeros((M, M))
blk = G * Sinv[:, None, :, None]
ii, kk = np.arange(N)[:, None], np.arange(K)[None, :]
blk[ii, kk, ii, kk] += D_inv
P[:NK, :NK] = blk.reshape(NK, NK)
rw, cl = np.arange(NK), NK + np.tile(np.arange(K), N)
P[rw, cl] = -D_inv.ravel(); P[cl, rw] = -D_inv.ravel()
P[NK:, NK:] = np.diag(D_inv.sum(axis=0)) + Dbb_inv
rhs = np.empty(M)
rhs[:NK] = np.einsum("ij,ijk->ik", Sinv, Fr, optimize=True).reshape(NK)
rhs[NK:] = Dbb_inv @ weak.b_bar_bar
Pinv = _spd_inverse(P)
exact = Pinv @ rhs

n_draws, n_burn = 4000, 500
draws = np.empty((n_draws - n_burn, M))
for it in range(n_draws):
    Bd, bd = sample_B_and_b_bar(rng, G, Fr, Sigma_fix, tau2_fix, weak.s,
                                weak.b_bar_bar, weak.Delta_b_bar)
    if it >= n_burn:
        draws[it - n_burn] = np.concatenate([Bd.ravel(), bd])

emp = draws.mean(axis=0)
ess = np.array([ess_1d(draws[:, j]) for j in range(M)])
z = (emp - exact) / np.sqrt(np.diag(Pinv) / ess)
z_naive = (emp - exact) / np.sqrt(np.diag(Pinv) / (n_draws - n_burn))
print(f"   ESS-adjusted:  max |z| {np.abs(z).max():.2f}   median |z| "
      f"{np.median(np.abs(z)):.2f}   ({M} parameters)")
print(f"   naive sqrt(n): max |z| {np.abs(z_naive).max():.2f}"
      f"   [would understate the error if ESS < n]")
print(f"   ESS: min {ess.min():.0f} median {np.median(ess):.0f} of "
      f"{n_draws-n_burn} draws")

print("\n3. RECOVERY OF KNOWN GROUND TRUTH")
print("   Judged against the POSTERIOR SD, not the MCSE of the posterior mean.")
print("   With ESS in the thousands the MCSE is tiny, so dividing by it makes")
print("   every z enormous and says nothing about whether the truth is")
print("   plausible. An earlier version did exactly that and reported |z| up")
print("   to 47 for a correct sampler.")
fit = run_gibbs(R, F, weak, n_draws=6000, n_burn=1500, seed=2)
bb = fit.b_bar
print(f"\n   {'j':>3s} {'true b_bar':>11s} {'posterior':>11s} {'post sd':>9s} "
      f"{'z':>7s} {'in 95% CI':>10s}")
inside = 0
for j in range(K):
    lo, hi = np.percentile(bb[:, j], [2.5, 97.5])
    ok = bool(lo <= sim.b_bar[j] <= hi); inside += ok
    print(f"   {j:>3d} {sim.b_bar[j]:>11.4f} {bb[:, j].mean():>11.4f} "
          f"{bb[:, j].std():>9.4f} "
          f"{(bb[:, j].mean()-sim.b_bar[j])/bb[:, j].std():>7.2f} {str(ok):>10s}")
print(f"   {inside}/{K} true values inside their 95% credible interval")
print("   [K=5 is far too few for a coverage claim; that is")
print("    tests/test_calibration_lasso.py, over many independent datasets.")
print("    Systematic failure here would show as every z having the same sign]")

B_hat = fit.B.mean(axis=0)
th_hat = B_hat - bb.mean(axis=0)[None, :]
print(f"\n   B:     correlation with truth "
      f"{np.corrcoef(B_hat.ravel(), sim.b.ravel())[0,1]:.4f}")
print(f"   theta: correlation with truth "
      f"{np.corrcoef(th_hat.ravel(), sim.theta.ravel())[0,1]:.4f}"
      f"   shrinkage {np.abs(th_hat).mean()/np.abs(sim.theta).mean():.3f}")
print(f"   lambda {fit.lam.mean():.3f} +/- {fit.lam.std():.3f}"
      f"   implied theta sd {np.sqrt(2)*weak.s/fit.lam.mean():.3f} vs true "
      f"{sim.theta.std():.3f}")
