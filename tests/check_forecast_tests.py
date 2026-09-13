"""
check_forecast_tests.py

Verifies the two forecast-comparison tests in src/evaluation/metrics.py, and
establishes why BOTH are needed rather than one.

Reproduces Appendix A.7.6, the size and power of the forecast comparison
tests, and supports Section 4.6's statement that both tests were checked for
size and power before use. The tests themselves are in
src/evaluation/metrics.py; this script validates them.

Nested and non-nested forecast comparisons require different tests. The
historical-mean benchmark is nested inside a predictive regression: set every
slope to zero and the model becomes the benchmark. Under the null that the
extra predictors are useless, the larger model still estimates them, and that
estimation noise inflates its squared error, so the model is expected to
LOSE on mean squared error even when the null is true. Section 5 below
measures how badly a plain Diebold-Mariano test fails when misapplied there.

Recorded results (seed 0):
    Newey-West vs plain variance on AR(1) rho=0.7 : 7.433 vs 1.945
    DM under the null  : mean -0.058, sd 1.013, rejects 4.5% at 5%
    DM with signal     : large negative, sign flips exactly on swapping
    CW under the null  : mean -0.032, sd 1.020, rejects 2.3% (conservative)
    DM MISAPPLIED to the nested null: mean +5.4, benchmark "wins" 99.7%
"""
import numpy as np
from src.evaluation.metrics import diebold_mariano, clark_west, _newey_west_variance

rng = np.random.default_rng(0)
N, T = 25, 479

print("1. Newey-West long-run variance vs the plain variance")
print("   Annual refits mean twelve consecutive months share one coefficient")
print("   vector, so loss differentials are serially correlated. Using the")
print("   plain variance would inflate every statistic.")
x = np.zeros(2000)
for t in range(1, 2000):
    x[t] = 0.7*x[t-1] + rng.standard_normal()
print("   AR(1) rho=0.7: plain var %.3f | Newey-West %.3f | analytic long-run %.3f"
      % (x.var(), _newey_west_variance(x), 1/(1-0.7)**2))
print("   ignoring the correction would inflate statistics by ~%.1fx"
      % np.sqrt(_newey_west_variance(x)/x.var()))

print("\n2. SIZE of Diebold-Mariano under the null of equal accuracy")
stats = np.array([diebold_mariano(rng.standard_normal((N, T))*0.05,
                                  rng.standard_normal((N, T))*0.01,
                                  rng.standard_normal((N, T))*0.01)["statistic"]
                  for _ in range(400)])
print("   mean %.3f [want ~0]  sd %.3f [want ~1]  rejects at 5%%: %.1f%% [want ~5%%]"
      % (stats.mean(), stats.std(), 100*np.mean(np.abs(stats) > 1.96)))

print("\n3. POWER of Diebold-Mariano, and the sign convention")
signal = rng.standard_normal((N, T))*0.02
r = rng.standard_normal((N, T))*0.05 + signal
fwd = diebold_mariano(r, signal, np.zeros((N, T)))
rev = diebold_mariano(r, np.zeros((N, T)), signal)
print("   A = true signal, B = zero : stat %+.2f  better = %s" % (fwd["statistic"], fwd["better"]))
print("   arguments swapped         : stat %+.2f  better = %s" % (rev["statistic"], rev["better"]))
print("   a NEGATIVE statistic favours the first argument")

print("\n4. Clark-West: size under the nested null, and power with signal")
cw = np.array([clark_west(rng.standard_normal((N, T))*0.05,
                          rng.standard_normal((N, T))*0.005,
                          np.zeros((N, T)))["statistic"] for _ in range(300)])
print("   null: mean %.3f sd %.3f  rejects at 5%%: %.1f%%  [one-sided; conservative"
      % (cw.mean(), cw.std(), 100*np.mean(cw > 1.645)))
print("                                                    by construction]")
print("   with real signal: stat %+.2f" % clark_west(r, signal, np.zeros((N, T)))["statistic"])

print("\n5. WHY BOTH TESTS ARE NEEDED: DM applied to the SAME nested null.")
print("   The 'model' here has no signal at all: it adds only estimation noise.")
dm = np.array([diebold_mariano(rng.standard_normal((N, T))*0.05,
                               rng.standard_normal((N, T))*0.005,
                               np.zeros((N, T)))["statistic"] for _ in range(300)])
print("   DM mean %+.2f  [biased POSITIVE: the model wrongly looks worse]" % dm.mean())
print("   DM declares the uninformative benchmark the winner in %.1f%% of cases"
      % (100*np.mean(dm > 1.96)))
print("   Not a mild bias: a test that almost always reaches the wrong")
print("   conclusion. Clark-West is therefore used for model-vs-benchmark and")
print("   Diebold-Mariano only for the non-nested model-vs-model comparisons.")
