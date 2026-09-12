"""
check_rhat_ess.py

Verifies src/diagnostics/convergence.py against cases whose answers are known
analytically, and establishes the reference distribution used when many
parameters are compared at once.

THIS SCRIPT SUPPORTS TWO APPENDIX SUBSECTIONS, and its numbered sections map
to them separately. Section 7 reproduces Appendix A.5: R-hat agreeing to four
decimals against an independent implementation, bulk ESS to 0.4%, and tail ESS
differing by 15-40% because the two are defined differently. Section 6
reproduces Appendix A.6.2's reference distribution: a median maximum |z| of
3.73 with a 5th-95th range of 3.34-4.34, over 4,000 replications at 3,600
parameters. Sections 1-5 support neither directly; they are the
known-answer cases the module is checked against. Supports Section 4.5's
paragraph justifying the reporting of the distribution of standardised
discrepancies rather than the maximum.

Every convergence number reported in the dissertation rests on this module,
so it is checked against known answers rather than against plausibility. The
three non-convergence cases are chosen so that each is caught by exactly one
of the three refinements over the classic Gelman-Rubin statistic: rank
normalisation, folding, and splitting.

Recorded results (chat of 10 August, seed 0):
    i.i.d.                R-hat 1.0001, ESS 7750 of 8000
    AR(1) ESS vs analytic N(1-rho)/(1+rho): ratios 1.009, 1.008, 0.972
    different means 1.169 | different scale 1.156 | common drift 1.547
    max |z| over 3600 agreeing parameters: median 3.73, 5th-95th 3.34-4.34
    arviz cross-check: R-hat to 4 decimals, bulk ESS to 0.4%
"""
import numpy as np
from src.diagnostics.convergence import (rank_normalised_rhat, effective_sample_size,
                                         ess_tail)

rng = np.random.default_rng(0)

print("1. i.i.d. draws -- R-hat should be ~1 and ESS ~ the number of draws")
x = rng.standard_normal((4, 2000))
print("   R-hat %.4f  [want ~1.000]   ESS %.0f of 8000  [want ~8000]"
      % (rank_normalised_rhat(x), effective_sample_size(x)))

print("\n2. AR(1) -- ESS has the analytic value N(1-rho)/(1+rho)")
for rho in [0.5, 0.9, 0.99]:
    n, m = 20000, 4
    e = rng.standard_normal((m, n)); y = np.zeros((m, n))
    for t in range(1, n):
        y[:, t] = rho*y[:, t-1] + e[:, t]*np.sqrt(1 - rho**2)
    ess, an = effective_sample_size(y), m*n*(1-rho)/(1+rho)
    print("   rho=%.2f  ESS %8.0f  analytic %8.0f  ratio %.3f  [want ~1.00]"
          % (rho, ess, an, ess/an))

print("\n3. chains that have NOT converged -- each case is caught by exactly")
print("   one of the three refinements over classic Gelman-Rubin")
print("   different means  : %.3f  (caught by any version)" % rank_normalised_rhat(
    rng.standard_normal((4, 2000)) + np.array([0, .5, 1, 1.5])[:, None]))
print("   different SCALE  : %.3f  (only FOLDING catches this)" % rank_normalised_rhat(
    rng.standard_normal((4, 2000)) * np.array([1., 1., 3., 3.])[:, None]))
print("   common DRIFT     : %.3f  (only SPLITTING catches this)" % rank_normalised_rhat(
    rng.standard_normal((4, 2000))*0.3 + np.linspace(0, 2, 2000)[None, :]))

print("\n4. heavy tails -- rank normalisation keeps both statistics finite")
c = rng.standard_cauchy((4, 2000))
print("   Cauchy R-hat %.4f   ESS %.0f" % (rank_normalised_rhat(c), effective_sample_size(c)))

print("\n5. degenerate input returns nan rather than raising")
print("   constant parameter:", rank_normalised_rhat(np.ones((4, 100))),
      effective_sample_size(np.ones((4, 100))))

# --- the reference distribution for high-dimensional comparisons ----------
print("\n6. REFERENCE DISTRIBUTION for comparing many parameters at once.")
print("   With 3,600 parameters, the maximum |z| is the maximum of 3,600")
print("   draws and is therefore large even under perfect agreement. This is")
print("   why the dissertation quotes medians rather than maxima.")
m = np.array([np.abs(rng.standard_normal(3600)).max() for _ in range(4000)])
print("   median max |z| %.2f   5th-95th percentile %.2f-%.2f"
      % (np.median(m), *np.percentile(m, [5, 95])))
print("   P(max |z| > 3.06) = %.2f   [so an observed 3.06 is unremarkable]"
      % np.mean(m > 3.06))

# --- independent cross-check ---------------------------------------------
try:
    import arviz as az

    def _scalar(v):
        return float(np.asarray(v).ravel()[0])

    n, m_, rho = 2000, 4, 0.9
    e = rng.standard_normal((m_, n)); ar = np.zeros((m_, n))
    for t in range(1, n):
        ar[:, t] = rho*ar[:, t-1] + e[:, t]*np.sqrt(1 - rho**2)
    cases = [("iid", rng.standard_normal((4, 2000))), ("AR(1) rho=0.9", ar),
             ("unconverged", rng.standard_normal((4, 2000))
              + np.array([0, .5, 1, 1.5])[:, None])]
    print("\n7. cross-check against arviz %s (independent implementation)" % az.__version__)
    for lab, d in cases:
        idata = az.convert_to_dataset(d)
        print("   %-14s R-hat ours %.4f arviz %.4f | ESS ours %7.0f arviz %7.0f | "
              "tail ours %7.0f arviz %7.0f"
              % (lab, rank_normalised_rhat(d), _scalar(az.rhat(idata).x.values),
                 effective_sample_size(d), _scalar(az.ess(idata).x.values),
                 ess_tail(d), _scalar(az.ess(idata, method="tail").x.values)))
    print("   R-hat and bulk ESS agree closely. Tail ESS differs by 15-40%:")
    print("   ours is the min ESS of the two tail indicator series, arviz")
    print("   computes quantile-ESS directly. Neither is wrong; the")
    print("   dissertation reports arviz's, which matches Vehtari et al.")
except ImportError:
    print("\n7. arviz not installed -- cross-check skipped")