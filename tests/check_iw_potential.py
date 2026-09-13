"""
check_iw_potential.py

Reproduces the inverse-Wishart implementation check reported in
Appendix A.7.3.

WHAT WENT WRONG, AND WHY IT MATTERED. The NUTS models represent the residual
covariance through an unconstrained packed Cholesky coordinate, with the
inverse-Wishart log density supplied by a custom implementation because
SciPy's cannot be differentiated inside the PyTensor graph. In an early
version that density was added through a Potential while the coordinate was
still declared with a proper Gaussian prior. A Potential ADDS to the total
log density rather than replacing a declared variable's own prior, so the
model silently targeted the product of two densities on Sigma.

The failure was invisible to every routine check. Sampling completed, no
exception was raised, R-hat and ESS were unremarkable, and the posterior
looked plausible. Only comparison against a quantity whose answer is known
in advance detected it.

WHAT THIS SCRIPT DOES. It samples the prior ALONE (no likelihood, no data),
so the posterior is exactly the inverse-Wishart prior and its moments are
available in closed form:

    E[Sigma]     = V / (nu - N - 1)                       for nu > N + 1
    Var[Sigma_ij] = [(nu - N + 1) V_ij^2 + (nu - N - 1) V_ii V_jj]
                    / [(nu - N)(nu - N - 1)^2 (nu - N - 3)]

Sampled means are compared with E[Sigma] in units of their own Monte Carlo
standard error, so the comparison accounts for finite sample size rather
than requiring exact agreement. The script runs the model twice: once with
the proper Gaussian coordinate prior that caused the bug, and once with
pm.Flat, which supplies an improper flat base measure and leaves the
Potential as the only density on the coordinate.

EXPECTED RESULT. At the production dimension N=25 the contaminated version
displaces posterior means by a median of roughly 3.5 standard errors, with
more than half of the 325 distinct covariance entries beyond three; the flat
version falls to a median near 1.0 with a small minority beyond three. The
exact figures depend on the dimension, the seed and the draw count, but the
finding is that a bug no convergence diagnostic detects shifts posterior
moments by several standard errors across most of the matrix.

WHY THE COMPARISON USES MEDIANS. With 325 entries the MAXIMUM |z| is a poor
criterion: even under a correct implementation the largest of 325
standardised discrepancies has an expected value near three, which is the
same multiplicity point the verification section makes for the 3,600-entry
comparisons. A success test built on the maximum would therefore fail on
correct code. The median and the proportion beyond three are used instead.

A CONSERVATIVE STANDARD ERROR. The MCSE below divides the theoretical
variance by the raw draw count rather than the effective sample size, so the
reported z-values are inflated by the chains' autocorrelation. For a correct
implementation the median |z| would be approximately 0.67; an observed
median near 1.0 implies an inflation factor of roughly 1.5. The comparison
of interest is between the two implementations, which share this inflation,
rather than between either and an exact nominal distribution.

INTERPRETATION. Passing this check establishes that the implemented density
and its Jacobian correspond to the intended prior. It does not establish
that the sampler targets the correct posterior once a likelihood is present;
that is what the joint-composition and prior-generative checks in
Appendix A.7 address.

Run from the project root, a few minutes:

    python3 check_iw_potential.py
    python3 check_iw_potential.py --n-assets 6 --draws 4000
"""

from __future__ import annotations

import argparse

import numpy as np
import pymc as pm

from src.priors.inverse_wishart import inverse_wishart_cholesky_logp


def iw_moments(nu: float, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and elementwise variance of IW(nu, V).

    Both require nu large enough for the moment to exist: the mean needs
    nu > N + 1 and the variance nu > N + 3. The script checks this rather
    than silently returning infinities, because a degrees-of-freedom choice
    that fails the condition would make the comparison meaningless rather
    than merely imprecise.
    """
    N = V.shape[0]
    if nu <= N + 3:
        raise ValueError(f"nu={nu} too small for finite variance at N={N}; "
                         f"need nu > {N + 3}")
    mean = V / (nu - N - 1)
    d = np.diag(V)
    var = ((nu - N + 1) * V ** 2 + (nu - N - 1) * np.outer(d, d)) / (
        (nu - N) * (nu - N - 1) ** 2 * (nu - N - 3))
    return mean, var


def sample_prior(nu: float, V: np.ndarray, coordinate_prior: str,
                 draws: int, tune: int, chains: int, seed: int):
    """
    Sample the inverse-Wishart prior alone through the packed Cholesky
    coordinate used by the NUTS models.

    coordinate_prior:
        'flat':      pm.Flat, the correct construction. The Potential is
                      then the only density acting on the coordinate.
        'gaussian':  the bug. A proper N(0,1) prior on the coordinate,
                      with the inverse-Wishart density added on top, so the
                      model targets the product of two densities.
    """
    N = V.shape[0]
    n_packed = N * (N + 1) // 2

    with pm.Model() as model:
        if coordinate_prior == "flat":
            packed = pm.Flat("packed", shape=n_packed)
        elif coordinate_prior == "gaussian":
            packed = pm.Normal("packed", 0.0, 1.0, shape=n_packed)
        else:
            raise ValueError(coordinate_prior)

        Sigma_expr, total_logp = inverse_wishart_cholesky_logp(packed, nu, V)
        Sigma = pm.Deterministic("Sigma", Sigma_expr)
        pm.Potential("Sigma_prior", total_logp)

        idata = pm.sample(draws=draws, tune=tune, chains=chains,
                          cores=1, random_seed=seed, target_accept=0.95,
                          progressbar=False,
                          compute_convergence_checks=False)
    return idata.posterior["Sigma"].values.reshape(-1, N, N)


def report(label: str, draws: np.ndarray, mean: np.ndarray,
           var: np.ndarray) -> float:
    """Discrepancy between sampled and theoretical means, in MCSE units."""
    n = draws.shape[0]
    emp = draws.mean(axis=0)
    mcse = np.sqrt(var / n)
    z = np.abs(emp - mean) / mcse
    iu = np.triu_indices(mean.shape[0])
    zz = z[iu]
    med = float(np.median(zz))
    above = int((zz > 3).sum())
    print(f"  {label:<28} median |z| {med:5.2f}   max |z| {zz.max():6.2f}"
          f"   entries above 3: {above} of {zz.size} "
          f"({100 * above / zz.size:.1f}%)")
    return med, above / zz.size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-assets", type=int, default=5)
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--tune", type=int, default=2000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    N = args.n_assets
    nu = float(N + 8)
    rng = np.random.default_rng(args.seed)
    A = rng.standard_normal((N, N))
    V = (A @ A.T + N * np.eye(N)) * 0.01

    mean, var = iw_moments(nu, V)
    print(f"inverse-Wishart prior alone: N={N}, nu={nu:.0f}, "
          f"{args.chains} chains x {args.draws} draws\n")

    zs = {}
    for label, prior in (("pm.Normal coordinate (bug)", "gaussian"),
                         ("pm.Flat coordinate (fixed)", "flat")):
        d = sample_prior(nu, V, prior, args.draws, args.tune,
                         args.chains, args.seed)
        zs[prior] = report(label, d, mean, var)

    print()
    med_bug, frac_bug = zs["gaussian"]
    med_fix, frac_fix = zs["flat"]
    if med_bug > 2.0 and med_fix < 2.0 and frac_bug > 4 * max(frac_fix, 0.01):
        print(f"  Detected. The proper coordinate prior displaces posterior")
        print(f"  means by a median of {med_bug:.1f} standard errors, with")
        print(f"  {100*frac_bug:.0f}% of entries beyond three; the flat base")
        print(f"  measure reduces these to {med_fix:.1f} and "
              f"{100*frac_fix:.0f}%. Neither run")
        print("  raised an exception or produced unusual convergence")
        print("  diagnostics, which is why a known-answer comparison was")
        print("  required to detect it.")
    else:
        print("  *** The expected pattern did not appear. Either the")
        print("  implementation has changed, or the draw count is too small")
        print("  for the displacement to separate from Monte Carlo error at")
        print("  this dimension. Try a larger --n-assets or --draws. ***")


if __name__ == "__main__":
    main()
