"""
check_iw_potential.py

Reproduces the inverse-Wishart implementation check reported in
Appendix A.6.3.

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

WHAT THIS SCRIPT DOES. It samples the prior ALONE -- no likelihood, no data
-- so the posterior is exactly the inverse-Wishart prior and its moments are
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

EXPECTED RESULT. The Gaussian-prior version should show discrepancies of
roughly 8-11 MCSE; the flat version should fall to approximately 1.2. The
exact figures depend on the seed and draw count, but the ORDER of the
discrepancy is the finding: a bug that no convergence diagnostic detects
shifts posterior moments by many standard errors.

INTERPRETATION. Passing this check establishes that the implemented density
and its Jacobian correspond to the intended prior. It does not establish
that the sampler targets the correct posterior once a likelihood is present;
that is what the joint-composition and prior-generative checks in
Appendix A.6 address.

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
        'flat'     -- pm.Flat, the correct construction. The Potential is
                      then the only density acting on the coordinate.
        'gaussian' -- the bug. A proper N(0,1) prior on the coordinate,
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

        # returns the reconstructed covariance together with the total log
        # density in terms of the packed coordinate -- the IW density plus
        # the Jacobian of the Cholesky and log-diagonal transformations
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
    # MCSE of a sample mean, using the THEORETICAL variance rather than the
    # sampled one: under the bug the sampled variance is itself wrong, and
    # using it would partly absorb the error being tested for.
    mcse = np.sqrt(var / n)
    z = np.abs(emp - mean) / mcse
    iu = np.triu_indices(mean.shape[0])
    zz = z[iu]
    print(f"  {label:<28} max |z| {zz.max():6.2f}   median |z| "
          f"{np.median(zz):5.2f}   entries above 3: "
          f"{int((zz > 3).sum())} of {zz.size}")
    return float(zz.max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-assets", type=int, default=5)
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--tune", type=int, default=2000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    N = args.n_assets
    # nu comfortably above N + 3 so both moments exist and the prior is not
    # so diffuse that the Monte Carlo error swamps the comparison
    nu = float(N + 8)
    rng = np.random.default_rng(args.seed)
    A = rng.standard_normal((N, N))
    V = (A @ A.T + N * np.eye(N)) * 0.01      # arbitrary but well conditioned

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
    if zs["gaussian"] > 3.0 and zs["flat"] < 3.0:
        print("  Detected. The proper coordinate prior shifts posterior means")
        print("  by several Monte Carlo standard errors; the flat base measure")
        print("  restores agreement. Neither run raised an exception.")
    else:
        print("  *** The expected pattern did not appear. Either the")
        print("  implementation has changed or the draw count is too small")
        print("  for the discrepancy to exceed Monte Carlo error. ***")


if __name__ == "__main__":
    main()