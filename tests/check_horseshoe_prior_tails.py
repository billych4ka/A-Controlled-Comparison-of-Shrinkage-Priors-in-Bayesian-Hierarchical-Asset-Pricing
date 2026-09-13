"""
check_horseshoe_prior_tails.py

Demonstrates why full-dimensional prior-generative calibration of the plain
horseshoe was abandoned, as reported in Appendix A.7.7.

THE PROBLEM. Prior-generative calibration draws parameters from the model's
own prior, simulates data, refits, and checks that credible intervals cover
the drawn values at their nominal rate. It requires the fit to succeed on
every simulated dataset, or the exercise conditions on the draws that
happened to be tractable, which biases coverage towards easier regions of
the prior and defeats the purpose.

At production dimensions the plain horseshoe has NK = 3,600 half-Cauchy
local scales. A half-Cauchy has no finite mean, and the maximum of n
independent draws grows roughly linearly in n rather than logarithmically as
it would for a light-tailed distribution. A regime that behaves at a few
hundred scales therefore produces coefficient vectors thousands of residual
standard errors from zero at 3,600: datasets that are not merely slow to
fit but effectively unfittable.

WHY A SCRIPT THAT DOES NOT FIT ANYTHING. The original attempt ran for over
five hours before being stopped. Reproducing the failure by repeating it
would cost the same time to establish something the prior's order statistics
show directly. This script therefore samples from the prior only and reports
the distribution of the largest deviation, at increasing numbers of local
scales, which is the mechanism rather than one instance of its consequence.

WHAT THIS DOES NOT SHOW. Extreme prior draws do not imply the model
misbehaves on observed data, where the likelihood constrains the same tails.
The finding is a limitation of prior-generative calibration for unbounded
global-local priors at high dimension, not evidence about the fitted
posterior. Section 4.5 and Appendix A.7.7 state this distinction; the
reduced-dimension calibration reported in Table 4.5 is what establishes
coverage.

Run from the project root, seconds:

    python3 check_horseshoe_prior_tails.py
    python3 check_horseshoe_prior_tails.py --draws 20000 --seed 7
"""

from __future__ import annotations

import argparse

import numpy as np


def half_cauchy(rng: np.random.Generator, size) -> np.ndarray:
    """|X| for X ~ Cauchy(0,1), i.e. tan of a uniform on (0, pi/2)."""
    return np.abs(np.tan(np.pi * (rng.random(size) - 0.5)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=10000,
                    help="prior draws per dimension setting")
    ap.add_argument("--tau0", type=float, default=3.945463e-04,
                    help="calibrated global scale; production value for "
                         "size_bm_25 at p0=23")
    ap.add_argument("--sigma", type=float, default=0.05565703,
                    help="pooled df-corrected residual scale")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    settings = [(6, 5), (10, 30), (25, 30), (25, 72), (25, 144)]

    print(f"plain horseshoe prior draws: tau_0 = {args.tau0:.4e}, "
          f"sigma = {args.sigma:.6f}")
    print(f"{args.draws:,} prior draws per setting\n")
    print(f"{'N':>4}{'K':>5}{'NK':>7}"
          f"{'median max|theta|/sigma':>25}{'90th pct':>11}{'max':>12}"
          f"{'P(>1000 sigma)':>16}")

    for N, K in settings:
        nk = N * K
        med, p90, mx, frac = [], [], [], 0
        worst = np.empty(args.draws)
        for d in range(args.draws):
            tau = args.tau0 * half_cauchy(rng, 1)[0]
            lam = half_cauchy(rng, nk)
            z = rng.standard_normal(nk)
            theta = tau * lam * z
            worst[d] = np.abs(theta).max() / args.sigma
        print(f"{N:>4}{K:>5}{nk:>7}{np.median(worst):>25.1f}"
              f"{np.percentile(worst, 90):>11.1f}{worst.max():>12.1f}"
              f"{100 * np.mean(worst > 1000):>15.2f}%")

    print("\n  [Entries are the largest asset-predictor deviation in a prior")
    print("   draw, expressed in residual standard errors. The median grows")
    print("   with the number of local scales because the maximum of n")
    print("   half-Cauchy draws grows roughly linearly in n; a dataset")
    print("   simulated from such a draw has a signal far outside the range")
    print("   the sampler can resolve in reasonable time.]")
    print("\n  The reduced dimensions used for calibration keep this maximum")
    print("  within a range where every simulated dataset can be fitted, so")
    print("  coverage is not conditioned on tractable draws. The cost is that")
    print("  the exercise cannot speak to the extreme tails of the")
    print("  full-dimensional prior.")


if __name__ == "__main__":
    main()
