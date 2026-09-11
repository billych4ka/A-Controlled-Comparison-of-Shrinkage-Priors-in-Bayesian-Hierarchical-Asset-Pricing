"""
check_scale_transportability.py

Regenerates every number in the Section 4.3 scale table from the data, so that
no figure in it rests on a hand calculation.

This script exists because the earlier "three independent calibration routes
agree to within 8%" claim did rest on hand calculations, and the routes turned
out not to be computed on a common basis. Every route is therefore reported on
both possible bases, so the comparison is explicit rather than assumed.

TWO QUANTITIES ARE REPORTED FOR EVERY ROUTE.

    deviation sd   the per-coefficient prior sd of theta_ij alone
    total sd       the per-coefficient prior sd of b_ij = b_bar_j + theta_ij,
                   i.e. sqrt(Delta_b_bar_jj + E[Delta_theta_jj])

These differ by a fixed factor sqrt(1 + 33.33) = 5.86, because the calibration
preserves Feng & He's common-to-deviation variance ratio. Comparing a deviation
sd from one route against a total sd from another produces spurious agreement;
that is precisely the error this script exists to prevent.

THE EIGENVALUE ROUTE DEPENDS ON WHICH COVARIANCE IS USED.

    P = F'(Sigma^-1 (x) I_T) F

is the data precision for the stacked coefficient vector B under asset-major
ordering. Sigma is not pinned down by the calibration principle: the prior-side
choice is the raw return covariance (available before any fit, and the natural
scale for a prior-side diagnostic), while the likelihood-side choice is the OLS
residual covariance (what Omega is actually built from). Both are computed
below. If they disagree materially, that disagreement belongs in the write-up
rather than in a footnote.

AND ON WHICH PRIOR PRECISION IS TAKEN TO COMPETE WITH THE DATA.

    conditional reading  in the B-update the precision is
                         P + I_N (x) Delta_theta^-1, so the prior precision
                         competing with the data is Delta_theta^-1 and
                         1/sqrt(median eig) is a DEVIATION sd
    marginal reading     the marginal per-coefficient prior precision on b_ij
                         is (Delta_b_bar + E[Delta_theta])^-1, under which
                         1/sqrt(median eig) is a TOTAL sd

Both are reported. Neither is exact: marginally over b_bar the stacked prior
covariance is I_N (x) Delta_theta + J_N (x) Delta_b_bar rather than a Kronecker
product, because the shared common component induces cross-asset correlation.
The route is a heuristic under either reading, and the write-up should state
which reading it uses.

Run from the project root:

    python3 check_scale_transportability.py
    python3 check_scale_transportability.py --universe size_op_25
    python3 check_scale_transportability.py --skip-eigen      # fast

The eigendecomposition is of an NK x NK matrix (3,600 x 3,600 at production
size) and takes on the order of a minute.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.gibbs.baseline_gaussian import (default_hyperparameters,
                                         prior_implied_r2,
                                         rescaled_hyperparameters)
from src.gibbs.bayesian_lasso import pooled_residual_scale

# Feng & He's disclosed prior variances, on their own predictor scale.
#   common     Delta_b_bar = 0.1 I_K                        (their Eq. 9)
#   deviation  E[Delta_b]  = V_b / (nu_b - K - 1) = 3/1000  (derived; Sec 4.2.1)
FH_COMMON = 0.1
FH_DEVIATION = 0.003

# Feng & He's effective regressor scale. Their macroeconomic predictors are in
# raw units (dividend yield ~0.03, T-bill ~0.05), giving a typical standard
# deviation near 0.015; this project's expanding-window z-scores give ~1.
FH_REGRESSOR_SD = 0.015

# Recorded value for the eigenvalue route, for regression-testing the
# reconstruction. Reported, not asserted: this is the figure whose provenance
# was in question.
RECORDED_EIGEN_SD = 7.64e-04


def data_precision(F: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """
    P = F'(Sigma^-1 (x) I_T)F, the NK x NK data precision for the stacked
    coefficient vector under asset-major ordering.

    Block (i,j) is [Sigma^-1]_ij * F_i' F_j. Built blockwise rather than by
    forming the NT x NT Kronecker product, which would be 17,975^2 here.
    """
    N, T, K = F.shape
    Sinv = np.linalg.inv(Sigma)
    # Axis order (i,k,j,l), NOT (i,j,k,l): the stacked vector is asset-major,
    # so row i*K+k must pair with column j*K+l and the reshape below only does
    # that if the asset and predictor axes already alternate. optimize=True
    # routes the contraction over t through BLAS; without it this is a
    # Python-level loop over 9.3e9 products.
    G = np.einsum("itk,jtl->ikjl", F, F, optimize=True)      # (N,K,N,K)
    P = (Sinv[:, None, :, None] * G).reshape(N * K, N * K)
    return 0.5 * (P + P.T)                        # symmetrise against fp drift


def eigen_route(F: np.ndarray, Sigma: np.ndarray) -> tuple[float, float]:
    """
    The scale at which prior precision equals the median data precision.
    Returns (median eigenvalue, 1/sqrt(median eigenvalue)).
    """
    eig = np.linalg.eigvalsh(data_precision(F, Sigma))
    med = float(np.median(eig))
    return med, float(1.0 / np.sqrt(med))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="size_bm_25")
    ap.add_argument("--target-r2", type=float, default=0.05)
    ap.add_argument("--skip-eigen", action="store_true",
                    help="skip the NK x NK eigendecomposition")
    args = ap.parse_args()

    path = Path("data/processed") / f"{args.universe}_arrays.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Available: "
            f"{sorted(p.name for p in Path('data/processed').glob('*_arrays.npz'))}"
        )
    d = np.load(path, allow_pickle=True)
    F, R = d["F"], d["R"]
    N, T, K = F.shape

    # The design-dependent constant. Every V_prior below is s^2 times this.
    C = np.cov(F.reshape(-1, K).T)
    var_r = float(R.var())
    design = float(np.trace(C) / var_r)

    hp = rescaled_hyperparameters(R, F, K, target_r2=args.target_r2)
    s2_dev = float(hp.V_b[0, 0] / (hp.nu_b - K - 1))
    s2_total = float(hp.Delta_b_bar[0, 0]) + s2_dev
    ratio = np.sqrt(s2_total / s2_dev)

    print(f"{args.universe}: N={N} T={T} K={K}")
    print(f"trace(C)          {np.trace(C):.6f}   (mean sq. regressor scale "
          f"{np.trace(C) / K:.3f})")
    print(f"Var(r)            {var_r:.6e}")
    print(f"trace(C)/Var(r)   {design:,.1f}")
    print(f"adopted deviation sd  {np.sqrt(s2_dev):.4e}")
    print(f"adopted total sd      {np.sqrt(s2_total):.4e}   "
          f"(ratio {ratio:.3f})\n")

    # ---- transportability table -------------------------------------------
    s2_fh = FH_COMMON + FH_DEVIATION
    s2_fh_conv = s2_fh * FH_REGRESSOR_SD ** 2

    print("SCALE TRANSPORTABILITY  (all rows are TOTAL prior sd of b_ij)")
    print(f"{'specification':<52s} {'total sd':>13s} {'V_prior':>13s}")
    print("-" * 82)
    for label, s2 in (("Feng & He, applied verbatim", s2_fh),
                      (f"Feng & He, converted for regressor scale "
                       f"(sd={FH_REGRESSOR_SD})", s2_fh_conv),
                      (f"Adopted calibration (V_prior={args.target_r2})",
                       s2_total)):
        print(f"{label:<52s} {np.sqrt(s2):>13.4e} {s2 * design:>13,.4f}")
    print(f"\nconverted prior remains {s2_fh_conv / s2_total:,.0f}x looser "
          f"than the adopted calibration")
    print(f"(the conversion divides V_prior by (1/{FH_REGRESSOR_SD})^2 = "
          f"{1 / FH_REGRESSOR_SD ** 2:,.0f} exactly -- definitional)")

    # ---- eigenvalue route, both covariances and both readings -------------
    if not args.skip_eigen:
        print("\nEIGENVALUE ROUTE: prior precision = median data precision")
        print("  P = F'(Sigma^-1 (x) I_T)F; 1/sqrt(median eig) is the scale at")
        print("  which the two are equal. Reported under both readings.\n")

        S_return = np.cov(R, ddof=1)
        _, B_ols = pooled_residual_scale(R, F)
        E = R - np.einsum("itk,ik->it", F, B_ols)
        S_resid = np.cov(E, ddof=1)

        # The isotropic comparator holds the RAW RETURN covariance's average
        # magnitude fixed and removes only its cross-asset structure, because
        # that matrix is the one the check itself uses. Three plausible
        # scalings give three different answers -- unscaled I_N (62x), the
        # residual covariance's mean diagonal (3.1x) and this one (3.6x) --
        # so the choice is stated here rather than left implicit.
        ISO = "isotropic, mean diag of return cov"
        S_iso = float(np.mean(np.diag(S_return))) * np.eye(N)

        print(f"{'Sigma used':<40s} {'1/sqrt(med)':>13s} "
              f"{'vs dev sd':>12s} {'vs total sd':>13s}")
        print("-" * 82)
        results, medians = {}, {}
        for label, S in (("raw return covariance", S_return),
                         ("OLS residual covariance", S_resid),
                         (ISO, S_iso)):
            med, sd = eigen_route(F, S)
            results[label], medians[label] = sd, med
            print(f"{label:<40s} {sd:>13.4e} "
                  f"{sd / np.sqrt(s2_dev):>11.2f}x "
                  f"{sd / np.sqrt(s2_total):>12.2f}x")

        print("\n  conditional reading: 1/sqrt(med) is a DEVIATION sd; compare")
        print("    against the adopted deviation sd (column 'vs dev sd')")
        print("  marginal reading:    it is a TOTAL sd; compare against the")
        print("    adopted total sd (column 'vs total sd')")
        print("  A ratio near 1.00 in either column is agreement on that")
        print("  reading. State in the write-up which reading is intended.")

        # How much of the agreement is Sigma's cross-asset structure?
        # The isotropic row is NOT a calibration route -- it exists only to
        # show that the agreement is a property of THIS return covariance
        # (25 portfolios sharing a dominant market factor) rather than
        # something that would hold in any SUR design.
        ratio_iso = results[ISO] / np.sqrt(s2_total)
        inflation = medians["raw return covariance"] / medians[ISO]
        print(f"\n  Removing the cross-asset structure but holding the mean")
        print(f"  diagonal fixed gives {results[ISO]:.4e}, a factor of "
              f"{ratio_iso:.2f} larger")
        print(f"  than the adopted scale. The cross-asset correlation raises")
        print(f"  the median data precision {inflation:.2f}x "
              f"({np.sqrt(inflation):.2f}x on 1/sqrt);")
        print( "  without it the check would imply a substantially looser prior.")
        print( "  The agreement is therefore a property of this return")
        print( "  covariance, not a general feature of the construction.")

        got = results["raw return covariance"]
        ok = abs(got - RECORDED_EIGEN_SD) / RECORDED_EIGEN_SD < 0.01
        print(f"\n  recorded value {RECORDED_EIGEN_SD:.4e}; reconstruction with "
              f"the raw return covariance gives {got:.4e} "
              f"({'MATCH' if ok else 'NO MATCH'})")

        # Is the prior-side covariance choice consistent with V_Sigma?
        base_hp = default_hyperparameters(R, K)
        for label, S in (("raw return covariance", S_return),
                         ("OLS residual covariance", S_resid)):
            if np.allclose(base_hp.V_Sigma, S, rtol=1e-8):
                print(f"  V_Sigma equals the {label}")
                break
        else:
            print("  V_Sigma matches neither covariance exactly; check "
                  "default_hyperparameters before claiming that the "
                  "prior-side choice is consistent with the prior "
                  "specification")

    # ---- assertions -------------------------------------------------------
    got = s2_total * design
    assert abs(got - args.target_r2) < 1e-9, (
        f"adopted calibration implies V_prior={got}, expected {args.target_r2}"
    )
    assert abs(prior_implied_r2(hp, R, F) - got) < 1e-12, (
        "prior_implied_r2 disagrees with the trace formula used here"
    )
    base = default_hyperparameters(R, K)
    s2_disclosed = float(base.Delta_b_bar[0, 0]
                         + base.V_b[0, 0] / (base.nu_b - K - 1))
    assert abs(s2_disclosed - s2_fh) < 1e-12, (
        f"default_hyperparameters gives {s2_disclosed}, expected {s2_fh}"
    )
    assert abs(prior_implied_r2(base, R, F) - s2_fh * design) < 1e-9
    assert abs(ratio - np.sqrt(1 + FH_COMMON / FH_DEVIATION)) < 1e-6, (
        "the common-to-deviation ratio is not the preserved 33.33"
    )

    print("\nall assertions passed")


if __name__ == "__main__":
    main()