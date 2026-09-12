"""
check_sigma_truncation.py

Reproduces the truncation series reported in Section 5.5: the covariance
block's convergence diagnostics recomputed on progressively longer prefixes
of the same chains.

WHY THIS CHECK EXISTS. R-hat above the conventional threshold admits two
readings. Either the chains have settled in different regions of the
posterior, in which case the diagnostic is detecting a real failure and
longer runs will not help; or they agree in location but are individually
short, in which case R-hat reflects finite Monte Carlo error and falls as
the chains lengthen. The two are indistinguishable from a single R-hat
value, and they call for opposite responses.

Truncating the SAME chains to increasing lengths separates them. Under the
second reading R-hat falls monotonically while ESS rises, and the product
(R-hat - 1) x ESS stays roughly constant, because the excess of R-hat above
one scales inversely with effective sample size when the chains target the
same distribution. Under the first reading R-hat plateaus while ESS grows.

The check therefore establishes what a single diagnostic value cannot: that
the covariance chains are short rather than disagreeing. It is a
disclosure-supporting calculation, not a remedy -- Sigma's mixing remains
the weakest convergence result in the dissertation, and Section 5.5 reports
it as such.

WHAT IT DOES NOT SHOW. A stable product is consistent with slow mixing
towards a common target; it is not proof of convergence. The argument is
that the evidence favours one explanation over the other, not that
convergence has been established.

Run from the project root:

    python3 check_sigma_truncation.py
    python3 check_sigma_truncation.py --model regularised_horseshoe
    python3 check_sigma_truncation.py --universe size_op_25 --field Sigma
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

from src.diagnostics.convergence import diagnose, load_chains
from src.gibbs.baseline_gaussian import GibbsDraws
from src.gibbs.bayesian_lasso import LassoDraws
from src.nuts.horseshoe import HorseshoeDraws
from src.nuts.regularised_horseshoe import RegHorseshoeDraws

CLASSES = {
    "baseline_gaussian": (GibbsDraws, "rescaled_r2_0p05"),
    "bayesian_lasso": (LassoDraws, "rescaled_r2_0p05"),
    "horseshoe": (HorseshoeDraws, "p0_23_r2_0p05"),
    "regularised_horseshoe": (RegHorseshoeDraws, "p0_23_r2_0p05"),
}


def truncate(chain, n: int):
    """Return a copy of `chain` keeping only its first n draws.

    Every array field whose leading dimension is the draw axis is sliced;
    scalars and metadata are carried through unchanged. Truncating rather
    than thinning is deliberate: thinning would change the autocorrelation
    structure the ESS calculation depends on, whereas a prefix is exactly
    what a shorter run of the same sampler would have produced.
    """
    n_draws = None
    for f in chain.__dataclass_fields__:
        v = getattr(chain, f)
        if isinstance(v, np.ndarray) and v.ndim >= 1:
            n_draws = v.shape[0]
            break
    if n_draws is None:
        raise ValueError("no array fields found on the draws object")

    updates = {}
    for f in chain.__dataclass_fields__:
        v = getattr(chain, f)
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n_draws:
            updates[f] = v[:n]
    return replace(chain, **updates)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="size_bm_25")
    ap.add_argument("--model", default="horseshoe", choices=list(CLASSES))
    ap.add_argument("--field", default="Sigma",
                    help="parameter block to diagnose")
    ap.add_argument("--lengths", type=int, nargs="+", default=None,
                    help="retained-draw counts; defaults to successive "
                         "halvings of the full chain")
    args = ap.parse_args()

    cls, setting = CLASSES[args.model]
    chains = load_chains(args.universe, args.model, setting, cls)
    full = getattr(chains[0], args.field).shape[0]

    lengths = args.lengths
    if lengths is None:
        lengths = sorted({max(50, full // 2 ** k) for k in range(4)})

    print(f"{args.model} | {args.universe} | field {args.field}")
    print(f"{len(chains)} chains x {full} retained draws\n")
    print(f"{'draws/chain':>12}{'max R-hat':>12}{'min ESS':>10}"
          f"{'median ESS':>12}{'(Rhat-1) x ESS':>16}")

    for n in lengths:
        if n > full:
            continue
        d = diagnose([truncate(c, n) for c in chains], args.field)
        rhat = float(np.nanmax(d.rhat))
        ess_min = float(np.nanmin(d.ess_bulk))
        ess_med = float(np.nanmedian(d.ess_bulk))
        # evaluated at the WORST entry, since that is the one the threshold
        # comparison in Section 5.5 concerns
        i = int(np.nanargmax(d.rhat))
        prod = (rhat - 1.0) * float(d.ess_bulk[i])
        print(f"{n:>12}{rhat:>12.4f}{ess_min:>10.0f}{ess_med:>12.0f}"
              f"{prod:>16.2f}")

    print("\n  [Falling R-hat with rising ESS, at a roughly stable product,")
    print("   indicates finite Monte Carlo error rather than chains settled")
    print("   in different regions. A plateau in R-hat as ESS grows would")
    print("   indicate the opposite. The product is evaluated at the entry")
    print("   attaining the maximum R-hat at each length, so it tracks the")
    print("   same quantity the threshold comparison concerns.]")


if __name__ == "__main__":
    main()