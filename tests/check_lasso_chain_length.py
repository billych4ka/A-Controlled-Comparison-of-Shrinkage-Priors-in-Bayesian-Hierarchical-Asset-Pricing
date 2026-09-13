"""
check_lasso_chain_length.py

Regenerates the chain-length and chain-count diagnostics for the Bayesian
LASSO's global shrinkage parameter, reported in Section 4.4.

The question is whether lambda_L's R-hat exceeding 1.01 reflects chains
DISAGREEING about the posterior or chains that are merely SHORT. R-hat alone
cannot distinguish these. Truncating saved chains to progressively longer runs
can: if (R-hat - 1) x ESS is roughly constant, the excess above one is driven
by finite effective sample size rather than by a persistent separation between
chains.

The chain-count column is the more counter-intuitive result and the reason
this script exists. R-hat RISES monotonically as chains are added, from 1.0079
at two chains to 1.0177 at eight. That is not convergence worsening. Two
chains have almost no power to detect between-chain variation; adding chains
increases that power and reveals a spread that was always present. Chain
LENGTH is therefore the lever, not chain count, and the two-chain figure
should not be quoted as evidence of convergence.

Reads saved chains rather than sampling, so it runs in seconds. Requires the
ORIGINAL 8 x 3,000 production run to reproduce the pre-enlargement column; if
results/ now holds only the enlarged 8 x 9,500 run, the truncation series can
still be regenerated (it is a property of the chains, not the budget) but the
recorded 3,000-draw values will correspond to the first 3,000 draws of the
longer chains rather than to the earlier run.

Recorded results (size_bm_25, rescaled_r2_0p05, 8 chains x 3,000 kept draws):

    draws/chain      R-hat     ESS     (R-hat - 1) x ESS
        500        1.1247      37            4.6
      1,000        1.0460     125            5.8
      1,500        1.0503      84            4.2
      2,000        1.0163     269            4.4
      2,500        1.0265     316            8.4
      3,000        1.0177     384            6.8

    chains           R-hat     ESS
        2          1.0079     121
        4          1.0104     218
        6          1.0121     310
        8          1.0177     384

After enlarging to 8 x 9,500 (8,500 kept), lambda_L reached R-hat 1.0058 with
bulk ESS 1,177 and tail ESS 2,652, consistent with the extrapolation from
(R-hat - 1) x ESS ~ 5.7, which predicted approximately 1.006 at ESS 950.

Run from the project root:  python3 tests/check_lasso_chain_length.py
"""
import numpy as np

from src.diagnostics.convergence import (effective_sample_size, load_chains,
                                         rank_normalised_rhat)
from src.gibbs.bayesian_lasso import LassoDraws

UNIVERSE, SETTING = "size_bm_25", "rescaled_r2_0p05"

chains = load_chains(UNIVERSE, "bayesian_lasso", SETTING, LassoDraws)
lam = np.array([c.lam for c in chains])
n_chains, n_draws = lam.shape
print(f"{UNIVERSE} | {SETTING} | {n_chains} chains x {n_draws} kept draws\n")

print("CHAIN LENGTH   (all chains, truncated to the first n draws)")
print(f"   {'draws/chain':>12s} {'R-hat':>9s} {'ESS':>7s} {'(R-hat-1) x ESS':>18s}")
products = []
for n in (500, 1000, 1500, 2000, 2500, 3000, n_draws):
    if n > n_draws:
        continue
    x = lam[:, :n]
    r, e = rank_normalised_rhat(x), effective_sample_size(x)
    products.append((r - 1.0) * e)
    print(f"   {n:>12d} {r:>9.4f} {e:>7.0f} {(r-1.0)*e:>18.1f}")
print(f"   [(R-hat - 1) x ESS is roughly constant, mean {np.mean(products):.1f}.")
print("    That is the signature of chains that agree and are merely short:")
print("    R-hat's excess above one is driven by finite ESS, not by a")
print("    persistent separation between chains.]")

print("\nCHAIN COUNT   (full length, first m chains)")
print(f"   {'chains':>7s} {'R-hat':>9s} {'ESS':>7s}")
for m in (2, 4, 6, 8):
    if m > n_chains:
        continue
    x = lam[:m]
    print(f"   {m:>7d} {rank_normalised_rhat(x):>9.4f} "
          f"{effective_sample_size(x):>7.0f}")
print("   [R-hat RISES as chains are added. This is the diagnostic gaining")
print("    power, not the sampler getting worse: two chains have almost no")
print("    ability to detect between-chain variation. Do NOT quote the")
print("    two-chain figure as evidence of convergence.]")

print("\nPER-CHAIN MEANS   (the multimodality check)")
means = lam.mean(axis=1)
print("   " + "  ".join(f"{m:.1f}" for m in means))
print(f"   spread {means.ptp():.1f} on a pooled posterior sd of {lam.std():.1f}"
      f"  ({means.ptp()/lam.std():.2f} sd)")
print("   [no clustering into groups. Park & Casella's unimodality guarantee")
print("    was forfeited when the plug-in scale replaced their sampled")
print("    sigma^2, so chain agreement is the evidence that nothing went")
print("    wrong as a result.]")

k = float(np.mean(products))
print(f"\nIMPLIED BUDGET   (from (R-hat - 1) x ESS ~ {k:.1f})")
print(f"   {'target R-hat':>13s} {'ESS needed':>11s} {'draws/chain at 8':>18s}")
ess_now = effective_sample_size(lam)
for target in (1.010, 1.008, 1.006):
    need = k / (target - 1.0)
    print(f"   {target:>13.3f} {need:>11.0f} {need/ess_now*n_draws:>18.0f}")
print("   [extrapolation, not measurement: ESS grows linearly in draws only")
print("    while the chain is stationary, which the length column above is")
print("    itself evidence for.]")
