"""
check_reg_backtest_budget.py: how many draws does a regularised horseshoe
window fit need?

The backtest runs 40 fits per universe and uses only the POSTERIOR MEAN of B.
The production budget (4 chains x 1,500 draws) is sized for credible intervals
on tau and c, which the backtest never reports.

The plain horseshoe's answer was 750/500, chosen by the same procedure. That
figure CANNOT simply be inherited: this model's posterior is far better
conditioned (tree depth 7 against 9, step size 0.028 against 0.008, zero
divergences against 0.9%, 41 minutes against 3h12m), which argues for fewer
draws, but c is a NEW global parameter and tau mixes somewhat worse
(ESS 709 against 1,012), which argues for more.

TWO GLOBAL PARAMETERS MUST HAVE TRAVELLED, not one. tau starts at tau_0 and
the data moves it to 0.091 tau_0; c starts at slab_scale and the data pulls it
to 0.69 of prior E[c]. Retained draws taken before either has arrived sample B
under the wrong shrinkage level: a systematic error repeated in all 40
windows, not a Monte Carlo one. Both ratios are reported per candidate, and a
budget failing either is disqualified however good its z looks.

THE BINDING FRACTION IS ALSO REPORTED per candidate. If a short budget leaves
the slab binding for a materially different fraction than production's 12.4%,
the backtest is not fitting the same model the production run described.

RULE, FIXED BEFORE THE NUMBERS ARE SEEN: adopt the SMALLEST budget with
median z below ~1, everything within 4 SE, correlation above 0.999, tau within
20% of the reference, c within 20%, and binding within 3 percentage points of
12.4%. Do not pick the cheapest that passes after the fact, and do not pick
the most expensive because it looks marginally better.

Run from the project root. At the measured 0.34 s/iteration each candidate is
a few minutes:

    caffeinate -i python3 check_reg_backtest_budget.py
"""
from __future__ import annotations

import argparse
from pathlib import Path
from time import time

import numpy as np

from src.diagnostics.convergence import load_chains
from src.nuts.backtest_adapter import validate_budget_reg_horseshoe
from src.nuts.regularised_horseshoe import (RegHorseshoeDraws,
                                            reg_horseshoe_hyperparameters)

DEFAULT_GRID = [(300, 300), (500, 500), (750, 500), (1000, 1000)]
ESS_REF = 5591.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="size_bm_25")
    ap.add_argument("--setting", default="p0_23_r2_0p05")
    ap.add_argument("--candidates", default=None,
                    help="semicolon-separated draws,tune pairs, e.g. '500,500;750,500'")
    ap.add_argument("--start", type=int, default=240)
    ap.add_argument("--windows", default="both", choices=["both", "early", "full"])
    args = ap.parse_args()

    grid = DEFAULT_GRID
    if args.candidates:
        grid = [tuple(int(x) for x in p.split(",")) for p in args.candidates.split(";")]

    d = np.load(Path("../data/processed") / f"{args.universe}_arrays.npz",
                allow_pickle=True)
    F, R = d["F"], d["R"]
    N, T, K = F.shape

    chains = load_chains(args.universe, "regularised_horseshoe", args.setting,
                         RegHorseshoeDraws)
    B_ref = np.concatenate([c.B for c in chains], axis=0).mean(axis=0)
    tau_ref = float(np.concatenate([c.tau for c in chains]).mean())
    c_ref = float(np.concatenate([c.c for c in chains]).mean())
    bind_ref = float(np.mean([(c.lam_tilde / c.lam_local < 0.99).mean()
                              for c in chains]))
    hp = reg_horseshoe_hyperparameters(R, F, K)

    print("=" * 82)
    print(f"REG. HORSESHOE BACKTEST BUDGET | {args.universe} | {args.setting}")
    print(f"reference: {len(chains)} chains x {chains[0].B.shape[0]} draws, "
          f"B median bulk ESS {ESS_REF:.0f}")
    print(f"reference tau {tau_ref:.4e} ({tau_ref/hp.tau_0:.4f} x tau_0), "
          f"c {c_ref:.4e} ({c_ref/hp.slab_sd:.4f} x E[c]), "
          f"binding {100*bind_ref:.1f}%")
    print("=" * 82)
    print("\nRULE, fixed before the numbers: adopt the SMALLEST budget with median")
    print("z below ~1, all within 4 SE, correlation above 0.999, tau AND c within")
    print("20% of reference, and binding within 3pp of the reference.")

    windows = []
    if args.windows in ("both", "early"):
        windows.append((f"early t={args.start}", R[:, :args.start],
                        F[:, :args.start, :]))
    if args.windows in ("both", "full"):
        windows.append((f"full t={T}", R, F))

    for wlabel, Rw, Fw in windows:
        Tw = Rw.shape[1]
        print(f"\n{'='*82}\n{wlabel}   ({Tw} months, {Tw-K} residual df)\n{'='*82}")
        if Tw < T:
            print("   NOTE: compared against the FULL-SAMPLE reference B, which is")
            print("   not this window's posterior mean. Read tau/ref, c/ref, the")
            print("   binding fraction and the timing here; the z statistics are")
            print("   only interpretable on the full window.")
        print(f"\n   {'draws':>6}{'tune':>6}{'med z':>8}{'max z':>8}{'<3SE':>7}"
              f"{'corr':>9}{'tau/ref':>9}{'c/ref':>8}{'bind%':>8}{'div':>5}{'min':>7}")
        for n_draws, n_tune in grid:
            t0 = time()
            o = validate_budget_reg_horseshoe(Rw, Fw, B_ref, ESS_REF,
                                              n_draws=n_draws, n_tune=n_tune)
            print(f"   {n_draws:>6d}{n_tune:>6d}{o['median_z']:>8.2f}"
                  f"{o['max_z']:>8.2f}{100*o['frac_within_3']:>6.1f}%"
                  f"{o['correlation']:>9.4f}{o['tau_mean']/tau_ref:>9.3f}"
                  f"{o['c_mean']/c_ref:>8.3f}{100*o['frac_binding']:>7.1f}%"
                  f"{o['divergences']:>5d}{(time()-t0)/60:>7.1f}", flush=True)

    print("\n   Read the MEDIAN z, not the max: with 3,600 parameters two perfectly")
    print("   agreeing estimates give a max |z| whose own median is 3.73. The plain")
    print("   horseshoe's accepted 750/500 gave median 0.60, 99.8% within 3 SE,")
    print("   correlation 0.9990, at 21.1 min per full-window fit.")
    print("\n   Multiply the chosen budget's time by 40 for one universe, and note")
    print("   fits get SLOWER as the window expands: the plain horseshoe's cost")
    print("   grew FASTER than linearly in T, because a longer window sharpens the")
    print("   posterior, lowers the step size and raises tree depth.")


if __name__ == "__main__":
    main()
