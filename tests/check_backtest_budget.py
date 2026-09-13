"""
check_backtest_budget.py: how many draws does a horseshoe window fit need?

The backtest runs 40 fits per universe and uses only the POSTERIOR MEAN of B.
The production budget (4 chains x 1,500 draws, 1,000 tune) is sized for
credible intervals on tau, which the backtest never reports. This script finds
the smallest per-window budget that still recovers the production posterior
mean of B, and measures what it costs, BEFORE 12 hours of compute run on a
guess.

Both Gibbs models had this done. The baseline settled on 500 sweeps / 100
burn after measuring median z 0.72, 100% within 3 SE, correlation 0.9989
against its full posterior; the LASSO on 1,200 / 700, its extra draws going
to burn-in rather than to B, because lambda had to travel from its hyperprior
centre before retained draws were sampling B under the right shrinkage level.

The horseshoe has the same burn-in concern in a sharper form: tau starts at
tau_0 and the data moves it to about 0.05 tau_0, a factor of twenty. Draws
retained before tau has travelled sample B under the wrong shrinkage: a
systematic error repeated in all 40 windows, not a Monte Carlo one. So this
script reports tau per candidate budget alongside the agreement statistics: a
budget whose tau has not reached ~0.05 tau_0 is disqualified however good its
z looks.

TWO WINDOWS ARE TESTED, and the early one is the point. At t=240 there are 96
residual degrees of freedom against 575 on the full sample, so sigma_pooled is
noisier, the likelihood is less informative, the prior has more say, and the
geometry is harder. A budget validated only on the full sample would be
validated where the problem is easiest.

DO NOT pick the cheapest budget that passes after seeing the results; that
is selection on outcomes, the same genre as revising target_r2. Decide the
rule first: the smallest budget with median z below about 1, everything
within 4 SE, correlation above 0.999, and tau within 20% of the production
value.

Run from the project root. Each candidate is one fit; expect 5-20 minutes
each, so roughly 1-2 hours for the default grid:

    caffeinate -i python3 check_backtest_budget.py
    python3 check_backtest_budget.py --candidates 500,500 --windows full
"""
from __future__ import annotations

import argparse
from pathlib import Path
from time import time

import numpy as np

from src.diagnostics.convergence import load_chains
from src.nuts.backtest_adapter import validate_budget_horseshoe
from src.nuts.horseshoe import HorseshoeDraws, horseshoe_hyperparameters

DEFAULT_GRID = [(300, 300), (500, 500), (750, 500), (1000, 1000)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="size_bm_25")
    ap.add_argument("--setting", default="p0_23_r2_0p05")
    ap.add_argument("--candidates", default=None,
                    help="comma-separated draws,tune pairs; e.g. '500,500;750,500'")
    ap.add_argument("--windows", default="both", choices=["both", "early", "full"])
    ap.add_argument("--start", type=int, default=240,
                    help="the early window: the first month forecast")
    args = ap.parse_args()

    grid = DEFAULT_GRID
    if args.candidates:
        grid = [tuple(int(x) for x in pair.split(","))
                for pair in args.candidates.split(";")]

    d = np.load(Path("../data/processed") / f"{args.universe}_arrays.npz",
                allow_pickle=True)
    F, R = d["F"], d["R"]
    N, T, K = F.shape

    chains = load_chains(args.universe, "horseshoe", args.setting, HorseshoeDraws)
    n_draw = chains[0].B.shape[0]
    B_ref = np.concatenate([c.B for c in chains], axis=0).mean(axis=0)
    tau_ref = float(np.concatenate([c.tau for c in chains]).mean())
    hp_full = horseshoe_hyperparameters(R, F, K)

    ESS_REF = 5591.0

    print("=" * 78)
    print(f"BACKTEST BUDGET | {args.universe} | {args.setting}")
    print(f"reference: {len(chains)} chains x {n_draw} draws, B median bulk "
          f"ESS {ESS_REF:.0f}")
    print(f"reference tau {tau_ref:.4e} = {tau_ref/hp_full.tau_0:.4f} x tau_0")
    print("=" * 78)
    print("\nRULE, fixed before the numbers are seen: adopt the SMALLEST budget")
    print("with median z below ~1, everything within 4 SE, correlation above")
    print("0.999, and tau within 20% of the reference. Do not pick the cheapest")
    print("that passes after the fact.")

    windows = []
    if args.windows in ("both", "early"):
        windows.append(("early t=%d" % args.start, R[:, :args.start],
                        F[:, :args.start, :]))
    if args.windows in ("both", "full"):
        windows.append(("full t=%d" % T, R, F))

    for wlabel, Rw, Fw in windows:
        Tw = Rw.shape[1]
        print(f"\n{'='*78}\n{wlabel}   ({Tw} months, {Tw - K} residual df)\n{'='*78}")
        if Tw < T:
            print("   NOTE: compared against the FULL-SAMPLE reference B, which is")
            print("   not this window's posterior mean. Read the tau column and the")
            print("   timing here; the z statistics are only interpretable on the")
            print("   full window.")
        print(f"\n   {'draws':>7}{'tune':>7}{'median z':>11}{'max z':>9}"
              f"{'<3 SE':>9}{'corr':>10}{'tau/ref':>10}{'div':>6}{'min':>8}")
        for n_draws, n_tune in grid:
            t0 = time()
            out = validate_budget_horseshoe(Rw, Fw, B_ref, ESS_REF,
                                            n_draws=n_draws, n_tune=n_tune)
            mins = (time() - t0) / 60
            print(f"   {n_draws:>7d}{n_tune:>7d}{out['median_z']:>11.2f}"
                  f"{out['max_z']:>9.2f}{100*out['frac_within_3']:>8.1f}%"
                  f"{out['correlation']:>10.4f}"
                  f"{out['tau_mean']/tau_ref:>10.3f}{out['divergences']:>6d}"
                  f"{mins:>8.1f}", flush=True)

    print("\n   Read the MEDIAN z, not the max: with 3,600 parameters two")
    print("   perfectly agreeing estimates give a max |z| whose own median is")
    print("   3.73 (5th-95th 3.34-4.34). The baseline's accepted budget gave")
    print("   median 0.72, 100% within 3 SE, correlation 0.9989.")
    print("\n   Multiply the chosen budget's time by 40 for one universe, and")
    print("   note fits get SLOWER as the window expands, so the early-window")
    print("   time is a lower bound on the average.")


if __name__ == "__main__":
    main()
