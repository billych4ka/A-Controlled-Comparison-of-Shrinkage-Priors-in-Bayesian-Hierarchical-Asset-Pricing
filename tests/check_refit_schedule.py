"""
check_refit_schedule.py

Measures what coarsening the refit schedule actually costs, on the real data
rather than on simulation.

Reproduces Appendix A.3, the refit-schedule check, and supports Section 4.6's
sentence that the annual schedule reduces computational cost with little
apparent loss. NOTE what Appendix A.3's headline figures are: 200 fits reduced
to 17, moving out-of-sample R^2 by 0.15 percentage points and the Sharpe ratio
by 0.04. Those come from the SIMULATED comparison described below, not from
this script, and the paragraph below argues they are a poor guide at this
application's operating point. This script measures the same quantity where it
actually applies.

Section 4.6 justifies annual refits partly by computational feasibility and
partly by the claim that finer schedules add little. That second claim was
originally supported by a simulated-data comparison (N=8, K=5, T=400, ridge
fit, linear DGP) which gave out-of-sample R2 of +0.2863 at 200 fits against
+0.2848 at 17, and Sharpe +6.56 against +6.52. Those are real numbers but a
poor guide here: the simulated problem had R2 of +0.29 and Sharpe of +6.5,
whereas this application has R2 of about -0.05 and Sharpe of about 0.15. A
degradation of 0.0015 in R2 is negligible against 0.29 and is 3% of 0.05.

This script measures the same thing at the actual operating point. It also
matters practically: if NUTS forces a coarser grid for the horseshoes, every
model must be re-run on that grid, because refit frequency defines the
information set. Knowing the cost in advance decides whether that is
acceptable.

The 12-month result is loaded from the saved backtest rather than recomputed,
so the runtime below covers only the coarser schedules.

    python3 check_refit_schedule.py                     (60-month only, ~20 min)
    python3 check_refit_schedule.py --schedules 24 60   (~70 min)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from time import time

import numpy as np

from src.evaluation.backtest import expanding_window
from src.evaluation.metrics import (certainty_equivalent, out_of_sample_r2,
                                    portfolio_returns, sharpe_ratio)
from src.gibbs.backtest_adapter import gaussian_fit_fn

ap = argparse.ArgumentParser()
ap.add_argument("--universe", default="size_bm_25")
ap.add_argument("--target-r2", type=float, default=0.05)
ap.add_argument("--start", type=int, default=240)
ap.add_argument("--schedules", type=int, nargs="*", default=[60],
                help="coarser refit intervals to test, in months")
ap.add_argument("--draws", type=int, default=500)
ap.add_argument("--burn", type=int, default=100)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

d = np.load(Path("data/processed") / f"{args.universe}_arrays.npz", allow_pickle=True)
F, R = d["F"], d["R"]
N, T, K = F.shape


def metrics(realised, predicted, benchmark):
    pr = portfolio_returns(realised, predicted)
    return (out_of_sample_r2(realised, predicted, benchmark),
            sharpe_ratio(pr), certainty_equivalent(pr))


rows = []

saved = (Path("results") / args.universe / "baseline_gaussian" / "backtest"
         / f"baseline_gaussian_rescaled_r2_0p{str(args.target_r2)[2:]}_backtest.npz")
if not saved.exists():
    raise SystemExit(f"{saved} not found; run the 12-month backtest first, or "
                     f"point this script at the correct filename")
z = np.load(saved)
n_fits_12 = len(z["refit_index"])
rows.append((12, n_fits_12, *metrics(z["realised"], z["predicted"], z["benchmark"]), 0.0))
print(f"loaded 12-month schedule from disk: {n_fits_12} fits\n")

for every in sorted(args.schedules):
    fit_fn = gaussian_fit_fn(prior="rescaled", target_r2=args.target_r2,
                             n_draws=args.draws, n_burn=args.burn, seed=args.seed)
    n_fits = len(range(args.start, T, every))
    print(f"--- refit every {every} months: {n_fits} fits ---", flush=True)
    t0 = time()
    bt = expanding_window(fit_fn, R, F, start=args.start, refit_every=every,
                          progress=True)
    rows.append((every, n_fits, *metrics(bt.realised, bt.predicted, bt.benchmark),
                 time() - t0))
    print(f"    done in {time()-t0:.0f}s\n", flush=True)

base_r2, base_sr, base_ce = rows[0][2], rows[0][3], rows[0][4]
print("=" * 74)
print(f"REFIT SCHEDULE SENSITIVITY | {args.universe} | rescaled, "
      f"target R2 = {args.target_r2:g}")
print("=" * 74)
print("   %-8s %6s %10s %9s %10s %12s %10s"
      % ("every", "fits", "OOS R2", "Sharpe", "CE", "dR2 vs 12m", "dSharpe"))
for every, n_fits, r2, sr, ce, secs in rows:
    if every == 12:
        print("   %-8d %6d %+10.4f %+9.3f %+10.4f %12s %10s"
              % (every, n_fits, r2, sr, ce, "--", "--"))
    else:
        print("   %-8d %6d %+10.4f %+9.3f %+10.4f %+12.4f %+10.3f"
              % (every, n_fits, r2, sr, ce, r2 - base_r2, sr - base_sr))
print("\n   dR2 is in R2 units, not percentage points: 0.0015 here would be")
print("   0.15 percentage points. Judge it against the level, which is about")
print("   -0.05, not against the +0.29 of the original simulated comparison.")
print("   Sharpe standard error over 479 months is approximately 0.158, so a")
print("   difference well inside that is not a detected effect either way.")
