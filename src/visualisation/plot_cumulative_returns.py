"""
src/visualisation/plot_cumulative_returns.py

Cumulative return of the cross-sectional long-short strategy of Section 4.6,
one panel per portfolio universe, four models per panel. Supports Table 5.4
by showing WHEN the horseshoe family's Sharpe advantage accrued rather than
only that it did.

Portfolio returns are recomputed from the saved backtest forecasts through
src.evaluation.metrics.portfolio_returns, the same function that produced
Table 5.4, so the figure cannot drift away from the reported numbers. The
cumulative series is the running sum of log(1 + r_p,t), matching the
cumulative log excess return convention of Figure 3.1.

Dates: oos_index is an integer index into the modelling sample, whose first
month is January 1966 (Section 3.1). The default --sample-start reflects
that; override it only if the data pipeline's sample start changes.

Run from the project root:

    python3 -m src.visualisation.plot_cumulative_returns

Optional:
    --stacked      add a fourth panel with the equal-weight stack of the three
                   sleeves (the "Stacked" column of Table 5.5)
    --format png   raster output instead of pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.metrics import portfolio_returns, sharpe_ratio


UNIVERSES = (
    ("size_bm_25", r"Size$\times$BM"),
    ("size_op_25", r"Size$\times$OP"),
    ("size_inv_25", r"Size$\times$Inv"),
)

MODELS = (
    ("Gaussian baseline", "baseline_gaussian", "rescaled_r2_0p05",
     "#4A4A4A", "-", 1.15),
    ("Bayesian LASSO", "bayesian_lasso", "rescaled_r2_0p05",
     "#3B75AF", "--", 1.15),
    ("Horseshoe", "horseshoe", "p0_23_r2_0p05",
     "#D05A3A", "-", 1.15),
    ("Regularised horseshoe", "regularised_horseshoe", "p0_23_r2_0p05",
     "#3A9679", "-.", 1.15),
)


STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "stixsans",
    "font.size": 8.0,
    "axes.labelsize": 8.0,
    "axes.labelweight": "normal",
    "axes.linewidth": 0.60,
    "axes.edgecolor": "#333333",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": "white",
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 2.8,
    "ytick.major.size": 2.8,
    "xtick.major.width": 0.50,
    "ytick.major.width": 0.50,
    "xtick.major.pad": 2.0,
    "ytick.major.pad": 2.0,
    "lines.solid_capstyle": "round",
    "lines.dash_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "legend.frameon": False,
    "legend.fontsize": 7.0,
    "figure.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
    "savefig.transparent": False,
}


def load_backtest(results_dir: Path, universe: str, model: str, setting: str):
    p = results_dir / universe / model / "backtest" / f"{model}_{setting}_backtest.npz"
    if not p.exists():
        raise FileNotFoundError(p)
    d = np.load(p)
    return d["predicted"], d["realised"], d["oos_index"]


def cumulative_log_return(r: np.ndarray) -> np.ndarray:
    """Running sum of log(1 + r), starting from zero at the first month."""
    return np.cumsum(np.log1p(r))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--format", default="pdf", choices=["pdf", "png"])
    ap.add_argument("--sample-start", default="1966-01",
                    help="calendar month of index 0 of the modelling sample")
    ap.add_argument("--stacked", action="store_true",
                    help="add a fourth panel: equal-weight stack of the three sleeves")
    args = ap.parse_args()

    results_dir = Path(args.results)
    sample_start = pd.Timestamp(args.sample_start)

    series: dict[str, dict[str, np.ndarray]] = {}
    dates: pd.DatetimeIndex | None = None

    for utag, _ in UNIVERSES:
        series[utag] = {}
        for label, model, setting, *_ in MODELS:
            try:
                pred, real, oos = load_backtest(results_dir, utag, model, setting)
            except FileNotFoundError as e:
                print(f"  ({e} not found, omitted)")
                continue
            r = portfolio_returns(real, pred)
            series[utag][label] = r
            this_dates = pd.DatetimeIndex(
                [sample_start + pd.DateOffset(months=int(t)) for t in oos])
            if dates is None:
                dates = this_dates
            elif not dates.equals(this_dates):
                raise ValueError(f"oos_index differs for {utag}/{model}")
            print(f"  {utag:<12s} {label:<24s} Sharpe {sharpe_ratio(r):+.3f}  "
                  f"cum log return {cumulative_log_return(r)[-1]:+.3f}")

    if dates is None:
        raise SystemExit("no backtest files found")

    panels = [(utag, ulabel, series[utag]) for utag, ulabel in UNIVERSES
              if series[utag]]

    if args.stacked:
        stacked: dict[str, np.ndarray] = {}
        for label, *_ in MODELS:
            sleeves = [series[u][label] for u, _ in UNIVERSES if label in series[u]]
            if len(sleeves) == len(UNIVERSES):
                stacked[label] = np.mean(sleeves, axis=0)
                print(f"  {'stacked':<12s} {label:<24s} Sharpe "
                      f"{sharpe_ratio(stacked[label]):+.3f}")
        if stacked:
            panels.append(("stacked", "Stacked", stacked))

    plt.rcParams.update(STYLE)
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(6.5, 2.35), sharey=True)
    axes = np.atleast_1d(axes)

    styles = {label: (c, ls, lw) for label, _, _, c, ls, lw in MODELS}
    handles: dict[str, object] = {}

    for ax, (utag, ulabel, by_model) in zip(axes, panels):
        for label, r in by_model.items():
            c, ls, lw = styles[label]
            (h,) = ax.plot(dates, cumulative_log_return(r), color=c,
                           linestyle=ls, linewidth=lw, label=label, zorder=3)
            handles.setdefault(label, h)

        ax.axhline(0.0, color="#B0B0B0", linewidth=0.5, zorder=1)
        ax.text(0.03, 0.95, ulabel, transform=ax.transAxes,
                ha="left", va="top", fontsize=7.6)
        ax.xaxis.set_major_locator(mdates.YearLocator(10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_xlim(dates[0], dates[-1])
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.45, zorder=0)
        ax.tick_params(axis="x", which="major", length=2.8)

    axes[0].set_ylabel("Cumulative log return", labelpad=4)

    fig.legend(handles.values(), handles.keys(), loc="lower center",
               ncol=len(handles), bbox_to_anchor=(0.5, -0.02),
               handlelength=2.6, columnspacing=1.6)

    fig.subplots_adjust(left=0.085, right=0.985, top=0.96, bottom=0.20,
                        wspace=0.10)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    suffix = "_stacked" if args.stacked else ""
    out = outdir / f"cumulative_returns{suffix}.{args.format}"
    fig.savefig(out, dpi=300 if args.format == "png" else None)
    print(f"\n  saved {out}")


if __name__ == "__main__":
    main()
