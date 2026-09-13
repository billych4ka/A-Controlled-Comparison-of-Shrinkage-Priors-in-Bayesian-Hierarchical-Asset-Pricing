"""
src/visualisation/plot_return_history.py

The realised return history of the three portfolio universes, over the
modelling sample.

WHY THIS FIGURE EXISTS. Table 3.1 characterises the cross-section: how much
the 25 portfolios within a universe differ in mean, volatility and Sharpe
ratio. It says nothing about the time dimension. This figure supplies it, and
in doing so makes two properties of the sample visible that the rest of the
dissertation assumes rather than shows: that the three universes are
alternative partitions of one underlying market rather than distinct
investment opportunities, and that volatility varies by a factor of several
across the sample, so a constant residual covariance is an approximation.

TWO PANELS.

  (a) Cumulative log excess return of the equally weighted portfolio of each
      universe. The three tracks are nearly indistinguishable, which is the
      point: the sorts differ in how they slice the cross-section, not in
      what they hold in aggregate.

  (b) Rolling 36-month annualised volatility of the same three series, with
      the cross-sectional dispersion of the primary universe shaded behind
      it. Both the common time variation and the persistent spread across
      portfolios are visible in one panel.

Equal weighting is used rather than the value weighting within each portfolio
because the object here is the universe as a whole, not any one portfolio, and
an equally weighted average keeps the three universes comparable.

PRESENTATION. Follows the dissertation's common visual style: sans-serif
typography with STIX sans mathematics, thin dark-grey axes, the same restrained
palette, and line styles that differ as well as colours so the figure remains
readable in monochrome.

Run from the project root:

    python3 -m src.visualisation.plot_return_history
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


UNIVERSES = (
    ("Size$\\times$BM", "size_bm_25", "#4A4A4A", "-"),
    ("Size$\\times$OP", "size_op_25", "#3B75AF", "--"),
    ("Size$\\times$Inv", "size_inv_25", "#D05A3A", "-."),
)

PRIMARY = "size_bm_25"

START_YEAR = 1966
START_MONTH = 1

ROLL = 36


STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "stixsans",
    "font.size": 8.0,

    "axes.labelsize": 8.2,
    "axes.labelweight": "normal",
    "axes.linewidth": 0.65,
    "axes.edgecolor": "#333333",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": "white",

    "xtick.labelsize": 7.4,
    "ytick.labelsize": 7.4,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.major.width": 0.55,
    "ytick.major.width": 0.55,

    "lines.linewidth": 1.15,
    "lines.solid_capstyle": "round",
    "lines.dash_capstyle": "round",

    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.5,

    "legend.frameon": False,
    "legend.fontsize": 7.2,

    "figure.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
    "savefig.transparent": False,
}


def load_universe(universe: str) -> np.ndarray:
    """Return R, the (N, T) matrix of monthly excess returns."""
    path = Path("data/processed") / f"{universe}_arrays.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Available: "
            f"{sorted(p.name for p in Path('data/processed').glob('*_arrays.npz'))}"
        )
    return np.load(path, allow_pickle=True)["R"]


def decimal_years(T: int) -> np.ndarray:
    """Month index as a decimal year, from the start of the modelling sample."""
    return START_YEAR + (START_MONTH - 1 + np.arange(T)) / 12.0


def rolling_vol(x: np.ndarray, window: int) -> np.ndarray:
    """Annualised rolling standard deviation; NaN before the window fills."""
    T = x.shape[-1]
    out = np.full(T, np.nan)
    for t in range(window - 1, T):
        out[t] = x[..., t - window + 1:t + 1].std(ddof=1) * np.sqrt(12.0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roll", type=int, default=ROLL,
                    help="rolling volatility window in months")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--format", default="pdf", choices=["pdf", "png"])
    args = ap.parse_args()

    loaded = {stem: load_universe(stem) for _, stem, _, _ in UNIVERSES}
    T = loaded[PRIMARY].shape[1]
    if any(R.shape[1] != T for R in loaded.values()):
        raise SystemExit("universes have different sample lengths")
    years = decimal_years(T)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6))

        ax = axes[0]
        for label, stem, colour, ls in UNIVERSES:
            ew = loaded[stem].mean(axis=0)
            cum = np.cumsum(np.log1p(ew))
            ax.plot(years, cum, color=colour, linestyle=ls, label=label,
                    zorder=3)
        ax.axhline(0.0, color="#BBBBBB", linewidth=0.5, zorder=1)
        ax.set_ylabel("Cumulative log excess return")
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
        ax.text(0.02, 0.95, "(a)", transform=ax.transAxes,
                fontsize=8.2, fontweight="bold", va="top")

        ax = axes[1]

        Rp = loaded[PRIMARY]
        per_asset = np.array([rolling_vol(Rp[i], args.roll)
                              for i in range(Rp.shape[0])])
        lo = np.nanmin(per_asset, axis=0)
        hi = np.nanmax(per_asset, axis=0)
        ax.fill_between(years, lo, hi, color="#4A4A4A", alpha=0.12,
                        linewidth=0.0, zorder=1,
                        label="Size$\\times$BM, range across portfolios")

        for label, stem, colour, ls in UNIVERSES:
            ew = loaded[stem].mean(axis=0)
            ax.plot(years, rolling_vol(ew, args.roll), color=colour,
                    linestyle=ls, zorder=3)

        ax.set_ylabel(f"{args.roll}-month volatility, annualised")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
        ax.text(0.02, 0.95, "(b)", transform=ax.transAxes,
                fontsize=8.2, fontweight="bold", va="top")

        for ax in axes:
            ax.set_xlim(years[0], years[-1])
            ax.set_xlabel("")

        handles, labels = [], []
        for a in axes:
            for h, l in zip(*a.get_legend_handles_labels()):
                if l not in labels:
                    handles.append(h)
                    labels.append(l)
        fig.legend(handles, labels, loc="lower center", ncol=4,
                   bbox_to_anchor=(0.5, -0.06))

        fig.tight_layout()

        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        out = outdir / f"return_history.{args.format}"
        fig.savefig(out, bbox_inches="tight")
        print(f"wrote {out}")

        for label, stem, _, _ in UNIVERSES:
            ew = loaded[stem].mean(axis=0)
            v = rolling_vol(ew, args.roll)
            print(f"  {label:<16s} ann. vol range "
                  f"{np.nanmin(v):.1%} to {np.nanmax(v):.1%}, "
                  f"total log return {np.cumsum(np.log1p(ew))[-1]:.2f}")
        corr = np.corrcoef(np.array([loaded[s].mean(axis=0)
                                     for _, s, _, _ in UNIVERSES]))
        print(f"  pairwise correlation of the three equally weighted series: "
              f"{corr[0,1]:.3f}, {corr[0,2]:.3f}, {corr[1,2]:.3f}")


if __name__ == "__main__":
    main()
