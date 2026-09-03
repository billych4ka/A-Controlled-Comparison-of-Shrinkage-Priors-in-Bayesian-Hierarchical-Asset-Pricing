"""
src/visualisation/plot_economic_differences.py

Change in economic performance relative to the Gaussian baseline, by model and
portfolio universe.

WHY DIFFERENCES RATHER THAN LEVELS. A grouped bar chart of the twelve Sharpe
ratios and certainty equivalents would be a graphical transcription of the
results table: it shows which bars are taller, which the table already shows.
The experimental question is different. The design holds the likelihood, the
common-component prior and the residual-covariance prior fixed and changes
only the prior on the asset-specific deviations, so the quantity of interest
is the INCREMENT in economic performance attributable to that change. Plotting

    Sharpe(model) - Sharpe(Gaussian baseline)

answers that directly, and it gives zero a common meaning in both panels: no
improvement over the experimental control. The Gaussian baseline is not
plotted because it is identically zero by construction.

WHY NO UNCERTAINTY BAND. The standard error of an individual annualised
Sharpe ratio over the 479-month evaluation period is 0.158, and it is
tempting to show it here. That would be wrong. It is the standard error of a
LEVEL, not of a DIFFERENCE, and the four portfolios are constructed from the
same 25 assets over the same months, so their returns are highly correlated
and the standard error of a difference between two of them is generally
smaller than that of either. Displaying 0.158 around these points would
therefore overstate the uncertainty in the direction that happens to favour
caution, which is not a good reason to display a wrong quantity. No formal
test is reported instead, because a test for dependent Sharpe ratios requires
a covariance estimate the design does not otherwise need; the accompanying
text states this.

WHY DOTS RATHER THAN BARS. There is no continuous ordering between the three
prior families, so connecting or stacking them would imply structure that
does not exist. Points offset within each universe row keep the comparison
within a universe immediate while allowing the pattern across universes to be
read down the column.

PRESENTATION. This figure follows the dissertation's common visual style:
clean sans-serif typography with STIX sans mathematics, thin dark-grey axes,
the same model palette used throughout, subtle reference lines, internal bold
panel labels, and a compact shared legend. Marker shape varies with colour so
the figure remains interpretable in monochrome.

Run from the project root:

    python3 -m src.visualisation.plot_economic_differences
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from src.evaluation.metrics import (
    certainty_equivalent,
    portfolio_returns,
    sharpe_ratio,
)


# ---------------------------------------------------------------------------
# Portfolio universes
# ---------------------------------------------------------------------------

UNIVERSES = (
    ("Size$\\times$BM", "size_bm_25"),
    ("Size$\\times$OP", "size_op_25"),
    ("Size$\\times$Inv", "size_inv_25"),
)


# ---------------------------------------------------------------------------
# Experimental control
# ---------------------------------------------------------------------------

BASELINE = (
    "baseline_gaussian",
    "rescaled_r2_0p05",
)


# ---------------------------------------------------------------------------
# Alternative deviation priors
# ---------------------------------------------------------------------------

# Colours are identical to the dissertation house palette.
#
# (label, directory, setting tag, colour, marker)
MODELS = (
    (
        "Bayesian LASSO",
        "bayesian_lasso",
        "rescaled_r2_0p05",
        "#3B75AF",
        "o",
    ),
    (
        "Horseshoe",
        "horseshoe",
        "p0_23_r2_0p05",
        "#D05A3A",
        "s",
    ),
    (
        "Regularised horseshoe",
        "regularised_horseshoe",
        "p0_23_r2_0p05",
        "#3A9679",
        "D",
    ),
)


# ---------------------------------------------------------------------------
# Publication-style plotting defaults
# ---------------------------------------------------------------------------

STYLE = {
    # Typography
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Arial",
        "Helvetica",
        "Liberation Sans",
        "DejaVu Sans",
    ],
    "mathtext.fontset": "stixsans",
    "font.size": 8.0,

    # Axes
    "axes.labelsize": 8.2,
    "axes.labelweight": "normal",
    "axes.linewidth": 0.65,
    "axes.edgecolor": "#333333",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.facecolor": "white",

    # Ticks
    "xtick.labelsize": 7.4,
    "ytick.labelsize": 7.4,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.0,
    "ytick.major.size": 0.0,
    "xtick.minor.size": 1.6,
    "ytick.minor.size": 0.0,
    "xtick.major.width": 0.55,
    "ytick.major.width": 0.0,
    "xtick.minor.width": 0.40,
    "ytick.minor.width": 0.0,
    "xtick.major.pad": 2.5,
    "ytick.major.pad": 4.0,

    # Legend
    "legend.frameon": False,
    "legend.fontsize": 7.0,

    # Figure
    "figure.facecolor": "white",

    # Vector export
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    # Figure export
    "savefig.facecolor": "white",
    "savefig.transparent": False,
}


def load_metrics(
    universe: str,
    model: str,
    setting: str,
) -> tuple[float, float]:
    """
    Sharpe ratio and certainty equivalent, recomputed from the saved backtest
    rather than read from a summary, so the figure cannot drift away from the
    stored forecasts.
    """

    p = (
        Path("results")
        / universe
        / model
        / "backtest"
        / f"{model}_{setting}_backtest.npz"
    )

    d = np.load(p)

    r = portfolio_returns(
        d["realised"],
        d["predicted"],
    )

    return (
        sharpe_ratio(r),
        certainty_equivalent(r),
    )


def padded_limits(
    data: dict[str, list[float]],
    *,
    left_fraction: float = 0.08,
    right_fraction: float = 0.08,
) -> tuple[float, float]:
    """
    Data-driven horizontal limits with enough room around zero and the most
    extreme point for the markers not to sit against the frame.

    Zero is always included because it is the experimental-control reference.
    This affects presentation only.
    """

    values = np.concatenate(
        [
            np.asarray(v, dtype=float)
            for v in data.values()
        ]
    )

    values = values[
        np.isfinite(values)
    ]

    lo = min(
        0.0,
        float(values.min()),
    )

    hi = max(
        0.0,
        float(values.max()),
    )

    span = hi - lo

    if span == 0:
        span = 1.0

    return (
        lo - left_fraction * span,
        hi + right_fraction * span,
    )


def main() -> None:

    # -----------------------------------------------------------------------
    # Arguments
    # -----------------------------------------------------------------------

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--outdir",
        default="figures",
    )

    ap.add_argument(
        "--format",
        default="pdf",
        choices=["pdf", "png"],
    )

    args = ap.parse_args()

    # -----------------------------------------------------------------------
    # Gaussian-baseline performance
    # -----------------------------------------------------------------------

    base_s = {}
    base_c = {}

    for _, universe in UNIVERSES:

        s, c = load_metrics(
            universe,
            *BASELINE,
        )

        base_s[universe] = s
        base_c[universe] = c

    # -----------------------------------------------------------------------
    # Differences relative to Gaussian baseline
    # -----------------------------------------------------------------------

    d_sharpe = {}
    d_ce = {}

    for (
        label,
        model,
        setting,
        _,
        _,
    ) in MODELS:

        ds = []
        dc = []

        for _, universe in UNIVERSES:

            try:

                s, c = load_metrics(
                    universe,
                    model,
                    setting,
                )

            except FileNotFoundError:

                print(
                    f"  ({model} / {universe} not found)"
                )

                s = np.nan
                c = np.nan

            ds.append(
                s - base_s[universe]
            )

            dc.append(
                c - base_c[universe]
            )

        d_sharpe[label] = ds
        d_ce[label] = dc

        print(
            f"  {label:<24s}"
            f" dSharpe "
            + " ".join(
                f"{v:+.3f}"
                for v in ds
            )
            + "   dCE "
            + " ".join(
                f"{v:+.4f}"
                for v in dc
            )
        )

    # -----------------------------------------------------------------------
    # Figure setup
    # -----------------------------------------------------------------------

    plt.rcParams.update(STYLE)

    fig, (a1, a2) = plt.subplots(
        1,
        2,
        figsize=(6.5, 2.35),
    )

    fig.patch.set_facecolor("white")

    # First universe at the top.
    rows = np.arange(
        len(UNIVERSES)
    )[::-1]

    ticks = [
        label
        for label, _ in UNIVERSES
    ]

    # Small symmetric vertical dodge within each universe.
    #
    # This is large enough to distinguish the three markers but small enough
    # that they continue to read as one universe-level comparison.


    # -----------------------------------------------------------------------
    # Shared plotting function
    # -----------------------------------------------------------------------

    panels = (
        (
            a1,
            d_sharpe,
            r"Change in annualised Sharpe ratio",
        ),
        (
            a2,
            d_ce,
            r"Change in certainty equivalent",
        ),
    )

    for ax, data, xlabel in panels:

        # One very light horizontal guide through each universe. This groups
        # the three vertically dodged points without creating heavy banding.
        for r in rows:

            ax.axhline(
                r,
                color="#EAEAEA",
                linewidth=0.45,
                zorder=0,
            )

        # Zero has a substantive interpretation: no change relative to the
        # Gaussian experimental control. It is therefore darker than the
        # ordinary reference grid.
        ax.axvline(
            0.0,
            color="#555555",
            linewidth=0.70,
            zorder=1,
        )

        # Subtle vertical major grid behind the data.
        ax.grid(
            which="major",
            axis="x",
            color="#EAEAEA",
            linewidth=0.45,
            zorder=0,
        )

        ax.set_axisbelow(True)

        for i, (
            label,
            _,
            _,
            colour,
            marker,
        ) in enumerate(MODELS):

            y = rows

            ax.scatter(
                data[label],
                y,
                s=28,
                facecolor=colour,
                edgecolor="white",
                linewidth=0.70,
                marker=marker,
                zorder=3,
                label=label if ax is a1 else None,
            )

        ax.set_yticks(
            rows
        )

        ax.set_ylim(
            -0.50,
            len(UNIVERSES) - 0.50,
        )

        ax.set_xlabel(
            xlabel,
            labelpad=4,
        )

        ax.set_xlim(
            *padded_limits(data)
        )

        ax.tick_params(
            axis="both",
            which="both",
            pad=2,
        )

    # -----------------------------------------------------------------------
    # Panel (a): Sharpe differences
    # -----------------------------------------------------------------------

    a1.set_yticklabels(
        ticks
    )

    a1.text(
        0.025,
        0.965,
        "(a)",
        transform=a1.transAxes,
        fontsize=8.5,
        fontweight="bold",
        ha="left",
        va="top",
    )

    # -----------------------------------------------------------------------
    # Panel (b): certainty-equivalent differences
    # -----------------------------------------------------------------------

    # Universe labels are shown once, on the left panel only. The horizontal
    # guides make correspondence across the two panels immediate.
    a2.set_yticklabels(
        []
    )

    # Prevent Matplotlib from introducing an offset such as 1e-3 above the
    # axis. The differences are small enough that direct decimal labels are
    # easier to read.
    a2.xaxis.set_major_formatter(
        FuncFormatter(
            lambda v, _: f"{v:.3f}"
        )
    )

    a2.text(
        0.025,
        0.965,
        "(b)",
        transform=a2.transAxes,
        fontsize=8.5,
        fontweight="bold",
        ha="left",
        va="top",
    )

    # -----------------------------------------------------------------------
    # Shared legend
    # -----------------------------------------------------------------------

    handles, labels = (
        a1.get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(
            0.5,
            0.018,
        ),
        ncol=3,
        frameon=False,
        fontsize=7.0,
        handlelength=1.0,
        handletextpad=0.4,
        columnspacing=1.8,
        borderaxespad=0.0,
        scatterpoints=1,
    )

    # -----------------------------------------------------------------------
    # Layout
    # -----------------------------------------------------------------------

    fig.subplots_adjust(
        left=0.105,
        right=0.985,
        top=0.95,
        bottom=0.275,
        wspace=0.25,
    )

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------

    outdir = Path(
        args.outdir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out = (
        outdir
        / f"economic_differences.{args.format}"
    )

    fig.savefig(
        out,
        bbox_inches="tight",
        pad_inches=0.025,
        dpi=600,
    )

    plt.close(fig)

    print(
        f"\nsaved -> {out}"
    )


if __name__ == "__main__":
    main()