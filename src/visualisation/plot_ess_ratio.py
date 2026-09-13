""""
src/visualisation/plot_ess_ratio.py

Empirical distributions of the effective sample size ratio across all monitored
parameters, for the four models.

WHY THIS FIGURE. The results report the minimum and median effective sample
size for each parameter block, which is the right numerical summary but does
not reveal how broadly low-ESS behaviour is distributed across parameters.
A single low minimum may reflect one isolated element, whereas a gradual lower
tail indicates a more general mixing limitation.

The figure therefore plots the empirical cumulative distribution function
(ECDF) of the bulk effective sample size ratio

    ESS_bulk / retained posterior draws

for every monitored element in each parameter block.

The ratio rather than the raw count is plotted because the four models use
different numbers of retained draws. A ratio of one corresponds to the
effective information in approximately the same number of independent draws.

HOW TO READ THE FIGURE. At any horizontal value x, the vertical position gives
the proportion of monitored parameters whose bulk ESS ratio is less than or
equal to x. Curves that rise sharply near zero therefore indicate that a large
share of the corresponding parameter block mixes slowly. Curves concentrated
toward one indicate substantially more efficient sampling.

A vertical dashed line at 0.6 marks the threshold reported by Feng and He
(2020, Figure 5), who report all monitored elements exceeding this value. The
comparison is contextual rather than a like-for-like benchmark because their
model, sampler and retained draw counts differ from those used here.

A light reference line at one marks the effective-sample-size ratio associated
with the same amount of information as an equal number of independent draws.

PRESENTATION. Panels correspond to models and curves to parameter blocks. ECDFs
are used instead of overlapping histograms because they avoid arbitrary binning,
place all parameter blocks on the same interpretable vertical scale, and make
the lower tail directly readable. The figure follows the dissertation house
style: sans-serif typography with STIX sans mathematics, thin dark-grey axes,
restrained colours, subtle horizontal guides and a shared frameless legend.

Computation is O(n_params x n_draws log n_draws) and B has 3,600 elements per
model, so expect a minute or two.

Run from the project root:

    python3 -m src.visualisation.plot_ess_ratio

or:

    python3 -m src.visualisation.plot_ess_ratio --universe size_op_25
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.diagnostics.convergence import diagnose, load_chains
from src.gibbs.baseline_gaussian import GibbsDraws
from src.gibbs.bayesian_lasso import LassoDraws
from src.nuts.horseshoe import HorseshoeDraws
from src.nuts.regularised_horseshoe import RegHorseshoeDraws


MODELS = (
    (
        "Gaussian baseline",
        "baseline_gaussian",
        "rescaled_r2_0p05",
        GibbsDraws,
    ),
    (
        "Bayesian LASSO",
        "bayesian_lasso",
        "rescaled_r2_0p05",
        LassoDraws,
    ),
    (
        "Horseshoe",
        "horseshoe",
        "p0_23_r2_0p05",
        HorseshoeDraws,
    ),
    (
        "Regularised horseshoe",
        "regularised_horseshoe",
        "p0_23_r2_0p05",
        RegHorseshoeDraws,
    ),
)


BLOCKS = (
    (
        "B",
        r"$b_{ij}$",
        "#4A4A4A",
        "-",
    ),
    (
        "b_bar",
        r"$\bar b_j$",
        "#3B75AF",
        "--",
    ),
    (
        "Sigma",
        r"$\Sigma$",
        "#D05A3A",
        "-.",
    ),
)


FENG_HE_THRESHOLD = 0.6


STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Arial",
        "Helvetica",
        "Liberation Sans",
        "DejaVu Sans",
    ],
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

    "lines.linewidth": 1.20,
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


def main() -> None:

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--universe",
        default="size_bm_25",
    )

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

    ratios = {}

    for (
        label,
        model,
        setting,
        cls,
    ) in MODELS:

        try:

            chains = load_chains(
                args.universe,
                model,
                setting,
                cls,
            )

        except FileNotFoundError:

            print(
                f"  ({model} not found, panel omitted)"
            )

            continue

        per_block = {}

        for (
            field,
            flabel,
            colour,
            linestyle,
        ) in BLOCKS:

            if not hasattr(
                chains[0],
                field,
            ):
                continue

            d = diagnose(
                chains,
                field,
            )

            n_total = (
                d.n_draws
                * d.n_chains
            )

            r = (
                np.asarray(
                    d.ess_bulk,
                    dtype=float,
                )
                / n_total
            )

            r = r[
                np.isfinite(r)
            ]

            per_block[
                flabel
            ] = (
                r,
                colour,
                linestyle,
            )

            print(
                f"  {label:<24s}"
                f" {flabel:<10s}"
                f" n={len(r):>5d}"
                f"  median {np.nanmedian(r):.3f}"
                f"  min {np.nanmin(r):.3f}"
                f"  below {FENG_HE_THRESHOLD}: "
                f"{100 * np.nanmean(r < FENG_HE_THRESHOLD):.1f}%"
            )

        ratios[
            label
        ] = per_block

    if not ratios:
        raise SystemExit(
            "no chains found"
        )

    plt.rcParams.update(
        STYLE
    )

    n = len(
        ratios
    )

    fig, axes = plt.subplots(
        1,
        n,
        figsize=(6.5, 2.05),
        sharey=True,
    )

    axes = np.atleast_1d(
        axes
    )

    fig.patch.set_facecolor(
        "white"
    )

    x_max = 1.25

    for panel_index, (
        ax,
        (
            label,
            blocks,
        ),
    ) in enumerate(
        zip(
            axes,
            ratios.items(),
        )
    ):

        for (
            flabel,
            (
                r,
                colour,
                linestyle,
            ),
        ) in blocks.items():

            x = np.sort(
                r
            )

            y = (
                np.arange(
                    1,
                    len(x) + 1,
                )
                / len(x)
            )

            ax.step(
                x,
                y,
                where="post",
                color=colour,
                linestyle=linestyle,
                linewidth=1.20,
                label=flabel,
                zorder=3,
            )

        ax.axvline(
            FENG_HE_THRESHOLD,
            color="#777777",
            linewidth=0.75,
            linestyle=(0, (3, 2)),
            zorder=2,
        )

        ax.axvline(
            1.0,
            color="#D5D5D5",
            linewidth=0.65,
            linestyle="-",
            zorder=1,
        )

        ax.set_xlim(
            0,
            x_max,
        )

        ax.set_ylim(
            0,
            1.02,
        )

        ax.set_title(
            label,
            fontsize=7.8,
            fontweight="normal",
            pad=5,
        )

        ax.grid(
            which="major",
            axis="y",
            color="#EAEAEA",
            linewidth=0.45,
            zorder=0,
        )

        ax.set_axisbelow(
            True
        )

        ax.tick_params(
            axis="both",
            which="both",
            pad=2,
        )

        if panel_index > 0:

            ax.tick_params(
                axis="y",
                labelleft=False,
            )

    axes[0].set_ylabel(
        "Cumulative proportion",
        labelpad=5,
    )

    fig.supxlabel(
        "Bulk ESS / retained posterior draws",
        fontsize=8.0,
        y=0.105,
    )

    handles, legend_labels = (
        axes[0].get_legend_handles_labels()
    )

    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(
            0.5,
            0.005,
        ),
        ncol=3,
        frameon=False,
        fontsize=7.0,
        handlelength=2.4,
        handletextpad=0.5,
        columnspacing=1.8,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.99,
        top=0.85,
        bottom=0.27,
        wspace=0.20,
    )

    outdir = Path(
        args.outdir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out = (
        outdir
        / f"ess_ratio_{args.universe}.{args.format}"
    )

    fig.savefig(
        out,
        bbox_inches="tight",
        pad_inches=0.025,
        dpi=600,
    )

    plt.close(
        fig
    )

    print(
        f"\nsaved -> {out}"
    )


if __name__ == "__main__":
    main()
