"""
src/visualisation/plot_global_scale.py

Posterior distributions of the horseshoe global scale relative to the scale
used for prior calibration.

WHY THIS FIGURE. The results report the posterior mean of the global scale as
a ratio to its calibrated prior scale (roughly 0.035 to 0.053 for the plain
horseshoe and 0.076 to 0.091 for the regularised horseshoe) and note that
every posterior interval excludes the calibrated value. Those numerical
summaries do not show how concentrated the posterior is relative to the size
of the revision.

The figure therefore plots the posterior distribution of

    lambda_H / lambda_0,H

rather than lambda_H itself. This is a change of graphical scale only: every
posterior draw is divided by the calibrated value already used as the
comparison point for that universe. The calibrated value is consequently one
in every panel. This removes irrelevant variation in the raw calibration scale
between universes and makes the degree of posterior revision directly
comparable across the three portfolio sorts.

WHAT IS PLOTTED. Kernel density estimates of the pooled posterior draws of the
global scale divided by the corresponding calibrated scale, for the plain and
regularised horseshoe in each portfolio universe. The vertical reference line
at one denotes the calibrated scale.

A logarithmic horizontal axis is used because the comparison is inherently
multiplicative. Values below one indicate a posterior global scale tighter
than calibration; for example, 0.05 corresponds to a scale twenty times
smaller than the calibrated value.

Rows are portfolio universes so that the two horseshoe specifications can be
compared directly within each setting. The regularised horseshoe's consistently
larger global scale is the visual counterpart of the slab absorbing part of
the regularisation that the global scale must otherwise supply.

PRESENTATION. The figure follows the dissertation house style: clean
sans-serif typography with STIX sans mathematics, the common horseshoe colour
palette, thin dark-grey axes, subtle logarithmic reference lines, compact
spacing and a shared frameless legend.

Run from the project root:

    python3 -m src.visualisation.plot_global_scale
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from src.diagnostics.convergence import load_chains
from src.nuts.horseshoe import (
    HorseshoeDraws,
    horseshoe_hyperparameters,
)
from src.nuts.regularised_horseshoe import RegHorseshoeDraws


UNIVERSES = (
    ("Size$\\times$BM", "size_bm_25"),
    ("Size$\\times$OP", "size_op_25"),
    ("Size$\\times$Inv", "size_inv_25"),
)


MODELS = (
    (
        "Horseshoe",
        "horseshoe",
        "p0_23_r2_0p05",
        HorseshoeDraws,
        "#D05A3A",
        "-",
    ),
    (
        "Regularised horseshoe",
        "regularised_horseshoe",
        "p0_23_r2_0p05",
        RegHorseshoeDraws,
        "#3A9679",
        "-.",
    ),
)


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

    "axes.labelsize": 8.2,
    "axes.labelweight": "normal",
    "axes.linewidth": 0.65,
    "axes.edgecolor": "#333333",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.facecolor": "white",

    "xtick.labelsize": 7.4,
    "ytick.labelsize": 7.4,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.0,
    "xtick.minor.size": 1.6,
    "ytick.major.size": 0.0,
    "xtick.major.width": 0.55,
    "xtick.minor.width": 0.40,

    "lines.linewidth": 1.30,
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
        "--outdir",
        default="figures",
    )

    ap.add_argument(
        "--format",
        default="pdf",
        choices=["pdf", "png"],
    )

    ap.add_argument(
        "--bw",
        type=float,
        default=0.35,
        help="kernel bandwidth; larger is smoother",
    )

    args = ap.parse_args()

    draws = {}
    prior_scale = {}

    for _, universe in UNIVERSES:

        d = np.load(
            Path("data/processed")
            / f"{universe}_arrays.npz",
            allow_pickle=True,
        )

        F = d["F"]
        R = d["R"]

        prior_scale[universe] = horseshoe_hyperparameters(
            R,
            F,
            F.shape[2],
        ).tau_0

        for (
            label,
            model,
            setting,
            cls,
            _,
            _,
        ) in MODELS:

            try:
                ch = load_chains(
                    universe,
                    model,
                    setting,
                    cls,
                )

            except FileNotFoundError:

                print(
                    f"  ({model} / {universe} not found)"
                )

                continue

            g = np.concatenate(
                [
                    c.tau
                    for c in ch
                ]
            )

            draws[(universe, label)] = g

            lo, hi = np.percentile(
                g,
                [2.5, 97.5],
            )

            print(
                f"  {universe:<13s}"
                f" {label:<22s}"
                f" mean {g.mean():.4e}"
                f"  [{lo:.4e}, {hi:.4e}]"
                f"  ratio "
                f"{g.mean() / prior_scale[universe]:.4f}"
                f"  prior excluded: "
                f"{'yes' if not lo <= prior_scale[universe] <= hi else 'NO'}"
            )

    if not draws:
        raise SystemExit(
            "no chains found"
        )

    ratios = {}

    for (universe, label), g in draws.items():

        ratios[(universe, label)] = (
            g / prior_scale[universe]
        )

    all_ratios = np.concatenate(
        list(ratios.values())
    )

    lo_x = (
        np.floor(
            np.log10(
                all_ratios.min()
            )
        )
        - 0.05
    )

    hi_x = 0.08

    grid = np.linspace(
        lo_x,
        hi_x,
        500,
    )

    xgrid = 10 ** grid

    plt.rcParams.update(
        STYLE
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(6.5, 3.35),
        sharex=True,
    )

    axes = np.atleast_1d(
        axes
    )

    for panel_index, (
        ax,
        (universe_label, universe),
    ) in enumerate(
        zip(
            axes,
            UNIVERSES,
        )
    ):

        for (
            label,
            _,
            _,
            _,
            colour,
            linestyle,
        ) in MODELS:

            r = ratios.get(
                (
                    universe,
                    label,
                )
            )

            if r is None:
                continue

            kde = gaussian_kde(
                np.log10(r),
                bw_method=args.bw,
            )

            density = kde(
                grid
            )

            ax.plot(
                xgrid,
                density,
                color=colour,
                linestyle=linestyle,
                linewidth=1.30,
                label=(
                    label
                    if panel_index == 0
                    else None
                ),
                zorder=3,
            )

        ax.axvline(
            1.0,
            color="#555555",
            linewidth=0.80,
            linestyle=(0, (4, 2)),
            zorder=4,
        )

        ax.set_xscale(
            "log"
        )

        ax.set_xlim(
            10 ** lo_x,
            10 ** hi_x,
        )

        ax.set_ylim(
            bottom=0,
        )

        ax.set_yticks(
            []
        )

        ax.text(
            0.018,
            0.86,
            universe_label,
            transform=ax.transAxes,
            fontsize=8.2,
            color="#333333",
            ha="left",
            va="top",
        )

        ax.grid(
            which="major",
            axis="x",
            color="#EAEAEA",
            linewidth=0.45,
            zorder=0,
        )

        ax.set_axisbelow(
            True
        )

        ax.tick_params(
            axis="x",
            which="both",
            pad=2,
        )

    top = axes[0]

    top.annotate(
        "calibration",
        xy=(1.0, top.get_ylim()[1] * 0.68),
        xytext=(-6, 0),
        textcoords="offset points",
        fontsize=7.2,
        color="#555555",
        ha="right",
        va="center",
    )

    axes[1].set_ylabel(
        "Density",
        labelpad=7,
    )

    axes[-1].set_xlabel(
        "Global shrinkage scale (relative to calibration)",
        labelpad=4,
    )

    handles, labels = (
        top.get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(
            0.5,
            0.012,
        ),
        ncol=2,
        frameon=False,
        fontsize=7.0,
        handlelength=2.5,
        handletextpad=0.5,
        columnspacing=2.0,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(
        left=0.055,
        right=0.985,
        top=0.975,
        bottom=0.205,
        hspace=0.18,
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
        / f"global_scale_posteriors.{args.format}"
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
