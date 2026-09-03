"""
src/visualisation/plot_trace.py

Trace plots for the four models, following the convergence display of Feng and
He (2020, Appendix C).

WHY THIS FORM. The parameter vector is far too large to display in full: B
alone has NK = 3,600 elements. Feng and He address this by plotting one
randomly selected element of each key parameter block and reporting that all
others behave similarly. The same approach is taken here, with two additions.
First, the element is selected with a FIXED seed and its index is printed, so
the choice is reproducible and cannot be suspected of having been made after
inspecting the chains. Second, all chains are overplotted rather than one, so
between-chain agreement is visible alongside within-chain stationarity -- the
property that rank-normalised split-R-hat quantifies and that a single-chain
trace cannot show.

Trace plots are illustrative, not evidential. Stationarity and between-chain
agreement are established quantitatively by the R-hat and effective sample
size figures reported in the results; these panels show what those statistics
describe. The accompanying caption should say so, since a reader could
otherwise take a visually clean trace as the evidence itself.

LAYOUT. Rows are models and columns are parameter blocks: an asset-specific
coefficient, a common coefficient, a residual-covariance element, and the
model's global deviation-prior parameter. The final column differs across
models by construction: the Gaussian baseline displays an element of the
deviation covariance, while the three shrinkage models display their global
scale parameter.

PRESENTATION. This figure follows the dissertation's common visual style:
clean sans-serif typography with STIX sans mathematics, thin dark-grey axes,
restrained colours and explicit spacing. Chain colours are used only to make
independently sampled chains distinguishable; they do not encode substantive
model categories and therefore receive no legend. Trace lines are deliberately
thin and semi-transparent so that overlap between chains remains visible.

Run from the project root:

    python3 -m src.visualisation.plot_trace

or:

    python3 -m src.visualisation.plot_trace --universe size_op_25 --seed 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.diagnostics.convergence import load_chains
from src.gibbs.baseline_gaussian import GibbsDraws
from src.gibbs.bayesian_lasso import LassoDraws
from src.nuts.horseshoe import HorseshoeDraws
from src.nuts.regularised_horseshoe import RegHorseshoeDraws


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

# (
#     row label,
#     directory,
#     setting tag,
#     dataclass,
#     global-parameter field,
#     global-parameter label,
# )
MODELS = (
    (
        "Gaussian baseline",
        "baseline_gaussian",
        "rescaled_r2_0p05",
        GibbsDraws,
        "Delta_b_diag",
        r"$\Delta_{\theta,jj}$",
    ),
    (
        "Bayesian LASSO",
        "bayesian_lasso",
        "rescaled_r2_0p05",
        LassoDraws,
        "lam",
        r"$\lambda_L$",
    ),
    (
        "Horseshoe",
        "horseshoe",
        "p0_23_r2_0p05",
        HorseshoeDraws,
        "tau",
        r"$\lambda_H$",
    ),
    (
        "Regularised horseshoe",
        "regularised_horseshoe",
        "p0_23_r2_0p05",
        RegHorseshoeDraws,
        "tau",
        r"$\lambda_H$",
    ),
)


# ---------------------------------------------------------------------------
# Chain colours
# ---------------------------------------------------------------------------

# These colours are deliberately restrained. They distinguish chains without
# competing with the substantive four-model palette used in the main results
# figures.
CHAIN_COLOURS = (
    "#4A4A4A",
    "#3B75AF",
    "#D05A3A",
    "#3A9679",
    "#7A6F9B",
    "#9A9A9A",
    "#B86F91",
    "#6E9FBF",
)


# ---------------------------------------------------------------------------
# Dissertation house style
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
    "font.size": 7.6,

    # Axes
    "axes.labelsize": 7.8,
    "axes.labelweight": "normal",
    "axes.linewidth": 0.55,
    "axes.edgecolor": "#333333",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": "white",

    # Ticks
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.50,
    "ytick.major.width": 0.50,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.pad": 2.0,
    "ytick.major.pad": 2.0,

    # Lines
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",

    # Figure
    "figure.facecolor": "white",

    # Vector export
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    # Figure export
    "savefig.facecolor": "white",
    "savefig.transparent": False,
}


def main() -> None:

    # -----------------------------------------------------------------------
    # Arguments
    # -----------------------------------------------------------------------

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--universe",
        default="size_bm_25",
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "fixed seed for element selection, so the choice is reproducible "
            "and independent of the chains"
        ),
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

    # -----------------------------------------------------------------------
    # Load chains
    # -----------------------------------------------------------------------

    rng = np.random.default_rng(
        args.seed
    )

    loaded = {}

    for (
        label,
        model,
        setting,
        cls,
        gfield,
        glabel,
    ) in MODELS:

        try:

            loaded[label] = (
                load_chains(
                    args.universe,
                    model,
                    setting,
                    cls,
                ),
                gfield,
                glabel,
            )

        except FileNotFoundError:

            print(
                f"  ({model} not found, row omitted)"
            )

    if not loaded:
        raise SystemExit(
            "no chains found"
        )

    # -----------------------------------------------------------------------
    # Select representative elements
    # -----------------------------------------------------------------------

    # Element indices are drawn once and then shared across models, so the
    # same coefficient and covariance entry are displayed in every row.
    first = (
        next(
            iter(
                loaded.values()
            )
        )[0][0]
    )

    N = first.B.shape[1]
    K = first.B.shape[2]

    i_a = int(
        rng.integers(N)
    )

    j_a = int(
        rng.integers(K)
    )

    j_b = int(
        rng.integers(K)
    )

    i_s = int(
        rng.integers(N)
    )

    j_s = int(
        rng.integers(N)
    )

    print(
        f"  elements shown (seed {args.seed}): "
        f"b[{i_a},{j_a}], "
        f"b_bar[{j_b}], "
        f"Sigma[{i_s},{j_s}]"
    )

    # -----------------------------------------------------------------------
    # Figure setup
    # -----------------------------------------------------------------------

    plt.rcParams.update(
        STYLE
    )

    n_rows = len(
        loaded
    )

    fig, axes = plt.subplots(
        n_rows,
        4,
        figsize=(
            6.5,
            1.05 * n_rows + 0.65,
        ),
    )

    axes = np.atleast_2d(
        axes
    )

    fig.patch.set_facecolor(
        "white"
    )

    # Shared column headings.
    col_titles = (
        rf"$b_{{{i_a},{j_a}}}$",
        rf"$\bar b_{{{j_b}}}$",
        rf"$\Sigma_{{{i_s},{j_s}}}$",
        "Global parameter",
    )

    # -----------------------------------------------------------------------
    # Trace panels
    # -----------------------------------------------------------------------

    for r, (
        label,
        (
            chains,
            gfield,
            glabel,
        ),
    ) in enumerate(
        loaded.items()
    ):

        series = [
            [
                c.B[:, i_a, j_a]
                for c in chains
            ],
            [
                c.b_bar[:, j_b]
                for c in chains
            ],
            [
                c.Sigma[:, i_s, j_s]
                for c in chains
            ],
            None,
        ]

        # The global parameter is scalar for the three shrinkage models and
        # vector-valued for the Gaussian baseline, whose deviation covariance
        # has one entry per predictor.
        g = [
            getattr(
                c,
                gfield,
            )
            for c in chains
        ]

        if g[0].ndim == 2:

            series[3] = [
                x[:, j_b]
                for x in g
            ]

        else:

            series[3] = g

        # -------------------------------------------------------------------
        # Columns
        # -------------------------------------------------------------------

        for col in range(4):

            ax = axes[
                r,
                col,
            ]

            # Overplot every chain.
            n_chains = len(
                series[col]
            )

            chain_alpha = (
                0.72
                if n_chains <= 4
                else 0.48
            )

            for k, s in enumerate(
                    series[col]
            ):
                ax.plot(
                    s,
                    color=CHAIN_COLOURS[
                        k % len(CHAIN_COLOURS)
                        ],
                    linewidth=0.42,
                    alpha=chain_alpha,
                    zorder=2,
                )

            ax.set_xlim(
                0,
                len(
                    series[col][0]
                ),
            )

            # Scientific notation only where Matplotlib judges it necessary
            # under the stated limits.
            ax.ticklabel_format(
                axis="y",
                style="sci",
                scilimits=(-2, 3),
                useMathText=True,
            )

            ax.yaxis.get_offset_text().set_fontsize(
                5.8
            )

            # Column titles appear only once, on the first row.
            if r == 0:

                ax.set_title(
                    col_titles[col],
                    fontsize=7.8,
                    fontweight="normal",
                    pad=5,
                )

            # The final column has a different parameter by model. Keep the
            # shared column heading and place the model-specific notation
            # quietly inside each panel.
            if col == 3:

                ax.text(
                    0.97,
                    0.90,
                    glabel,
                    transform=ax.transAxes,
                    fontsize=6.8,
                    color="#555555",
                    ha="right",
                    va="top",
                )

            # Only the final row needs draw labels and x tick labels.
            if r == n_rows - 1:

                ax.set_xlabel(
                    "Draw",
                    labelpad=2,
                )

            else:

                ax.tick_params(
                    axis="x",
                    labelbottom=False,
                )

            # Model names identify rows.
            if col == 0:

                ax.set_ylabel(
                    label,
                    fontsize=7.5,
                    labelpad=5,
                )

            # Trace plots are already visually dense. Grids add little and
            # would create a distracting 4 x 4 lattice.
            ax.grid(
                False
            )

            ax.tick_params(
                axis="both",
                which="both",
                pad=2,
            )

    # -----------------------------------------------------------------------
    # Layout
    # -----------------------------------------------------------------------

    fig.subplots_adjust(
        left=0.105,
        right=0.99,
        top=0.92,
        bottom=0.12,
        wspace=0.28,
        hspace=0.30,
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
        / f"trace_plots_{args.universe}.{args.format}"
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