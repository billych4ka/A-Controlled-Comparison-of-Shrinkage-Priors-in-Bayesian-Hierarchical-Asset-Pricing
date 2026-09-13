"""
src/visualisation/plot_shrinkage_factors.py

The shrinkage factor kappa for all four deviation priors.

WHY THE AXIS IS 1 - kappa. Every model concentrates near kappa = 1, because
the deviations are heavily shrunk in all four specifications. Plotted on kappa
directly, all four collapse into a narrow region at the right-hand edge and the
differences between the shrinkage mechanisms are difficult to see. The
informative quantity is 1 - kappa, the share of the coefficient left free by
the prior, which spans several orders of magnitude and separates the models
cleanly.

TWO PANELS.

  (a) The density of log10(1 - kappa), displayed on the corresponding
      1 - kappa scale. This shows that the priors differ in the SHAPE of their
      shrinkage distribution and not only in its average level. The
      horseshoe's mode sits further left than the Bayesian LASSO's (it pools
      the bulk of coefficients more strongly) while its right tail extends
      much further. The short vertical marks on the horizontal axis give each
      model's least-shrunk coefficient. Without them panel (a) reads as "the
      horseshoe shrinks hardest", because out where the four floors lie the
      densities are visually indistinguishable from zero.

  (b) The upper tail, as the percentage of coefficients exceeding a given
      level of 1 - kappa. This is where the substantive comparison lies,
      because the curves CROSS: the horseshoe leaves fewer coefficients
      moderately free than the Bayesian LASSO while leaving a small number
      very much freer. The vertical drop terminating each curve marks that
      model's least-shrunk coefficient.

WHAT IS PLOTTED. One kappa per asset-predictor coefficient, evaluated at the
POSTERIOR MEAN of the model's deviation-prior variance:

    kappa_ij = 1 / (1 + T sigma^-2 v_ij q_j^2)

with q_j the root mean square of predictor j and v_ij the deviation-prior
variance implied by the model. The same construction is used for all four
models, which is necessary for comparability: the Bayesian LASSO retains
posterior means of its local scales together with full traces for only a fixed
subset of coefficients, so a draw-level summary is not available uniformly.

q_j is the root mean SQUARE, not the standard deviation. The intercept column
has mean one and standard deviation exactly zero, so using the standard
deviation would assign it kappa = 1 by construction rather than by inference.

PRESENTATION. The figure is designed for a standard 6.5-inch dissertation or
journal text block. The visual treatment is deliberately restrained: clean
sans-serif typography with STIX sans mathematics, thin axes, subtle grid lines,
compact panel spacing, and a shared legend. Line styles differ as well as colours so the figure remains readable
in monochrome. The secondary kappa axis is deliberately omitted because it is
a deterministic transformation of 1 - kappa and adds substantial visual
clutter without adding information.

Run from the project root:

    python3 -m src.visualisation.plot_shrinkage_factors

or, for another universe:

    python3 -m src.visualisation.plot_shrinkage_factors --universe size_op_25
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, NullFormatter
from scipy.stats import gaussian_kde

from src.diagnostics.convergence import load_chains
from src.gibbs.baseline_gaussian import GibbsDraws
from src.gibbs.bayesian_lasso import LassoDraws
from src.nuts.horseshoe import HorseshoeDraws, horseshoe_hyperparameters
from src.nuts.regularised_horseshoe import RegHorseshoeDraws

MODELS = (
    (
        "Gaussian baseline",
        "baseline_gaussian",
        "rescaled_r2_0p05",
        GibbsDraws,
        "#4A4A4A",
        "-",
        1.30,
    ),
    (
        "Bayesian LASSO",
        "bayesian_lasso",
        "rescaled_r2_0p05",
        LassoDraws,
        "#3B75AF",
        "--",
        1.30,
    ),
    (
        "Horseshoe",
        "horseshoe",
        "p0_23_r2_0p05",
        HorseshoeDraws,
        "#D05A3A",
        "-",
        1.30,
    ),
    (
        "Regularised horseshoe",
        "regularised_horseshoe",
        "p0_23_r2_0p05",
        RegHorseshoeDraws,
        "#3A9679",
        "-.",
        1.30,
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
    "axes.xmargin": 0.0,
    "axes.facecolor": "white",
    "xtick.labelsize": 7.4,
    "ytick.labelsize": 7.4,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.minor.size": 1.6,
    "ytick.minor.size": 1.6,
    "xtick.major.width": 0.55,
    "ytick.major.width": 0.55,
    "xtick.minor.width": 0.40,
    "ytick.minor.width": 0.40,
    "xtick.major.pad": 2.5,
    "ytick.major.pad": 2.5,
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


def kappa_from_variance(
    v: np.ndarray,
    F: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    kappa = 1/(1 + T sigma^-2 v q_j^2), with q_j the root mean square of
    predictor j. v is (N, K) or broadcastable to it.
    """
    N, T, K = F.shape

    q = np.sqrt((F**2).mean(axis=1))

    return 1.0 / (1.0 + (T / sigma**2) * v * q**2)


def deviation_prior_variance(
    label: str,
    chains,
    N: int,
) -> np.ndarray:
    """
    Posterior-mean deviation-prior variance v_ij, shape (N, K).

    The four models differ ONLY in this quantity, which is what makes kappa
    a common axis on which they can be compared.
    """

    if label == "Gaussian baseline":

        d = np.mean(
            [c.Delta_b_diag.mean(axis=0) for c in chains],
            axis=0,
        )

        return np.tile(d, (N, 1))

    if label == "Bayesian LASSO":

        s = float(chains[0].meta["s"])

        return s**2 * np.mean(
            [c.tau2_mean for c in chains],
            axis=0,
        )

    if label == "Horseshoe":

        tau = np.mean([c.tau.mean() for c in chains])

        lam = np.mean(
            [c.lam_local.mean(axis=0) for c in chains],
            axis=0,
        )

        return (tau * lam) ** 2

    if label == "Regularised horseshoe":

        tau = np.mean([c.tau.mean() for c in chains])

        lam = np.mean(
            [c.lam_tilde.mean(axis=0) for c in chains],
            axis=0,
        )

        return (tau * lam) ** 2

    raise ValueError(label)


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

    ap.add_argument(
        "--bw",
        type=float,
        default=0.30,
        help="kernel bandwidth for panel (a); larger is smoother",
    )

    args = ap.parse_args()

    d = np.load(
        Path("data/processed") / f"{args.universe}_arrays.npz",
        allow_pickle=True,
    )

    F = d["F"]
    R = d["R"]

    N, T, K = F.shape

    sigma = horseshoe_hyperparameters(
        R,
        F,
        K,
    ).sigma_pooled

    kappas = {}
    styles = {}

    for (
        label,
        model,
        setting,
        cls,
        colour,
        ls,
        lw,
    ) in MODELS:

        try:

            ch = load_chains(
                args.universe,
                model,
                setting,
                cls,
            )

        except FileNotFoundError:

            print(f"  ({model} not found, omitted)")

            continue

        k = kappa_from_variance(
            deviation_prior_variance(
                label,
                ch,
                N,
            ),
            F,
            sigma,
        ).ravel()

        kappas[label] = k

        styles[label] = (
            colour,
            ls,
            lw,
        )

        print(
            f"  {label:<24s}"
            f" mean {k.mean():.4f}"
            f"  p1 {np.percentile(k, 1):.4f}"
            f"  p5 {np.percentile(k, 5):.4f}"
            f"  min {k.min():.4f}"
            f"  max {k.max():.4f}"
            f"  K_free {(1 - k.mean()) * K:.2f}"
        )

    if not kappas:
        raise SystemExit("no chains found")

    plt.rcParams.update(STYLE)

    fig, (a1, a2) = plt.subplots(
        1,
        2,
        figsize=(6.5, 2.35),
    )

    fig.patch.set_facecolor("white")

    lo = float(np.floor(np.log10(min(1 - k.max() for k in kappas.values()))))

    hi = -0.05

    grid = np.linspace(
        lo,
        hi,
        500,
    )

    for lab, k in kappas.items():

        c, ls, lw = styles[lab]

        density = gaussian_kde(
            np.log10(1 - k),
            bw_method=args.bw,
        )(grid)

        a1.plot(
            grid,
            density,
            color=c,
            linestyle=ls,
            linewidth=lw,
            label=lab,
            zorder=3,
        )

    a1.set_xlim(
        lo,
        hi,
    )

    a1.set_ylim(
        0,
        None,
    )

    decades = list(
        range(
            int(lo),
            0,
        )
    )

    a1.set_xticks(decades)

    a1.set_xticklabels([rf"$10^{{{e}}}$" for e in decades])

    a1.set_xlabel(
        r"$1-\kappa_{ij}$",
        labelpad=3,
    )

    a1.set_ylabel(
        "Density",
        labelpad=4,
    )

    rug = 0.075 * a1.get_ylim()[1]

    for lab, k in kappas.items():

        c, _, _ = styles[lab]

        a1.vlines(
            np.log10(1 - k.min()),
            -0.35 * rug,
            0.75 * rug,
            color=c,
            linewidth=1.2,
            clip_on=False,
            zorder=4,
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

    t = np.logspace(
        lo,
        0,
        500,
    )

    for lab, k in kappas.items():

        c, ls, lw = styles[lab]

        tail = [((1 - k) >= x).mean() * 100 for x in t]

        a2.plot(
            t,
            tail,
            color=c,
            linestyle=ls,
            linewidth=lw,
            label=lab,
            zorder=3,
        )

    a2.set_xscale("log")

    a2.set_yscale("log")

    a2.set_xlim(
        10**lo,
        1,
    )

    n_coef = len(next(iter(kappas.values())))

    a2.set_ylim(
        100 / n_coef * 0.6,
        320,
    )

    a2.set_xlabel(
        r"$1-\kappa_{ij}$",
        labelpad=3,
    )

    a2.set_ylabel(
        "Coefficients exceeding\nthreshold (%)",
        labelpad=4,
        linespacing=1.5,
    )

    a2.set_yticks([0.01, 0.1, 1, 10, 100])

    a2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

    a2.yaxis.set_minor_formatter(NullFormatter())

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

    for ax in (a1, a2):

        ax.grid(
            which="major",
            axis="y",
            color="#EAEAEA",
            linewidth=0.45,
            zorder=0,
        )

        ax.set_axisbelow(True)

        ax.tick_params(
            axis="both",
            which="both",
            pad=2,
        )

    handles, labels = a1.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(
            0.5,
            0.018,
        ),
        ncol=4,
        frameon=False,
        fontsize=7.0,
        handlelength=2.5,
        handletextpad=0.5,
        columnspacing=1.45,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(
        left=0.085,
        right=0.985,
        top=0.95,
        bottom=0.255,
        wspace=0.30,
    )

    outdir = Path(args.outdir)

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out = outdir / (f"kappa_densities_" f"{args.universe}." f"{args.format}")

    fig.savefig(
        out,
        bbox_inches="tight",
        pad_inches=0.025,
        dpi=600,
    )

    plt.close(fig)

    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
