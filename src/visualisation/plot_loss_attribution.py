"""
src/visualisation/plot_loss_attribution.py

Figure companion to Table 5.7: attribution of out-of-sample forecast loss
between the common component b_bar and the deviations theta across the
four Gaussian calibrations. Values are typed from Table 5.7 so the figure
cannot disagree with the table; the assert below checks the components sum
to the reported -R2_OOS.

Panel (a): share of benchmark-relative loss attributable to b_bar and
theta (cross term omitted, as in the table's share columns).
Panel (b): the same decomposition in absolute terms, stacked, so that the
Feng-He row is visible as a different regime rather than only a different
split.

Run from the project root:

    python3 -m src.visualisation.plot_loss_attribution
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROWS = [
    ("Feng–He",              0.1820, 0.0472, 0.1776, -0.0033, 56.8, 43.2, -0.3817),
    (r"$V_{\rm prior}=0.10$", 0.0228, 0.0370, 0.0006,  0.0004, 98.3,  1.7, -0.0611),
    (r"$V_{\rm prior}=0.05$", 0.0175, 0.0312, 0.0003,  0.0003, 99.0,  1.0, -0.0493),
    (r"$V_{\rm prior}=0.01$", 0.0133, 0.0179, 0.0000,  0.0001, 99.7,  0.3, -0.0313),
]

COL_BBAR = "#4A4A4A"
COL_THETA = "#D05A3A"

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "stixsans",
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.linewidth": 0.60,
    "axes.edgecolor": "#333333",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 2.8,
    "ytick.major.size": 2.8,
    "xtick.major.width": 0.50,
    "ytick.major.width": 0.50,
    "legend.frameon": False,
    "legend.fontsize": 7.5,
    "figure.facecolor": "white",
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--format", default="pdf", choices=["pdf", "png"])
    args = ap.parse_args()

    labels = [r[0] for r in ROWS]
    bbar = np.array([r[1] + r[2] for r in ROWS])
    theta = np.array([r[3] + r[4] for r in ROWS])
    share_bbar = np.array([r[5] for r in ROWS])
    share_theta = np.array([r[6] for r in ROWS])
    r2 = np.array([r[7] for r in ROWS])

    for i in range(1, 4):
        assert abs((bbar[i] + theta[i]) - (-r2[i])) < 1.5e-3, labels[i]

    plt.rcParams.update(STYLE)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(5.9, 2.05),
                                     gridspec_kw={"width_ratios": [1.0, 1.0]})
    y = np.arange(len(ROWS))[::-1]
    h = 0.50

    ax_a.barh(y, share_bbar, height=h, color=COL_BBAR, label=r"Common component $\bar b$", zorder=3)
    ax_a.barh(y, share_theta, left=share_bbar, height=h, color=COL_THETA,
              label=r"Deviations $\theta$", zorder=3)
    for yi, sb, st in zip(y, share_bbar, share_theta):
        ax_a.text(101.5, yi, f"{st:.1f}%", va="center", ha="left", fontsize=7.3,
                  color=COL_THETA)
    ax_a.set_xlim(0, 116)
    ax_a.tick_params(axis="y", length=0)
    ax_a.set_xticks([0, 25, 50, 75, 100])
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(labels)
    ax_a.set_xlabel("Share of benchmark-relative loss (%)")
    ax_a.text(0.0, 1.03, "(a)", transform=ax_a.transAxes, fontsize=8.0, color="#333333")
    ax_a.grid(axis="x", color="#E6E6E6", linewidth=0.45, zorder=0)

    ax_b.barh(y, bbar, height=h, color=COL_BBAR, zorder=3)
    ax_b.barh(y, theta, left=bbar, height=h, color=COL_THETA, zorder=3)
    for k, (yi, val) in enumerate(zip(y, r2)):
        ax_b.text(bbar[k] + max(theta[k], 0) + 0.006, yi,
                  rf"$R^2_{{\rm OOS}}={val:.4f}$", va="center", ha="left",
                  fontsize=6.9, color="#555555")
    ax_b.set_xlim(0, 0.56)
    ax_b.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax_b.set_yticks(y)
    ax_b.set_yticklabels([])
    ax_b.tick_params(axis="y", length=0)
    ax_b.spines["left"].set_visible(False)
    ax_b.set_xlabel(r"Loss contribution (units of $-R^2_{\rm OOS}$)")
    ax_b.text(0.0, 1.03, "(b)", transform=ax_b.transAxes, fontsize=8.0, color="#333333")
    ax_b.grid(axis="x", color="#E6E6E6", linewidth=0.45, zorder=0)

    fig.legend(*ax_a.get_legend_handles_labels(), loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, 0.0), handlelength=1.6, columnspacing=2.0)
    fig.subplots_adjust(left=0.145, right=0.985, top=0.91, bottom=0.31, wspace=0.07)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"loss_attribution.{args.format}"
    fig.savefig(p, dpi=300 if args.format == "png" else None)
    print(f"saved {p}")


if __name__ == "__main__":
    main()
