"""
src/visualisation/plot_predictor_correlation.py

Figure for Section 3.2: correlation structure of the 144-column design matrix.

WHAT THE TWO PANELS DO. Panel (a) is the correlation matrix, which shows the
block structure: common predictors, asset-level predictors, and the
interactions that inherit from both. Panel (b) is the point of the figure: the
off-diagonal correlations sorted by magnitude, on a log axis. The design is
nearly orthogonal, with a median absolute off-diagonal correlation of about
0.05, but sixteen pairs sit above 0.9 and are separated from the bulk by a
visible gap. Those sixteen are the near-duplicate pairs described in the text:
the 12-month rolling mean and 12-1 momentum, which share eleven months of
return history, together with the fifteen counterpart pairs generated when
each common predictor is interacted with both of them.

Panel (b) exists because the heatmap alone cannot carry the claim. Thirty-two
affected columns out of 144 is 32 cells out of 20,736, which is invisible at
print size. The sorted plot makes the count readable and shows that the
redundancy is concentrated rather than diffuse, which matters for the
sparsity argument, since a sparse procedure here faces a small number of
genuine near-ties rather than a general collinearity problem.

WHY POOLED. The correlation matrix is computed on the design pooled across
assets, F.reshape(-1, K), which is the same object as C in the prior-implied
variance ratio of Section 4.3.

Run from the project root:

    python3 -m src.visualisation.plot_predictor_correlation
    python3 -m src.visualisation.plot_predictor_correlation --universe size_op_25
    python3 -m src.visualisation.plot_predictor_correlation --format png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "stixsans",
    "font.size": 8.0,
    "axes.labelsize": 8.2,
    "axes.linewidth": 0.65,
    "axes.edgecolor": "#333333",
    "axes.facecolor": "white",
    "xtick.labelsize": 7.4,
    "ytick.labelsize": 7.4,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.major.width": 0.55,
    "ytick.major.width": 0.55,
    "legend.frameon": False,
    "legend.fontsize": 7.0,
    "figure.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
    "savefig.transparent": False,
}

OVERLAP_PAIR = ("roll_mean12", "mom_12_1")

ACCENT = "#D05A3A"
GREY = "#4A4A4A"


def load_universe(universe: str) -> np.ndarray:
    path = Path("data/processed") / f"{universe}_arrays.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Available: "
            f"{sorted(p.name for p in Path('data/processed').glob('*_arrays.npz'))}"
        )
    return np.load(path, allow_pickle=True)["F"]


def predictor_columns(universe: str, K: int) -> list[str]:
    for name in (f"{universe}_metadata.json", f"{universe}.json"):
        path = Path("data/processed") / name
        if path.exists():
            try:
                cols = json.load(open(path)).get("predictor_columns")
                if cols and len(cols) == K:
                    return list(cols)
            except (json.JSONDecodeError, OSError):
                pass
    return [f"predictor {j}" for j in range(K)]


def block_edges(cols: list[str]) -> list[tuple[str, int, int]]:
    """
    (label, start, stop) per block. The intercept occupies column 0 and is not
    labelled: one column out of 144 cannot be seen, and a label for it collides
    with the common-predictor label. It is noted in the caption instead.
    """
    first_interaction = next(
        (j for j, c in enumerate(cols) if "_x_" in c), len(cols)
    )
    return [
        ("Common", 1, 16),
        ("Asset", 16, first_interaction),
        ("Interactions", first_interaction, len(cols)),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="size_bm_25")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--format", default="pdf", choices=["pdf", "png"])
    ap.add_argument("--threshold", type=float, default=0.90)
    args = ap.parse_args()

    F = load_universe(args.universe)
    N, T, K = F.shape
    cols = predictor_columns(args.universe, K)
    print(f"{args.universe}: N={N} T={T} K={K}")

    X = F.reshape(-1, K)
    keep = X.std(axis=0) > 0
    C = np.full((K, K), np.nan)
    C[np.ix_(keep, keep)] = np.corrcoef(X[:, keep].T)

    lower = [c.lower() for c in cols]
    n_pairs = None
    try:
        a, b = lower.index(OVERLAP_PAIR[0]), lower.index(OVERLAP_PAIR[1])
        print(f"\n  corr({cols[a]}, {cols[b]}) = {C[a, b]:.3f}")
        pairs = []
        for j, cj in enumerate(lower):
            if not cj.endswith("_x_" + OVERLAP_PAIR[0]):
                continue
            stem = cj[: -len("_x_" + OVERLAP_PAIR[0])]
            k = next((m for m, cm in enumerate(lower)
                      if cm == stem + "_x_" + OVERLAP_PAIR[1]), None)
            if k is not None:
                pairs.append((cols[j], cols[k], C[j, k]))
        if pairs:
            strongest = max(pairs, key=lambda p: abs(p[2]))
            n_pairs = len(pairs) + 1
            print(f"  {len(pairs)} interaction counterparts; strongest "
                  f"|corr| = {abs(strongest[2]):.3f} "
                  f"({strongest[0]}, {strongest[1]})")
            print(f"  near-duplicate pairs including the parent: {n_pairs}, "
                  f"spanning {2 * n_pairs} of {K} columns")
    except ValueError:
        print(f"\n  could not locate {OVERLAP_PAIR} in the recorded column "
              f"names; adjust OVERLAP_PAIR at the top of this script")

    iu = np.triu_indices(K, k=1)
    r = np.abs(C[iu])
    r = r[np.isfinite(r)]
    n_above = int((r > args.threshold).sum())
    print(f"  pairs with |corr| > {args.threshold}: {n_above}")
    print(f"  median |corr| off-diagonal: {np.median(r):.3f}")
    print(f"  99th percentile: {np.percentile(r, 99):.3f}\n")

    plt.rcParams.update(STYLE)
    fig, (a1, a2) = plt.subplots(
        1, 2, figsize=(6.5, 2.9),
        gridspec_kw={"width_ratios": [1.0, 1.15], "wspace": 0.32},
    )
    fig.patch.set_facecolor("white")

    im = a1.imshow(
        C, cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0),
        interpolation="nearest", origin="upper",
    )
    edges = block_edges(cols)
    for _, start, stop in edges:
        for pos in (start, stop):
            a1.axhline(pos - 0.5, color="#333333", linewidth=0.5)
            a1.axvline(pos - 0.5, color="#333333", linewidth=0.5)
    a1.set_xticks([])
    a1.set_yticks([])
    a1.tick_params(length=0)
    a1.set_xlim(-0.5, K - 0.5)
    a1.set_ylim(K - 0.5, -0.5)
    a1.set_title("(a) Correlation matrix", fontsize=8.2, pad=6)

    cb = fig.colorbar(im, ax=a1, fraction=0.046, pad=0.03)
    cb.outline.set_linewidth(0.5)
    cb.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    cb.ax.tick_params(length=2.0, width=0.5)

    s = np.sort(r)[::-1]
    rank = np.arange(1, s.size + 1)
    a2.plot(rank, s, color=GREY, linewidth=1.0, zorder=2)
    if n_above:
        a2.plot(rank[:n_above], s[:n_above], color=ACCENT, linewidth=1.6,
                zorder=4)
        a2.scatter(rank[:n_above], s[:n_above], s=7, color=ACCENT,
                   zorder=5, linewidths=0)
    a2.axhline(args.threshold, color=ACCENT, linewidth=0.6, linestyle=":",
               zorder=1)
    a2.set_xscale("log")
    a2.set_xlim(1, s.size)
    a2.set_ylim(0, 1.0)
    a2.set_xlabel("Rank of pair", labelpad=3)
    a2.set_ylabel(r"$|\mathrm{corr}|$", labelpad=4)
    a2.set_title("(b) Off-diagonal correlations, sorted",
                 fontsize=8.2, pad=6)
    a2.spines["top"].set_visible(False)
    a2.spines["right"].set_visible(False)
    a2.grid(axis="y", color="#DDDDDD", linewidth=0.5, zorder=0)
    a2.set_axisbelow(True)

    if n_above:
        a2.annotate(
            f"{n_above} pairs above {args.threshold:g}",
            xy=(rank[n_above // 2], s[n_above // 2]),
            xytext=(0.45, 0.55), textcoords="axes fraction",
            fontsize=7.0, color=ACCENT,
            arrowprops=dict(arrowstyle="-", color=ACCENT, linewidth=0.55,
                            shrinkA=1, shrinkB=2),
        )
    a2.annotate(
        f"median {np.median(r):.3f}",
        xy=(0.97, np.median(r)), xycoords=("axes fraction", "data"),
        xytext=(0, 5), textcoords="offset points",
        fontsize=7.0, color=GREY, ha="right", va="bottom",
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"predictor_correlation_{args.universe}.{args.format}"
    fig.savefig(out, dpi=400, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
