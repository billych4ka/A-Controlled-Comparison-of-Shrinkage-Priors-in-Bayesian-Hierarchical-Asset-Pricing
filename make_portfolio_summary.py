"""
make_portfolio_summary.py

Generates Table 3.1 (tab:portfolio-summary): descriptive statistics for the
25 portfolios in each universe, over the modelling sample.

The table reports the MINIMUM, MEDIAN and MAXIMUM across the 25 portfolios
rather than one row per portfolio. Seventy-five rows would be unreadable, and
the cross-sectional spread is the thing Section 3.1 actually claims: the
min-to-max range IS the evidence for "substantial cross-sectional variation in
return behaviour". If that range turns out to be narrow, the claim in the text
needs softening -- the script prints the ratios so this can be checked.

Statistics reported, all computed on excess returns over the modelling sample
(January 1966 to November 2025, T = 719):

    mean        monthly mean excess return, per cent
    sd          monthly standard deviation, per cent
    Sharpe      annualised, mean/sd * sqrt(12); no risk-free subtraction
                because R already holds excess returns
    AR(1)       first-order autocorrelation of monthly excess returns

Run from the project root:

    python3 make_portfolio_summary.py
    python3 make_portfolio_summary.py --outfile tables/portfolio_summary.tex
    python3 make_portfolio_summary.py --no-ar1     # drop the AR(1) row
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

UNIVERSES = [
    ("size_bm_25", r"Size$\times$BM"),
    ("size_op_25", r"Size$\times$OP"),
    ("size_inv_25", r"Size$\times$Inv"),
]


def load_returns(universe: str) -> tuple[np.ndarray, str, str]:
    """Return the (N, T) excess-return matrix and the sample end points."""
    path = Path("data/processed") / f"{universe}_arrays.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Available: "
            f"{sorted(p.name for p in Path('data/processed').glob('*_arrays.npz'))}"
        )
    R = np.load(path, allow_pickle=True)["R"]

    start = end = None
    meta = Path("data/processed") / f"{universe}_metadata.json"
    if meta.exists():
        try:
            rng = json.load(open(meta)).get("date_range")
            if rng:
                start, end = rng[0][:7], rng[1][:7]
        except (json.JSONDecodeError, OSError):
            pass
    return R, start, end


def ar1(x: np.ndarray) -> float:
    """First-order autocorrelation of a single series."""
    x = x - x.mean()
    denom = float(x @ x)
    return float(x[:-1] @ x[1:] / denom) if denom > 0 else float("nan")


def statistics(R: np.ndarray) -> dict[str, np.ndarray]:
    """Per-portfolio statistics. R is (N, T) monthly excess returns."""
    mean = R.mean(axis=1) * 100.0
    sd = R.std(axis=1, ddof=1) * 100.0
    sharpe = (R.mean(axis=1) / R.std(axis=1, ddof=1)) * np.sqrt(12.0)
    return {
        "Mean (\\% p.m.)": mean,
        "SD (\\% p.m.)": sd,
        "Sharpe (annualised)": sharpe,
        "AR(1)": np.array([ar1(R[i]) for i in range(R.shape[0])]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outfile", default="portfolio_summary.tex")
    ap.add_argument("--no-ar1", action="store_true",
                    help="omit the first-order autocorrelation row")
    ap.add_argument("--decimals", type=int, default=2)
    args = ap.parse_args()

    rows = ["Mean (\\% p.m.)", "SD (\\% p.m.)", "Sharpe (annualised)"]
    if not args.no_ar1:
        rows.append("AR(1)")

    stats, spans = {}, []
    for universe, _ in UNIVERSES:
        R, start, end = load_returns(universe)
        N, T = R.shape
        stats[universe] = statistics(R)
        spans.append((universe, N, T, start, end))
        print(f"{universe}: N={N} T={T} sample {start} to {end}")

    # ---- the claim being checked -----------------------------------------
    print("\nCross-sectional spread across the 25 portfolios "
          "(max/min, for the 'substantial variation' claim in 3.1):")
    for universe, label in UNIVERSES:
        s = stats[universe]
        m, sd = s["Mean (\\% p.m.)"], s["SD (\\% p.m.)"]
        print(f"  {universe:<12s} mean {m.min():6.3f} to {m.max():6.3f} "
              f"({m.max() / m.min():4.1f}x)   "
              f"sd {sd.min():5.2f} to {sd.max():5.2f} "
              f"({sd.max() / sd.min():4.1f}x)")
    print("  If these ratios are close to one, soften the sentence in 3.1.\n")

    # ---- build the table --------------------------------------------------
    d = args.decimals
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Descriptive statistics for the three portfolio universes}",
        r"\label{tab:portfolio-summary}",
        r"\begin{tabular}{l" + "ccc" * len(UNIVERSES) + "}",
        r"\toprule",
        " & " + " & ".join(
            rf"\multicolumn{{3}}{{c}}{{{label}}}" for _, label in UNIVERSES
        ) + r" \\",
        " ".join(rf"\cmidrule(lr){{{2 + 3 * k}-{4 + 3 * k}}}"
                 for k in range(len(UNIVERSES))),
        " & " + " & ".join(["Min", "Median", "Max"] * len(UNIVERSES)) + r" \\",
        r"\midrule",
    ]

    for row in rows:
        cells = []
        for universe, _ in UNIVERSES:
            v = stats[universe][row]
            cells += [f"${v.min():.{d}f}$", f"${np.median(v):.{d}f}$",
                      f"${v.max():.{d}f}$"]
        lines.append(f"{row} & " + " & ".join(cells) + r" \\")

    _, N, T, start, end = spans[0]
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\begin{flushleft}",
        r"\footnotesize",
        rf"\textit{{Note:}} Each universe contains $N={N}$ value-weighted "
        r"portfolios. Statistics are computed on monthly excess returns over "
        rf"the modelling sample ({start} to {end}, $T={T}$), and summarised "
        r"by their minimum, median and maximum across the portfolios within "
        r"each universe. Sharpe ratios are annualised by $\sqrt{12}$; no "
        r"risk-free subtraction is applied because the returns are already "
        r"in excess of the one-month rate.",
        r"\end{flushleft}",
        r"\end{table}",
    ]

    out = Path(args.outfile)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()