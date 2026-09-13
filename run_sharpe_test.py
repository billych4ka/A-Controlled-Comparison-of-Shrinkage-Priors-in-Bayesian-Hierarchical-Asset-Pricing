"""
run_sharpe_test.py

Ledoit and Wolf (2008) test for the difference between two Sharpe ratios,
applied to the horseshoe-family versus Gibbs-family comparisons of
Section 5.3. Reference:

    Ledoit, O. and Wolf, M. (2008). Robust performance hypothesis testing
    with the Sharpe ratio. Journal of Empirical Finance, 15(5), 850-859.
    doi:10.1016/j.jempfin.2008.03.002

The test is a studentised circular block bootstrap of

    Delta = SR_1 - SR_2,     SR_k = mu_k / sqrt(gamma_k - mu_k^2),

where gamma_k = E[r_k^2]. The two return series enter jointly, so the
bootstrap preserves both serial dependence (blocks) and the cross-strategy
dependence that arises because the four strategies trade the same assets
over the same months. The p-value is the two-sided bootstrap p-value of the
studentised statistic; annualisation cancels and does not affect it.

Implementation choices, stated so the dissertation can describe them:
  * Outer standard error: HAC with a Bartlett kernel (Newey and West 1987),
    the same estimator used for the Diebold-Mariano tests. Ledoit and Wolf
    use a prewhitened Parzen kernel; both are consistent and the bootstrap
    p-value is what is reported.
  * Bootstrap standard error: the natural block estimator of Ledoit and
    Wolf's Section 3.1, computed from the block sums of the resampled
    series.
  * Block length b = 5 by default, with a sensitivity sweep over b in
    {2, 5, 10, 20} because the data-driven selection of Ledoit and Wolf's
    Appendix is not implemented.

Portfolio returns are recomputed from the saved backtest forecasts through
src.evaluation.metrics.portfolio_returns, so the tested series are exactly
those behind Table 5.4.

Run from the project root:

    python3 run_sharpe_test.py            (B = 4999, b = 5)
    python3 run_sharpe_test.py --sweep    (also b in {2, 10, 20})

Output: printed table, and logs/sharpe_test.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluation.metrics import portfolio_returns, sharpe_ratio  # noqa: E402

UNIVERSES = ("size_bm_25", "size_op_25", "size_inv_25")

MODELS = {
    "baseline": ("baseline_gaussian", "rescaled_r2_0p05"),
    "lasso": ("bayesian_lasso", "rescaled_r2_0p05"),
    "horseshoe": ("horseshoe", "p0_23_r2_0p05"),
    "reg_horseshoe": ("regularised_horseshoe", "p0_23_r2_0p05"),
}

COMPARISONS = (
    ("horseshoe", "baseline"),
    ("reg_horseshoe", "baseline"),
    ("horseshoe", "lasso"),
    ("reg_horseshoe", "lasso"),
    ("lasso", "baseline"),
    ("horseshoe", "reg_horseshoe"),
)


def _sharpe_from_moments(mu: float, gamma: float) -> float:
    return mu / np.sqrt(gamma - mu * mu)


def _gradient(mu1, gamma1, mu2, gamma2) -> np.ndarray:
    """Gradient of Delta = f(mu1, mu2, gamma1, gamma2) w.r.t. (mu1, mu2, gamma1, gamma2)."""
    s1 = (gamma1 - mu1**2) ** 1.5
    s2 = (gamma2 - mu2**2) ** 1.5
    return np.array([
        gamma1 / s1,
        -gamma2 / s2,
        -0.5 * mu1 / s1,
        0.5 * mu2 / s2,
    ])


def _moment_series(r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
    """T x 4 matrix of (r1, r2, r1^2, r2^2)."""
    return np.column_stack([r1, r2, r1**2, r2**2])


def _hac_bartlett(y: np.ndarray, lag: int) -> np.ndarray:
    """Newey-West long-run covariance of the mean of a T x k series."""
    T = y.shape[0]
    yc = y - y.mean(axis=0)
    psi = yc.T @ yc / T
    for j in range(1, lag + 1):
        w = 1.0 - j / (lag + 1.0)
        g = yc[j:].T @ yc[:-j] / T
        psi += w * (g + g.T)
    return psi


def _block_psi(y: np.ndarray, b: int) -> np.ndarray:
    """
    Natural covariance estimator for a circular-block-bootstrap sample
    (Ledoit and Wolf 2008, Section 3.1): average outer product of the
    normalised block sums of the demeaned resampled series.
    """
    T = y.shape[0]
    l = T // b
    yc = (y - y.mean(axis=0))[: l * b]
    zeta = yc.reshape(l, b, -1).sum(axis=1) / np.sqrt(b)
    return zeta.T @ zeta / l


def _delta_and_se(y: np.ndarray, psi: np.ndarray) -> tuple[float, float]:
    T = y.shape[0]
    mu1, mu2, g1, g2 = y.mean(axis=0)
    delta = _sharpe_from_moments(mu1, g1) - _sharpe_from_moments(mu2, g2)
    grad = _gradient(mu1, g1, mu2, g2)
    se = np.sqrt(grad @ psi @ grad / T)
    return float(delta), float(se)


def ledoit_wolf_test(r1: np.ndarray, r2: np.ndarray, *, b: int = 5,
                     B: int = 4999, hac_lag: int | None = None,
                     seed: int = 0) -> dict:
    """
    Two-sided studentised circular block bootstrap test of SR_1 = SR_2.

    Returns dict with delta (monthly), delta_annual, se (HAC), t_stat,
    p_value, and the bootstrap settings.
    """
    r1 = np.asarray(r1, float)
    r2 = np.asarray(r2, float)
    if r1.shape != r2.shape or r1.ndim != 1:
        raise ValueError("r1 and r2 must be 1-D arrays of equal length")
    T = r1.shape[0]
    if hac_lag is None:
        hac_lag = int(np.floor(4 * (T / 100) ** (2 / 9)))

    y = _moment_series(r1, r2)
    delta_hat, se_hat = _delta_and_se(y, _hac_bartlett(y, hac_lag))
    t_hat = delta_hat / se_hat

    rng = np.random.default_rng(seed)
    l = int(np.ceil(T / b))
    idx_base = np.arange(b)
    t_star = np.empty(B)
    for i in range(B):
        starts = rng.integers(0, T, size=l)
        idx = ((starts[:, None] + idx_base[None, :]) % T).ravel()[:T]
        y_star = y[idx]
        d_star, se_star = _delta_and_se(y_star, _block_psi(y_star, b))
        t_star[i] = (d_star - delta_hat) / se_star

    p_value = (np.sum(np.abs(t_star) >= abs(t_hat)) + 1.0) / (B + 1.0)

    return {
        "delta": delta_hat,
        "delta_annual": delta_hat * np.sqrt(12),
        "se_annual": se_hat * np.sqrt(12),
        "t_stat": t_hat,
        "p_value": float(p_value),
        "block_length": b,
        "B": B,
        "hac_lag": hac_lag,
        "T": T,
    }


def load_returns(results_dir: Path, universe: str, key: str) -> np.ndarray:
    model, setting = MODELS[key]
    p = results_dir / universe / model / "backtest" / f"{model}_{setting}_backtest.npz"
    d = np.load(p)
    return portfolio_returns(d["realised"], d["predicted"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--B", type=int, default=4999)
    ap.add_argument("--block", type=int, default=5)
    ap.add_argument("--sweep", action="store_true",
                    help="also run b in {2, 10, 20}")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="logs/sharpe_test.csv")
    args = ap.parse_args()

    results_dir = Path(args.results)
    blocks = [args.block] + ([2, 10, 20] if args.sweep else [])
    blocks = sorted(set(blocks))

    rets: dict[str, dict[str, np.ndarray]] = {}
    for u in UNIVERSES:
        rets[u] = {k: load_returns(results_dir, u, k) for k in MODELS}
    rets["stacked"] = {k: np.mean([rets[u][k] for u in UNIVERSES], axis=0)
                       for k in MODELS}

    rows = []
    hdr = (f"{'universe':<12s} {'comparison':<28s} {'SR1':>6s} {'SR2':>6s} "
           f"{'dSR':>7s} {'se':>6s} {'t':>6s}" + "".join(f"{'p(b=%d)' % b:>9s}" for b in blocks))
    print(hdr)
    print("-" * len(hdr))

    for u in list(UNIVERSES) + ["stacked"]:
        for k1, k2 in COMPARISONS:
            r1, r2 = rets[u][k1], rets[u][k2]
            sr1, sr2 = sharpe_ratio(r1), sharpe_ratio(r2)
            res_by_b = {b: ledoit_wolf_test(r1, r2, b=b, B=args.B, seed=args.seed)
                        for b in blocks}
            base = res_by_b[args.block]
            line = (f"{u:<12s} {k1 + ' vs ' + k2:<28s} {sr1:6.3f} {sr2:6.3f} "
                    f"{base['delta_annual']:7.3f} {base['se_annual']:6.3f} "
                    f"{base['t_stat']:6.2f}"
                    + "".join(f"{res_by_b[b]['p_value']:9.3f}" for b in blocks))
            print(line)
            row = {"universe": u, "strategy_1": k1, "strategy_2": k2,
                   "sharpe_1": sr1, "sharpe_2": sr2,
                   "delta_annual": base["delta_annual"],
                   "se_annual": base["se_annual"], "t_stat": base["t_stat"],
                   "hac_lag": base["hac_lag"], "T": base["T"], "B": args.B}
            for b in blocks:
                row[f"p_b{b}"] = res_by_b[b]["p_value"]
            rows.append(row)
        print()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with out.open("w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(f"{r[k]:.6g}" if isinstance(r[k], float) else str(r[k])
                             for k in keys) + "\n")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
