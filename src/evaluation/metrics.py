"""
src/evaluation/metrics.py

Out-of-sample evaluation metrics, shared identically by all four models.

Every function here is a pure function of predicted and realised returns:
no model objects, no samplers, no priors. That is deliberate: the four models
must be judged by exactly the same yardstick, and the cleanest way to
guarantee that is for the yardstick not to know which model produced its
input. Same argument as sur.py enforcing one likelihood.

Three metrics, answering three different questions:
  - out-of-sample R^2 : does the model forecast the level of returns better
    than a naive benchmark? A brutal test at monthly frequency.
  - Sharpe ratio      : does trading on the forecasts make money per unit of
    risk? Forecasts can be nearly useless by R^2 and still profitable, since
    only the sign and relative ordering matter for a portfolio.
  - certainty equivalent : what guaranteed return would a mean-variance
    investor accept in place of the strategy? Penalises volatility explicitly
    rather than dividing by it, so a strategy is not flattered by being small.
"""

from __future__ import annotations

import numpy as np

MONTHS_PER_YEAR = 12


def predict(F: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Predicted excess returns r_hat[i,t] = f_{i,t}' b_i.

    Identical arithmetic to sur.py's sur_predicted_returns; repeated here
    only so that the evaluation module has no dependency on the likelihood
    module. F : (N,T,K), b : (N,K) -> (N,T).
    """
    return np.einsum("itk,ik->it", F, b)


def out_of_sample_r2(realised: np.ndarray, predicted: np.ndarray,
                     benchmark: np.ndarray) -> float:
    """
    Campbell & Thompson (2008) out-of-sample R^2:

        R2_oos = 1 - sum (r - r_hat)^2 / sum (r - r_bench)^2

    The benchmark is the HISTORICAL MEAN computed from data available before
    each forecast, not zero. This matters: predicting zero is a much weaker
    benchmark than predicting the running average, and measuring against zero
    would flatter every model. A negative value means the model forecasts
    worse than simply extrapolating the average return so far, which is the
    normal outcome for monthly return prediction and not by itself a sign of
    a bug.

    All three arrays are (N, T_oos), aligned on the same asset-month grid.
    """
    realised, predicted, benchmark = map(np.asarray, (realised, predicted, benchmark))
    if not (realised.shape == predicted.shape == benchmark.shape):
        raise ValueError(f"shape mismatch: {realised.shape}, {predicted.shape}, "
                         f"{benchmark.shape}")
    sse_model = np.sum((realised - predicted) ** 2)
    sse_bench = np.sum((realised - benchmark) ** 2)
    if sse_bench == 0:
        return np.nan
    return float(1.0 - sse_model / sse_bench)


def r2_by_asset(realised: np.ndarray, predicted: np.ndarray,
                benchmark: np.ndarray) -> np.ndarray:
    """
    The same statistic computed separately for each asset, (N,).

    Pooled R^2 is dominated by the most volatile assets, so a model can post
    a respectable pooled figure while failing on most of the cross-section.
    Reporting the per-asset distribution alongside the pooled number guards
    against that.
    """
    sse_m = np.sum((realised - predicted) ** 2, axis=1)
    sse_b = np.sum((realised - benchmark) ** 2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sse_b > 0, 1.0 - sse_m / sse_b, np.nan)


def proportional_weights(predicted: np.ndarray) -> np.ndarray:
    """
    Zero-cost long-short weights proportional to demeaned predictions.

        w[i,t] = (r_hat[i,t] - mean_i r_hat[i,t]) / sum_i |demeaned|

    Two properties, both deliberate. Demeaning makes the weights sum to zero,
    so the portfolio is self-financing and its return reflects the
    CROSS-SECTIONAL ranking rather than a bet on the market's overall
    direction. Dividing by total absolute weight fixes gross exposure at 1,
    so returns across models and dates are comparable and no model can look
    better merely by taking larger positions.

    predicted : (N, T) -> weights (N, T). A period with identical predictions
    for every asset gets zero weights, which is the correct response to a
    forecast that expresses no view.
    """
    demeaned = predicted - predicted.mean(axis=0, keepdims=True)
    gross = np.abs(demeaned).sum(axis=0, keepdims=True)
    scale = np.abs(predicted).sum(axis=0, keepdims=True)
    return np.divide(demeaned, gross, out=np.zeros_like(demeaned),
                     where=gross > 1e-12 * np.maximum(scale, 1e-300))


def portfolio_returns(realised: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """
    Realised return of the proportional long-short portfolio, (T,).
    Weights are formed from the forecast for period t and applied to the
    realised return of period t; the forecast uses only information dated
    t-1 or earlier, which is enforced upstream by the backtest loop.
    """
    return (proportional_weights(predicted) * realised).sum(axis=0)


def sharpe_ratio(returns: np.ndarray, annualise: bool = True) -> float:
    """
    Mean divided by standard deviation of the portfolio return series.

    No risk-free subtraction: the inputs are already excess returns and the
    portfolio is zero-cost, so its return is an excess return by
    construction. Annualisation multiplies by sqrt(12), the usual convention
    for monthly data, which assumes serial independence, an approximation
    worth stating but standard.
    """
    returns = np.asarray(returns, dtype=float)
    sd = returns.std(ddof=1)
    scale = np.abs(returns).max()
    if not np.isfinite(sd) or sd <= 1e-12 * scale:
        return np.nan
    sr = returns.mean() / sd
    return float(sr * np.sqrt(MONTHS_PER_YEAR) if annualise else sr)


def certainty_equivalent(returns: np.ndarray, risk_aversion: float = 3.0,
                         annualise: bool = True) -> float:
    """
    Certainty-equivalent return for a mean-variance investor:

        CE = mean - (gamma / 2) * variance

    The guaranteed return the investor would swap the strategy for. Unlike
    the Sharpe ratio this penalises variance additively rather than dividing
    by volatility, so it does not flatter a strategy simply for being small,
    and it is expressed in return units that can be read directly.

    risk_aversion = 3 is the conventional default in this literature
    (e.g. Campbell & Thompson 2008; Rapach, Strauss & Zhou 2010).
    """
    returns = np.asarray(returns, dtype=float)
    ce = returns.mean() - 0.5 * risk_aversion * returns.var(ddof=1)
    return float(ce * MONTHS_PER_YEAR if annualise else ce)


def expanding_mean_benchmark(R: np.ndarray, start: int) -> np.ndarray:
    """
    Historical-mean benchmark, (N, T - start): for each asset and each
    forecast date t >= start, the mean of that asset's returns over periods
    0 .. t-1.

    Strictly point-in-time: the forecast for period t uses returns up to
    t-1 only, never t itself. Computed by cumulative sum rather than a loop
    so it is O(T) and obviously free of off-by-one drift.
    """
    csum = np.cumsum(R, axis=1)
    counts = np.arange(1, R.shape[1] + 1, dtype=float)
    running = csum / counts
    return running[:, start - 1:-1]


def _newey_west_variance(d: np.ndarray, lags: int | None = None) -> float:
    """
    Long-run variance of a series, corrected for autocorrelation
    (Newey & West 1987).

    A plain variance would understate the sampling error of the mean whenever
    the series is serially correlated, and forecast-error differences are,
    because the same coefficient vector is reused for every month between
    refits, so consecutive errors share an estimation error. Ignoring that
    would inflate the test statistic and manufacture significance.

    Default lag length floor(4 (T/100)^(2/9)) is the standard rule of thumb.
    Bartlett weights (1 - k/(lags+1)) guarantee a non-negative estimate.
    """
    d = np.asarray(d, dtype=float)
    T = len(d)
    if lags is None:
        lags = int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))
    dc = d - d.mean()
    gamma0 = float(dc @ dc / T)
    total = gamma0
    for k in range(1, min(lags, T - 1) + 1):
        gamma_k = float(dc[k:] @ dc[:-k] / T)
        total += 2.0 * (1.0 - k / (lags + 1.0)) * gamma_k
    return max(total, 0.0)


def diebold_mariano(realised: np.ndarray, predicted_a: np.ndarray,
                    predicted_b: np.ndarray, lags: int | None = None) -> dict:
    """
    Diebold & Mariano (1995) test of equal forecast accuracy between two
    models, under squared-error loss.

    The loss differential per period is

        d_t = mean_i (r_it - a_it)^2 - mean_i (r_it - b_it)^2

    averaged across assets so the test is over the T out-of-sample months,
    which is the dimension along which the observations are (approximately)
    independent. Assets within a month are strongly correlated, so treating
    N*T asset-months as independent would badly overstate the sample size.

    A NEGATIVE statistic means model A has lower loss, i.e. A forecasts
    better. The statistic is asymptotically standard normal, so |DM| > 1.96
    is significant at 5%.

    Caveat worth stating in the write-up: the classical DM test assumes the
    two forecasts are non-nested and does not account for parameter
    estimation error. Comparing a shrinkage model against the historical
    mean is a nested comparison, where DM is known to be undersized and
    Clark & West (2007) is the appropriate correction. Comparing two
    different shrinkage priors against each other is non-nested, which is
    the case DM is designed for, so this function is the right tool for
    model-versus-model, and clark_west below for model-versus-benchmark.
    """
    realised, predicted_a, predicted_b = map(np.asarray,
                                             (realised, predicted_a, predicted_b))
    if not (realised.shape == predicted_a.shape == predicted_b.shape):
        raise ValueError("realised and both forecast arrays must have the same shape")

    loss_a = ((realised - predicted_a) ** 2).mean(axis=0)
    loss_b = ((realised - predicted_b) ** 2).mean(axis=0)
    d = loss_a - loss_b
    T = len(d)

    lrv = _newey_west_variance(d, lags)
    if lrv <= 0:
        return {"statistic": np.nan, "p_value": np.nan, "mean_loss_diff": float(d.mean()),
                "n_periods": T, "better": None}

    stat = float(d.mean() / np.sqrt(lrv / T))
    from scipy.stats import norm as _norm
    p = float(2.0 * (1.0 - _norm.cdf(abs(stat))))
    return {"statistic": stat, "p_value": p, "mean_loss_diff": float(d.mean()),
            "n_periods": T, "better": ("A" if stat < 0 else "B") if p < 0.05 else None}


def clark_west(realised: np.ndarray, predicted: np.ndarray,
               benchmark: np.ndarray, lags: int | None = None) -> dict:
    """
    Clark & West (2007) test for NESTED forecast comparison, the right test
    for "does the model beat the historical mean?".

    Why a different test is needed: the benchmark is nested inside the model
    (set all slope coefficients to zero and the model becomes the historical
    mean). Under the null that the extra predictors are useless, the larger
    model still estimates them, and that estimation noise inflates its
    squared error. So the model is expected to LOSE on mean squared error
    even when the null is true, and a plain DM test would systematically
    fail to detect genuine predictability.

    Clark & West adjust the loss differential by the term that estimation
    noise contributes:

        f_t = (r - bench)^2 - [ (r - model)^2 - (bench - model)^2 ]

    and test whether mean(f) > 0. One-sided by construction: the alternative
    is that the model helps.
    """
    realised, predicted, benchmark = map(np.asarray, (realised, predicted, benchmark))
    if not (realised.shape == predicted.shape == benchmark.shape):
        raise ValueError("all three arrays must have the same shape")

    e_bench = (realised - benchmark) ** 2
    e_model = (realised - predicted) ** 2
    adj = (benchmark - predicted) ** 2
    f = (e_bench - (e_model - adj)).mean(axis=0)
    T = len(f)

    lrv = _newey_west_variance(f, lags)
    if lrv <= 0:
        return {"statistic": np.nan, "p_value": np.nan, "mean_f": float(f.mean()),
                "n_periods": T}

    stat = float(f.mean() / np.sqrt(lrv / T))
    from scipy.stats import norm as _norm
    p = float(1.0 - _norm.cdf(stat))
    return {"statistic": stat, "p_value": p, "mean_f": float(f.mean()), "n_periods": T}
