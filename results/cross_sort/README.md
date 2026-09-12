# results/cross_sort

Pooled cross-sort portfolio. Not a universe and not a fitted model -- a
re-allocation of forecasts that already exist.

`cross_sort_results.json` is written by `run_cross_sort_portfolio.py`, which
reads the saved backtest predictions from all three 25-portfolio universes
(`size_bm_25`, `size_op_25`, `size_inv_25`) and combines them into one
long-short allocation over 75 assets and 479 forecast months. NOTHING IS
REFITTED: same posterior means, same forecast months, same refit dates as the
per-universe backtests. Only the portfolio rule changes, so any difference is
attributable to the allocation, not to the models. See Section 5.3 / 6.3.

Two pooling rules are recorded per model:

  `stacked_*`  three within-sort market-neutral sleeves at equal weight.
               Capital cannot move between sorts. Reference point only.
  `pooled_*`   demeaned across all 75 predictions each month, so a sort short
               of attractive positions can cede capital to another. This is
               the headline construction.

| model | pooled Sharpe | stacked Sharpe | pooled CE | pooled R^2 |
|---|---|---|---|---|
| Gaussian baseline | 0.151 | 0.190 | 0.00352 | -0.0369 |
| Bayesian LASSO | 0.179 | 0.228 | 0.00457 | -0.0375 |
| Horseshoe | 0.241 | 0.319 | 0.00748 | -0.0354 |
| Regularised horseshoe | 0.238 | 0.309 | 0.00696 | -0.0358 |

All four: n_assets = 75, n_months = 479.

The pooled R^2 is negative for every model, the baseline included. That is
the usual out-of-sample result for monthly return prediction and is not a
symptom of the pooling; the economic ordering, not the R^2, is what this
table is for. Regenerate with `python3 run_cross_sort_portfolio.py`.
