# tests/

These scripts reproduce the verification results reported in Section 4.5 and
Appendix A.6 of the dissertation. They are not unit tests. Each one compares
the implementation against a quantity whose answer is known in advance --- an
analytic conditional, a closed-form prior moment, a brute-force reconstruction
of an algebraic shortcut, or a nominal coverage rate --- and reports the
agreement rather than asserting a boolean. Several take minutes to hours,
because the quantity being checked is only available by sampling. Each script's
own docstring records the results it produced, so a run that disagrees with the
docstring is the signal to investigate.

## Script to dissertation section

| script | dissertation section | approximate runtime |
|---|---|---|
| `check_lasso_conditionals.py` | Appendix A.6.1, first paragraph; Table 4.5, row "Analytic conditionals"; supports Section 4.4 | seconds |
| `check_blocked_sur_kronecker.py` | Appendix A.6.1, second paragraph; Section 4.4, the blocked (B, b_bar) update | seconds |
| `check_lasso_blocked_sur.py` | Appendix A.6.1, second paragraph; A.6.2, first paragraph; Table 4.5, rows "Analytic conditionals" and "Distributional moments" | under a minute (60,000 draws) |
| `check_lasso_composition.py` | Appendix A.6.2, second paragraph; A.6.5, first paragraph; Table 4.5, row "Joint composition" | minutes |
| `check_iw_potential.py` | Appendix A.6.3; Section 4.4, the custom differentiable inverse-Wishart density; Section 4.5 | minutes |
| `check_calibration_baseline_gaussian.py` | Table 4.6, Gaussian baseline column; Appendix A.6.4, first paragraph; Section 4.5 | minutes |
| `check_calibration_bayesian_lasso.py` | Table 4.6, Bayesian LASSO column; Appendix A.6.4; Section 4.5; A.6.5, second example | about 3 minutes |
| `check_calibration_horseshoe.py` | Table 4.6, horseshoe column and the three excluded datasets; Appendix A.6.4; A.6.7 via `--full-scale`; Section 4.5 | about 101 minutes |
| `check_calibration_regularised_horseshoe.py` | Table 4.6, regularised horseshoe column; Section 4.5 | 30-40 minutes |
| `check_prior_predictive_horseshoe.py` | Section 4.3, the m_eff = 30.04 discrepancy against the nominal p_0 = 23; Appendix A.6.7, second paragraph; supports Section 4.2.4 | a few seconds |
| `check_horseshoe_prior_tails.py` | Appendix A.6.7, why full-dimensional calibration was abandoned | seconds |
| `check_sparsity_recovery_horseshoe.py` | Section 5.6 and Table 5.10; Appendix A.9, the simulation design; A.6.5, third example | about 1.8 hours (5.3 min per fit, 10 sparse + 10 dense) |
| `check_scale_transportability.py` | Section 4.3, Tables 4.3 and 4.4; Appendix A.7 | about a minute |
| `check_forecast_tests.py` | Appendix A.6.6, size and power of the forecast comparison tests; Section 4.6; validates `src/evaluation/metrics.py` | seconds |
| `check_eq_17_collapse.py` | the disclosed correction to Feng & He's eq. (17) | seconds |
| `check_rhat_ess.py` | Appendix A.5 (script section 7) and Appendix A.6.2 (script section 6); Section 4.5; verifies `src/diagnostics/convergence.py` | seconds |
| `check_lasso_chain_length.py` | Section 4.4, whether lambda_L's R-hat reflects disagreement or short chains | seconds (reads saved chains) |
| `check_sigma_truncation.py` | Section 5.5, the covariance block's truncation series | seconds (reads saved chains) |
| `check_divergences_horseshoe.py` | the `target_accept` choice behind the production NUTS settings | 8, 12 and 20+ minutes per setting |
| `check_backtest_budget.py` | the per-window budget for the horseshoe backtest | 1-2 hours for the default grid |
| `check_reg_backtest_budget.py` | the per-window budget for the regularised horseshoe backtest | 21.1 minutes per full-window fit |
| `check_refit_schedule.py` | Appendix A.2, the refit-schedule check; Section 4.6 | about 20 minutes (60-month only); about 70 minutes with `--schedules 24 60` |
| `check_gross_exposure.py` | Section 5.3, whether the pooled and stacked cross-sort portfolios are comparable on leverage | seconds |

Runtimes are those recorded in each script's own docstring where it records
one. The rest are order-of-magnitude only: they are fast scripts that do no
sampling, but no measured figure is on record for them.

## Filenames are load-bearing

Appendix A of the dissertation carries a table mapping each appendix
subsection to its script, so these filenames are cited in the write-up and
must not drift. If a name here and a name in the appendix disagree, RENAME THE
FILE --- do not edit the table.

    check_lasso_conditionals.py
    check_blocked_sur_kronecker.py
    check_lasso_blocked_sur.py
    check_lasso_composition.py
    check_iw_potential.py
    check_calibration_baseline_gaussian.py
    check_calibration_bayesian_lasso.py
    check_calibration_horseshoe.py
    check_calibration_regularised_horseshoe.py
    check_forecast_tests.py
    check_scale_transportability.py
    check_sparsity_recovery_horseshoe.py
    check_prior_predictive_horseshoe.py
    check_rhat_ess.py
    check_refit_schedule.py

The four calibration scripts are cited collectively as `check_calibration_*.py`
and must keep that prefix exactly. `check_rhat_ess.py` and
`check_refit_schedule.py` were added to this list once they too were cited
in Appendix A (A.5/A.6.2 and A.2 respectively); the same no-drift rule
applies to them. The regularised horseshoe's script was
briefly named `check_calibration_reg_horseshoe.py`, which the wildcard still
matched but which a reader scanning for the regularised horseshoe would not
recognise; abbreviations in this group are what the prefix rule exists to
prevent.

## Running these scripts

The scripts import from `src/` using absolute imports rooted at the project
root, so they must be run from the project root, and several were written to
sit there. To run one, move it to the project root first:

    mv tests/check_lasso_composition.py .
    python3 check_lasso_composition.py

Do not run them from inside `tests/` --- the imports will fail.

Note that four scripts load data with a path relative to `tests/`
(`../data/processed`): `check_scale_transportability.py`,
`check_backtest_budget.py`, `check_reg_backtest_budget.py` and
`check_divergences_horseshoe.py`. Moving those to the project root as above
requires changing that path to `data/processed`, or they will not find the
dataset.

## What is not a script here

Appendix A.6.8's look-ahead corruption test has no standalone script. It is
implemented as `assert_no_lookahead` in `src/evaluation/backtest.py` and
invoked from the backtest runners themselves --- `run_backtest_lasso.py`,
`run_backtest_horseshoe.py` and `run_backtest_regularised_horseshoe.py`. It
runs as part of the backtest rather than separately, which is the right place
for it, but note that `run_backtest_gaussian.py` does NOT invoke it: three of
the four runners do.

## Where the output lives

`logs/` holds the recorded console output of the runs that are too slow to
repeat routinely --- the horseshoe and regularised horseshoe calibrations, the
sparsity recovery run, the divergence scans, and the abandoned
production-dimension designs. `results/` holds the production and backtest logs
alongside the saved draws, one folder per universe and model, with a README in
each universe folder recording what was run.
