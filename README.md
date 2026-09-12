# A Controlled Comparison of Shrinkage Priors in Bayesian Hierarchical Asset PricingBayesian Hierarchical Shrinkage in Asset Pricing

Code accompanying the MSc dissertation *A Controlled Comparison of Shrinkage
Priors in Bayesian Hierarchical Asset Pricing* (MSc Statistical Science,
University of Oxford, 2026).

Four continuous shrinkage priors are placed on the asset-specific coefficient
deviations of a Bayesian hierarchical SUR model, with the likelihood, the
common-component prior and the residual-covariance specification held fixed,
so that differences in fitted behaviour are attributable to the deviation
prior alone.

| Model                 | Deviation Prior                     | Sampler                                |
|-----------------------|-------------------------------------|----------------------------------------|
| Gaussian baseline     | Gaussian covariance, no local layer | blocked Gibbs                          |
| Bayesian LASSO        | normal–exponential scale mixture    | blocked Gibbs, collapsed global update |
| Horseshoe             | half-Cauchy local scales            | NUTS, non-centred                      |
| Regularised horseshoe | half-Cauchy with finite slab        | NUTS, non-centred                      |

---

## Environment

```
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the exact environment every reported result was
produced in, including the full transitive closure. The pins are exact rather
than floors: MCMC output is bit-stable only within a fixed PyTensor/NumPy
pair, and the diagnostics are written against ArviZ 1.2.0 specifically.

All NUTS sampling runs with `cores=1`. Parallel execution caused repeated
worker deaths during development.

---

## Layout

```
src/
  data_pipeline/      returns, predictor construction, standardisation
  likelihood/sur.py   SUR likelihood, NumPy and PyTensor, shared by all models
  gibbs/              Gaussian baseline and Bayesian LASSO
  nuts/               horseshoe and regularised horseshoe
  evaluation/         expanding-window backtest, forecasts, portfolios
  diagnostics/        rank-normalised split-R-hat and effective sample size
  simulate/           generative process for recovery and calibration tests
  visualisation/      figures
tests/                calibration, prior-predictive and pipeline checks
logs/                 output of verification runs
results/              per-universe, per-model sampling and backtest output
```

`run_<model>.py`, `diagnose_<model>.py` and `run_backtest_<model>.py` at the
project root are the entry points.

---

## Reproducing the results

Build the datasets, then fit, diagnose and backtest one model on one universe:

```
python3 -m src.data_pipeline.build_dataset
python3 run_baseline_gaussian.py   --universe size_bm_25 --prior rescaled
python3 diagnose_baseline_gaussian.py --universe size_bm_25
python3 run_backtest_gaussian.py   --universe size_bm_25 --settings rescaled_r2_0p05
```

The three universes are `size_bm_25` (primary), `size_op_25` and
`size_inv_25`. Substitute `bayesian_lasso`, `horseshoe` or
`regularised_horseshoe` for the other models.

`--prior rescaled` is required for the baseline: the default `feng_he`
reproduces the untransported positive control rather than the primary
specification. The other three models have no such switch, since their
calibration is derived rather than chosen.

Approximate wall clock on one machine, per universe: under three hours for a
Gibbs production fit; 19–20 hours for a horseshoe backtest and 6 for the
regularised horseshoe.

---

## Calibration

Hyperparameters are derived from the data rather than transported as fixed
numerical values, and are recomputed within each expanding-window training
sample. `check_scale_transportability.py` reproduces the scale calculations
reported in the dissertation and asserts them against the implementation used
for the fitted models, so those figures fail loudly if the calibration
changes.

---

## Verification

Nine implementation errors capable of producing plausible output were found
during development, almost none of which raised an exception. Verification
therefore proceeds by comparison with quantities whose answers are known:
analytic conditionals against independent references, distributional moments
against theory, composed samplers against an analytically Gaussian target,
prior-generative calibration, and an end-to-end look-ahead corruption test.

`tests/` holds the runnable checks; `logs/` holds the recorded output of runs
too slow to repeat routinely.

---

## Output

`results/<universe>/<model>/` holds `run_log.txt`, `diagnose_log.txt`,
`backtest_log.txt` and the backtest summary JSON for every reported run.
Posterior draws (`.npz`) are excluded from version control as regenerable;
the logs and JSON records of what was run are versioned.