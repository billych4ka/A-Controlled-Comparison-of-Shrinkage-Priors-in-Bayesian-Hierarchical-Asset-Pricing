# results/size_inv_25

25 Size x Investment portfolios. N=25, T=719 (1966-01 to
2025-11), K=144, built with `drop_columns=None` (no months dropped).

SECONDARY universe. Run to test whether the Chapter 5 ranking established on
`size_bm_25` survives a change of sort variable. The prior calibration is NOT
re-derived per universe by hand -- every hyperparameter is recomputed from
this universe's own R and F by the same functions, so `target_r2 = 0.05` and
`p_0 = 23` carry across unchanged and only the data differs.

Layout: one folder per model. Filenames repeat the model name so a file
remains self-identifying if moved or copied.

    <model>/<model>_<setting>_chain<k>.npz        posterior draws
    <model>/<model>_<setting>_chain<k>_meta.json  settings + timings
    <model>/backtest/                             expanding-window output
    <model>/run_log.txt, diagnose_log.txt         console output, where kept

`<k>` is the chain's SEED, not its loop index. With the default `--seed0 0`
the two coincide, which is the case for every file here.

Where filename and sidecar disagree, trust the sidecar.

| model | setting | chains (seeds) | budget |
|---|---|---|---|
| `baseline_gaussian` | `rescaled_r2_0p05` | 2 (0-1) | 3,000 sweeps, 1,000 burn-in |
| `bayesian_lasso` | `rescaled_r2_0p05` | 4 (0-3) | 4,000 sweeps, 1,000 burn-in |
| `horseshoe` | `p0_23_r2_0p05` | 4 (0-3) | 1,500 draws, 1,000 tune |
| `regularised_horseshoe` | `p0_23_r2_0p05` | 4 (0-3) | 1,500 draws, 1,000 tune |

All four at `target_r2 = 0.05`; both horseshoes at `p_0 = 23`. The baseline
runs 2 chains rather than 4 because it is the reference point, not a result
in its own right -- the sensitivity settings that justified it were run once,
on `size_bm_25`, and are not repeated per universe.

No `feng_he` verbatim-prior run here. That setting exists to document the
scale-transportability failure, which is a property of the hyperparameters
and the predictor standardisation, not of the sort variable; repeating it
would add nothing. See `results/size_bm_25/README.md`.

This universe also feeds the pooled cross-sort in `results/cross_sort/`.
