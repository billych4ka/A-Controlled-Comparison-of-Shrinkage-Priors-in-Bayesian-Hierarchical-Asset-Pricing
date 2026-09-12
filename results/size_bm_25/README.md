# results/size_bm_25

25 Size x Book-to-Market portfolios. N=25, T=719 (1966-01 to 2025-11), K=144.

Layout: one folder per model. Filenames repeat the model name so a file
remains self-identifying if moved or copied.

    <model>/<model>_<setting>_chain<k>.npz        posterior draws
    <model>/<model>_<setting>_chain<k>_meta.json  settings + timings

Every .npz has a sidecar recording `prior`, `target_r2`, `blocked`, `seed`,
`n_draws`, `n_burn` and runtimes. Where filename and sidecar disagree, trust
the sidecar -- except for the `feng_he_sequential` files, which predate those
metadata keys (see below).

## baseline_gaussian

Sampling budget throughout: 3,000 sweeps, 1,000 burn-in.

| setting | chains | Delta_b_bar | V_b | prior implied R^2 | notes |
|---|---|---|---|---|---|
| `feng_he_sequential` | 4 | 1.000e-01 | 3.000e+00 | 8756 | ORIGINAL RUN. Feng & He's disclosed mild setting, verbatim. Used the SEQUENTIAL scan (their eq. 14 then eq. 16). Metadata lacks `prior`/`target_r2`/`blocked` -- written before those keys existed. |
| `feng_he` | 4 | 1.000e-01 | 3.000e+00 | 8756 | Same prior, BLOCKED sampler. Reproduces the sequential run to ~0.2 MCMC standard errors -- a cross-sampler convergence check. |
| `rescaled_r2_0p05` | 4 | 5.710e-07 | 1.713e-05 | 0.05 | **PRIMARY.** Fair-comparison setting; this is what the three shrinkage priors are compared against. |
| `rescaled_r2_0p01` | 2 | 1.142e-07 | 3.426e-06 | 0.01 | sensitivity |
| `rescaled_r2_0p1` | 2 | 1.142e-06 | 3.426e-05 | 0.10 | sensitivity |

Shared by all settings: b_bar_bar = 0, nu_b = 1145, nu_Sigma = 27,
V_Sigma = S_hat (sample covariance of excess returns).

### Why two prior settings

Feng & He's hyperparameters are calibrated for THEIR predictor scale
(characteristics on [-1,1], Welch-Goyal macro predictors in raw units,
sd ~0.015-0.5). This project uses expanding-window z-scores, sd ~1. Prior
hyperparameters are not scale-invariant: applied verbatim, their prior implies
E[R^2] = 8,756 and shrinks nothing (0.0% of 3,600 eigendirections). The
rescaled settings preserve their common/deviation variance ratio (33.33:1)
exactly and change only the overall magnitude.

`target_r2 = 0.05` was fixed a priori and must NOT be revised after seeing
results. The derivation is in `rescaled_hyperparameters`
(`src/gibbs/baseline_gaussian.py`).

### Why two samplers

At the calibrated hyperparameters Delta_b ~ 1.7e-8, and Feng & He's sequential
scan is a centred parameterisation, which mixes very poorly at small
hierarchical variance: lag-1 autocorrelation of ||b_bar|| measured at 1.000,
chain still drifting after 250 sweeps. All runs from `rescaled_r2_0p05` onward
draw (B, b_bar) JOINTLY, which fixes it (lag-1 0.190, stationary in one sweep)
and leaves the target unchanged.

`feng_he_sequential` is still valid -- at Delta_b = 0.003 the coupling is weak
and it mixed fine -- and is retained as the cross-sampler check.

## bayesian_lasso

| setting | chains (seeds) | budget | notes |
|---|---|---|---|
| `rescaled_r2_0p05` | 8 (0-7) | 9,500 sweeps, 1,000 burn-in | **PRIMARY.** Eight chains and the long budget are sized for lambda's ESS, which mixes far more slowly than B. |

## horseshoe

| setting | chains (seeds) | budget | notes |
|---|---|---|---|
| `p0_23_r2_0p05` | 4 (0-3) | 1,500 draws, 1,000 tune | **PRIMARY.** `target_accept = 0.99`. |

## regularised_horseshoe

| setting | chains (seeds) | budget | slab scale s | notes |
|---|---|---|---|---|
| `p0_23_r2_0p05` | 4 (0-3) | 1,500 draws, 1,000 tune | 1.357e-03 | **PRIMARY.** s derived from the same p_0 = 23 that sets tau_0. |
| `p0_23_r2_0p05_s2` | 4 (0-3) | 300 draws, 500 tune | 2.0 | Piironen & Vehtari's illustrative slab, applied verbatim. Short budget: it exists to DEMONSTRATE the failure, not to be interpreted. |

### Why the s2 run exists

At P&V's illustrative s = 2.0 the slab standard deviation is 2.83 -- 51
residual standard deviations, and 1,470x the calibrated coefficient scale --
so it binds for 0.017 of 3,600 coefficients and the regularised horseshoe
collapses to the plain horseshoe numerically. The `_s2` chains are kept as the
evidence for that claim. This is the fourth instance of the project's
scale-transportability finding, and the only one where the untransportable
hyperparameter would have silently collapsed one model into another rather
than raising a visible error.

## Shared across all four models

Delta_b_bar = 5.710e-07 propagates to all four -- `lasso_hyperparameters`,
`horseshoe_hyperparameters` and the regularised horseshoe's builder all call
`rescaled_hyperparameters` rather than recomputing it, so the shared hierarchy
holds by construction rather than by three copies of a formula agreeing.
