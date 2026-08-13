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
results. See GAUSSIAN_BASELINE_SUMMARY.md.

### Why two samplers

At the calibrated hyperparameters Delta_b ~ 1.7e-8, and Feng & He's sequential
scan is a centred parameterisation, which mixes very poorly at small
hierarchical variance: lag-1 autocorrelation of ||b_bar|| measured at 1.000,
chain still drifting after 250 sweeps. All runs from `rescaled_r2_0p05` onward
draw (B, b_bar) JOINTLY, which fixes it (lag-1 0.190, stationary in one sweep)
and leaves the target unchanged.

`feng_he_sequential` is still valid -- at Delta_b = 0.003 the coupling is weak
and it mixed fine -- and is retained as the cross-sampler check.

## bayesian_lasso / horseshoe / regularised_horseshoe

Empty. See GAUSSIAN_BASELINE_SUMMARY.md section 7 before running any of them:
Delta_b_bar = 5.710e-07 must propagate to all four models, and the regularised
horseshoe's slab width needs recalibrating from s=2 to s=1.357e-03 or it is
numerically identical to the plain horseshoe.
