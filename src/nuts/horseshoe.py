"""
src/nuts/horseshoe.py

Model 3 of 4: the Horseshoe prior on the coefficient deviations, sampled with
NUTS. The likelihood, the b_i = b_bar + theta_i decomposition, b_bar's prior
and Sigma's prior are IDENTICAL to the Gaussian baseline and the Bayesian
LASSO; only theta's prior changes.

    r_{i,t+1} = f_{i,t}' b_i + eps,        eps_t ~ N(0, Sigma)
    b_i       = b_bar + theta_i
    theta_ij  | lambda_ij, tau ~ N(0, tau^2 lambda_ij^2)
    lambda_ij ~ half-Cauchy(0, 1)
    tau       ~ half-Cauchy(0, tau_0)
    b_bar     ~ N(0, Delta_b_bar)
    Sigma     ~ IW(nu_Sigma, V_Sigma)

Delta_b does not exist in this model, exactly as it does not in the LASSO.
nu_b and V_b enter only as the source of sd_target, the per-coefficient
deviation scale the baseline's rescaled prior implies, which is retained here
purely as a cross-check on tau_0's magnitude.

WHY EVERY NUMBER IS COMPUTED AND NONE IS HARDCODED
--------------------------------------------------
Delta_b_bar is imported from baseline_gaussian.rescaled_hyperparameters rather
than re-derived, which makes "all models share one hierarchy" true by
construction instead of by two copies of a formula agreeing. sigma comes from
bayesian_lasso.pooled_residual_scale, the same df-corrected quantity the LASSO
calibrates against -- the same quantity computed two different ways across two
models is precisely what the hold-the-procedure-fixed discipline exists to
prevent. Everything else is derived from R and F, so the backtest recomputes
per window from training data only and the robustness universes recalibrate
automatically.

tau_0 AND THE CHOICE OF p_0
---------------------------
tau_0 = [p_0 / (K - p_0)] * sigma / sqrt(n)                (P&V Eq. 3.12)

p_0 = 23, fixed a priori before any horseshoe result was seen, on the same
discipline applied to target_r2 = 0.05. Piironen & Vehtari characterise p_0
explicitly as a prior guess at the number of relevant variables rather than
an estimate, and note that Eq. (3.12) is derived for the linear Gaussian
model; for more complicated models they recommend drawing from the prior at
different tau and studying the implied sparsity instead. This model --
hierarchical SUR with cross-asset error correlation and a common coefficient
vector -- is not the model the construction was derived for, so tau_0 is a
ballpark rather than a derivation, and implied_m_eff below is the P&V-
recommended substitute.

Three considerations set p_0 = 23. (1) The 144 predictors are not 144
independent opportunities: the mechanical interaction construction produces
systematic redundancy, roll_mean12 and mom_12_1 sharing eleven of twelve
months and correlating at 0.945, a near-duplicate pair inherited by each of
the fifteen macro interactions and reaching 0.955 at its strongest. (2)
Monthly return predictability in this literature is close to nil, so a large
active set is implausible. (3) The construction is asymmetric in its errors:
p_0 enters as p_0/(K - p_0), so understating it imposes shrinkage the data
cannot subsequently undo, and P&V themselves observe that setting p_0 slightly
above the true value can give better results. p_0 = 12 is reported as a
sensitivity.

The justification is deliberately a claim about HOW MANY coefficients are
active and never about WHICH. An earlier version argued that the 120
mechanically-generated interactions were predominantly noise; that claim is
withdrawn, having been contradicted by three converged prior settings of the
Gaussian baseline, by the Bayesian LASSO independently, and by both models on
two further portfolio sorts.

TWO DISCLOSURES
---------------
1. p_0 is a PER-ASSET count. p_0 = 23 means 23 of each asset's K = 144
   deviations, hence 575 of the NK = 3,600 in the model, not 23 in total.
2. Eq. (3.10), from which (3.12) is obtained, assumes unit-variance
   predictors. At this project's standardisation the interaction columns have
   a median standard deviation of 1.41, so tau_0 in fact centres the prior on
   30.0 active coefficients per asset rather than 23 -- an overshoot in the
   same direction consideration (3) prefers. implied_m_eff reports both.

n = T, NOT NT
-------------
Each b_i has T months of direct evidence; cross-asset information enters only
through Sigma in the likelihood, a much weaker channel than 25x more data.
n = NT is the disclosed robustness alternative, giving tau_0 = 7.891e-05.
sigma is POOLED rather than per-asset because tau is a global scale -- the
same argument that made the LASSO's plug-in scale pooled.

SAMPLER SETTINGS LIVE IN THE DATACLASS
--------------------------------------
target_accept, init and max_treedepth are fields here rather than arguments in
the runner, so the backtest adapter cannot silently use different settings from
the production run. target_accept = 0.99 was chosen by measurement (13 Aug):
divergences fell 16/200 -> 7/200 -> 3/200 across 0.90 / 0.95 / 0.99 at a cost
of 0.375 -> 0.540 -> 1.012 seconds per iteration. The setting is applied
uniformly to production and backtest fits, validated by the finding that the
posterior mean of b at 0.90 and 0.99 correlates at 0.9944 -- the same
agreement the LASSO showed between random seeds -- so the uniform choice is
conservative rather than necessary.

VERIFIED (check_chunk1.py, 13 August, 20 checks, all passed)
------------------------------------------------------------
Shared hierarchy, compared against the source functions at ZERO tolerance
rather than against recalled constants:
    Delta_b_bar   5.71048286e-07   identical to rescaled_hyperparameters
                                   and to lasso_hyperparameters
    nu_Sigma      27               = N + 2
    sigma_pooled  0.05565703       identical to the LASSO's s
    sd_target     1.30887160e-04   identical to the LASSO's

tau_0, recomputed independently from P&V Eq. (3.12):
    3.945463e-04   p0 = 23, n = T    PRIMARY
    1.886961e-04   p0 = 12, n = T    sensitivity
    7.890927e-05   p0 = 23, n = NT   sensitivity

implied_m_eff round-trips to EXACTLY 23.0 and 12.0 at unit-variance
predictors, which is what proves Eq. (3.12) was implemented as specified
rather than merely producing a plausible small number. With the real
predictor standard deviations it gives 30.04 per asset, 751 of 3,600 in
total -- the disclosed overshoot. At n = NT it gives 7.26, confirming the
diagnostic does not collapse to p0 by construction.

    tau_0 / sd_target = 3.0144

Two unrelated constructions -- one from a target R^2, one from a sparsity
guess -- landing within a factor of three, with the horseshoe's the looser
of the two, in the generous direction the asymmetry argument prefers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time

import numpy as np

from src.gibbs.baseline_gaussian import rescaled_hyperparameters
from src.gibbs.bayesian_lasso import pooled_residual_scale


@dataclass
class HorseshoeHyperparams:
    """
    Fixed hyperparameters of the Horseshoe model -- chosen once, before
    sampling, and never updated by the sampler.

    b_bar_bar     : (K,)   prior mean of b_bar -- zeros, Feng & He footnote 15
    Delta_b_bar   : (K,K)  prior covariance of b_bar. SHARED with all four
                           models; must be identical or the comparison is
                           confounded. NOT to be confused with Delta_b, the
                           baseline's covariance of the DEVIATIONS, which does
                           not exist in this model and differs by 33.3x
    nu_Sigma      : IW degrees of freedom for Sigma -- N + 2
    V_Sigma       : (N,N)  IW scale for Sigma -- S_hat
    tau_0         : scale of tau's half-Cauchy hyperprior, derived below
    p0            : prior guess at the number of active deviations PER ASSET
    n_choice      : "T" or "NT" -- which sample size entered tau_0. A string
                    rather than an integer so the disclosed robustness setting
                    appears by name in every metadata sidecar
    sigma_pooled  : df-corrected pooled residual sd; the sigma in tau_0
    sd_target     : the deviation sd the baseline's rescaled prior implies.
                    The only role nu_b and V_b play here, kept as a cross-check
                    on tau_0's magnitude, not used in sampling
    target_accept, init, max_treedepth : sampler settings, carried here so
                    production and backtest fits cannot diverge from each other
    """

    b_bar_bar: np.ndarray
    Delta_b_bar: np.ndarray
    nu_Sigma: float
    V_Sigma: np.ndarray
    tau_0: float
    p0: int
    n_choice: str
    sigma_pooled: float
    sd_target: float
    target_accept: float = 0.99
    init: str = "adapt_diag"
    max_treedepth: int = 10

    @property
    def K(self) -> int:
        return self.b_bar_bar.shape[0]

    @property
    def N(self) -> int:
        return self.V_Sigma.shape[0]

    @property
    def tau0_over_sd_target(self) -> float:
        """tau_0 as a multiple of the independently-calibrated deviation scale.

        A corroboration statistic, not an input: two unrelated constructions
        -- one from a target R^2, one from a sparsity guess -- landing within
        a factor of a few is the cross-check. Reported as a ratio so the
        write-up quotes a measured number rather than a recalled one.
        """
        return float(self.tau_0 / self.sd_target)


def horseshoe_hyperparameters(R: np.ndarray, F: np.ndarray, K: int,
                              p0: int = 23, target_r2: float = 0.05,
                              n_choice: str = "T",
                              target_accept: float = 0.99,
                              ) -> HorseshoeHyperparams:
    """
    Build the Horseshoe's hyperparameters from the data, at the same prior
    scale for the shared hierarchy as the baseline and the LASSO.

    Delta_b_bar, nu_Sigma and V_Sigma come straight from
    rescaled_hyperparameters(); sigma comes from pooled_residual_scale().
    Neither is recomputed here. Only tau_0 is new.

    R, F      : (N,T) and (N,T,K); everything is computed from these, so the
                backtest's per-window recomputation is automatic
    p0        : 23 primary, 12 sensitivity. Fixed a priori; NOT to be revised
                after seeing results, and in particular not to be moved toward
                wherever the posterior tau lands -- that would centre the prior
                on the answer and destroy the result it appears to support
    target_r2 : 0.05, fixed a priori, sets Delta_b_bar via the baseline
    n_choice  : "T" (adopted) or "NT" (disclosed robustness)
    """
    if not 0 < p0 < K:
        raise ValueError(f"p0={p0} must lie strictly between 0 and K={K}: "
                         f"tau_0 divides by (K - p0)")
    if n_choice not in ("T", "NT"):
        raise ValueError(f"n_choice={n_choice!r} must be 'T' or 'NT'")

    N, T = R.shape
    base = rescaled_hyperparameters(R, F, K, target_r2=target_r2)
    sigma_pooled, _ = pooled_residual_scale(R, F)

    n = T if n_choice == "T" else N * T
    tau_0 = (p0 / (K - p0)) * sigma_pooled / np.sqrt(n)

    # the deviation scale the baseline's rescaled prior implies -- the same
    # quantity the LASSO calibrates lambda against, retained only as a
    # cross-check on tau_0
    sd_target = float(np.sqrt(base.V_b[0, 0] / (base.nu_b - K - 1)))

    return HorseshoeHyperparams(
        b_bar_bar=base.b_bar_bar,
        Delta_b_bar=base.Delta_b_bar,
        nu_Sigma=base.nu_Sigma,
        V_Sigma=base.V_Sigma,
        tau_0=float(tau_0),
        p0=int(p0),
        n_choice=n_choice,
        sigma_pooled=float(sigma_pooled),
        sd_target=sd_target,
        target_accept=float(target_accept),
    )


def implied_m_eff(hp: HorseshoeHyperparams, F: np.ndarray,
                  tau: float | None = None, unit_variance: bool = False
                  ) -> float:
    """
    The effective number of nonzero coefficients per asset that a given tau
    implies -- the Horseshoe's counterpart to baseline_gaussian.prior_implied_r2
    and bayesian_lasso.lasso_prior_implied_r2.

        E[m_eff | tau, sigma] = sum_j a_j / (1 + a_j),
        a_j = tau * sigma^-1 * sqrt(n) * s_j                (P&V Eqs. 3.8, 3.5)

    with s_j the ROOT MEAN SQUARE of predictor j -- P&V write it as that
    predictor's standard deviation, which is the same thing only under their
    zero-mean assumption; the body comment below says why this project
    computes the mean square instead. This is the quantity Piironen & Vehtari
    recommend inspecting directly when the model is more complicated than the
    linear regression Eq. (3.12) was derived for, which is the case here.

    WHY n IS ALWAYS T, EVEN WHEN tau_0 WAS BUILT WITH n = NT
    --------------------------------------------------------
    tau_0 is proportional to 1/sqrt(n) and a_j is proportional to tau*sqrt(n),
    so n CANCELS: evaluating this at the same n the construction used would
    return p_0 by construction whatever n_choice was, and the n = NT
    robustness setting would look identical to the primary. Fixing n = T makes
    the diagnostic evaluate the CONSEQUENCE of the construction rather than
    adopt its assumption, and T is the honest per-coefficient sample size --
    each theta_ij is informed by asset i's T months. The n = NT setting then
    shows up as what it is, a fourfold tightening: 7.26 rather than 30.04.

    unit_variance : if True, sets every s_j = 1, reproducing P&V's
                    idealisation. At tau = tau_0 and n_choice = "T" this must
                    return EXACTLY p_0 -- the round-trip that proves the
                    construction was implemented as specified. With the real
                    s_j it returns 30.0 at p_0 = 23, the disclosed overshoot.
    tau           : defaults to hp.tau_0; pass a posterior mean to express a
                    fitted global scale in the same units, which is how the
                    Horseshoe is compared with the LASSO (tau and lambda are
                    not commensurable; m_eff and the shrinkage factors are)

    F -> (N,T,K). Returns the PER-ASSET expected count; multiply by N for the
    implied total across all NK deviations.
    """
    N, T, K = F.shape
    if K != hp.K:
        raise ValueError(f"F has K={K} but hyperparameters have K={hp.K}")

    tau = hp.tau_0 if tau is None else float(tau)
    # s_j is the ROOT MEAN SQUARE of column j, not its standard deviation.
    # P&V's a_j = tau sigma^-1 sqrt(n) s_j comes from X'X ~ n diag(s_j^2), so
    # s_j^2 is the mean square; they assume zero-mean predictors, under which
    # mean square and variance coincide, and write it as a variance. This
    # project's expanding-window z-scores do not have exactly zero mean
    # (median |mean| 0.116, max 0.74), and the intercept column has mean 1 and
    # standard deviation exactly 0 -- so F.std would hand the intercept a
    # scale of zero and assign it no information at all.
    s = np.ones((N, K)) if unit_variance else np.sqrt((F ** 2).mean(axis=1))

    a = tau / hp.sigma_pooled * np.sqrt(T) * s
    return float((a / (1.0 + a)).sum(axis=1).mean())

# =========================================================================
# CHUNK 2 -- append to src/nuts/horseshoe.py, below horseshoe_hyperparameters
# and implied_m_eff. The imports below go at the top of the file with the
# others; pymc and pytensor are imported INSIDE build_model so that
# horseshoe_hyperparameters and implied_m_eff stay importable without them
# (the backtest adapter and the diagnostics both need the cheap half).
# =========================================================================


def _sigma_reference(R: np.ndarray, F: np.ndarray):
    """
    Sigma's reference point and per-entry coordinate scales.

    NUTS cannot sample a constrained matrix, so Sigma is reconstructed from an
    unconstrained vector: the lower triangle of a Cholesky factor, with the
    diagonal stored as a log. Two refinements on top of that, both about the
    SAMPLER's coordinates and neither changing the target:

    1. The vector is centred at the OLS residual covariance rather than at
       zero. pm.Flat's support point is 0, which means L = I and Sigma = I --
       residual variance 1 against an actual 0.003, an enormous distance for
       warmup to travel for no reason.

    2. Each entry is scaled to unit natural size. For T observations the
       Cholesky log-diagonals have posterior sd about 1/sqrt(2T) and the
       off-diagonals in column j about L_jj/sqrt(T); here that is 2.6e-02 and
       5.5e-04 to 2.5e-03, against 1 for z and log lambda. An identity initial
       metric across a 2,000-fold spread of scales is what a diagonal mass
       matrix has to undo before it can start; supplying it costs nothing.

    Why this matters, concretely: the first attempt at this model left the
    coordinates unscaled and used PyMC's default init="jitter+adapt_diag",
    whose U(-1,1) jitter is 1,300 prior standard deviations on b_bar. The
    logp stayed finite, nothing raised, and dual averaging drove the step size
    to 9.1e-22 -- a sampler that never moved, reporting a 12-second "fit".

    The scaling is a linear map with a constant Jacobian, so it shifts the log
    density by a constant and leaves the posterior identical.

    -> (packed0, packed_scale, S0, B_ols)
    """
    N, T = R.shape
    _, B_ols = pooled_residual_scale(R, F)
    E = R - np.einsum("itk,ik->it", F, B_ols)
    S0 = np.cov(E, ddof=1)
    L0 = np.linalg.cholesky(S0)

    tri = np.tril_indices(N)
    is_diag = tri[0] == tri[1]

    packed0 = L0[tri].copy()
    packed0[is_diag] = np.log(np.diag(L0))

    packed_scale = np.empty(packed0.size)
    packed_scale[is_diag] = 1.0 / np.sqrt(2.0 * T)
    packed_scale[~is_diag] = np.diag(L0)[tri[1][~is_diag]] / np.sqrt(T)

    return packed0, packed_scale, S0, B_ols


def build_model(R: np.ndarray, F: np.ndarray, hp: HorseshoeHyperparams):
    """
    The PyMC model, returned together with its starting point.

    Returns (model, initvals) as a PAIR deliberately. Sigma's coordinate is
    DEFINED as packed0 + packed_scale * packed_z, so a starting point that
    does not agree with that definition is not merely suboptimal, it is
    wrong -- and the failure is silent, since any finite logp will sample.
    Keeping the two in separate functions invites exactly that mismatch.

    Parameterisation
    ----------------
    theta_ij = z_ij * lambda_ij * tau, with z ~ N(0,1): NON-CENTRED, so the
    scales and the standardised deviation are separate coordinates rather than
    nested ones.

    This is Piironen & Vehtari's Appendix C.1. Their Appendix C.2, which
    decomposes each half-Cauchy into a half-normal times the square root of an
    inverse-gamma, was built and tested against it on 13 August and REJECTED:
    13 divergences against 16 (inside binomial noise -- two runs of the
    identical C.1 model gave 10 and 16) for 1.90x the wall clock. The two
    parameterisations agreed on the posterior (b_bar median |z| 0.71, 99.3%
    within 3 SE; tau |z| 0.54), which is retained as evidence that C.1 is
    sampling correctly and that tau's position is not a coordinate artefact.

    Every sampled coordinate is O(1) by construction: b_bar_z, z, log lambda,
    log tau_z and Sigma_packed_z. b_bar and tau are recovered as Deterministics
    at their true scales.

    On the intercept
    ----------------
    theta_i0 carries the same horseshoe prior as every other deviation. P&V
    give the intercept a separate wide prior, but their intercept is a level
    while this one is a DEVIATION from the common b_bar_0, which already has
    its own prior through Delta_b_bar. Shrinking per-asset alphas toward the
    common alpha is the same treatment the baseline and the LASSO apply, and
    departing from it here would make the models differ in two places.

    Note the intercept column has predictor sd exactly 0 but mean square 1.
    The shrinkage diagnostics use the mean square for exactly this reason;
    an earlier version used the standard deviation and assigned the intercept
    no information at all, giving every asset's alpha deviation a shrinkage
    factor of exactly 1. See implied_m_eff.
    """
    import pymc as pm

    from src.likelihood.sur import sur_log_likelihood_pytensor
    from src.priors.inverse_wishart import inverse_wishart_cholesky_logp

    N, T, K = F.shape
    if K != hp.K or N != hp.N:
        raise ValueError(f"F is (N={N}, T={T}, K={K}) but hyperparameters "
                         f"describe N={hp.N}, K={hp.K}")

    packed0, packed_scale, _, B_ols = _sigma_reference(R, F)
    sd_bbar = float(np.sqrt(hp.Delta_b_bar[0, 0]))
    if not np.allclose(np.diag(hp.Delta_b_bar), hp.Delta_b_bar[0, 0]):
        raise ValueError("Delta_b_bar is not isotropic; build_model takes its "
                         "scalar sd from entry [0,0] and would be wrong")

    with pm.Model() as model:
        b_bar_z = pm.Normal("b_bar_z", 0.0, 1.0, shape=K)
        b_bar = pm.Deterministic("b_bar", sd_bbar * b_bar_z)

        z = pm.Normal("z", 0.0, 1.0, shape=(N, K))
        lam = pm.HalfCauchy("lam", beta=1.0, shape=(N, K))
        tau_z = pm.HalfCauchy("tau_z", beta=1.0)
        tau = pm.Deterministic("tau", hp.tau_0 * tau_z)

        b = pm.Deterministic("b", b_bar[None, :] + z * lam * tau)

        # pm.Flat, NEVER a proper prior. pm.Potential ADDS to the model's total
        # logp; it does not replace the declared variable's own prior. A
        # pm.Normal(0,1) here silently stacked an extra density on Sigma and
        # biased E[Sigma] by 8-11 Monte Carlo standard errors; pm.Flat brought
        # it to under 1.3. check_chunk2.py tests this directly rather than
        # trusting the comment.
        packed_z = pm.Flat("Sigma_packed_z", shape=packed0.size)
        Sigma, sigma_logp = inverse_wishart_cholesky_logp(
            packed0 + packed_scale * packed_z, hp.nu_Sigma, hp.V_Sigma)
        Sigma = pm.Deterministic("Sigma", Sigma)
        pm.Potential("Sigma_prior", sigma_logp)

        pm.Potential("likelihood", sur_log_likelihood_pytensor(R, F, b, Sigma))

    initvals = {
        "b_bar_z": B_ols.mean(axis=0) / sd_bbar,   # b_bar at the average OLS fit
        "z": np.zeros((N, K)),                     # theta exactly 0
        "lam": np.ones((N, K)),                    # the half-Cauchy median
        "tau_z": 1.0,                              # tau at tau_0
        "Sigma_packed_z": np.zeros(packed0.size),  # Sigma at the OLS residual cov
    }
    return model, initvals


def transformed_point(model, initvals: dict) -> dict:
    """
    Convert an untransformed starting point into the dictionary PyMC's
    compiled logp and dlogp functions expect.

    lam and tau_z are positive, so PyMC samples their logs and names them
    lam_log__ and tau_z_log__. Doing that conversion explicitly, and asserting
    the key set, is more robust across PyMC versions than calling the transform
    objects -- and it fails loudly if a variable is renamed, rather than
    silently evaluating at the default point.

    pm.sample() takes the UNTRANSFORMED initvals; this is only for evaluating
    logp and gradients directly, as check_chunk2.py and the timing tests do.
    """
    point = model.initial_point()
    expected = {"b_bar_z", "z", "lam_log__", "tau_z_log__", "Sigma_packed_z"}
    if set(point) != expected:
        raise ValueError(f"model has value variables {sorted(point)}, "
                         f"expected {sorted(expected)}")
    point["b_bar_z"] = np.asarray(initvals["b_bar_z"], dtype=float)
    point["z"] = np.asarray(initvals["z"], dtype=float)
    point["lam_log__"] = np.log(np.asarray(initvals["lam"], dtype=float))
    point["tau_z_log__"] = np.log(np.asarray(initvals["tau_z"], dtype=float))
    point["Sigma_packed_z"] = np.asarray(initvals["Sigma_packed_z"], dtype=float)
    return point

# =========================================================================
# CHUNK 3 -- append to src/nuts/horseshoe.py, below build_model and
# transformed_point. Needs `from dataclasses import dataclass, field` and
# `from time import time` at the top of the file.
# =========================================================================


@dataclass
class HorseshoeDraws:
    """
    Posterior draws from one chain. Field names and shapes mirror GibbsDraws
    and LassoDraws so that io.py, convergence.py and the evaluation code work
    unchanged.

    B          : (n, N, K)  asset-specific coefficients, b_i = b_bar + theta_i.
                            The backtest's estimand and the only field it uses
    b_bar      : (n, K)     the common coefficient vector
    lam_local  : (n, N, K)  the 3,600 LOCAL scales
    tau        : (n,)       the global scale
    Sigma      : (n, N, N)  residual covariance

    NOT `lam`. LassoDraws.lam is that model's GLOBAL scale, so reusing the
    name here -- for the local scales, the opposite level of the hierarchy --
    would invite exactly the confusion Delta_b and Delta_b_bar already cause.
    The horseshoe's global scale is `tau`.

    NUTS sample statistics, one entry per retained draw. The brief requires
    divergences and tree-depth saturation to be reported; energy is kept
    because E-BFMI is computed from it and arviz 1.2.0's az.bfmi raises a
    TypeError on an InferenceData:

    diverging, tree_depth, n_steps, step_size, energy

    WHAT IS NOT STORED, AND WHY IT IS NOT A COMPROMISE
    --------------------------------------------------
    z, the standardised deviation in theta = z * lambda * tau, is a
    coordinate that exists for the sampler's geometry and is exactly
    recoverable as (B - b_bar) / (lam_local * tau). The shrinkage factors
    kappa are a deterministic function of lam_local, tau and the data, so
    they are recomputed on demand by shrinkage_factors() rather than stored.
    Every parameter in the model is therefore either retained in full or
    exactly reconstructible; nothing is summarised or subsampled.

    That is a real difference from the LASSO, where tau^2 had to be reduced
    to running means plus 200 tracked traces, so its convergence diagnostics
    cover a subsample and must be described as such. Here R-hat and ESS can
    be reported for all 3,600 local scales. NUTS needs far fewer draws for a
    given ESS than a Gibbs sampler with lag-1 autocorrelation near 0.93, so
    the model that looked like the storage problem is the one that avoids it.
    State this in the write-up: the horseshoe is held to a STRICTER
    diagnostic than the LASSO. It is a difference in how the models are
    checked, not in how they are specified, so it does not confound the
    comparison -- but a reader will notice one model reporting 3,600 R-hats
    and the other 200.

    At 1,500 retained draws a chain is about 91 MB, comparable to the
    baseline's 66 MB and the LASSO's ~100 MB.
    """

    B: np.ndarray
    b_bar: np.ndarray
    lam_local: np.ndarray
    tau: np.ndarray
    Sigma: np.ndarray
    diverging: np.ndarray
    tree_depth: np.ndarray
    n_steps: np.ndarray
    step_size: np.ndarray
    energy: np.ndarray
    meta: dict = field(default_factory=dict)


def ebfmi(energy: np.ndarray) -> float:
    """
    E-BFMI = sum (E_t - E_{t-1})^2 / sum (E_t - Ebar)^2, per Betancourt (2017).
    Below about 0.3 indicates the sampler is struggling to move between energy
    levels -- a failure distinct from divergences, and one a divergence count
    will not reveal. Three lines of arithmetic, computed here rather than
    through arviz, whose az.bfmi raises a TypeError on version 1.2.0.
    """
    e = np.asarray(energy, dtype=float).ravel()
    return float(np.sum(np.diff(e) ** 2) / np.sum((e - e.mean()) ** 2))


def shrinkage_factors(draws: HorseshoeDraws, F: np.ndarray,
                      hp: HorseshoeHyperparams) -> np.ndarray:
    """
    The per-coefficient shrinkage factors kappa_ij, P&V Eq. (2.4):

        kappa_ij = 1 / (1 + n sigma^-2 tau^2 lambda_ij^2 s_j^2)

    with s_j the ROOT MEAN SQUARE of predictor j, not its standard deviation:
    the intercept column has mean 1 and sd exactly 0 and would otherwise be
    assigned no information at all. See implied_m_eff.

    kappa near 1 is complete shrinkage toward the common b_bar; near 0 is a
    coefficient left free. This is THE quantity in which the four models are
    comparable. tau and the LASSO's lambda are not commensurable -- one is a
    global scale multiplying heavy-tailed local scales, the other the rate of
    an exponential on per-coefficient variances -- but every model in the
    design implies a kappa, including the Gaussian baseline, whose kappa is
    the same for every coefficient because it has no local layer at all.

    n = T, the per-asset sample size: each theta_ij is informed by asset i's T
    months. See implied_m_eff for why this is fixed at T rather than taken
    from hp.n_choice.

    Not stored in HorseshoeDraws because it is an exact function of fields
    that are. Returns (n_draws, N, K).
    """
    N, T, K = F.shape
    # root mean square, not standard deviation -- see implied_m_eff. With
    # F.std the intercept column gets s = 0, hence a = 0, hence kappa = 1
    # exactly: complete shrinkage of every asset's alpha deviation. With the
    # mean square, a = 0.190 and kappa = 0.965.
    s = np.sqrt((F ** 2).mean(axis=1))                   # (N, K)
    a = (np.sqrt(T) / hp.sigma_pooled) * draws.tau[:, None, None] * draws.lam_local * s
    return 1.0 / (1.0 + a ** 2)


def run_nuts(R: np.ndarray, F: np.ndarray, hp: HorseshoeHyperparams,
             n_draws: int = 1500, n_tune: int = 1000, chains: int = 4,
             cores: int = 4, seed0: int = 0, progressbar: bool = True
             ) -> list[HorseshoeDraws]:
    """
    Sample the horseshoe and return ONE HorseshoeDraws PER CHAIN.

    All chains are drawn in a single pm.sample call so PyMC can run them in
    parallel across cores -- roughly 41 minutes rather than 164 for a
    four-chain production run -- and the result is split afterwards. The
    per-chain files that convergence.load_chains globs are identical either
    way; what the runner does internally is not something any other module
    sees. Chain k is seeded seed0 + k, and the RUNNER must use seed0 + k in
    the filename too: run_bayesian_lasso.py uses the loop index instead, so
    --seed0 4 silently overwrote chains 0-3.

    Note the draws are not guaranteed bit-identical to four separate
    single-chain calls with the same seeds; PyMC's seed handling across chains
    is a version-dependent detail. The target is the same and R-hat across
    chains is what certifies agreement.

    Sampler settings come from hp, NOT from arguments here, so a backtest fit
    cannot silently differ from the production run: target_accept = 0.99,
    init = "adapt_diag", max_treedepth = 10.

    On init: NEVER "jitter+adapt_diag", PyMC's default. Its U(-1,1) jitter is
    applied in unconstrained space, which is 1,300 prior standard deviations
    on b_bar -- displacing b by 1.0 gives predicted returns with sd 27.5
    against an actual 0.058. The logp stays finite, nothing raises, and dual
    averaging drives the step size to 9.1e-22: a sampler that never moves,
    reporting a fast "fit". This is why the coordinates are non-dimensionalised
    in build_model and why init is pinned here.

    n_draws is RETAINED draws per chain, excluding n_tune -- the opposite of
    the Gibbs runners' convention, where n_draws includes burn-in. PyMC's
    argument means retained, and silently redefining it would be worse than
    the inconsistency. Budget: tau is the slowest parameter, ESS ~14 per 200
    draws in pilot runs, so ESS 400 needs roughly 5,700 retained draws in
    total; 4 chains x 1,500 clears it. B and b_bar reach ESS 400 far sooner.
    """
    import pymc as pm

    N, T, K = F.shape
    model, initvals = build_model(R, F, hp)
    seeds = [seed0 + k for k in range(chains)]

    t0 = time()
    with model:
        idata = pm.sample(draws=n_draws, tune=n_tune, chains=chains, cores=cores,
                          target_accept=hp.target_accept, init=hp.init,
                          max_treedepth=hp.max_treedepth, initvals=initvals,
                          random_seed=seeds, progressbar=progressbar,
                          compute_convergence_checks=False)
    elapsed = time() - t0

    po, ss = idata.posterior, idata.sample_stats
    out = []
    for k in range(chains):
        div = ss["diverging"].values[k]
        depth = (ss["tree_depth"].values[k] if "tree_depth" in ss
                 else np.full(n_draws, np.nan))
        steps = (ss["n_steps"].values[k] if "n_steps" in ss
                 else np.full(n_draws, np.nan))
        energy = ss["energy"].values[k]
        out.append(HorseshoeDraws(
            B=po["b"].values[k],
            b_bar=po["b_bar"].values[k],
            lam_local=po["lam"].values[k],
            tau=po["tau"].values[k],
            Sigma=po["Sigma"].values[k],
            diverging=div, tree_depth=depth, n_steps=steps,
            step_size=ss["step_size"].values[k], energy=energy,
            meta={"seed": seeds[k], "chain_index": k,
                  "n_draws": n_draws, "n_tune": n_tune,
                  "N": N, "T": T, "K": K,
                  "tau_0": hp.tau_0, "p0": hp.p0, "n_choice": hp.n_choice,
                  "sigma_pooled": hp.sigma_pooled, "sd_target": hp.sd_target,
                  "Delta_b_bar": float(hp.Delta_b_bar[0, 0]),
                  "nu_Sigma": float(hp.nu_Sigma),
                  "target_accept": hp.target_accept, "init": hp.init,
                  "max_treedepth": hp.max_treedepth,
                  "divergences": int(div.sum()),
                  "tree_depth_mean": float(np.nanmean(depth)),
                  "tree_depth_saturating": float(np.nanmean(depth >= hp.max_treedepth)),
                  "step_size_final": float(ss["step_size"].values[k][-1]),
                  "ebfmi": ebfmi(energy),
                  "sampling_seconds_all_chains": elapsed},
        ))

    total_div = sum(d.meta["divergences"] for d in out)
    bfmi_str = " ".join(f"{d.meta['ebfmi']:.2f}" for d in out)
    depth_str = " ".join(f"{d.meta['tree_depth_mean']:.2f}" for d in out)
    print(f"  {chains} chains x {n_draws} draws ({n_tune} tune) in {elapsed:.0f}s")
    print(f"  divergences {total_div} of {chains * n_draws} "
          f"({100 * total_div / (chains * n_draws):.1f}%) | "
          f"tree depth {depth_str} | E-BFMI {bfmi_str}")
    return out

