"""
src/nuts/regularised_horseshoe.py

Model 4 of 4: the REGULARISED Horseshoe on the coefficient deviations.
Everything except theta's prior is identical to the other three models:
the SUR likelihood, the b_i = b_bar + theta_i decomposition, b_bar's Normal
prior at Delta_b_bar, and Sigma's Inverse-Wishart.

    theta_ij | lambda_ij, tau, c ~ N(0, tau^2 lambda~_ij^2)
    lambda~_ij^2 = c^2 lambda_ij^2 / (c^2 + tau^2 lambda_ij^2)     P&V Eq. (2.8)
    lambda_ij ~ C+(0, 1)
    tau       ~ C+(0, tau_0)
    c^2       ~ Inv-Gamma(nu/2, nu s^2 / 2)

The whole model is the plain horseshoe with one change: lambda~ replaces
lambda. When tau^2 lambda^2 << c^2 the two coincide and the prior IS the
horseshoe; when tau^2 lambda^2 >> c^2 the prior approaches N(0, c^2), a
Gaussian slab. So c "soft-truncates" the horseshoe's Cauchy tails, and
c -> infinity recovers the plain horseshoe exactly. That nesting is what makes
models 3 and 4 a controlled pair: they differ in ONE hyperparameter, and
setting it to infinity collapses one into the other.

THE SLAB WIDTH, AND WHY P&V's DEFAULT IS UNUSABLE HERE
------------------------------------------------------
E[c^2] = nu s^2 / (nu - 2), so at nu = 4 the slab's typical scale is
sqrt(2) s. Piironen & Vehtari's illustrative default s = 2.0 gives a slab sd
of 2.83, which is 51 monthly return standard deviations and roughly 1,500
times the calibrated coefficient scale. A cap that far above anything the data
can produce NEVER BINDS, and the regularised horseshoe would run cleanly,
converge, and be numerically indistinguishable from the plain horseshoe:
four priors that are really three, with no error raised anywhere.

s is therefore derived from the SAME p_0 logic that gives tau_0. If p_0 = 23
coefficients carry the target R^2 between them, each has variance
R^2 Var(r) / (p_0 * mean var(f)), a signal scale of 1.919e-03 on size_bm_25.
Inverting E[c] = sqrt(nu/(nu-2)) s at nu = 4 gives s = 1.357e-03. Every input
is justified elsewhere in the dissertation (p_0, target_r2, nu, and
quantities read off the data), so no new free parameter is introduced.

This is the FOURTH instance of the project's scale theme, and the cleanest:
a dimensionless hyperparameter (nu = 4) transports safely, while a dimensional
one (s) does not. Running once at s = 2.0 should give a binding fraction of
essentially zero, which converts the recalibration from housekeeping into
evidence.

WHERE THE SLAB ACTUALLY BINDS: and a correction to the original prediction
--------------------------------------------------------------------------
The slab dominates when tau^2 lambda^2 > c^2, i.e. lambda > c/tau. The
project's earlier note predicted a binding fraction of about 12%, computed at
tau = tau_0. That was correct when written, but the plain horseshoe has since
MEASURED tau, and it lands at roughly 5% of tau_0 on all three universes.
A twentyfold smaller tau raises the threshold twentyfold:

    evaluated at            c/tau    P(lambda > c/tau)    of NK = 3,600
    tau_0 (prior centre)      4.9         12.91%              465
    posterior tau, BM        91.3          0.70%               25
    posterior tau, OP        98.9          0.64%               23
    posterior tau, Inv      138.2          0.46%               17
    s = 2.0, posterior tau  134565          0.0005%             0.02

So the honest prediction is ~0.7%, about 25 coefficients of 3,600, NOT 12%.
For the other 99.3% the two models place an IDENTICAL prior on theta.

That is not a reason to abandon the model. Those 25 are exactly the population
causing the trouble: the plain horseshoe's prior predictive showed the single
largest |theta| in a draw contributing a median implied R^2 of 1.28 on its
own. But it does sharpen the expectation: the two models will be very
similar, and any difference lives in a handful of extreme coefficients rather
than across the board.

PRE-REGISTERED, before model 4 is fitted:
  - binding fraction ~0.7% of local scales at the posterior tau (12.9% if
    evaluated at tau_0, and ~0% at s = 2.0)
  - OOS R^2 indistinguishable from the plain horseshoe: the slab touches theta,
    and theta carries ~1% of out-of-sample loss
  - fewer divergences and similar or better wall clock, per P&V Sec. 4.2/5
  - any performance difference appears in Sharpe or CE, via the demeaned
    portfolio weights, where one outsized coefficient can dominate a month's
    cross-sectional ranking
  - STOPPING RULE: a Sharpe difference against the plain horseshoe below one
    standard error (0.158 over 479 months) means the primary universe alone is
    reported; above it, OP and Inv are run. Fixed in advance.

    Building Notes:
    Verified 16 Aug (23 checks against an independent NumPy reference):
    τ₀, Δb̄, ν_Σ, V_Σ, σ and sd_target identical to horseshoe_hyperparameters at
    zero tolerance. s = 1.356930e-03, ν = 4, E[c] = 1.918988e-03 = 0.034σ = 14.7× sd_target.
     Binding fraction P(λ > c/τ): 12.91% at τ₀, 0.70% (25 of 3,600) at the measured posterior τ,
    and 0.017 of 3,600 at P&V's illustrative s = 2.0. λ̃ limits confirmed: c → ∞
    recovers the plain horseshoe exactly; τλ ≫ c gives τλ̃ → c; λ̃ ≤ λ always, with small λ shrunk by 0.021% —
    P&V's stated design that the prior always shrinks at least a little.
    The slab width is derived from the same p₀ = 23 that sets τ₀: if p₀ coefficients carry the target R² between them,
    each has a signal scale of 1.919e-03, which at ν = 4 inverts to s = 1.357e-03. Piironen & Vehtari's illustrative
    s = 2.0 implies a slab standard deviation of 2.83 — 51 residual standard deviations, and 1,470 times the
    calibrated coefficient scale — which at this data's scale binds for 0.017 of 3,600 coefficients. Applied verbatim,
    the regularised horseshoe would be numerically the plain horseshoe. This is the fourth instance of the project's
    scale finding, and the only one where the untransportable hyperparameter would have silently collapsed one model
    into another rather than producing a visible error.
    Verified 16 Aug (8 checks): total log-density at the starting point 32,874.997957,
    matching an independent NumPy reference and equal to the plain horseshoe's 32,875.6117 minus the
     caux term of −0.613743. Nesting confirmed: at c → ∞ a z perturbation gives −1.286883, identical to model 3's.
     With the calibrated c the same perturbation gives −1.239592, since λ̃/λ = 0.960233 at λ = 1 — the slab is active
     and shrinks a median local scale by 4%.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time

import numpy as np

from src.gibbs.baseline_gaussian import rescaled_hyperparameters
from src.gibbs.bayesian_lasso import pooled_residual_scale


@dataclass
class RegHorseshoeHyperparams:
    """
    Fixed hyperparameters of the Regularised Horseshoe.

    Identical to HorseshoeHyperparams except for slab_scale and slab_df, which
    are the only new quantities in the entire model. Deliberately a SEPARATE
    dataclass rather than optional fields on the horseshoe's: a model 3 object
    should not be constructible with a slab silently absent, and io.py stores
    whatever fields a dataclass declares.

    b_bar_bar     : (K,)   zeros
    Delta_b_bar   : (K,K)  SHARED with all four models; must be identical or
                           the comparison is confounded
    nu_Sigma      : N + 2
    V_Sigma       : (N,N)  S_hat
    tau_0         : the horseshoe's, unchanged. P&V note that Eq. (3.12) is
                    used for the regularised horseshoe too, with p_0 as the
                    prior guess for the number of coefficients FAR FROM ZERO,
                    remembering those will also be regularised by the slab
    slab_scale    : s, derived from p_0, NOT P&V's illustrative 2.0
    slab_df       : nu = 4. DIMENSIONLESS, so it transports safely
    p0, n_choice, sigma_pooled, sd_target : as for the horseshoe
    target_accept, init, max_treedepth : sampler settings, carried here so
                    production and backtest fits cannot diverge from each other
    """

    b_bar_bar: np.ndarray
    Delta_b_bar: np.ndarray
    nu_Sigma: float
    V_Sigma: np.ndarray
    tau_0: float
    slab_scale: float
    slab_df: float
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
    def slab_sd(self) -> float:
        """E[c] = sqrt(nu/(nu-2)) * s, the slab's typical scale.

        The quantity to quote in the write-up, not slab_scale itself: s is the
        Inv-Gamma's scale parameter and is smaller than the slab it implies by
        a factor of sqrt(2) at nu = 4.
        """
        return float(np.sqrt(self.slab_df / (self.slab_df - 2.0)) * self.slab_scale)

    def binding_threshold(self, tau: float | None = None) -> float:
        """The lambda_ij above which the slab dominates: lambda > c/tau.

        tau defaults to tau_0. PASS THE POSTERIOR tau INSTEAD when reporting:
        the data moves tau to about 5% of tau_0, which raises this threshold
        twentyfold and cuts the binding fraction from 12.9% to 0.7%.
        """
        return float(self.slab_sd / (self.tau_0 if tau is None else tau))

    def binding_fraction(self, tau: float | None = None) -> float:
        """P(lambda_ij > c/tau) under the half-Cauchy(0,1) prior.

        THE diagnostic for this model, and the one whose failure is silent: if
        it comes back near zero the regularised horseshoe has collapsed into
        the plain horseshoe and any reported difference between them is noise.
        Prior-side only: the posterior fraction is computed from the draws in
        diagnose_regularised_horseshoe.py, and the two should be compared.
        """
        return float(2.0 / np.pi * np.arctan(1.0 / self.binding_threshold(tau)))


def reg_horseshoe_hyperparameters(R: np.ndarray, F: np.ndarray, K: int,
                                  p0: int = 23, target_r2: float = 0.05,
                                  nu: float = 4.0, n_choice: str = "T",
                                  slab_scale: float | None = None,
                                  target_accept: float = 0.99,
                                  ) -> RegHorseshoeHyperparams:
    """
    Build the Regularised Horseshoe's hyperparameters from the data.

    Delta_b_bar, nu_Sigma and V_Sigma come from rescaled_hyperparameters();
    sigma from pooled_residual_scale(); tau_0 from P&V Eq. (3.12) exactly as
    for the plain horseshoe. Only slab_scale is new, and it is DERIVED, not
    supplied, so the robustness universes recalibrate automatically and the
    derivation lives next to the number.

    slab_scale : None (the default) derives s from p_0 as described in the
                 module docstring. Pass 2.0 to reproduce P&V's illustrative
                 default and demonstrate that it never binds; that run is
                 evidence, not a mistake, and belongs in the sensitivity table.
    nu         : 4, P&V. Dimensionless, so it transports across applications
                 where s does not.
    """
    if not 0 < p0 < K:
        raise ValueError(f"p0={p0} must lie strictly between 0 and K={K}")
    if n_choice not in ("T", "NT"):
        raise ValueError(f"n_choice={n_choice!r} must be 'T' or 'NT'")
    if nu <= 2.0:
        raise ValueError(f"nu={nu} must exceed 2, or E[c^2] = nu s^2/(nu-2) "
                         f"does not exist")

    N, T = R.shape
    base = rescaled_hyperparameters(R, F, K, target_r2=target_r2)
    sigma_pooled, _ = pooled_residual_scale(R, F)

    n = T if n_choice == "T" else N * T
    tau_0 = (p0 / (K - p0)) * sigma_pooled / np.sqrt(n)

    if slab_scale is None:
        mean_var_f = float(np.trace(np.cov(F.reshape(-1, K).T)) / K)
        signal_scale = np.sqrt(target_r2 * R.var() / (p0 * mean_var_f))
        slab_scale = float(signal_scale / np.sqrt(nu / (nu - 2.0)))

    sd_target = float(np.sqrt(base.V_b[0, 0] / (base.nu_b - K - 1)))

    return RegHorseshoeHyperparams(
        b_bar_bar=base.b_bar_bar,
        Delta_b_bar=base.Delta_b_bar,
        nu_Sigma=base.nu_Sigma,
        V_Sigma=base.V_Sigma,
        tau_0=float(tau_0),
        slab_scale=float(slab_scale),
        slab_df=float(nu),
        p0=int(p0),
        n_choice=n_choice,
        sigma_pooled=float(sigma_pooled),
        sd_target=sd_target,
        target_accept=float(target_accept),
    )


def lambda_tilde(lam: np.ndarray, tau: float | np.ndarray,
                 c: float | np.ndarray) -> np.ndarray:
    """
    The regularised local scale, P&V Eq. (2.8):

        lambda~^2 = c^2 lambda^2 / (c^2 + tau^2 lambda^2)

    Returns lambda~ (not its square). Used by build_model and by the
    diagnostics; kept here as a plain NumPy function so it can be tested
    against the PyTensor version and evaluated on saved draws.

    The two limits are worth checking numerically and were asserted against
    an independent NumPy reference: as c -> infinity, lambda~ -> lambda (the plain
    horseshoe); as tau*lambda >> c, lambda~ -> c/tau, so tau*lambda~ -> c and
    the prior becomes N(0, c^2) regardless of how large lambda is. That second
    limit is the whole point: it is what caps the tails.
    """
    lam2 = np.asarray(lam, dtype=float) ** 2
    c2 = np.asarray(c, dtype=float) ** 2
    return np.sqrt(c2 * lam2 / (c2 + np.asarray(tau, dtype=float) ** 2 * lam2))


def _sigma_reference(R: np.ndarray, F: np.ndarray):
    """
    Sigma's reference point and per-entry coordinate scales.

    IDENTICAL to the plain horseshoe's, deliberately imported rather than
    re-derived so the two models cannot drift apart in Sigma's
    parameterisation. See src/nuts/horseshoe.py for why the vector is centred
    at the OLS residual covariance and scaled per entry.
    """
    from src.nuts.horseshoe import _sigma_reference as _hs_sigma_reference
    return _hs_sigma_reference(R, F)


def build_model(R: np.ndarray, F: np.ndarray, hp: RegHorseshoeHyperparams):
    """
    The PyMC model, returned with its starting point.

    Structurally identical to the plain horseshoe's build_model with ONE
    change: lambda is replaced by lambda~ in theta's scale.

        caux        ~ InvGamma(nu/2, nu/2)
        c            = slab_scale * sqrt(caux)
        lambda~^2    = c^2 lambda^2 / (c^2 + tau^2 lambda^2)
        theta        = z * lambda~ * tau

    The caux parameterisation is Piironen & Vehtari's own (Appendix C.1):
    giving c^2 an Inv-Gamma(nu/2, nu s^2/2) prior is the same as giving
    caux an Inv-Gamma(nu/2, nu/2) prior and setting c = s sqrt(caux), and the
    latter keeps the sampled coordinate at unit scale, which is the whole
    reason this project's coordinates are non-dimensionalised. E[caux] =
    (nu/2)/(nu/2 - 1) = 2 at nu = 4, so E[c] = sqrt(2) s, and the slab a
    reader should be quoted is E[c], not s.

    P&V's Appendix C.2 parameterisation was tested for the plain horseshoe and
    gave no reduction in divergences (13 against 16, within binomial noise) at
    1.9x the wall clock, so C.1 is used for both models.

    THE NESTING, verified numerically rather than asserted:
    as c -> infinity, lambda~ -> lambda and this model becomes the plain
    horseshoe EXACTLY. That is what makes models 3 and 4 a controlled pair:
    one hyperparameter apart, with one limit collapsing the other.

    Every sampled coordinate is O(1) by construction: b_bar_z, z, log lambda,
    log tau_z, log caux and Sigma_packed_z.

    Sigma_packed_z is pm.Flat, never a proper prior: pm.Potential ADDS to the
    total logp rather than replacing a declared variable's own prior, and a
    pm.Normal there biased E[Sigma] by 8-11 MCSE.
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

        caux = pm.InverseGamma("caux", alpha=hp.slab_df / 2.0,
                               beta=hp.slab_df / 2.0)
        c = pm.Deterministic("c", hp.slab_scale * pm.math.sqrt(caux))

        lam2 = lam ** 2
        lam_tilde = pm.Deterministic(
            "lam_tilde", pm.math.sqrt(c ** 2 * lam2 / (c ** 2 + tau ** 2 * lam2)))

        b = pm.Deterministic("b", b_bar[None, :] + z * lam_tilde * tau)

        packed_z = pm.Flat("Sigma_packed_z", shape=packed0.size)
        Sigma, sigma_logp = inverse_wishart_cholesky_logp(
            packed0 + packed_scale * packed_z, hp.nu_Sigma, hp.V_Sigma)
        Sigma = pm.Deterministic("Sigma", Sigma)
        pm.Potential("Sigma_prior", sigma_logp)

        pm.Potential("likelihood", sur_log_likelihood_pytensor(R, F, b, Sigma))

    initvals = {
        "b_bar_z": B_ols.mean(axis=0) / sd_bbar,
        "z": np.zeros((N, K)),
        "lam": np.ones((N, K)),
        "tau_z": 1.0,
        "caux": 1.0,
        "Sigma_packed_z": np.zeros(packed0.size),
    }
    return model, initvals


def transformed_point(model, initvals: dict) -> dict:
    """
    Convert an untransformed starting point into the dictionary PyMC's
    compiled logp and dlogp functions expect.

    lam, tau_z and caux are positive, so PyMC samples their logs. Doing the
    conversion explicitly, and asserting the key set, is more robust across
    PyMC versions than calling the transform objects, and it fails loudly if
    a variable is renamed rather than silently evaluating at the default point.

    pm.sample() takes the UNTRANSFORMED initvals; this is only for evaluating
    logp and gradients directly, as the log-density checks do.
    """
    point = model.initial_point()
    expected = {"b_bar_z", "z", "lam_log__", "tau_z_log__", "caux_log__",
                "Sigma_packed_z"}
    if set(point) != expected:
        raise ValueError(f"model has value variables {sorted(point)}, "
                         f"expected {sorted(expected)}")
    point["b_bar_z"] = np.asarray(initvals["b_bar_z"], dtype=float)
    point["z"] = np.asarray(initvals["z"], dtype=float)
    point["lam_log__"] = np.log(np.asarray(initvals["lam"], dtype=float))
    point["tau_z_log__"] = np.log(np.asarray(initvals["tau_z"], dtype=float))
    point["caux_log__"] = np.log(np.asarray(initvals["caux"], dtype=float))
    point["Sigma_packed_z"] = np.asarray(initvals["Sigma_packed_z"], dtype=float)
    return point


@dataclass
class RegHorseshoeDraws:
    """
    Posterior draws from one chain. Mirrors HorseshoeDraws field for field,
    plus the two the slab adds, so io.py, convergence.py and the evaluation
    code work unchanged.

    B          : (n, N, K)  b_i = b_bar + theta_i. The backtest's estimand
    b_bar      : (n, K)
    lam_local  : (n, N, K)  the RAW local scales, before the slab
    lam_tilde  : (n, N, K)  the REGULARISED local scales actually used
    tau        : (n,)
    c          : (n,)       the slab width, = slab_scale * sqrt(caux)
    Sigma      : (n, N, N)

    BOTH lam_local AND lam_tilde are stored, and that is the point of this
    model's storage. The binding diagnostic is the ratio between them: where
    lam_tilde < lam_local the slab is doing something, and where they coincide
    this model IS the plain horseshoe. Storing only one would make the
    model's central claim unverifiable from the saved output.

    At 1,500 draws that is roughly 132 MB per chain, against the plain
    horseshoe's 91 MB; the extra 41 MB is lam_tilde. Acceptable, and it
    buys the one diagnostic the model exists to produce.

    NUTS sample statistics, one per retained draw:
    diverging, tree_depth, n_steps, step_size, energy.

    NOT stored, because exactly reconstructible: z = (B - b_bar)/(lam_tilde *
    tau), and the shrinkage factors kappa, which are a deterministic function
    of lam_tilde, tau and the data.
    """

    B: np.ndarray
    b_bar: np.ndarray
    lam_local: np.ndarray
    lam_tilde: np.ndarray
    tau: np.ndarray
    c: np.ndarray
    Sigma: np.ndarray
    diverging: np.ndarray
    tree_depth: np.ndarray
    n_steps: np.ndarray
    step_size: np.ndarray
    energy: np.ndarray
    meta: dict = field(default_factory=dict)


def ebfmi(energy: np.ndarray) -> float:
    """E-BFMI, Betancourt (2017). Below ~0.3 indicates the sampler is
    struggling to move between energy levels: a failure distinct from
    divergences. Computed here rather than through arviz, whose az.bfmi
    raises a TypeError on version 1.2.0."""
    e = np.asarray(energy, dtype=float).ravel()
    return float(np.sum(np.diff(e) ** 2) / np.sum((e - e.mean()) ** 2))


def shrinkage_factors(draws: RegHorseshoeDraws, F: np.ndarray,
                      hp: RegHorseshoeHyperparams) -> np.ndarray:
    """
    kappa_ij = 1 / (1 + n sigma^-2 tau^2 lambda~_ij^2 s_j^2).

    Uses lam_tilde, NOT lam_local; the regularised scale is the one that
    enters theta's prior, so it is the one that determines shrinkage. Using
    the raw lambda here would overstate how free the tail coefficients are and
    would make model 4's kappa look identical to model 3's by construction.

    s_j is the root MEAN SQUARE of predictor j, not its standard deviation:
    the intercept column has mean 1 and sd exactly 0 and would otherwise be
    assigned no information at all. See horseshoe.implied_m_eff.

    Returns (n_draws, N, K). Not stored, because it is an exact function of
    fields that are.
    """
    N, T, K = F.shape
    s = np.sqrt((F ** 2).mean(axis=1))
    a = (np.sqrt(T) / hp.sigma_pooled) * draws.tau[:, None, None] * draws.lam_tilde * s
    return 1.0 / (1.0 + a ** 2)


def binding_summary(draws: RegHorseshoeDraws, hp: RegHorseshoeHyperparams,
                    tol: float = 0.99) -> dict:
    """
    HOW MUCH IS THE SLAB ACTUALLY DOING? The diagnostic this model exists to
    produce, and the one whose failure is silent.

    Reports the POSTERIOR binding fraction, computed from the draws, against
    the PRIOR prediction from hp.binding_fraction(). The two should be
    compared: the prior figure is 12.91% evaluated at tau_0 but only about
    0.70% at the measured posterior tau, because the data moves tau to roughly
    5% of tau_0 and the binding threshold c/tau moves inversely.

    "Binding" is defined as lam_tilde / lam_local < tol. At tol = 0.99 that
    means the slab has shrunk a coefficient's local scale by more than 1%,
    a deliberately generous threshold, because P&V's design shrinks EVERY
    coefficient at least slightly (lambda~/lambda is 0.9998 at lambda = 0.1
    and 0.960 at lambda = 1 with the calibrated c), so a strict "any shrinkage
    at all" test would return 100% and mean nothing.

    IF frac_binding IS NEAR ZERO the regularised horseshoe has collapsed into
    the plain horseshoe and any reported difference between models 3 and 4 is
    noise. That is exactly what P&V's illustrative s = 2.0 would produce here
    (0.017 of 3,600 coefficients), which is why s is recalibrated.
    """
    ratio = draws.lam_tilde / draws.lam_local
    binding = ratio < tol
    heavy = ratio < 0.5
    return {
        "frac_binding": float(binding.mean()),
        "n_binding_per_draw": float(binding.sum(axis=(1, 2)).mean()),
        "frac_halved": float(heavy.mean()),
        "median_ratio": float(np.median(ratio)),
        "min_ratio": float(ratio.min()),
        "prior_frac_at_tau0": hp.binding_fraction(),
        "prior_frac_at_posterior_tau": hp.binding_fraction(float(draws.tau.mean())),
        "c_mean": float(draws.c.mean()),
        "c_over_tau": float(draws.c.mean() / draws.tau.mean()),
        "tol": tol,
    }


def run_nuts(R: np.ndarray, F: np.ndarray, hp: RegHorseshoeHyperparams,
             n_draws: int = 1500, n_tune: int = 1000, chains: int = 4,
             cores: int = 1, seed0: int = 0, progressbar: bool = True
             ) -> list[RegHorseshoeDraws]:
    """
    Sample the Regularised Horseshoe and return ONE RegHorseshoeDraws PER
    CHAIN. Interface identical to the plain horseshoe's run_nuts.

    cores DEFAULTS TO 1, unlike the horseshoe's 4. PyMC's multiprocessing
    killed a worker twice in this project: an EOFError with no traceback at
    reduced dimensions, and a production run that lost a process after 70
    minutes. Every single-core run has completed. Sequential chains cost about
    3 hours instead of 45 minutes for a production run, and that is the trade
    this project has already made.

    Chain k is seeded seed0 + k, and the RUNNER must use seed0 + k in the
    FILENAME too: run_bayesian_lasso.py uses the loop index, so --seed0 4
    silently overwrote chains 0-3.

    Sampler settings come from hp, not from arguments here, so a backtest fit
    cannot silently differ from the production run: target_accept = 0.99,
    init = "adapt_diag", max_treedepth = 10. NEVER "jitter+adapt_diag": its
    U(-1,1) jitter is ~1,300 prior standard deviations on b_bar and drives the
    step size to 9.1e-22 in the plain horseshoe, a sampler that never moves
    while appearing to fit quickly.

    n_draws is RETAINED draws per chain, EXCLUDING tuning.

    BUDGET: the plain horseshoe needed 4 x 1,500 because TAU had bulk ESS of
    only ~14 per 200 draws while B was superefficient. Whether the slab
    changes that is unknown (c is a new global parameter and may itself mix
    slowly), so the first production run's ESS on tau AND c should be read
    before this default is trusted for the robustness universes.
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
        ratio = po["lam_tilde"].values[k] / po["lam"].values[k]
        out.append(RegHorseshoeDraws(
            B=po["b"].values[k],
            b_bar=po["b_bar"].values[k],
            lam_local=po["lam"].values[k],
            lam_tilde=po["lam_tilde"].values[k],
            tau=po["tau"].values[k],
            c=po["c"].values[k],
            Sigma=po["Sigma"].values[k],
            diverging=div, tree_depth=depth, n_steps=steps,
            step_size=ss["step_size"].values[k], energy=energy,
            meta={"seed": seeds[k], "chain_index": k,
                  "n_draws": n_draws, "n_tune": n_tune,
                  "N": N, "T": T, "K": K,
                  "tau_0": hp.tau_0, "p0": hp.p0, "n_choice": hp.n_choice,
                  "slab_scale": hp.slab_scale, "slab_df": hp.slab_df,
                  "slab_sd_prior": hp.slab_sd,
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
                  "frac_binding_0p99": float((ratio < 0.99).mean()),
                  "sampling_seconds_all_chains": elapsed},
        ))

    total_div = sum(d.meta["divergences"] for d in out)
    bfmi_str = " ".join(f"{d.meta['ebfmi']:.2f}" for d in out)
    depth_str = " ".join(f"{d.meta['tree_depth_mean']:.2f}" for d in out)
    bind_str = " ".join(f"{100*d.meta['frac_binding_0p99']:.1f}%" for d in out)
    print(f"  {chains} chains x {n_draws} draws ({n_tune} tune) in {elapsed:.0f}s")
    print(f"  divergences {total_div} of {chains * n_draws} "
          f"({100 * total_div / (chains * n_draws):.1f}%) | "
          f"tree depth {depth_str} | E-BFMI {bfmi_str}")
    print(f"  slab binding (lam_tilde/lam < 0.99): {bind_str}")
    return out


