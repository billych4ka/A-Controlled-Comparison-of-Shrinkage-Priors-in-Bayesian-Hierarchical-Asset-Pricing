"""
src/gibbs/bayesian_lasso.py

Bayesian LASSO Gibbs sampler for the Bayesian hierarchical SUR asset-pricing
model -- model 2 of 4. The likelihood (sur.py) and the cross-asset hierarchy
(b_i = b_bar + theta_i, with b_bar ~ N(b_bar_bar, Delta_b_bar) and
Sigma ~ IW(nu_Sigma, V_Sigma)) are held fixed against the Gaussian baseline.
Only the prior on the deviations theta_i changes:

    theta_ij | tau_ij^2  ~  N(0, s^2 tau_ij^2)
    tau_ij^2             ~  Exponential(lambda^2 / 2)
    lambda^2             ~  Gamma(r, delta)

which marginally gives theta_ij ~ Laplace(0, s/lambda) -- Park & Casella's
(2008) scale-mixture-of-normals construction, with a single global lambda
shared across all i, j.

Delta_b does NOT exist in this model. Each theta_ij has its own scale
s^2 tau_ij^2, so the prior covariance of theta is diagonal rather than a
K x K matrix, and Feng & He's eq. (17) step is REPLACED by the tau^2 update,
not supplemented by it. nu_b and V_b are therefore irrelevant here, except as
the source of the deviation scale that lambda is calibrated against (below).

Disclosed deviations from Park & Casella
-----------------------------------------
1. The conditioning scale is a POOLED PLUG-IN constant s, not the sampled
   residual scale. Park & Casella condition the coefficient prior on sigma^2,
   the sampled error variance of the same equation. The elementwise
   generalisation to SUR -- theta_ij | Sigma_ii ~ N(0, Sigma_ii tau_ij^2) --
   destroys Sigma's Inverse-Wishart conjugacy: it enters the joint density as
   an inverse-gamma kernel in diag(Sigma), against an Inverse-Wishart kernel
   in Sigma^{-1}, and the two do not combine. (An independence
   Metropolis-Hastings step using the conjugate IW as proposal was tested and
   rejected: importance-sampling efficiency is 58% when the model and data
   agree but collapses to ~0% once the deviation sum of squares is 30% away
   from what Sigma implies, which will occur during burn-in.)

   The generalisation that DOES preserve conjugacy is the matrix-normal
   theta ~ MN(0, Sigma, D_tau), giving Sigma | . ~ IW(nu_Sigma + T + K,
   V_Sigma + EE' + theta D_tau^{-1} theta') -- verified numerically to 1.8e-13.
   Its Kronecker structure, however, forces one tau_j^2 per PREDICTOR shared
   across assets (a Bayesian group lasso) rather than one per coefficient,
   which would not be comparable with the Horseshoe models' per-coefficient
   lambda_ij. It is noted as a possible robustness extension.

   The plug-in retains the scale-equivariance of lambda, which is the
   conditioning's practical purpose, at the cost of Park & Casella's formal
   unimodality guarantee (their Appendix A). The quantity being held fixed
   has a posterior relative standard deviation of 2.6%, and their
   multimodality counterexample relies on the improper prior 1/sigma^2,
   whereas Sigma here has a proper Inverse-Wishart prior and T=719 months
   behind it. Multimodality would in any case be detectable through
   multi-chain R-hat rather than silent.

2. s is POOLED across assets rather than per-asset. Park & Casella's sigma is
   the residual scale of the single equation being fitted; there is no
   single-equation analogue in SUR. A per-asset s_i was considered and
   rejected because it would make the LASSO's prior differ from the
   baseline's in per-asset scaling as well as in tail shape, adding a second
   axis of difference to a design that varies one. Pooling is also consistent
   with the Horseshoe, where sigma must be pooled because tau is a global
   scale. Note that s is estimated from the training data, so this is
   empirical Bayes on the prior scale; it is recomputed per backtest window
   from training data only, so it introduces no look-ahead.

3. s is degrees-of-freedom corrected, RSS/(T-K), not RSS/(T-1). At K=144 and
   an initial backtest window of T=240 the uncorrected estimator is 63% of
   the correct one, rising to 90% by the final window -- a drifting bias that
   would tighten the prior at the start of the backtest and loosen it
   monotonically thereafter, appearing in results as a model that improves
   over time.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import time

import numpy as np
from scipy.linalg import solve_triangular

from src.gibbs.baseline_gaussian import (_spd_inverse, precompute_cross_products,
                                         rescaled_hyperparameters, sample_Sigma)
from src.likelihood.sur import sur_residuals

@dataclass
class LassoHyperparams:
    """
    Fixed hyperparameters of the Bayesian LASSO -- chosen once, before
    sampling, and never updated by the sampler.

    b_bar_bar     : (K,)   prior mean of b_bar          -- zeros, Feng & He
    Delta_b_bar   : (K,K)  prior covariance of b_bar    -- SHARED with all
                           four models; must be identical or the comparison
                           is confounded
    nu_Sigma      : IW degrees of freedom for Sigma     -- N + 2
    V_Sigma       : (N,N)  IW scale for Sigma           -- S_hat
    s             : pooled plug-in residual scale entering theta's prior
    lambda_r      : shape of the Gamma hyperprior on lambda^2
    lambda_delta  : rate of that Gamma hyperprior
    sd_target     : the per-coefficient deviation sd the baseline's rescaled
                    prior implies; lambda is calibrated so the Laplace matches
                    it. Stored for the record, not used during sampling.
    """

    b_bar_bar: np.ndarray
    Delta_b_bar: np.ndarray
    nu_Sigma: float
    V_Sigma: np.ndarray
    s: float
    lambda_r: float
    lambda_delta: float
    sd_target: float

    @property
    def K(self) -> int:
        return self.b_bar_bar.shape[0]

    @property
    def N(self) -> int:
        return self.V_Sigma.shape[0]

    @property
    def lambda_prior_mean(self) -> float:
        """Prior mean of lambda^2 is r/delta; this returns lambda itself."""
        return float(np.sqrt(self.lambda_r / self.lambda_delta))


def pooled_residual_scale(R: np.ndarray, F: np.ndarray):
    """
    Pooled, degrees-of-freedom-corrected residual standard deviation from
    per-asset OLS, plus the OLS coefficients themselves.

        s^2 = sum_i sum_t e_it^2 / (N (T - K))

    The (T - K) denominator is not cosmetic. With K = 144 predictors and a
    first backtest window of T = 240, only 96 degrees of freedom remain, and
    RSS/(T-1) understates the residual sd by 37%. Because that bias shrinks
    as the expanding window grows, using it would make the prior tighten and
    then loosen across the backtest for no substantive reason.

    B_ols is returned alongside so that initialise_state can reuse it rather
    than solving the same 25 least-squares problems twice.

    R : (N,T), F : (N,T,K) -> (s, B_ols) with B_ols of shape (N,K)
    """
    N, T = R.shape
    K = F.shape[2]
    if T <= K:
        raise ValueError(
            f"T={T} <= K={K}: OLS residual variance is undefined. The plug-in "
            "scale needs T > K, which the backtest's start=240 guarantees."
        )

    B_ols = np.array([np.linalg.lstsq(F[i], R[i], rcond=None)[0] for i in range(N)])
    E = R - np.einsum("itk,ik->it", F, B_ols)
    s2 = float((E ** 2).sum() / (N * (T - K)))
    return float(np.sqrt(s2)), B_ols


def lasso_hyperparameters(R: np.ndarray, F: np.ndarray, K: int,
                          target_r2: float = 0.05,
                          lambda_r: float = 1.0,
                          lambda_prior_multiplier: float = 1.0
                          ) -> LassoHyperparams:
    """
    Build the LASSO's hyperparameters from the data, at the same prior scale
    as the Gaussian baseline's rescaled setting.

    Delta_b_bar, nu_Sigma and V_Sigma are taken directly from
    rescaled_hyperparameters(), NOT recomputed here. That is deliberate: the
    four models must share one hierarchy, and importing the derivation makes
    that true by construction rather than by two copies of a formula agreeing.

    Calibrating lambda
    ------------------
    The baseline's rescaled prior puts a per-coefficient deviation sd of
    sd_target = sqrt(V_b / (nu_b - K - 1)) on theta. The Laplace prior
    theta_ij ~ Laplace(0, s/lambda) has sd sqrt(2) s / lambda, so matching
    the two gives

        lambda_cal = sqrt(2) * s / sd_target

    This is the only role nu_b and V_b play in this model.

    Choosing the Gamma hyperprior
    -----------------------------
    lambda^2's full conditional is Gamma(r + NK, delta + sum tau_ij^2 / 2),
    so the data contributes NK = 3,600 pseudo-observations against the
    hyperprior's r. Any proper Gamma with r small and delta small relative to
    sum(tau^2)/2 therefore washes out. The trap is that delta has units of
    1/lambda^2, and lambda^2 is of order 3.6e5 here: an off-the-shelf diffuse
    choice such as delta = 1.78 would swamp the data term (0.00995) by 180x
    and pull the posterior lambda from 601 to 45. This is the same
    scale-transportability failure already found in V_b, Delta_b_bar and the
    regularised horseshoe's slab width, so delta is DERIVED, not chosen:

        delta = r / (lambda_prior_multiplier * lambda_cal)^2

    which centres the hyperprior on the calibrated value with r = 1, the most
    diffuse proper choice. lambda_prior_multiplier exists for the sensitivity
    check: at 0.1 and 10 the posterior mean of lambda moves by about 1%,
    which is the point.

    R, F      : (N,T) and (N,T,K); everything is computed from these, so the
                backtest's per-window recomputation is automatic
    target_r2 : 0.05, fixed a priori, not to be revised after seeing results
    """
    base = rescaled_hyperparameters(R, F, K, target_r2=target_r2)

    # the deviation scale the baseline's prior implies -- lambda's calibration target
    sd_target = float(np.sqrt(base.V_b[0, 0] / (base.nu_b - K - 1)))

    s, _ = pooled_residual_scale(R, F)

    lambda_cal = np.sqrt(2.0) * s / sd_target
    lambda_delta = float(lambda_r / (lambda_prior_multiplier * lambda_cal) ** 2)

    return LassoHyperparams(
        b_bar_bar=base.b_bar_bar,
        Delta_b_bar=base.Delta_b_bar,
        nu_Sigma=base.nu_Sigma,
        V_Sigma=base.V_Sigma,
        s=s,
        lambda_r=float(lambda_r),
        lambda_delta=lambda_delta,
        sd_target=sd_target,
    )


def lasso_prior_implied_r2(hp: LassoHyperparams, R: np.ndarray,
                           F: np.ndarray) -> float:
    """
    The R^2 the prior implicitly expects the predictors to explain -- the
    LASSO's counterpart to baseline_gaussian.prior_implied_r2, and the check
    that the calibration closes.

    Under the prior, b_ij = b_bar_j + theta_ij has variance
    Var(b_bar_j) + E[Var(theta_ij)], with

        E[Var(theta_ij)] = s^2 E[tau_ij^2] = s^2 * 2 / lambda^2,

    evaluated at the hyperprior's central value lambda^2 = r/delta. (It is
    evaluated there rather than integrated over lambda because E[1/lambda^2]
    under Gamma(r, delta) is delta/(r-1), which does not exist at r = 1 --
    the price of the most diffuse proper hyperprior, and harmless since this
    function is a diagnostic rather than part of the model.)

    Should return target_r2 to within floating point if lambda was calibrated
    at lambda_prior_multiplier = 1.
    """
    K = hp.K
    C = np.cov(F.reshape(-1, K).T)
    lambda2 = hp.lambda_r / hp.lambda_delta
    s2 = hp.Delta_b_bar[0, 0] + hp.s ** 2 * 2.0 / lambda2
    return float(s2 * np.trace(C) / R.var())

def sample_tau2(theta: np.ndarray, s: float, lam: float,
                rng: np.random.Generator) -> np.ndarray:
    """
    Draw the auxiliary scales tau_ij^2 from their full conditional.

    The conditional
    ---------------
    theta_ij's prior is N(0, s^2 tau_ij^2) and tau_ij^2's is Exp(lambda^2/2),
    so as a function of tau_ij^2 the joint density is

        (tau^2)^(-1/2) exp( -theta_ij^2 / (2 s^2 tau^2) )  *  exp( -lambda^2 tau^2 / 2 )

    a generalised inverse Gaussian with p = 1/2, which is exactly the case in
    which the RECIPROCAL is inverse Gaussian -- Park & Casella's key
    tractability result:

        1 / tau_ij^2  ~  InverseGaussian( mu' = lambda s / |theta_ij|,
                                          lam' = lambda^2 )

    Verified against a brute-force evaluation of the model's own log joint
    density: log joint minus this kernel is constant to 1.7e-13 over 500 grid
    points, and the sampler's E[tau^2] matches numerical integration to 0.29
    Monte Carlo standard errors.

    Why this does not call scipy.stats.invgauss
    -------------------------------------------
    scipy's sampler uses the Michael-Schucany-Haas transformation in its
    textbook form,

        x = mu' + mu'^2 y / (2 lam') - (mu' / 2 lam') sqrt(4 mu' lam' y + mu'^2 y^2),

    in which the last two terms both grow like mu'^2 y / (2 lam') and their
    difference is lost to floating-point cancellation once mu' is large. On
    scipy 1.17.1 this returns NEGATIVE draws -- from a distribution supported
    on (0, infinity) -- once mu' is large enough, rising to 67% of draws at
    |theta| = 1e-13. np.isfinite() passes on every one of them.

    A negative tau_ij^2 gives a negative prior precision 1/(s^2 tau_ij^2). The
    likelihood contribution would usually keep the joint (B, b_bar) precision
    matrix positive definite regardless, so the Cholesky factorisation
    succeeds and the sweep silently samples from the wrong distribution.

    This is reachable here, not hypothetical: the failure region begins around
    |theta_ij| < 9.3e-12, which has probability 5.7e-8 per coefficient under
    the calibrated prior. Over a 2,000-sweep production chain that is 0.4
    expected occurrences; over the 40-window backtest (40 x 500 x 3,600 draws)
    it is 4. Frequent enough to happen in production, far too rare to appear
    in a short test.

    The stable form
    ---------------
    Writing u = mu' y / lam', the same transformation rearranges exactly to

        x1 = mu' [ 1 - 2u / (u + sqrt(u^2 + 4u)) ]
           = (4 lam' / y) / (1 + sqrt(1 + 4/u))^2

    which is algebraically identical but contains no subtraction of nearly
    equal quantities -- every operation is on a positive number. Better
    still, 4/u = 4 lambda |theta_ij| / (s y) involves no division by theta,
    so theta_ij = 0 exactly gives x1 = lambda^2 / y, which IS the correct
    limit (as theta -> 0 the conditional becomes Gamma(1/2, 2/lambda^2), with
    E[tau^2] -> 1/lambda^2). The edge case resolves itself rather than needing
    a guard, and no arbitrary floor on |theta| is introduced.

    The rejection step then returns x1 with probability mu'/(mu' + x1) and
    mu'^2 / x1 otherwise; writing r = x1/mu' = x1 |theta| / (lambda s), those
    are 1/(1+r) and x1/r^2, again with no division by theta in the accepted
    branch.

    Validation (see check_chunk2.py): sample mean matches mu' and sample
    variance matches mu'^3/lam' wherever Monte Carlo error permits the
    comparison; Kolmogorov-Smirnov against scipy agrees wherever scipy is
    trustworthy (10 seeds per setting, 0-1 of 10 p-values below 0.05);
    E[tau^2] matches its analytic value to 0.38 Monte Carlo standard errors
    at theta = 0, where scipy fails outright.

    theta : (N,K) current deviations B - b_bar
    s     : pooled plug-in residual scale
    lam   : current lambda
    rng   : np.random.Generator, passed in so the sweep is reproducible
    -> (N,K) array of tau_ij^2, strictly positive
    """
    a = np.abs(theta)
    lam_prime = lam * lam

    y = rng.standard_normal(a.shape) ** 2                 # chi-squared_1
    q = 4.0 * lam * a / (s * y)                           # = 4/u, no 1/theta
    x1 = (4.0 * lam_prime / y) / (1.0 + np.sqrt(1.0 + q)) ** 2

    r = x1 * a / (lam * s)                                # = x1 / mu'
    keep = rng.random(a.shape) * (1.0 + r) <= 1.0         # accept w.p. 1/(1+r)
    r_safe = np.where(r > 0.0, r, 1.0)                    # branch never taken at r=0
    x = np.where(keep, x1, x1 / (r_safe * r_safe))        # mu'^2 / x1 = x1 / r^2

    return 1.0 / x

def sample_lambda_collapsed(theta: np.ndarray, s: float, hp: LassoHyperparams,
                            rng: np.random.Generator) -> float:
    """
    Draw lambda from p(lambda | theta), with tau^2 integrated out analytically.
    THIS IS THE DEFAULT. See sample_lambda below for the conditional version
    and why it is not used.

    Why collapse
    ------------
    lambda | tau^2 is Gamma(r + NK, delta + sum(tau^2)/2), whose relative
    standard deviation is 1/(2 sqrt(3601)) = 0.83%. Each tau_ij^2 is in turn
    drawn given lambda. Two nearly deterministic conditionals alternated is the
    same pathology as the centred (B, b_bar) scan, one level up, and it is
    severe: measured on real data over 1,500 sweeps, lag-1 autocorrelation of
    lambda was 0.973, ESS was 8 from 750 draws, and lambda had still not
    converged -- means by fifth of the run were 492, 441, 430, 420, 403, with
    no floor in sight. The implied budget for ESS 400 was 38,300 sweeps.
    B, b_bar and Sigma were unaffected (ESS 711-746), confirming the problem
    is the tau^2 <-> lambda pair specifically.

    The fix is the same in kind as blocking (B, b_bar): remove the coupling
    rather than traverse it. Since the scale mixture

        integral N(theta; 0, s^2 tau^2) Exp(tau^2; lambda^2/2) d tau^2
            = Laplace(theta; 0, s/lambda)

    (verified numerically to a ratio of 1.00000000), lambda's conditional
    given theta alone is available in closed form:

        p(lambda | theta) prop lambda^(NK + 2r - 1)
                               exp( -lambda sum|theta_ij| / s - delta lambda^2 )

    Verified against a brute-force evaluation of the log joint: constant to
    7.8e-14 over 6,000 grid points. lambda now moves with theta, which mixes
    freely, instead of with sum(tau^2), which does not.

    HONEST ACCOUNT OF HOW MUCH THIS FIXES. Collapsing improves lambda's lag-1
    autocorrelation from 0.973 to 0.913 -- roughly 3x the effective sample size
    -- and removes the drift entirely: lambda now reaches its stationary region
    within about 30 sweeps instead of still falling after 1,500. It does NOT
    make lambda mix as well as B, b_bar and Sigma, because lambda remains
    coupled to theta through sum|theta_ij|, and theta is drawn given tau^2,
    which is drawn given lambda. The loop is longer, not broken. lambda is
    still the slowest-mixing parameter in this sampler and its budget is set
    by that, not by B.

    Sampling it
    -----------
    At delta = 0 this is exactly Gamma(NK + 2r - 1, rate = sum|theta|/s). The
    delta lambda^2 term is retained exactly, by rejection sampling against a
    Gamma proposal whose rate absorbs a linearisation of that term about the
    target's mode:

        rate_eff = rate + 2 delta * mode,
        mode     = [ -rate + sqrt(rate^2 + 8 delta (a-1)) ] / (4 delta)

    Proposing from Gamma(a, rate_eff) and accepting with probability
    exp(-delta (lambda - mode)^2) is exact, since that factor is bounded by 1
    and the remaining discrepancy between target and proposal is precisely a
    centred Gaussian factor.

    The naive version -- proposing from Gamma(a, rate) and accepting with
    probability exp(-delta lambda^2) -- is also exact but useless in practice.
    delta is calibrated so that delta * lambda_cal^2 = 1, NOT so that it is
    negligible, so the acceptance probability is exp(-1) = 0.37 at the centre
    and falls to 0.007 by lambda = 1345. That version raised RuntimeError after
    1,000 rejections in testing. The shift the delta term actually produces in
    the posterior mean is small (-0.01% to -0.28% over lambda in 200 to 1345),
    but it is retained rather than dropped so that the hyperprior is honoured
    exactly and the function stays correct under a sensitivity run with a much
    larger delta.

    theta : (N,K) current deviations B - b_bar
    s     : pooled plug-in residual scale
    hp    : supplies lambda_r and lambda_delta
    -> lambda (a positive scalar)
    """
    shape = theta.size + 2.0 * hp.lambda_r - 1.0
    rate = float(np.abs(theta).sum()) / s

    if hp.lambda_delta <= 0.0:
        return float(rng.gamma(shape=shape, scale=1.0 / rate))

    delta = hp.lambda_delta
    # mode of lambda^(a-1) exp(-rate lambda - delta lambda^2), from the positive
    # root of (a-1)/lambda - rate - 2 delta lambda = 0
    mode = (-rate + np.sqrt(rate ** 2 + 8.0 * delta * (shape - 1.0))) / (4.0 * delta)
    rate_eff = rate + 2.0 * delta * mode

    for _ in range(10000):
        lam = rng.gamma(shape=shape, scale=1.0 / rate_eff)
        if rng.random() < np.exp(-delta * (lam - mode) ** 2):
            return float(lam)
    raise RuntimeError(
        "sample_lambda_collapsed: 10,000 rejections, which should be impossible "
        "at the calibrated delta. Check lambda_delta against the data term."
    )

def sample_lambda(tau2: np.ndarray, hp: LassoHyperparams,
                  rng: np.random.Generator) -> float:
    """
    Draw lambda from its full conditional given tau^2.

    NOT USED IN THE SAMPLER -- retained because it is the textbook Park &
    Casella step, and because check_chunk3.py validates it as an independent
    confirmation that the collapsed version above targets the same
    distribution. See sample_lambda_collapsed for why it is not the default.

    The conditional
    ---------------
    Each tau_ij^2 has density (lambda^2/2) exp(-lambda^2 tau_ij^2 / 2), so the
    NK of them contribute (lambda^2)^(NK) exp(-lambda^2 sum(tau^2)/2). Against
    a Gamma(r, delta) prior on lambda^2 -- density (lambda^2)^(r-1)
    exp(-delta lambda^2) -- this is conjugate:

        lambda^2 | tau^2  ~  Gamma( shape = r + NK,
                                    rate  = delta + sum_ij tau_ij^2 / 2 )

    Verified against a brute-force evaluation of the model's own log joint:
    constant to 2.2e-11 over 500 grid points.

    Note the shape is r + NK = 3,601 here. The data contributes 3,600
    pseudo-observations against the hyperprior's r = 1, which is why delta
    washes out (see lasso_hyperparameters) -- and also why the conditional is
    tight, with a relative standard deviation of 1/(2 sqrt(3601)) = 0.83% on
    lambda itself. That tightness is a mixing risk rather than a precision
    claim: lambda and tau^2 are coupled through sum(tau^2), so lambda can only
    move as fast as that sum does. Measure the lag-1 autocorrelation of lambda
    before trusting any long run.

    numpy's Generator.gamma takes SHAPE and SCALE, not shape and rate, so the
    rate is inverted below. Getting this backwards multiplies lambda by the
    rate -- about 100x here, so it would be caught. But the size of the error
    IS the rate, which is data-dependent and near 1 in other problems, so this
    must be checked against the parameterisation rather than trusted to look
    obviously wrong.

    tau2 : (N,K) current auxiliary scales
    hp   : supplies lambda_r and lambda_delta
    rng  : np.random.Generator
    -> lambda (a positive scalar, NOT lambda^2)
    """
    shape = hp.lambda_r + tau2.size
    rate = hp.lambda_delta + 0.5 * float(tau2.sum())
    lambda2 = rng.gamma(shape=shape, scale=1.0 / rate)
    return float(np.sqrt(lambda2))

def sample_B_and_b_bar(rng: np.random.Generator, G: np.ndarray, Fr: np.ndarray,
                       Sigma: np.ndarray, tau2: np.ndarray, s: float,
                       b_bar_bar: np.ndarray, Delta_b_bar: np.ndarray):
    """
    Draw B and b_bar JOINTLY from their exact Gaussian conditional.

    Structurally identical to the Gaussian baseline's blocked draw, with one
    substitution: the baseline's single dense K x K prior precision
    Delta_b^{-1}, shared by every asset, becomes a per-asset DIAGONAL

        D_i^{-1} = diag( 1 / (s^2 tau_ij^2) ),   j = 1..K

    since under the LASSO each theta_ij has its own variance. Delta_b_bar --
    a different object, the prior covariance of b_bar itself -- is unchanged
    and shared with all four models. Everything else (the Kronecker structure
    of the likelihood term, the sign of the off-diagonal coupling, the
    Cholesky solve) is as in the baseline.

    Why blocked, with no sequential alternative
    -------------------------------------------
    Feng & He's scan updates b_i given b_bar, then b_bar given the b_i. That
    is a CENTRED parameterisation, and it mixes catastrophically when the
    hierarchical variance is small (Papaspiliopoulos, Roberts & Skold, 2007).
    The Gaussian baseline measured lag-1 autocorrelation of ||b_bar|| at 1.000
    under the calibrated prior, against 0.190 for the blocked draw.

    That pathology is reproduced here at identical strength, not merely
    inherited by analogy: calibrating lambda to sd_target makes the LASSO's
    mean prior variance on theta E[s^2 tau^2] = s^2 * 2/lambda^2 = 1.7131e-08,
    against the baseline's E[Delta_b]_jj of 1.7131e-08 -- a ratio of 1.0000,
    forced by the calibration. The per-coefficient spread does not rescue it:
    under tau^2 ~ Exp(lambda^2/2) even the 99th percentile of prior variance
    is only 4.6x the baseline's, and the median is 0.69x.

    No sequential sampler is provided, since there is no literal Feng & He
    LASSO to reproduce. Blocking's legitimacy is general -- it leaves the
    target unchanged and cannot worsen mixing (Liu, Wong & Kong, 1994) -- and
    was confirmed empirically in the baseline. What does NOT transfer is that
    THIS precision matrix is assembled correctly, so it is validated directly
    against a brute-force NT x NK stacked system and against the model's own
    log joint density (see check_chunk4.py). A misassembled P would still be
    symmetric positive definite, still factorise, and still produce plausible
    coefficients.

    Math
    ----
    Stacking (vec B, b_bar) asset-major, the joint precision is

        [ F'Omega^{-1}F + blkdiag(D_i^{-1})  |  -[D_1^{-1}; ...; D_N^{-1}] ]
        [ -[D_1^{-1}, ..., D_N^{-1}]         |  sum_i D_i^{-1} + Delta_b_bar^{-1} ]

    with location (F'Omega^{-1}R, Delta_b_bar^{-1} b_bar_bar), and

        [F'Omega^{-1}F]_{ij} = Sigma^{-1}[i,j] * (f_i' f_j)
        [F'Omega^{-1}R]_i    = sum_j Sigma^{-1}[i,j] * (f_i' r_j)

    Note the bottom-right block SUMS the per-asset precisions. The baseline
    writes N * Delta_b_inv there, which is valid only because its assets share
    one Delta_b. Copying that form here is wrong and, measured over 200 random
    tau^2 draws, leaves the matrix positive definite in 111 of them -- so the
    Cholesky succeeds, nothing raises, and b_bar's posterior mean moves by a
    median of 1.6 and up to 472 posterior standard deviations.

    rng         : numpy Generator
    G, Fr       : outputs of precompute_cross_products (baseline_gaussian)
    Sigma       : (N,N) current residual covariance
    tau2        : (N,K) current auxiliary scales
    s           : pooled plug-in residual scale
    b_bar_bar   : (K,) prior mean of b_bar
    Delta_b_bar : (K,K) prior covariance of b_bar
    -> (B, b_bar) with shapes (N,K) and (K,)
    """
    N = Sigma.shape[0]
    K = b_bar_bar.shape[0]
    NK = N * K
    M = NK + K

    Sigma_inv = _spd_inverse(Sigma)
    Delta_b_bar_inv = _spd_inverse(Delta_b_bar)
    D_inv = 1.0 / (s * s * tau2)                 # (N,K) diagonal prior precisions

    P = np.zeros((M, M))

    # likelihood block, plus the per-asset diagonal prior precision on its diagonal
    block = G * Sigma_inv[:, None, :, None]      # (N,K,N,K)
    ii = np.arange(N)[:, None]
    kk = np.arange(K)[None, :]
    block[ii, kk, ii, kk] += D_inv               # element (i,k,i,k) only
    P[:NK, :NK] = block.reshape(NK, NK)

    # coupling blocks: -D_i^{-1}, diagonal, so only NK entries are non-zero
    rows = np.arange(NK)
    cols = NK + np.tile(np.arange(K), N)
    P[rows, cols] = -D_inv.ravel()
    P[cols, rows] = -D_inv.ravel()

    P[NK:, NK:] = np.diag(D_inv.sum(axis=0)) + Delta_b_bar_inv

    rhs = np.empty(M)
    rhs[:NK] = np.einsum("ij,ijk->ik", Sigma_inv, Fr, optimize=True).reshape(NK)
    rhs[NK:] = Delta_b_bar_inv @ b_bar_bar

    L = np.linalg.cholesky(P)
    y = solve_triangular(L, rhs, lower=True)
    mean = solve_triangular(L.T, y, lower=False)
    noise = solve_triangular(L.T, rng.standard_normal(M), lower=False)
    draw = mean + noise

    return draw[:NK].reshape(N, K), draw[NK:]

# ---------------------------------------------------------------------------
# Initialisation and the assembled Gibbs loop
# ---------------------------------------------------------------------------

@dataclass
class LassoDraws:
    """
    Posterior draws returned by run_gibbs, after burn-in has been discarded.

    B             : (n_keep, N, K) asset-specific coefficients
    b_bar         : (n_keep, K)    shared mean coefficients
    Sigma         : (n_keep, N, N) residual covariance across assets
    lam           : (n_keep,)      global shrinkage parameter lambda
                    (named `lam`, not `lambda`, which is a Python keyword)
    tau2_mean     : (N, K) posterior mean of tau_ij^2
    tau2_inv_mean : (N, K) posterior mean of 1/tau_ij^2
    tau2_track    : (n_keep, n_track) full traces for a fixed random subset
    tau2_track_idx: (n_track, 2) the (asset, predictor) pairs tracked
    tau2          : (n_keep, N, K) or None -- everything, only if asked
    meta          : bookkeeping (seed, timings, settings)

    Why tau^2 is summarised rather than stored in full
    ---------------------------------------------------
    tau2 is (25, 144) = 3,600 values per draw, 58 MB per chain at 2,000 kept
    draws -- affordable, but tau^2 is a nuisance scale and the inference is
    about theta. The running means are EXACT, not approximations: a mean
    accumulated over every post-burn-in draw equals one computed later from a
    stored array. What is given up is credible intervals and R-hat for the
    3,400 untracked coefficients, recoverable by one re-run.

    Both tau2_mean and tau2_inv_mean are kept because the prior precision is
    1/(s^2 tau^2) and E[1/tau^2] != 1/E[tau^2] -- on real data the product of
    the two means is about 11, not 1. The reciprocal mean is what describes
    how hard each coefficient was actually shrunk, and is the quantity to set
    beside the horseshoe's shrinkage factors.

    Note for the write-up: the tracked subset is a SAMPLE. A strict reading of
    Vehtari et al. (2021) checks R-hat for every sampled parameter. The subset
    is drawn with a fixed seed independent of the run seed, so it is not
    selected on results, but it should be described as a subsample rather than
    as full coverage.
    """

    B: np.ndarray
    b_bar: np.ndarray
    Sigma: np.ndarray
    lam: np.ndarray
    tau2_mean: np.ndarray
    tau2_inv_mean: np.ndarray
    tau2_track: np.ndarray
    tau2_track_idx: np.ndarray
    tau2: np.ndarray | None
    meta: dict


def initialise_state(R: np.ndarray, F: np.ndarray, hp: LassoHyperparams,
                     B0: np.ndarray | None = None) -> dict:
    """
    Data-driven starting values. These affect only how long burn-in takes, not
    what the chain converges to.

        B      : per-asset OLS estimates
        b_bar  : the average of those
        Sigma  : empirical covariance of the OLS residuals
        tau2   : the PRIOR MEAN 2/lambda^2
        lam    : lambda_cal, the hyperprior's central value

    tau2 is the exception, for the same reason the baseline starts Delta_b at
    its prior mean. The data-driven choice would be to draw tau^2 from its
    conditional given the OLS deviations, but unregularised OLS deviations are
    two orders of magnitude larger than anything the posterior supports, so
    that start would be far worse, not better.

    B0 may be passed in to avoid re-solving the N least-squares problems that
    pooled_residual_scale has already solved inside lasso_hyperparameters --
    worth doing across 40 backtest windows.
    """
    N, T = R.shape
    K = hp.K

    if B0 is None:
        B0 = np.array([np.linalg.lstsq(F[i], R[i], rcond=None)[0] for i in range(N)])

    E0 = sur_residuals(R, F, B0)
    Sigma0 = np.cov(E0, ddof=1)
    Sigma0 = 0.5 * (Sigma0 + Sigma0.T)
    if np.linalg.eigvalsh(Sigma0).min() <= 0:
        Sigma0 = hp.V_Sigma / (hp.nu_Sigma - N - 1)

    lam0 = hp.lambda_prior_mean
    return {
        "B": B0,
        "b_bar": B0.mean(axis=0),
        "Sigma": Sigma0,
        "tau2": np.full((N, K), 2.0 / lam0 ** 2),
        "lam": lam0,
    }


def run_gibbs(R: np.ndarray, F: np.ndarray, hp: LassoHyperparams,
              n_draws: int = 3000, n_burn: int = 1000, seed: int | None = None,
              n_track_tau2: int = 200, store_tau2_full: bool = False,
              B0: np.ndarray | None = None,
              progress_every: int | None = None) -> LassoDraws:
    """
    Run the Bayesian LASSO's four-step Gibbs sampler.

    Each sweep updates:
        (1) (B, b_bar) jointly   given tau^2, Sigma   -- blocked, see below
        (2) lambda               given theta          -- COLLAPSED, tau^2
                                                         integrated out
        (3) tau^2                given theta, lambda  -- data augmentation
        (4) Sigma                given B, R, F        -- identical to baseline

    Feng & He's Delta_b step has no counterpart: it is REPLACED by (2), not
    supplemented by it. Step (4) is imported from baseline_gaussian unchanged,
    which is the "one shared hierarchy" claim enforced in code -- theta's prior
    involves no Sigma under the plug-in scale, so Sigma's conditional is
    exactly the baseline's IW(nu_Sigma + T, V_Sigma + EE').

    Steps (2) and (3) are in this order deliberately: lambda is drawn from
    theta with tau^2 marginalised out, then tau^2 is drawn given that fresh
    lambda. Drawing lambda from tau^2 instead (the textbook Park & Casella
    step, kept as sample_lambda) leaves lambda with an ESS of 8 from 750 draws
    and unconverged after 1,500 sweeps -- see sample_lambda_collapsed.

    Step (1) is always blocked. No sequential option is offered: there is no
    literal Feng & He LASSO to reproduce, and the centred parameterisation it
    would correspond to is known to mix catastrophically at this prior scale
    (see sample_B_and_b_bar).

    Order matters only for efficiency. Any fixed scan over the full
    conditionals leaves the same joint posterior invariant.

    n_draws is the TOTAL, so the defaults keep 2,000 draws. Returns a
    LassoDraws with burn-in already discarded.
    """
    N, T = R.shape
    K = hp.K
    if hp.N != N:
        raise ValueError(f"hyperparameters built for N={hp.N}, data has N={N}")
    if F.shape != (N, T, K):
        raise ValueError(f"F has shape {F.shape}, expected {(N, T, K)}")

    rng = np.random.default_rng(seed)

    t_start = time()
    G, Fr = precompute_cross_products(F, R)
    t_precompute = time() - t_start

    state = initialise_state(R, F, hp, B0=B0)
    n_keep = n_draws - n_burn

    # which tau^2 entries to track: fixed independently of `seed`, so the same
    # coefficients are monitored across every run and every chain
    tracker_rng = np.random.default_rng(0)
    pick = tracker_rng.choice(N * K, size=min(n_track_tau2, N * K), replace=False)
    track_idx = np.column_stack(np.unravel_index(pick, (N, K)))

    out = {
        "B": np.empty((n_keep, N, K)),
        "b_bar": np.empty((n_keep, K)),
        "Sigma": np.empty((n_keep, N, N)),
        "lam": np.empty(n_keep),
        "tau2_track": np.empty((n_keep, len(track_idx))),
    }
    tau2_full = np.empty((n_keep, N, K)) if store_tau2_full else None
    tau2_sum = np.zeros((N, K))
    tau2_inv_sum = np.zeros((N, K))

    t0 = time()
    for it in range(n_draws):
        state["B"], state["b_bar"] = sample_B_and_b_bar(
            rng, G, Fr, state["Sigma"], state["tau2"], hp.s,
            hp.b_bar_bar, hp.Delta_b_bar)
        theta = state["B"] - state["b_bar"][None, :]
        # lambda BEFORE tau^2, and from theta directly: tau^2 is integrated out,
        # so lambda moves with theta (which mixes freely) instead of with
        # sum(tau^2) (which does not). tau^2 is then drawn given the fresh
        # lambda. See sample_lambda_collapsed.
        state["lam"] = sample_lambda_collapsed(theta, hp.s, hp, rng)
        state["tau2"] = sample_tau2(theta, hp.s, state["lam"], rng)
        state["Sigma"] = sample_Sigma(rng, R, F, state["B"],
                                      hp.nu_Sigma, hp.V_Sigma)

        if it >= n_burn:
            k = it - n_burn
            out["B"][k] = state["B"]
            out["b_bar"][k] = state["b_bar"]
            out["Sigma"][k] = state["Sigma"]
            out["lam"][k] = state["lam"]
            out["tau2_track"][k] = state["tau2"][track_idx[:, 0], track_idx[:, 1]]
            tau2_sum += state["tau2"]
            tau2_inv_sum += 1.0 / state["tau2"]
            if tau2_full is not None:
                tau2_full[k] = state["tau2"]

        if progress_every and (it + 1) % progress_every == 0:
            elapsed = time() - t0
            print(f"  sweep {it+1}/{n_draws}  ({elapsed:.0f}s elapsed, "
                  f"{elapsed/(it+1)*(n_draws-it-1):.0f}s remaining)", flush=True)

    return LassoDraws(
        **out,
        tau2_mean=tau2_sum / n_keep,
        tau2_inv_mean=tau2_inv_sum / n_keep,
        tau2_track_idx=track_idx,
        tau2=tau2_full,
        meta={"seed": seed, "n_draws": n_draws, "n_burn": n_burn,
              "N": N, "T": T, "K": K,
              "blocked": True,
              "s": hp.s, "lambda_r": hp.lambda_r, "lambda_delta": hp.lambda_delta,
              "sd_target": hp.sd_target,
              "precompute_seconds": t_precompute,
              "sampling_seconds": time() - t0},
    )


