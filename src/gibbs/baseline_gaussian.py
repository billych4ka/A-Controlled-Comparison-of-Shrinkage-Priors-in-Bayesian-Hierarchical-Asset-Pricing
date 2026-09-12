"""
src/gibbs/baseline_gaussian.py

Gaussian-baseline Gibbs sampler for the Bayesian hierarchical SUR
asset-pricing model (Feng & He, 2022), implementing their four-step
MCMC scheme (their eq. 14-18) by hand.

This is the reference model against which the three shrinkage priors
(Bayesian LASSO, Horseshoe, Regularised Horseshoe) are compared. The
likelihood and the cross-asset hierarchy are held fixed across all four;
only the prior on the deviations theta_i = b_i - b_bar changes. Here that
prior is Gaussian, theta_i ~ N(0, Delta_b), which is exactly Feng & He's
own b_i ~ N(b_bar, Delta_b) after a unit-Jacobian change of variables.

Disclosed deviations from the paper as printed
----------------------------------------------
1. Eq. (17)'s posterior scale matrix carries a spurious inverse. We
   implement the standard Normal-Inverse-Wishart conjugate update,
       Delta_b | B, b_bar ~ IW(nu_b + N, V_b + sum_i (b_i - b_bar)(b_i - b_bar)'),
   with no inverse. Taken literally, Eq. (17) shrinks Delta_b as the
   observed dispersion across assets GROWS, which inverts the model's
   own logic; it also fails to recover a known Delta_b in simulation
   (collapses to zero), while the un-inverted form recovers it. The
   magnitude of the posterior draws in their own Appendix C (Fig. 4,
   one off-diagonal element spanning about +/- 2e-4) matches the
   un-inverted form and is ~9x too large for the inverted one, so their
   code evidently does the correct thing -- this is a typesetting slip.
2. Eq. (7)'s asset-major stacking implies Omega = Sigma (x) I_T, not
   Sigma (x) I_N as printed (which is not even conformable at NT x NT).
   sur.py already implements the correct block structure.
3. Sigma's hyperparameters (nu_Sigma, V_Sigma) are never specified in
   the paper; ours are our own disclosed choice (see below).
4. Feng & He's prior hyperparameters are not scale-invariant, and our
   predictor standardisation differs from theirs (expanding-window
   z-scores, sd ~1, vs their [-1,1] characteristics and raw-unit macro
   predictors, sd ~0.015-0.5). Applied verbatim their prior implies an
   R^2 of ~8,800 and shrinks nothing. We therefore report both: their
   values exactly (default_hyperparameters) and a rescaled version
   preserving their common/deviation ratio but with a plausible implied
   R^2 (rescaled_hyperparameters).
5. We replace Feng & He's separate eq. (14) and eq. (16) updates with a
   single blocked draw of (B, b_bar). Their sequential scan is a centred
   parameterisation, which mixes very poorly when Delta_b is small: at our
   calibrated hyperparameters the lag-1 autocorrelation of ||b_bar|| was
   1.000 and the chain had not stabilised after 250 sweeps. Blocking leaves
   the target unchanged (verified: the two samplers agree to <0.1 combined
   MCMC standard errors where the sequential one mixes adequately) and costs
   ~25% more per sweep.
"""

from __future__ import annotations

from dataclasses import dataclass

from scipy.linalg import solve_triangular

from scipy.stats import invwishart

from time import time

from src.likelihood.sur import sur_residuals

import numpy as np


@dataclass
class GaussianBaselineHyperparams:
    """
    The fixed hyperparameters of the Gaussian baseline -- the numbers that
    are chosen once, before sampling, and never updated by the sampler.

    b_bar_bar   : (K,)   prior mean of b_bar         -- Feng & He's b_bar_bar = 0
    Delta_b_bar : (K,K)  prior covariance of b_bar   -- diag(0.1, K)
    nu_b        : IW degrees of freedom for Delta_b  -- 1001 + K
    V_b         : (K,K)  IW scale matrix for Delta_b -- diag(3, K)
    nu_Sigma    : IW degrees of freedom for Sigma    -- our own choice, N + 2
    V_Sigma     : (N,N)  IW scale for Sigma          -- our own choice,
                         (nu_Sigma - N - 1) * S_hat
    """

    b_bar_bar: np.ndarray
    Delta_b_bar: np.ndarray
    nu_b: float
    V_b: np.ndarray
    nu_Sigma: float
    V_Sigma: np.ndarray

    @property
    def K(self) -> int:
        return self.b_bar_bar.shape[0]

    @property
    def N(self) -> int:
        return self.V_Sigma.shape[0]


def sample_covariance(R: np.ndarray) -> np.ndarray:
    """
    Sample covariance matrix S_hat of the excess returns, across assets.

    R : (N,T) excess returns -> (N,N) covariance.

    Each entry S_hat[i,j] is the covariance, over the T time periods,
    between asset i's and asset j's excess returns, with the usual
    unbiased 1/(T-1) normalisation. This is not a model quantity -- it is
    a purely empirical summary of the data, used only to centre Sigma's
    prior somewhere plausible rather than at an arbitrary identity matrix.
    """
    S_hat = np.cov(R, ddof=1)
    # an IW scale matrix must be symmetric positive definite
    S_hat = 0.5 * (S_hat + S_hat.T)          # kill any 1e-17 asymmetry
    eigmin = np.linalg.eigvalsh(S_hat).min()
    if eigmin <= 0:
        raise ValueError(
            f"Sample covariance is not positive definite (min eigenvalue {eigmin:.3e}); "
            "check for duplicated or collinear return series."
        )
    return S_hat


def default_hyperparameters(R: np.ndarray, K: int) -> GaussianBaselineHyperparams:
    """
    Feng & He's "mild" prior setting, plus our own disclosed choice for
    Sigma's two hyperparameters (which they never specify).

    R : (N,T) excess returns -- used only to compute S_hat for V_Sigma
    K : number of predictors (144 for size_bm_25)

    The mean of IW(nu, V) is V / (nu - p - 1) with p the dimension, so
    E[Delta_b] = diag(3,K)/(1001+K-K-1) = diag(0.003, K): a prior SD of
    about 0.055 on each deviation theta_ij. (Cross-check: their "tight"
    setting nu_b = 5001+K gives 3/5000 -> SD 0.0245, matching the "around
    0.02" their own footnote claims -- confirming this reading of the
    parameterisation.)

    Sigma's hyperparameters are ours: nu_Sigma = N + 2 is the smallest
    degrees of freedom at which the IW mean exists (it needs nu > N + 1),
    i.e. the weakest proper choice, in the spirit of Pastor (2000).
    V_Sigma = (nu_Sigma - N - 1) * S_hat then centres the prior mean of
    Sigma exactly on the sample covariance; the multiplier is exactly 1
    at this nu_Sigma, so V_Sigma = S_hat.
    """
    N, T = R.shape

    S_hat = sample_covariance(R)

    nu_Sigma = N + 2.0
    V_Sigma = (nu_Sigma - N - 1) * S_hat

    return GaussianBaselineHyperparams(
        b_bar_bar=np.zeros(K),
        Delta_b_bar=np.eye(K) * 0.1,
        nu_b=1001.0 + K,
        V_b=np.eye(K) * 3.0,
        nu_Sigma=nu_Sigma,
        V_Sigma=V_Sigma,
    )

# ---------------------------------------------------------------------------
# Step (1) of Feng & He's Gibbs loop: update B  (their eq. 14-15)
# ---------------------------------------------------------------------------

def _spd_inverse(A: np.ndarray) -> np.ndarray:
    """
    Inverse of a symmetric positive-definite matrix, via its Cholesky factor.

    Preferred over np.linalg.inv here for two reasons: it exploits (and
    implicitly checks) positive-definiteness -- np.linalg.cholesky raises
    if A has drifted non-PD, which is exactly the failure we want to hear
    about loudly -- and it returns an exactly symmetric result, so
    round-off cannot accumulate asymmetry across thousands of sweeps.
    """
    L = np.linalg.cholesky(A)
    L_inv = solve_triangular(L, np.eye(A.shape[0]), lower=True)
    return L_inv.T @ L_inv


def precompute_cross_products(F: np.ndarray, R: np.ndarray):
    """
    Precompute the two data summaries the B-update needs, ONCE, before
    sampling starts. These depend only on (F, R), never on any parameter,
    so recomputing them inside the loop would repeat ~N^2 T K^2 work every
    sweep for no reason.

    F : (N,T,K), R : (N,T)
    -> G  : (N,K,N,K)  with G[i,:,j,:] = f_i' f_j     (K x K Gram blocks)
       Fr : (N,N,K)    with Fr[i,j,:]  = f_i' r_j     (K-vectors)

    G is stored in the axis order (i,k,j,l) -- deliberately NOT (i,j,k,l) --
    because that is exactly the layout the NK x NK precision matrix needs:
    reshaping (N,K,N,K) -> (NK,NK) then gives the right block structure with
    no transpose, and a transpose of a 104 MB array every sweep is the
    single most expensive avoidable operation in this sampler.
    """
    G = np.einsum("itk,jtl->ikjl", F, F, optimize=True)
    Fr = np.einsum("itk,jt->ijk", F, R, optimize=True)
    return G, Fr


def sample_B(rng: np.random.Generator, G: np.ndarray, Fr: np.ndarray,
             Sigma: np.ndarray, Delta_b: np.ndarray,
             b_bar: np.ndarray) -> np.ndarray:
    """
    Draw B from its full conditional (Feng & He eq. 14-15): a single joint
    multivariate normal over ALL N assets' coefficient vectors at once,
    dimension NK (= 3,600 for size_bm_25).

    Plain English: each asset's coefficients are pulled by three forces --
    its own data, the other assets' data (because residuals are correlated
    across assets, so another asset's surprise is informative about this
    one's), and the shared prior mean b_bar. This step resolves all three
    simultaneously. It is the "information sharing" step: the only place in
    the sampler where assets talk to each other through the likelihood.

    rng     : numpy Generator
    G, Fr   : outputs of precompute_cross_products
    Sigma   : (N,N) current residual covariance across assets
    Delta_b : (K,K) current covariance of the deviations theta_i
    b_bar   : (K,)  current shared prior mean
    -> (N,K) draw of B, one row per asset

    Math
    ----
    With Feng & He's asset-major stacking (their eq. 7), Omega = Sigma (x) I_T,
    so Omega^{-1} = Sigma^{-1} (x) I_T, and because F is block-diagonal:

        [F' Omega^{-1} F]_{ij} = Sigma^{-1}[i,j] * (f_i' f_j)          (K x K)
        [F' Omega^{-1} R]_i    = sum_j Sigma^{-1}[i,j] * (f_i' r_j)    (K,)

    The prior contributes precision I_N (x) Delta_b^{-1} (block-diagonal,
    Delta_b^{-1} on each diagonal block) and location term
    (I_N (x) Delta_b^{-1})(1_N (x) b_bar) = 1_N (x) (Delta_b^{-1} b_bar),
    i.e. the same K-vector repeated for every asset. Hence

        precision P = F'Omega^{-1}F + I_N (x) Delta_b^{-1}
        b*          = P^{-1} (F'Omega^{-1}R + 1_N (x) Delta_b^{-1} b_bar)
        B | .       ~ N(b*, P^{-1})
    """
    N = Sigma.shape[0]
    K = Delta_b.shape[0]

    Sigma_inv = _spd_inverse(Sigma)
    Delta_b_inv = _spd_inverse(Delta_b)

    # --- precision matrix, as a 4-D array then viewed as (NK, NK) ---
    # broadcasting Sigma_inv over the (i,j) axes multiplies each K x K Gram
    # block G[i,:,j,:] by the scalar Sigma_inv[i,j] -- this single line is
    # the whole of F' Omega^{-1} F.
    P = G * Sigma_inv[:, None, :, None]          # (N,K,N,K)
    diag = np.arange(N)
    P[diag, :, diag, :] += Delta_b_inv           # add I_N (x) Delta_b^{-1}
    P = P.reshape(N * K, N * K)

    # --- location term ---
    rhs = np.einsum("ij,ijk->ik", Sigma_inv, Fr, optimize=True)  # (N,K)
    rhs = rhs + (Delta_b_inv @ b_bar)[None, :]                   # broadcast over assets
    rhs = rhs.reshape(N * K)

    # --- draw: b = b* + L^{-T} z,  where P = L L' ---
    # Cov(L^{-T} z) = L^{-T} L^{-1} = (L L')^{-1} = P^{-1}, exactly the
    # target covariance -- so we never form or invert P itself.
    L = np.linalg.cholesky(P)
    y = solve_triangular(L, rhs, lower=True)             # L y   = rhs
    b_star = solve_triangular(L.T, y, lower=False)       # L' b* = y
    z = rng.standard_normal(N * K)
    noise = solve_triangular(L.T, z, lower=False)        # L' n  = z

    return (b_star + noise).reshape(N, K)

# ---------------------------------------------------------------------------
# Step (2) of Feng & He's Gibbs loop: update b_bar  (their eq. 16)
# ---------------------------------------------------------------------------

def sample_b_bar(rng: np.random.Generator, B: np.ndarray, Delta_b: np.ndarray,
                 b_bar_bar: np.ndarray, Delta_b_bar: np.ndarray) -> np.ndarray:
    """
    Draw b_bar from its full conditional (Feng & He eq. 16), a K-dimensional
    multivariate normal.

    Plain English: this is the "information grouping" step -- the N assets
    have just been given their own coefficient vectors, and this step asks
    what common value they are scattered around. It is the textbook
    conjugate update for the mean of a normal sample with known covariance:
    treat the N vectors b_1, ..., b_N as N observations drawn from
    N(b_bar, Delta_b), and combine their sample mean with the prior
    N(b_bar_bar, Delta_b_bar), weighting each by its precision.

    Note this conditional does NOT involve R, F or Sigma at all. Once B is
    known, the returns carry no further information about b_bar -- b_bar
    influences the data only through B. That conditional independence is
    what makes the hierarchy tractable and is why this step is cheap
    (K x K = 144 x 144) even though the previous one was NK x NK.

    B           : (N,K) current coefficient draws, one row per asset
    Delta_b     : (K,K) current deviation covariance
    b_bar_bar   : (K,)  prior mean (zero, in Feng & He's setting)
    Delta_b_bar : (K,K) prior covariance
    -> (K,) draw of b_bar

    Math
    ----
        precision  = Delta_b_bar^{-1} + N * Delta_b^{-1}
        location   = Delta_b_bar^{-1} b_bar_bar + Delta_b^{-1} sum_i b_i
        b_bar | .  ~ N(precision^{-1} location, precision^{-1})

    Feng & He write sum_i b_i as (1_N (x) I_K)' B, which is the same thing
    once B is de-stacked from an NK-vector into N rows of length K.
    """
    N, K = B.shape

    Delta_b_inv = _spd_inverse(Delta_b)
    Delta_b_bar_inv = _spd_inverse(Delta_b_bar)

    precision = Delta_b_bar_inv + N * Delta_b_inv
    location = Delta_b_bar_inv @ b_bar_bar + Delta_b_inv @ B.sum(axis=0)

    # same never-invert-the-precision trick as sample_B
    L = np.linalg.cholesky(precision)
    y = solve_triangular(L, location, lower=True)
    mean = solve_triangular(L.T, y, lower=False)
    noise = solve_triangular(L.T, rng.standard_normal(K), lower=False)

    return mean + noise

# ---------------------------------------------------------------------------
# Blocked update of (B, b_bar) -- see "Disclosed deviations", point 5
# ---------------------------------------------------------------------------

def sample_B_and_b_bar(rng: np.random.Generator, G: np.ndarray, Fr: np.ndarray,
                       Sigma: np.ndarray, Delta_b: np.ndarray,
                       b_bar_bar: np.ndarray, Delta_b_bar: np.ndarray):
    """
    Draw B and b_bar JOINTLY from their exact Gaussian conditional, replacing
    Feng & He's separate eq. (14) and eq. (16) steps with a single block.

    Why block them
    --------------
    Feng & He's scan updates b_i given b_bar, then b_bar given the b_i. When
    Delta_b is small each b_i is nearly pinned to b_bar and b_bar is nearly
    pinned to their mean, so the two conditionals almost determine each other
    and the chain moves in tiny steps -- the classic inefficiency of a CENTRED
    parameterisation at small hierarchical variance (Papaspiliopoulos, Roberts
    & Skold, 2007). At our calibrated hyperparameters (Delta_b ~ 1.7e-8) this
    is severe: measured lag-1 autocorrelation of ||b_bar|| was 1.000, with the
    chain still drifting monotonically after 250 sweeps. Drawing the two
    blocks together removes the coupling entirely; the same measurement gives
    0.190 with no residual trend, and the chain reaches its stationary region
    within a single sweep instead of thousands.

    Blocking leaves the target distribution unchanged and cannot worsen
    mixing (Liu, Wong & Kong, 1994), so this is a pure efficiency gain. It
    costs one Cholesky of dimension NK+K instead of NK -- about 25% more time
    per sweep at N=25, K=144.

    Math
    ----
    Stacking (vec B, b_bar), the joint precision is

        [ F'Omega^{-1}F + I_N (x) Delta_b^{-1}   |  -1_N (x) Delta_b^{-1} ]
        [ -1_N' (x) Delta_b^{-1}                 |  N Delta_b^{-1} + Delta_b_bar^{-1} ]

    with location (F'Omega^{-1}R , Delta_b_bar^{-1} b_bar_bar). The
    off-diagonal blocks are exactly the coupling that the sequential scan has
    to traverse one small step at a time.

    -> (B, b_bar) with shapes (N,K) and (K,)
    """
    N = Sigma.shape[0]
    K = Delta_b.shape[0]

    Sigma_inv = _spd_inverse(Sigma)
    Delta_b_inv = _spd_inverse(Delta_b)
    Delta_b_bar_inv = _spd_inverse(Delta_b_bar)

    M = N * K + K
    P = np.zeros((M, M))

    block = G * Sigma_inv[:, None, :, None]
    diag = np.arange(N)
    block[diag, :, diag, :] += Delta_b_inv
    P[:N * K, :N * K] = block.reshape(N * K, N * K)

    for i in range(N):
        P[i * K:(i + 1) * K, N * K:] = -Delta_b_inv
        P[N * K:, i * K:(i + 1) * K] = -Delta_b_inv
    P[N * K:, N * K:] = N * Delta_b_inv + Delta_b_bar_inv

    rhs = np.empty(M)
    rhs[:N * K] = np.einsum("ij,ijk->ik", Sigma_inv, Fr, optimize=True).reshape(N * K)
    rhs[N * K:] = Delta_b_bar_inv @ b_bar_bar

    L = np.linalg.cholesky(P)
    y = solve_triangular(L, rhs, lower=True)
    mean = solve_triangular(L.T, y, lower=False)
    noise = solve_triangular(L.T, rng.standard_normal(M), lower=False)
    draw = mean + noise

    return draw[:N * K].reshape(N, K), draw[N * K:]

# ---------------------------------------------------------------------------
# Steps (3) and (4): update Delta_b and Sigma  (their eq. 17-18)
# ---------------------------------------------------------------------------

def sample_Delta_b(rng: np.random.Generator, B: np.ndarray, b_bar: np.ndarray,
                   nu_b: float, V_b: np.ndarray) -> np.ndarray:
    """
    Draw Delta_b from its full conditional (Feng & He eq. 17, corrected --
    see the "Disclosed deviations" note at the top of this file).

        Delta_b | B, b_bar ~ IW(nu_b + N, V_b + sum_i (b_i - b_bar)(b_i - b_bar)')

    Plain English: having seen how far each asset's coefficients sit from
    the common b_bar, this step asks how spread out those deviations are.
    Delta_b is the model's own estimate of "how heterogeneous are the
    assets?", and it is what sets the strength of shrinkage applied back to
    B on the next sweep -- large Delta_b means assets are allowed to differ,
    small Delta_b pulls them all toward b_bar.

    The conjugate pattern is the same one as any Inverse-Wishart update:
    the degrees of freedom gain the number of observations (N assets), and
    the scale matrix gains their sum of squared deviations. The two must
    move TOGETHER -- that pairing is the entire content of conjugacy, and
    it is why Eq. (17)'s printed inverse on the scale matrix cannot be
    right (it would shrink Delta_b as observed dispersion grows).

    B      : (N,K) current coefficient draws
    b_bar  : (K,)  current shared mean
    nu_b, V_b : the IW prior hyperparameters
    -> (K,K) draw of Delta_b
    """
    N, K = B.shape

    theta = B - b_bar[None, :]                 # (N,K) deviations
    S = theta.T @ theta                        # (K,K) sum_i theta_i theta_i'

    scale = V_b + S
    scale = 0.5 * (scale + scale.T)            # enforce exact symmetry
    return invwishart.rvs(df=nu_b + N, scale=scale, random_state=rng)


def sample_Sigma(rng: np.random.Generator, R: np.ndarray, F: np.ndarray,
                 B: np.ndarray, nu_Sigma: float, V_Sigma: np.ndarray) -> np.ndarray:
    """
    Draw Sigma from its full conditional (Feng & He eq. 18):

        Sigma | B, R, F ~ IW(nu_Sigma + T, V_Sigma + E~' E~)

    where E~ is their T x N residual matrix. Our residuals come back from
    sur_residuals as (N,T), so E~' E~ is simply E @ E.T -- an N x N matrix
    whose (i,j) entry is sum_t e_it e_jt, the cross-asset co-movement of
    what the model failed to explain.

    Plain English: exactly the same conjugate pattern as Delta_b, one level
    down. There the "observations" were the N assets' deviations from
    b_bar; here they are the T months' residual vectors. Degrees of freedom
    gain T, the scale gains the residual sum of squares and cross-products.

    Residuals are computed with sur.py's sur_residuals rather than
    reimplemented here, so that all four models share one definition of the
    likelihood's residual, in code and not just in the write-up.

    R : (N,T), F : (N,T,K), B : (N,K)
    -> (N,N) draw of Sigma
    """
    N, T = R.shape

    E = sur_residuals(R, F, B)                 # (N,T)
    scale = V_Sigma + E @ E.T
    scale = 0.5 * (scale + scale.T)
    return invwishart.rvs(df=nu_Sigma + T, scale=scale, random_state=rng)

# ---------------------------------------------------------------------------
# Initialisation and the assembled Gibbs loop
# ---------------------------------------------------------------------------

@dataclass
class GibbsDraws:
    """
    Posterior draws returned by run_gibbs, after burn-in has been discarded.

    B            : (n_keep, N, K) asset-specific coefficients
    b_bar        : (n_keep, K)    shared mean coefficients
    Sigma        : (n_keep, N, N) residual covariance across assets
    Delta_b_diag : (n_keep, K)    diagonal of Delta_b
    Delta_b_off  : (n_keep, n_track) a fixed random sample of Delta_b's
                   off-diagonal entries, for trace plots and ESS
    Delta_b_off_idx : (n_track, 2) which (row, col) those entries are
    Delta_b_mean : (K,K) posterior mean of the full Delta_b, accumulated as
                   a running sum so the full matrix never has to be stored
    Delta_b      : (n_keep, K, K) or None -- the full matrix, only if asked
    meta         : bookkeeping (seed, timings, settings)
    """

    B: np.ndarray
    b_bar: np.ndarray
    Sigma: np.ndarray
    Delta_b_diag: np.ndarray
    Delta_b_off: np.ndarray
    Delta_b_off_idx: np.ndarray
    Delta_b_mean: np.ndarray
    Delta_b: np.ndarray | None
    meta: dict


def initialise_state(R: np.ndarray, F: np.ndarray,
                     hp: GaussianBaselineHyperparams) -> dict:
    """
    Data-driven starting values for the chain (NOT the prior -- this is only
    where the sampler begins, and has no effect on what it converges to;
    starting somewhere sensible just shortens burn-in). Standard practice,
    e.g. George & McCulloch (1993) initialise their SSVS sampler at the
    least-squares estimates in exactly this way.

        B        : per-asset OLS estimates
        b_bar    : the average of those OLS estimates
        Sigma    : the empirical covariance of the OLS residuals
        Delta_b  : the PRIOR MEAN, V_b / (nu_b - K - 1)

    Delta_b is the exception, and deliberately so: the obvious data-driven
    choice -- the sample covariance of the N deviation vectors -- is a K x K
    matrix built from only N observations. At N=25 and K=144 that matrix has
    rank at most 25 and is therefore singular, and the B-update needs its
    inverse on the very first sweep. The prior mean is well-conditioned by
    construction (condition number ~1 for diagonal V_b), so we start there.
    """
    N, T = R.shape
    K = hp.K

    B0 = np.array([np.linalg.lstsq(F[i], R[i], rcond=None)[0] for i in range(N)])

    E0 = sur_residuals(R, F, B0)
    Sigma0 = np.cov(E0, ddof=1)
    Sigma0 = 0.5 * (Sigma0 + Sigma0.T)
    if np.linalg.eigvalsh(Sigma0).min() <= 0:
        # can happen only if T <= N (not the case for our data, but small
        # simulation studies can hit it) -- fall back to the prior mean
        Sigma0 = hp.V_Sigma / (hp.nu_Sigma - N - 1)

    return {
        "B": B0,
        "b_bar": B0.mean(axis=0),
        "Sigma": Sigma0,
        "Delta_b": hp.V_b / (hp.nu_b - K - 1),
    }


def run_gibbs(R: np.ndarray, F: np.ndarray, hp: GaussianBaselineHyperparams,
              n_draws: int = 3000, n_burn: int = 1000, seed: int | None = None,
              n_track_offdiag: int = 200, store_Delta_b_full: bool = False,
              blocked: bool = True,
              progress_every: int | None = None) -> GibbsDraws:
    """
    Run Feng & He's four-step Gibbs sampler (their eq. 14-18) for the
    Gaussian baseline.

    Each sweep updates, in their order:
        (1) B        given b_bar, Delta_b, Sigma   -- information sharing
        (2) b_bar    given B, Delta_b              -- information grouping
        (3) Delta_b  given B, b_bar                -- how heterogeneous?
        (4) Sigma    given B, R, F                 -- how correlated are the errors?

    Order matters only for efficiency, not correctness: any fixed sweep
    order over the full conditionals leaves the same joint posterior
    invariant. We follow the paper's order so that differences from their
    reported results cannot be attributed to a different scan.

    n_draws / n_burn default to Feng & He's own budget (3,000 with 1,000
    discarded). Note n_draws is the TOTAL, so 2,000 draws are kept.

    Returns a GibbsDraws object with burn-in already discarded.
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

    state = initialise_state(R, F, hp)
    n_keep = n_draws - n_burn

    # which off-diagonal entries of Delta_b to track: fixed independently of
    # `seed`, so the same elements are monitored across every run and model
    tracker_rng = np.random.default_rng(0)
    iu = np.triu_indices(K, k=1)
    pick = tracker_rng.choice(len(iu[0]), size=min(n_track_offdiag, len(iu[0])),
                              replace=False)
    off_idx = np.column_stack([iu[0][pick], iu[1][pick]])

    out = {
        "B": np.empty((n_keep, N, K)),
        "b_bar": np.empty((n_keep, K)),
        "Sigma": np.empty((n_keep, N, N)),
        "Delta_b_diag": np.empty((n_keep, K)),
        "Delta_b_off": np.empty((n_keep, len(off_idx))),
    }
    Delta_b_full = np.empty((n_keep, K, K)) if store_Delta_b_full else None
    Delta_b_sum = np.zeros((K, K))

    t0 = time()
    for it in range(n_draws):
        if blocked:
            state["B"], state["b_bar"] = sample_B_and_b_bar(
                rng, G, Fr, state["Sigma"], state["Delta_b"],
                hp.b_bar_bar, hp.Delta_b_bar)
        else:
            state["B"] = sample_B(rng, G, Fr, state["Sigma"], state["Delta_b"],
                                  state["b_bar"])
            state["b_bar"] = sample_b_bar(rng, state["B"], state["Delta_b"],
                                          hp.b_bar_bar, hp.Delta_b_bar)
        state["Delta_b"] = sample_Delta_b(rng, state["B"], state["b_bar"],
                                          hp.nu_b, hp.V_b)
        state["Sigma"] = sample_Sigma(rng, R, F, state["B"],
                                      hp.nu_Sigma, hp.V_Sigma)

        if it >= n_burn:
            k = it - n_burn
            out["B"][k] = state["B"]
            out["b_bar"][k] = state["b_bar"]
            out["Sigma"][k] = state["Sigma"]
            out["Delta_b_diag"][k] = np.diag(state["Delta_b"])
            out["Delta_b_off"][k] = state["Delta_b"][off_idx[:, 0], off_idx[:, 1]]
            Delta_b_sum += state["Delta_b"]
            if Delta_b_full is not None:
                Delta_b_full[k] = state["Delta_b"]

        if progress_every and (it + 1) % progress_every == 0:
            elapsed = time() - t0
            print(f"  sweep {it+1}/{n_draws}  ({elapsed:.0f}s elapsed, "
                  f"{elapsed/(it+1)*(n_draws-it-1):.0f}s remaining)", flush=True)

    return GibbsDraws(
        **out,
        Delta_b_off_idx=off_idx,
        Delta_b_mean=Delta_b_sum / n_keep,
        Delta_b=Delta_b_full,
        meta={"seed": seed, "n_draws": n_draws, "n_burn": n_burn,
              "N": N, "T": T, "K": K,
              "blocked": blocked,
              "precompute_seconds": t_precompute,
              "sampling_seconds": time() - t0},
    )


# ---------------------------------------------------------------------------
# Scale-calibrated hyperparameters (see "Disclosed deviations", point 4)
# ---------------------------------------------------------------------------

def prior_implied_r2(hp: GaussianBaselineHyperparams, R: np.ndarray,
                     F: np.ndarray) -> float:
    """
    The R^2 that a prior implicitly expects the predictors to explain.

    A prior on regression coefficients is a statement about how much of the
    response the predictors can move, and that statement depends entirely on
    the units of the predictors. This function makes it explicit and so makes
    two priors comparable across different standardisations.

    Under the prior, b_ij = b_bar_j + theta_ij has variance
    Var(b_bar_j) + Var(theta_ij), isotropic across j in Feng & He's setting.
    The explained variance is then

        E[Var_t(f' b)] = trace(C * Cov(b)) = s^2 * trace(C),

    with C the empirical covariance of the predictors and s^2 the per-
    coefficient prior variance. Dividing by Var(r) gives the implied R^2.

    Values far above 1 mean the prior expects the predictable part of returns
    to be more variable than returns themselves -- incoherent, and a sign the
    hyperparameters were calibrated for a different predictor scale.
    """
    K = hp.K
    C = np.cov(F.reshape(-1, K).T)
    s2 = hp.Delta_b_bar[0, 0] + hp.V_b[0, 0] / (hp.nu_b - K - 1)
    return float(s2 * np.trace(C) / R.var())


def rescaled_hyperparameters(R: np.ndarray, F: np.ndarray, K: int,
                             target_r2: float = 0.05
                             ) -> GaussianBaselineHyperparams:
    """
    Feng & He's prior structure, rescaled so its implied R^2 is plausible at
    OUR predictor standardisation.

    Why this exists
    ---------------
    Feng & He standardise firm characteristics cross-sectionally to [-1,1]
    and use Welch-Goyal macro predictors in raw units (dividend yield ~0.03,
    T-bill ~0.05). We use expanding-window z-scores, so our predictors have
    standard deviation ~1 -- between 60 and 130 times larger. Prior
    hyperparameters are NOT scale-invariant: V_b = diag(3) is a statement
    about coefficient magnitude, and the same numbers therefore mean
    something entirely different under our scaling. Applied verbatim, their
    prior implies an R^2 of roughly 8,800 (see prior_implied_r2), and
    delivers no shrinkage in ANY of the NK eigendirections.

    What is preserved and what is changed
    -------------------------------------
    Their split between the common and deviation components -- 0.1 vs 0.003,
    i.e. 97.1% / 2.9% -- is a RATIO and so is scale-free. It encodes their
    substantive belief that assets are mostly alike, and is kept exactly.
    Only the overall magnitude changes, via a single factor applied to both
    Delta_b_bar and V_b. nu_b and b_bar_bar are untouched.

    The factor is set so that the prior's implied R^2 equals target_r2:
        s^2 = target_r2 * Var(r) / trace(C),   c = s^2 / (0.1 + 0.003)

    Two independent arguments agree on the resulting scale to within
    0.3-4.3%: equating prior precision to the median data precision, and
    targeting a plausible R^2. An earlier version of this note claimed three
    agreeing routes, the third being a match to Feng & He's own effective
    predictor scale. That comparison was not like-for-like -- it put a
    DEVIATION scale against two TOTAL scales -- and the route is withdrawn
    rather than reconciled.

    On the intercept: this shrinks b_bar_0 toward zero with a prior standard
    deviation of 0.91% annualised at target_r2 = 0.05 (0.076% monthly, x12).
    That is a belief-in-the-pricing-model prior on alpha, marginally TIGHTER
    than the range Pastor (2000) uses (1-3% annualised) rather than inside
    it -- close enough to the same order to be read as consistent with the
    source already cited for Sigma's hyperparameters, but the claim is
    "marginally tighter than", not "within". The figure is not tuned to
    Pastor: it falls out of target_r2 = 0.05 and the Feng-He ratio, and it
    is one of two independent scale arguments agreeing to within 0.3-4.3%,
    so landing just below the range is a consequence, not a calibration.
    The same intercept scale propagates to all four models, since
    bayesian_lasso, horseshoe and regularised_horseshoe each build on this
    function's output.

    target_r2 : prior expected R^2. Default 0.05 is deliberately generous:
                in-sample OLS gives 0.269 at K=144, but K/T = 0.20 means most
                of that is overfit (adjusted R^2 is 0.086), and realistic
                out-of-sample monthly predictability in this literature is
                0.5-1%.
    """
    N, T = R.shape

    base = default_hyperparameters(R, K)

    v_common = base.Delta_b_bar[0, 0]                       # 0.1
    v_dev = base.V_b[0, 0] / (base.nu_b - K - 1)            # 0.003
    v_total = v_common + v_dev

    C = np.cov(F.reshape(-1, K).T)
    s2 = target_r2 * R.var() / np.trace(C)
    c = s2 / v_total

    return GaussianBaselineHyperparams(
        b_bar_bar=base.b_bar_bar,
        Delta_b_bar=base.Delta_b_bar * c,
        nu_b=base.nu_b,
        V_b=base.V_b * c,
        nu_Sigma=base.nu_Sigma,
        V_Sigma=base.V_Sigma,
    )

