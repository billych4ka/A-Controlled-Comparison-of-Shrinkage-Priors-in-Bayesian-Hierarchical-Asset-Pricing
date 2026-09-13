"""
src/diagnostics/convergence.py

MCMC convergence diagnostics shared by all four models: rank-normalised
split-R-hat, and bulk/tail effective sample size.

Implementation follows Vehtari, Gelman, Simpson, Carpenter & Burkner (2021),
"Rank-normalization, folding, and localization: an improved R-hat for
assessing convergence of MCMC", Bayesian Analysis 16(2), 667-718. This is a
stricter standard than the classic Gelman-Rubin statistic that R's coda
reports (and that Feng & He quote), for three reasons worth stating in the
write-up:

  1. SPLIT. Each chain is cut in half and the halves treated as separate
     chains, so a single chain that drifts steadily is detected. Classic
     R-hat compares chains only to each other and is blind to a drift shared
     by all of them.
  2. RANK-NORMALISED. Values are replaced by normal scores of their ranks
     before the statistic is computed, so the diagnostic does not assume
     finite variance and is not dominated by a few extreme draws.
  3. FOLDED. The statistic is also computed on |x - median(x)| and the worse
     of the two reported, so chains that agree in location but differ in
     scale are caught.

Written in NumPy rather than calling arviz so that the definition used is
explicit and auditable, and so it can be verified against cases with known
answers (see tests). Cross-check against arviz separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import norm

from src.gibbs.io import load_draws


def load_chains(universe: str, model: str, setting: str, cls,
                results_root: str | Path = "results") -> list:
    """
    Load every chain for one (universe, model, setting) combination.

    Files are expected at
        <results_root>/<universe>/<model>/<model>_<setting>_chain*.npz
    which is the layout run_baseline_gaussian.py writes. Chains are returned
    in numerical order of the chain index, not filesystem order; otherwise
    chain10 would sort before chain2 and the pairing with seeds would be
    wrong in any run with 10+ chains.

    cls : the dataclass to rebuild (e.g. GibbsDraws). Passed explicitly so
          this module never has to import any model's code, keeping the four
          models independent of one another.
    """
    folder = Path(results_root) / universe / model
    pattern = f"{model}_{setting}_chain*.npz"
    paths = sorted(folder.glob(pattern),
                   key=lambda p: int(p.stem.rsplit("chain", 1)[1]))
    if not paths:
        available = sorted({p.stem.rsplit("_chain", 1)[0]
                            for p in folder.glob(f"{model}_*_chain*.npz")})
        raise FileNotFoundError(
            f"No files matching {folder / pattern}. "
            f"Settings present in that folder: {available}"
        )
    return [load_draws(p, cls) for p in paths]


def _rank_normalise(x: np.ndarray) -> np.ndarray:
    """
    Replace values by normal scores of their ranks, pooled across all chains
    and draws (Vehtari et al. 2021, eq. 14).

    r is the average rank (ties shared), S the total number of draws; the
    Blom transform z = Phi^{-1}((r - 3/8)/(S + 1/4)) maps ranks onto the
    normal scale. The point is that R-hat and ESS were derived for
    approximately normal quantities; a heavy-tailed or skewed parameter can
    otherwise produce a diagnostic dominated by a handful of draws.
    """
    shape = x.shape
    flat = x.ravel()
    order = flat.argsort()
    ranks = np.empty(len(flat), dtype=float)
    ranks[order] = np.arange(1, len(flat) + 1)
    uniq, inv, counts = np.unique(flat, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        sums = np.zeros(len(uniq))
        np.add.at(sums, inv, ranks)
        ranks = (sums / counts)[inv]

    S = len(flat)
    z = norm.ppf((ranks - 3.0 / 8.0) / (S + 1.0 / 4.0))
    return z.reshape(shape)


def _classic_rhat(x: np.ndarray) -> float:
    """
    Gelman-Rubin R-hat for x of shape (n_chains, n_draws), with no splitting
    or rank-normalisation: the building block the public function applies
    to already-split, already-transformed input.

        W        = mean within-chain variance
        B/n      = variance of the chain means
        var_plus = ((n-1)/n) W + B/n     (an over-estimate of the target
                                          variance if the chains have not
                                          mixed)
        R-hat    = sqrt(var_plus / W)

    R-hat is therefore the factor by which the pooled variance exceeds the
    average within-chain variance. It approaches 1 from above as chains
    become indistinguishable.
    """
    m, n = x.shape
    if n < 2:
        return np.nan
    chain_means = x.mean(axis=1)
    W = x.var(axis=1, ddof=1).mean()
    if W == 0 or not np.isfinite(W):
        return np.nan
    B_over_n = chain_means.var(ddof=1) if m > 1 else 0.0
    var_plus = (n - 1) / n * W + B_over_n
    return float(np.sqrt(var_plus / W))


def _split(x: np.ndarray) -> np.ndarray:
    """
    Split each chain into two halves, doubling the chain count. An odd draw
    in the middle is dropped so both halves are the same length.
    """
    m, n = x.shape
    half = n // 2
    return np.concatenate([x[:, :half], x[:, n - half:]], axis=0)


def rank_normalised_rhat(x: np.ndarray) -> float:
    """
    Rank-normalised, folded, split-R-hat for x of shape (n_chains, n_draws).

    Returns the LARGER of:
      - bulk R-hat  : rank-normalised split-R-hat of the values themselves,
                      sensitive to disagreement in location
      - folded R-hat: the same statistic applied to |x - median(x)|,
                      sensitive to disagreement in scale

    Reporting the max means a pair of chains centred identically but with
    different spread cannot pass. Convergence threshold in Vehtari et al.
    (2021) is 1.01, notably stricter than the 1.1 often quoted from older
    literature.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"expected (n_chains, n_draws), got shape {x.shape}")
    if not np.isfinite(x).all():
        return np.nan
    if np.ptp(x) == 0:
        return np.nan

    bulk = _classic_rhat(_split(_rank_normalise(x)))
    folded = _classic_rhat(_split(_rank_normalise(np.abs(x - np.median(x)))))
    return float(max(bulk, folded))


def _autocovariance(x: np.ndarray) -> np.ndarray:
    """
    Autocovariance of a single chain at lags 0..n-1, via FFT.

    Direct computation is O(n^2); the FFT route is O(n log n), which matters
    because this is called once per parameter and we have thousands. Zero-
    padding to at least 2n prevents the circular wrap-around that would
    otherwise contaminate long lags.
    """
    n = len(x)
    x = x - x.mean()
    n_fft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, n_fft)
    acov = np.fft.irfft(f * np.conjugate(f), n_fft)[:n]
    return acov / n


def effective_sample_size(x: np.ndarray) -> float:
    """
    Effective sample size for x of shape (n_chains, n_draws), following
    Vehtari et al. (2021) sec. 3: chains are split, then the multi-chain
    autocorrelation estimate is truncated by Geyer's initial positive
    sequence rule.

    What ESS means: MCMC draws are correlated, so N of them carry less
    information than N independent draws. ESS is the number of independent
    draws that would give the same Monte Carlo error, and the standard error
    of a posterior mean is sd/sqrt(ESS), NOT sd/sqrt(N).

    The estimator combines within-chain autocovariance with the multi-chain
    variance estimate var_plus:

        rho_t = 1 - (W - mean_chains(acov_t)) / var_plus

    Geyer's rule then sums CONSECUTIVE PAIRS (rho_{2k} + rho_{2k+1}) and stops
    at the first non-positive pair. Pairing is not a convenience: for a
    reversible chain the pair sums are provably positive and decreasing, so
    truncating there removes the noisy tail of the autocorrelation estimate
    without discarding real signal. Summing unpaired terms until one goes
    negative would systematically overstate ESS.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"expected (n_chains, n_draws), got shape {x.shape}")
    if not np.isfinite(x).all() or np.ptp(x) == 0:
        return np.nan

    x = _split(x)
    m, n = x.shape
    if n < 4:
        return np.nan

    acov = np.array([_autocovariance(chain) for chain in x])
    chain_var = acov[:, 0] * n / (n - 1)
    W = chain_var.mean()
    if W == 0 or not np.isfinite(W):
        return np.nan

    var_plus = (n - 1) / n * W
    if m > 1:
        var_plus += x.mean(axis=1).var(ddof=1)

    rho = 1.0 - (W - acov.mean(axis=0)) / var_plus
    rho[0] = 1.0

    pair_sum, t = 0.0, 0
    while t + 1 < n:
        p = rho[t] + rho[t + 1]
        if p <= 0:
            break
        pair_sum += p
        t += 2

    tau = -1.0 + 2.0 * pair_sum
    tau = max(tau, 1.0 / np.log10(max(m * n, 11)))
    return float(m * n / tau)


def ess_tail(x: np.ndarray) -> float:
    """
    Tail effective sample size: the smaller of the ESS of the indicator
    series 1{x < q05} and 1{x > q95}.

    Bulk ESS says how well the chain estimates the posterior MEAN; it can be
    perfectly adequate while the 2.5% and 97.5% quantiles, which is what a
    credible interval actually reports, are still badly estimated. Since
    every interval in the results tables is a tail quantity, this is the
    figure that licenses those intervals.
    """
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all() or np.ptp(x) == 0:
        return np.nan
    q05, q95 = np.quantile(x, [0.05, 0.95])
    return float(min(effective_sample_size((x < q05).astype(float)),
                     effective_sample_size((x > q95).astype(float))))


@dataclass
class ParameterDiagnostics:
    """
    Per-parameter R-hat and ESS for one named block of a model's output.

    name      : which block ("b_bar", "Sigma", "Delta_b_diag", ...)
    rhat      : (n_params,) rank-normalised folded split-R-hat
    ess_bulk  : (n_params,) effective sample size for posterior means
    ess_tail  : (n_params,) effective sample size for the 5%/95% tails
    n_draws   : kept draws per chain
    n_chains  : number of chains
    """

    name: str
    rhat: np.ndarray
    ess_bulk: np.ndarray
    ess_tail: np.ndarray
    n_draws: int
    n_chains: int

    def summary(self) -> str:
        """One line, the way it would be quoted in a results table."""
        total = self.n_draws * self.n_chains
        finite = np.isfinite(self.rhat)
        worst = np.nanmax(self.rhat) if finite.any() else np.nan
        n_bad = int(np.sum(self.rhat[finite] > 1.01))
        return (f"{self.name:<14s} n={len(self.rhat):>5d}  "
                f"max R-hat {worst:.4f}  ({n_bad} > 1.01)  "
                f"ESS bulk min {np.nanmin(self.ess_bulk):>7.0f} "
                f"median {np.nanmedian(self.ess_bulk):>7.0f} of {total}  "
                f"tail min {np.nanmin(self.ess_tail):>7.0f}")


def _stack_chains(chains: list, field: str) -> np.ndarray:
    """
    Stack one field across chains into (n_chains, n_draws, n_params).

    Fields arrive with different shapes: b_bar is (draws, K), Sigma is
    (draws, N, N), B is (draws, N, K), so everything past the draw axis is
    flattened to a single parameter axis. Matrix-valued fields therefore
    include duplicated off-diagonal entries; harmless for diagnostics, since
    a duplicated parameter simply gets an identical R-hat.
    """
    arrays = [getattr(c, field) for c in chains]
    if any(a is None for a in arrays):
        raise ValueError(f"field '{field}' is None in at least one chain")
    n_draws = min(a.shape[0] for a in arrays)
    if len({a.shape[1:] for a in arrays}) != 1:
        raise ValueError(f"field '{field}' has inconsistent shapes across chains")
    stacked = np.stack([a[:n_draws] for a in arrays])
    return stacked.reshape(stacked.shape[0], n_draws, -1)


def diagnose(chains: list, field: str) -> ParameterDiagnostics:
    """
    Compute R-hat and both ESS variants for every scalar parameter in one
    field, across all supplied chains.

    chains : list of GibbsDraws-like objects, e.g. from load_chains
    field  : attribute name, e.g. "b_bar" or "Sigma"

    Note on cost: this is O(n_params * n_draws log n_draws). For B at
    N=25, K=144 that is 3,600 parameters and takes a few seconds; for
    Delta_b_diag it is instant.
    """
    x = _stack_chains(chains, field)
    m, n, p = x.shape
    rhat = np.empty(p)
    bulk = np.empty(p)
    tail = np.empty(p)
    for j in range(p):
        col = x[:, :, j]
        rhat[j] = rank_normalised_rhat(col)
        bulk[j] = effective_sample_size(col)
        tail[j] = ess_tail(col)
    return ParameterDiagnostics(field, rhat, bulk, tail, n_draws=n, n_chains=m)


def posterior_mean(chains: list, field: str) -> np.ndarray:
    """
    Posterior mean of a field, pooling every draw from every chain.

    Pooling (rather than averaging per-chain means) is correct when the
    chains have equal length, which run_gibbs guarantees, and is what the
    results tables report.
    """
    arrays = [getattr(c, field) for c in chains]
    n_draws = min(a.shape[0] for a in arrays)
    return np.concatenate([a[:n_draws] for a in arrays], axis=0).mean(axis=0)


def credible_interval(chains: list, field: str, level: float = 0.95):
    """
    Equal-tailed credible interval of a field, pooled across chains.
    Returns (lower, upper), each with the field's per-draw shape.
    """
    arrays = [getattr(c, field) for c in chains]
    n_draws = min(a.shape[0] for a in arrays)
    pooled = np.concatenate([a[:n_draws] for a in arrays], axis=0)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(pooled, [alpha, 1.0 - alpha], axis=0)
    return lo, hi


def shrinkage_ratio(chains: list, reference_chains: list,
                    field: str = "b_bar") -> dict:
    """
    How much smaller are the coefficients under one setting than under a
    reference setting?

    Reports the ratio of L2 norms of the posterior means, the ratio of
    medians of |coefficient|, and the correlation between the two coefficient
    vectors. The last matters as much as the first: a prior that shrinks
    everything proportionally leaves the correlation near 1, whereas one that
    reorders which predictors matter does not, and only the second is
    really "selecting" anything.
    """
    a = posterior_mean(chains, field).ravel()
    b = posterior_mean(reference_chains, field).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return {
        "norm_ratio": float(np.linalg.norm(a) / np.linalg.norm(b)),
        "median_abs_ratio": float(np.median(np.abs(a)) / np.median(np.abs(b))),
        "correlation": float(np.corrcoef(a, b)[0, 1]),
    }


def rank_stability(settings: dict, field: str = "b_bar", top: int = 20) -> dict:
    """
    Do different prior settings agree on WHICH predictors matter?

    settings : {label: list-of-chains}

    For every pair of settings, reports Spearman rank correlation of
    |posterior mean| across all parameters, and the overlap among each
    setting's top-`top` predictors by |posterior mean|.

    This is the test that decides whether a hyperparameter choice is
    cosmetic. Coefficients MUST get smaller as the prior tightens; that is
    mechanical and says nothing. What matters is whether the ordering
    survives. High rank correlation lets the choice be reported in one
    sentence; low correlation means it has to be defended.
    """
    labels = list(settings)
    means = {k: np.abs(posterior_mean(v, field).ravel()) for k, v in settings.items()}
    tops = {k: set(np.argsort(-v)[:top]) for k, v in means.items()}

    def spearman(u, v):
        ru = np.empty(len(u)); ru[u.argsort()] = np.arange(len(u))
        rv = np.empty(len(v)); rv[v.argsort()] = np.arange(len(v))
        return float(np.corrcoef(ru, rv)[0, 1])

    out = {}
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            out[(a, b)] = {
                "spearman": spearman(means[a], means[b]),
                f"top{top}_overlap": len(tops[a] & tops[b]) / top,
            }
    return out


def chains_agree(chains_a: list, chains_b: list, field: str = "b_bar") -> dict:
    """
    Do two independent runs targeting the SAME posterior agree, in units of
    Monte Carlo standard error?

    Used for the sequential-versus-blocked comparison. Both samplers target
    the same distribution, so their posterior means must agree up to Monte
    Carlo noise, and the right yardstick is the combined standard error
    sqrt(sd_a^2/ESS_a + sd_b^2/ESS_b), not a raw difference, since a
    difference of 1e-4 is either trivial or damning depending on how precise
    the estimates are.

    Reports the worst and median discrepancy in those units. Anything with a
    worst case under about 4 is agreement.
    """
    xa = _stack_chains(chains_a, field)
    xb = _stack_chains(chains_b, field)
    if xa.shape[2] != xb.shape[2]:
        raise ValueError("different numbers of parameters")

    p = xa.shape[2]
    z = np.empty(p)
    for j in range(p):
        ca, cb = xa[:, :, j], xb[:, :, j]
        ea, eb = effective_sample_size(ca), effective_sample_size(cb)
        se = np.sqrt(ca.var(ddof=1) / ea + cb.var(ddof=1) / eb)
        z[j] = np.abs(ca.mean() - cb.mean()) / se if se > 0 else np.nan

    finite = np.isfinite(z)
    return {
        "max_z": float(np.nanmax(z)),
        "median_z": float(np.nanmedian(z)),
        "frac_within_3": float(np.mean(z[finite] < 3)) if finite.any() else np.nan,
        "n_params": p,
    }

def _spearman(u: np.ndarray, v: np.ndarray) -> float:
    """Spearman rank correlation, no ties expected in posterior means."""
    ru = np.empty(len(u)); ru[u.argsort()] = np.arange(len(u))
    rv = np.empty(len(v)); rv[v.argsort()] = np.arange(len(v))
    return float(np.corrcoef(ru, rv)[0, 1])


def within_setting_rank_stability(chains: list, field: str = "b_bar",
                                  top: int = 20, n_reps: int = 20,
                                  seed: int = 0) -> dict:
    """
    How well does a setting agree with ITSELF on the ranking of |b_bar|?

    Why this is needed
    ------------------
    Comparing rankings ACROSS prior settings (rank_stability) only means
    something relative to how reproducible a ranking is WITHIN one setting.
    When posterior means are tightly compressed (as they are under a
    strongly shrinking prior, where the 2nd through 8th largest coefficients
    can differ by under 10%), the ordering is dominated by Monte Carlo noise,
    and even two halves of the same chains will disagree. A between-setting
    Spearman of 0.63 can be called "unstable" only if the within-setting
    figure is materially higher.

    Method: repeatedly split the pooled draws into two disjoint halves at
    random, rank |mean| within each half, and compare. This isolates Monte
    Carlo error, holding the prior, the data and the sampler fixed, and so
    gives the CEILING that any between-setting comparison could achieve.

    Random halves rather than first-half/second-half: a temporal split would
    also pick up any residual non-stationarity, conflating two different
    things. Here we want Monte Carlo error alone.

    Returns median and 10th percentile of the Spearman correlations and of
    the top-`top` overlap across n_reps splits.
    """
    arrays = [getattr(c, field) for c in chains]
    n_draws = min(a.shape[0] for a in arrays)
    pooled = np.concatenate([a[:n_draws] for a in arrays], axis=0)
    pooled = pooled.reshape(pooled.shape[0], -1)

    rng = np.random.default_rng(seed)
    n = pooled.shape[0]
    spear, overlap = [], []
    for _ in range(n_reps):
        idx = rng.permutation(n)
        a = np.abs(pooled[idx[: n // 2]].mean(axis=0))
        b = np.abs(pooled[idx[n // 2:]].mean(axis=0))
        spear.append(_spearman(a, b))
        overlap.append(len(set(np.argsort(-a)[:top]) & set(np.argsort(-b)[:top])) / top)

    spear, overlap = np.array(spear), np.array(overlap)
    return {
        "spearman_median": float(np.median(spear)),
        "spearman_p10": float(np.percentile(spear, 10)),
        f"top{top}_overlap_median": float(np.median(overlap)),
        f"top{top}_overlap_p10": float(np.percentile(overlap, 10)),
        "n_reps": n_reps,
    }


def stable_core(settings: dict, field: str = "b_bar", top: int = 20) -> dict:
    """
    Which parameters appear in the top-`top` of EVERY setting?

    A stable core with an unstable tail is the signature of rank noise among
    near-ties rather than genuine disagreement between priors, so this is the
    natural companion to the two rank-stability measures.

    Returns the intersection (indices), plus how many settings each parameter
    in the union appears in.
    """
    tops = {k: list(np.argsort(-np.abs(posterior_mean(v, field).ravel()))[:top])
            for k, v in settings.items()}
    sets = {k: set(v) for k, v in tops.items()}
    core = set.intersection(*sets.values()) if sets else set()
    union = set().union(*sets.values()) if sets else set()
    counts = {j: sum(j in s for s in sets.values()) for j in union}
    return {
        "core": sorted(core),
        "counts": counts,
        "n_settings": len(settings),
        "per_setting_top": tops,
    }
