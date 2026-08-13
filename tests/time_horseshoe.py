"""
time_horseshoe_v2.py  --  the timing measurement, second attempt.

v1 produced a step size of 9.1e-22: the sampler collapsed and never moved, so
its 12-second "fit" measured nothing. Two causes, both fixed here.

  1. init="jitter+adapt_diag" (PyMC's default) adds U(-1,1) jitter in
     UNCONSTRAINED space on top of any supplied initvals. b_bar's prior sd is
     7.56e-04, so +-1 is ~1,300 prior sds; displacing b by 1.0 gives predicted
     returns with sd 27.5 against an actual return sd of 0.058. The logp stays
     finite, so nothing raises -- dual averaging just drives the step size to
     zero. Fixed by init="adapt_diag".

  2. The sampled coordinates spanned 1 (z, log lam, log tau) down to 2.4e-04
     (Sigma's packed off-diagonals), against an identity initial metric. Fixed
     by non-dimensionalising every coordinate to unit scale.

Change 2 alters the sampler's COORDINATES, not the target -- the same
relationship blocking has to the Gibbs models. That is verified rather than
asserted: the log-density shifts by exactly K*log(sd_b_bar) and nothing else
(tau's log-space Jacobian cancels its rescaling; pm.Flat contributes zero
either way), so --reference-logp checks the new value against v1's.

Run from the project root:

    caffeinate -i python3 time_horseshoe_v2.py --tune 1000 --draws 200

A VALID run looks like: final step size 1e-3 to 1e-1, mean tree depth 5-9,
few or no divergences. A step size below ~1e-6 or a tree depth pinned at 10
means the geometry is still wrong and the timing is again meaningless -- send
the output rather than the extrapolation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np


def load_universe(universe: str):
    path = Path("../data/processed") / f"{universe}_arrays.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found (run from the project root)")
    d = np.load(path, allow_pickle=True)
    return d["F"], d["R"], d["asset_names"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="size_bm_25")
    ap.add_argument("--p0", type=int, default=23)
    ap.add_argument("--target-r2", type=float, default=0.05)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--target-accept", type=float, default=0.9)
    ap.add_argument("--n-evals", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reference-logp", type=float, default=33910.7,
                    help="v1's logp at the same point; the new value must differ "
                         "by exactly K*log(sd_b_bar)")
    args = ap.parse_args()

    import pymc as pm
    import pytensor

    from src.gibbs.baseline_gaussian import rescaled_hyperparameters
    from src.gibbs.bayesian_lasso import pooled_residual_scale
    from src.likelihood.sur import sur_log_likelihood_pytensor
    from src.priors.inverse_wishart import inverse_wishart_cholesky_logp

    print(f"pymc {pm.__version__} | pytensor {pytensor.__version__} | "
          f"floatX {pytensor.config.floatX}")

    F, R, _ = load_universe(args.universe)
    N, T, K = F.shape
    print(f"\n{args.universe}: N={N} T={T} K={K}")

    # ------------------------------------------------------------ hyperparameters
    base = rescaled_hyperparameters(R, F, K, target_r2=args.target_r2)
    sigma_pooled, B_ols = pooled_residual_scale(R, F)
    tau0 = (args.p0 / (K - args.p0)) * sigma_pooled / np.sqrt(T)
    sd_bbar = float(np.sqrt(base.Delta_b_bar[0, 0]))
    nu_Sigma = float(base.nu_Sigma)
    V_Sigma = np.asarray(base.V_Sigma, dtype=float)

    print(f"  Delta_b_bar = {base.Delta_b_bar[0,0]:.6e}   sd_b_bar = {sd_bbar:.6e}")
    print(f"  sigma pooled = {sigma_pooled:.6f}   tau_0 = {tau0:.6e}")

    n_packed = N * (N + 1) // 2

    # ------------------------------------------------- Sigma's starting point and scales
    # Start Sigma at the OLS residual covariance rather than at pm.Flat's
    # support point of 0, which would mean L = I and Sigma = I -- residual
    # variance 1 against an actual 0.003.
    E_ols = R - np.einsum("itk,ik->it", F, B_ols)
    S0 = np.cov(E_ols, ddof=1)
    L0 = np.linalg.cholesky(S0)
    tri = np.tril_indices(N)
    is_diag = tri[0] == tri[1]

    packed0 = L0[tri].copy()
    packed0[is_diag] = np.log(np.diag(L0))      # the diagonal is stored as log

    # per-entry scale, so the unconstrained coordinate is O(1). For T
    # observations the Cholesky log-diagonals have sd ~ 1/sqrt(2T) and the
    # off-diagonals in column j have sd ~ L_jj/sqrt(T).
    packed_scale = np.empty(n_packed)
    packed_scale[is_diag] = 1.0 / np.sqrt(2.0 * T)
    packed_scale[~is_diag] = np.diag(L0)[tri[1][~is_diag]] / np.sqrt(T)
    print(f"  Sigma packed scales: log-diag {packed_scale[is_diag][0]:.2e}, "
          f"off-diag {packed_scale[~is_diag].min():.2e}-{packed_scale[~is_diag].max():.2e}")

    # ------------------------------------------------------------ the model
    # Every sampled coordinate is O(1). b_bar_z, z, log lam_z, log tau_z and
    # Sigma_packed_z all have unit natural scale, so an identity initial metric
    # is approximately right for all 7,670 of them.
    with pm.Model() as model:
        b_bar_z = pm.Normal("b_bar_z", mu=0.0, sigma=1.0, shape=K)
        b_bar = pm.Deterministic("b_bar", sd_bbar * b_bar_z)

        z = pm.Normal("z", mu=0.0, sigma=1.0, shape=(N, K))
        lam = pm.HalfCauchy("lam", beta=1.0, shape=(N, K))
        tau_z = pm.HalfCauchy("tau_z", beta=1.0)
        tau = pm.Deterministic("tau", tau0 * tau_z)

        b = b_bar[None, :] + z * lam * tau

        packed_z = pm.Flat("Sigma_packed_z", shape=n_packed)
        packed_raw = packed0 + packed_scale * packed_z
        Sigma, sigma_logp = inverse_wishart_cholesky_logp(packed_raw, nu_Sigma, V_Sigma)
        pm.Potential("Sigma_prior", sigma_logp)

        pm.Potential("likelihood", sur_log_likelihood_pytensor(R, F, b, Sigma))

    initvals = {
        "b_bar_z": B_ols.mean(axis=0) / sd_bbar,
        "z": np.zeros((N, K)),
        "lam": np.ones((N, K)),
        "tau_z": 1.0,
        "Sigma_packed_z": np.zeros(n_packed),
    }

    ip = model.initial_point()
    expected = {"b_bar_z", "z", "lam_log__", "tau_z_log__", "Sigma_packed_z"}
    if set(ip) != expected:
        raise SystemExit(f"unexpected point keys {sorted(ip)}; expected {sorted(expected)}")
    ip["b_bar_z"] = initvals["b_bar_z"]
    ip["z"] = np.zeros((N, K))
    ip["lam_log__"] = np.zeros((N, K))
    ip["tau_z_log__"] = np.zeros(())
    ip["Sigma_packed_z"] = np.zeros(n_packed)

    # --------------------------------------------------- logp, gradient, and the check
    t0 = perf_counter()
    fn_logp = model.compile_logp()
    fn_dlogp = model.compile_dlogp()
    print(f"\ncompile: {perf_counter()-t0:.1f}s")

    lp = float(fn_logp(ip))
    offset = K * np.log(sd_bbar)
    predicted = args.reference_logp + offset
    print(f"  logp at start = {lp:,.1f}")
    print(f"  v1 was {args.reference_logp:,.1f}; the reparameterisation shifts it by "
          f"K*log(sd_b_bar) = {offset:,.1f}, so this should be {predicted:,.1f}")
    if not np.isfinite(lp):
        raise SystemExit("logp is not finite -- stop and diagnose.")
    if abs(lp - predicted) > 1.0:
        print(f"  *** MISMATCH of {lp - predicted:,.2f}: the reparameterisation is NOT "
              f"the same model. Do not trust anything below. ***")
    else:
        print(f"  agrees to {abs(lp - predicted):.3f} -- same target, new coordinates.")

    def timeit(fn, n):
        fn(ip)
        t = perf_counter()
        for _ in range(n):
            fn(ip)
        return (perf_counter() - t) / n

    t_logp = timeit(fn_logp, args.n_evals)
    t_grad = timeit(fn_dlogp, args.n_evals)
    print(f"  logp     : {t_logp*1e3:8.3f} ms")
    print(f"  gradient : {t_grad*1e3:8.3f} ms   (v1 measured 1.460 ms)")

    # ------------------------------------------------------------- the sampling run
    print(f"\nsampling {args.tune} tune + {args.draws} draws, 1 chain, "
          f"init=adapt_diag (NO jitter), target_accept={args.target_accept} ...")
    t0 = perf_counter()
    with model:
        idata = pm.sample(draws=args.draws, tune=args.tune, chains=1, cores=1,
                          target_accept=args.target_accept, initvals=initvals,
                          init="adapt_diag", random_seed=args.seed, progressbar=True,
                          compute_convergence_checks=False)
    t_sample = perf_counter() - t0

    ss = idata.sample_stats
    get = lambda k: ss[k].values.ravel() if k in ss else None
    div, nsteps, depth, step = get("diverging"), get("n_steps"), get("tree_depth"), get("step_size")

    print(f"\nshort run: {t_sample:.0f}s for {args.tune}+{args.draws} iterations")
    if div is not None:
        print(f"  divergences        : {int(div.sum())} of {div.size}")
    if depth is not None:
        print(f"  tree depth         : mean {depth.mean():.2f}, max {int(depth.max())}, "
              f"saturating (>=10) {np.mean(depth >= 10)*100:.1f}%")
    if nsteps is not None:
        print(f"  leapfrogs per draw : mean {nsteps.mean():.1f}, max {int(nsteps.max())}")
    if step is not None:
        ok = "OK" if step[-1] > 1e-6 else "*** COLLAPSED -- timing is meaningless ***"
        print(f"  final step size    : {step[-1]:.3e}   {ok}")

    # did anything actually move?
    po = idata.posterior
    print(f"\n  posterior sd: b_bar {float(po['b_bar'].std(dim=('chain','draw')).mean()):.3e}"
          f"  tau {float(po['tau'].std()):.3e}  tau mean {float(po['tau'].mean()):.3e}"
          f"  (tau_0 = {tau0:.3e})")

    # ------------------------------------------------------------------ extrapolation
    if nsteps is not None and step is not None and step[-1] > 1e-6:
        per_draw = nsteps.mean() * t_grad
        print(f"\nextrapolation: {nsteps.mean():.0f} leapfrogs x {t_grad*1e3:.2f} ms "
              f"= {per_draw:.3f} s per post-warmup draw")
        rows = [("production, 4 chains x 2,000 (1,000 tune), sequential", 4 * 2000),
                ("production, 4 chains x 2,000, 4 cores in parallel", 2000),
                ("one backtest fit, 1,500 (500 tune)", 1500),
                ("backtest, 40 fits x 1,500", 40 * 1500),
                ("backtest, 20 fits x 1,500 (24-month refit)", 20 * 1500)]
        for label, n_it in rows:
            secs = n_it * per_draw
            print(f"  {label:<52s} {secs/60:8.1f} min  ({secs/3600:5.2f} h)")
        print("\nAgainst the Gibbs costs: baseline backtest 40 fits ~1.7 h, LASSO ~3.8 h. "
              "Up to ~10 h keeps the annual schedule for all four models.")


if __name__ == "__main__":
    main()