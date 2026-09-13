"""
check_divergences_horseshoe.py: does target_accept fix the horseshoe's divergences,
and where are they actually coming from?

WHAT THE FIRST ROUND ESTABLISHED (check_divergences_horseshoe.py, 13 Aug)

  simple (P&V Appendix C.1)      16/200 divergences, 471 s, tau/tau_0 = 0.0521
  decomposed (P&V Appendix C.2)  13/200 divergences, 893 s, tau/tau_0 = 0.0538

  C.2 REJECTED: 16 vs 13 is inside binomial noise (sd ~3.5 at n=200; three runs
  of the IDENTICAL simple model gave 10 and 16), for 1.90x the wall clock. The
  location scan showed why it was never going to help: divergent draws sat at
  the 60th percentile of log tau under C.1 and the 44th under C.2, either side
  of 50 and disagreeing with each other, so THE DIVERGENCES ARE NOT IN THE
  LOW-TAU NECK. C.2 is aimed at a neck that is not the problem.

  Kept as evidence: the two parameterisations agree (b_bar median |z| 0.71, max
  3.45, 99.3% within 3 SE; tau |z| 0.54), which validates the simple form and
  rules out tau's twentyfold gap being a coordinate artefact.

WHAT THIS SCRIPT DOES

  1. A wider location scan: log tau, mean AND max log lambda, ||b_bar_z||,
     ||Sigma_packed_z||, and the energy. The first scan checked two quantities
     and both came back null, so the cause is in something not yet looked at.
  2. E-BFMI computed directly from the energy trace. arviz 1.2.0's az.bfmi
     raised a TypeError; the quantity is three lines of arithmetic and does not
     need the API.
  3. Saves the posterior mean of b, so the settings can be compared on the
     quantity the backtest actually uses.

USAGE: the ta=0.9 run must be REPEATED here, since the earlier file predates
the scan fields:

    caffeinate -i python3 check_divergences_horseshoe.py --target-accept 0.9
    caffeinate -i python3 check_divergences_horseshoe.py --target-accept 0.95
    caffeinate -i python3 check_divergences_horseshoe.py --target-accept 0.99
    python3 check_divergences_horseshoe.py --compare 0.9 0.99

Expect roughly 8, 12 and 20+ minutes: the step size falls as target_accept
rises, so tree depth goes 8 -> 9 and the cost per draw doubles at each step.
That doubling IS the decision-relevant number as much as the divergence count.

READING IT: if divergences fall to ~0 by 0.99, apply that setting uniformly
(option A) and the backtest costs ~11 h per universe. If they plateau around
5%, target_accept is not the remedy and the question becomes disclosure,
or the regularised horseshoe, whose bounded tails address exactly this
geometry. Do not reach for that first.
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


def tag_for(param: str, ta: float) -> str:
    return f"divcheck2_{param}_ta{ta:g}".replace(".", "p") + ".npz"


def ebfmi(energy: np.ndarray) -> float:
    """E-BFMI = sum (E_t - E_{t-1})^2 / sum (E_t - Ebar)^2. Below 0.3 is a warning."""
    e = np.asarray(energy, dtype=float).ravel()
    return float(np.sum(np.diff(e) ** 2) / np.sum((e - e.mean()) ** 2))


def run(args) -> None:
    import pymc as pm

    from src.gibbs.baseline_gaussian import rescaled_hyperparameters
    from src.gibbs.bayesian_lasso import pooled_residual_scale
    from src.likelihood.sur import sur_log_likelihood_pytensor
    from src.priors.inverse_wishart import inverse_wishart_cholesky_logp

    F, R, _ = load_universe(args.universe)
    N, T, K = F.shape

    base = rescaled_hyperparameters(R, F, K, target_r2=args.target_r2)
    sigma_pooled, B_ols = pooled_residual_scale(R, F)
    tau0 = (args.p0 / (K - args.p0)) * sigma_pooled / np.sqrt(T)
    sd_bbar = float(np.sqrt(base.Delta_b_bar[0, 0]))
    V_Sigma = np.asarray(base.V_Sigma, dtype=float)

    E_ols = R - np.einsum("itk,ik->it", F, B_ols)
    S0 = np.cov(E_ols, ddof=1)
    L0 = np.linalg.cholesky(S0)
    tri = np.tril_indices(N)
    is_diag = tri[0] == tri[1]
    packed0 = L0[tri].copy()
    packed0[is_diag] = np.log(np.diag(L0))
    packed_scale = np.empty(packed0.size)
    packed_scale[is_diag] = 1.0 / np.sqrt(2.0 * T)
    packed_scale[~is_diag] = np.diag(L0)[tri[1][~is_diag]] / np.sqrt(T)

    with pm.Model() as model:
        b_bar_z = pm.Normal("b_bar_z", 0.0, 1.0, shape=K)
        b_bar = pm.Deterministic("b_bar", sd_bbar * b_bar_z)
        z = pm.Normal("z", 0.0, 1.0, shape=(N, K))
        lam = pm.HalfCauchy("lam", beta=1.0, shape=(N, K))
        tau_z = pm.HalfCauchy("tau_z", beta=1.0)
        tau = pm.Deterministic("tau", tau0 * tau_z)
        b = pm.Deterministic("b", b_bar[None, :] + z * lam * tau)

        packed_z = pm.Flat("Sigma_packed_z", shape=packed0.size)
        Sigma, sigma_logp = inverse_wishart_cholesky_logp(
            packed0 + packed_scale * packed_z, float(base.nu_Sigma), V_Sigma)
        pm.Potential("Sigma_prior", sigma_logp)
        pm.Potential("likelihood", sur_log_likelihood_pytensor(R, F, b, Sigma))

    initvals = {"b_bar_z": B_ols.mean(axis=0) / sd_bbar,
                "z": np.zeros((N, K)), "lam": np.ones((N, K)), "tau_z": 1.0,
                "Sigma_packed_z": np.zeros(packed0.size)}

    print(f"{args.universe} | tau_0={tau0:.6e} | target_accept={args.target_accept}")
    t0 = perf_counter()
    with model:
        idata = pm.sample(draws=args.draws, tune=args.tune, chains=args.chains,
                          cores=args.cores, target_accept=args.target_accept,
                          initvals=initvals, init="adapt_diag",
                          random_seed=args.seed, progressbar=True,
                          compute_convergence_checks=False)
    elapsed = perf_counter() - t0

    ss, po = idata.sample_stats, idata.posterior
    div = ss["diverging"].values.ravel().astype(bool)
    depth = ss["tree_depth"].values.ravel()
    nsteps = ss["n_steps"].values.ravel()
    step = ss["step_size"].values.ravel()
    energy = ss["energy"].values

    lam_v = po["lam"].values
    scan = {
        "log_tau": np.log(po["tau"].values).ravel(),
        "mean_log_lam": np.log(lam_v).mean(axis=(-1, -2)).ravel(),
        "max_log_lam": np.log(lam_v).max(axis=(-1, -2)).ravel(),
        "norm_b_bar_z": np.linalg.norm(po["b_bar_z"].values, axis=-1).ravel(),
        "norm_Sigma_z": np.linalg.norm(po["Sigma_packed_z"].values, axis=-1).ravel(),
        "energy": energy.ravel(),
    }

    print(f"\ntarget_accept={args.target_accept}: {elapsed:.0f}s")
    print(f"  divergences    : {int(div.sum())} of {div.size} ({100*div.mean():.1f}%)"
          f"   [binomial sd ~{np.sqrt(div.size*div.mean()*(1-div.mean())):.1f}]")
    print(f"  tree depth     : mean {depth.mean():.2f}, max {int(depth.max())}, "
          f"saturating {100*np.mean(depth >= 10):.1f}%")
    print(f"  leapfrogs/draw : mean {nsteps.mean():.1f}   step size {step[-1]:.3e}")
    print(f"  E-BFMI         : {' '.join(f'{ebfmi(e):.3f}' for e in energy)}"
          f"   (below 0.3 is a warning)")
    tau_d = po["tau"].values.ravel()
    print(f"  tau            : mean {tau_d.mean():.4e}, sd {tau_d.std():.4e}, "
          f"tau/tau_0 = {tau_d.mean()/tau0:.4f}")

    if div.sum() > 0:
        print(f"\n  where the {int(div.sum())} divergent draws sit, as a percentile of "
              f"the non-divergent ones:")
        for name, v in scan.items():
            dv, nd = v[div], v[~div]
            pct = 100.0 * np.mean(nd[None, :] < dv[:, None])
            marker = "  <-- LOCALISED" if (pct < 25 or pct > 75) else ""
            print(f"    {name:<14s} {pct:5.1f}th   (divergent mean {dv.mean():+11.4g} "
                  f"vs {nd.mean():+11.4g}){marker}")
        print("    anything below 25 or above 75 is a real concentration; "
              "everything near 50 means\n    the divergences are not associated with "
              "any of these coordinates.")

    out = Path(f"results/{args.universe}/horseshoe/diagnostics")
    out.mkdir(parents=True, exist_ok=True)
    path = out / tag_for(args.param, args.target_accept)
    np.savez_compressed(path, b_bar=po["b_bar"].values, tau=po["tau"].values,
                        b_mean=po["b"].values.mean(axis=(0, 1)),
                        b_sd=po["b"].values.std(axis=(0, 1)),
                        diverging=div, tree_depth=depth, n_steps=nsteps,
                        step_size=step, elapsed=elapsed, tau0=tau0,
                        target_accept=args.target_accept, **scan)
    print(f"\nsaved -> {path}")


def compare(args) -> None:
    """Do two target_accept settings give the same answer, in MCSE units?"""
    import arviz as az
    ta_a, ta_b = args.compare
    out = Path(f"results/{args.universe}/horseshoe/diagnostics")
    fa, fb = out / tag_for(args.param, ta_a), out / tag_for(args.param, ta_b)
    for f in (fa, fb):
        if not f.exists():
            raise SystemExit(f"{f} missing; run it first")
    A, B = np.load(fa, allow_pickle=True), np.load(fb, allow_pickle=True)

    print(f"target_accept {ta_a:g} vs {ta_b:g}\n")
    for lbl, d in ((f"{ta_a:g}", A), (f"{ta_b:g}", B)):
        dv = d["diverging"]
        print(f"  ta={lbl:<5s} divergences {int(dv.sum())}/{dv.size} "
              f"({100*dv.mean():4.1f}%)  depth {d['tree_depth'].mean():.2f}  "
              f"{float(d['elapsed']):5.0f}s  tau/tau_0 {d['tau'].mean()/float(d['tau0']):.4f}")

    for field in ("b_bar", "tau"):
        a, b = A[field], B[field]
        fa_, fb_ = a.reshape(a.shape[0], a.shape[1], -1), b.reshape(b.shape[0], b.shape[1], -1)
        ma, mb = fa_.mean(axis=(0, 1)), fb_.mean(axis=(0, 1))
        sa, sb = fa_.std(axis=(0, 1)), fb_.std(axis=(0, 1))
        ea = np.array([az.ess(fa_[:, :, j]) for j in range(fa_.shape[2])])
        eb = np.array([az.ess(fb_[:, :, j]) for j in range(fb_.shape[2])])
        se = np.sqrt(sa**2 / np.maximum(ea, 1) + sb**2 / np.maximum(eb, 1))
        zs = np.abs(ma - mb) / np.where(se > 0, se, np.nan)
        print(f"\n  {field:<7s} n={zs.size:4d}  median |z| {np.nanmedian(zs):5.2f}  "
              f"max |z| {np.nanmax(zs):6.2f}  within 3 SE {100*np.nanmean(zs<3):5.1f}%"
              f"  median ESS {np.median(ea):.0f} vs {np.median(eb):.0f}")

    mba, mbb = A["b_mean"].ravel(), B["b_mean"].ravel()
    sd = 0.5 * (A["b_sd"].ravel() + B["b_sd"].ravel())
    print(f"\n  posterior mean of b, the ONLY thing the backtest uses:")
    print(f"    correlation {np.corrcoef(mba, mbb)[0,1]:.6f}   "
          f"max |diff| {np.max(np.abs(mba-mbb))/np.mean(sd):.3f} of b's own posterior sd")
    print("\n  Read the MEDIAN, not the max: with 144 comparisons two perfectly agreeing\n"
          "  estimates still give a max |z| near 3.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", nargs=2, type=float, metavar=("TA_A", "TA_B"))
    ap.add_argument("--param", default="simple")
    ap.add_argument("--universe", default="size_bm_25")
    ap.add_argument("--p0", type=int, default=23)
    ap.add_argument("--target-r2", type=float, default=0.05)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--chains", type=int, default=1)
    ap.add_argument("--cores", type=int, default=1)
    ap.add_argument("--target-accept", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    compare(args) if args.compare else run(args)


if __name__ == "__main__":
    main()

