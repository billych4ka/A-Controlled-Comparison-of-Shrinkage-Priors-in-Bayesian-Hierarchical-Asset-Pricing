import numpy as np
from src.gibbs.bayesian_lasso import lasso_hyperparameters, run_gibbs
d = np.load('data/processed/size_bm_25_arrays.npz', allow_pickle=True)
F, R = d['F'], d['R']
Tw = 240
hp = lasso_hyperparameters(R[:, :Tw], F[:, :Tw, :], 144)
Bs, lams = [], []
for sd in range(3):
    m = run_gibbs(R[:, :Tw], F[:, :Tw, :], hp, n_draws=1200, n_burn=700, seed=sd)
    Bs.append(m.B.mean(axis=0)); lams.append(m.lam.mean())
    print(f"seed {sd}: lambda {m.lam.mean():.0f} +- {m.lam.std():.0f}")
print(f"\ncorrelation of B across seeds: "
      f"{np.corrcoef(Bs[0].ravel(), Bs[1].ravel())[0,1]:.4f}, "
      f"{np.corrcoef(Bs[0].ravel(), Bs[2].ravel())[0,1]:.4f}")
print(f"max |B diff| / sd(B): {np.abs(Bs[0]-Bs[1]).max()/Bs[0].std():.3f}")