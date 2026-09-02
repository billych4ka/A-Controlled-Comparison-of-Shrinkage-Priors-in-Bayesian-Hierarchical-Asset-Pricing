import numpy as np
from src.diagnostics.convergence import load_chains
from src.nuts.horseshoe import HorseshoeDraws
ch = load_chains("size_bm_25","horseshoe","p0_23_r2_0p05",HorseshoeDraws)
B = np.concatenate([c.B for c in ch], axis=0)
# MCSE of the 2.5% quantile ≈ 2.675 * sd / sqrt(ESS)
sd = B.std(axis=0)
print("posterior sd of B, median:", np.median(sd))
print("at ESS 500, MCSE of a 2.5% quantile ≈", 2.675*np.median(sd)/np.sqrt(500))
print("as a fraction of the interval half-width:", 2.675/np.sqrt(500)/1.96)