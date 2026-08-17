import numpy as np
from src.diagnostics.convergence import load_chains
from src.nuts.horseshoe import HorseshoeDraws
ch = load_chains("size_op_25", "horseshoe", "p0_23_r2_0p05", HorseshoeDraws)
m = np.array([c.b_bar.mean(0) for c in ch])
for k in range(4):
    others = np.delete(np.arange(4), k)
    print(f"chain {k}: corr with the other three "
          f"{np.corrcoef(m[k], m[others].mean(0))[0,1]:.5f}")