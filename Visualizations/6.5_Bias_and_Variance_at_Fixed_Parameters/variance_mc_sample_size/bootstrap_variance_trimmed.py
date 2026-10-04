"""Match the dimensional study: 200 replicate-bootstrap resamples, 95% percentiles."""
from pathlib import Path
import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
d = pd.read_csv(H / "variance_bias_results.csv")
d = d[d.regime.isin(["rosenbrock_extreme", "hier_k5"])].copy()
for i, row in d.iterrows():
    step, M, R = int(row.step), int(row.M), int(row.replicates)
    cell = H / row.regime / f"step_{step}_M_{M}"
    files = sorted(
        cell.glob("gradients_r*.npz"), key=lambda p: int(p.stem.split("_r")[1])
    )
    G = np.concatenate([np.load(p)["gradients"] for p in files])
    assert len(G) == R, (cell, len(G), R)
    np.testing.assert_allclose(G.var(0, ddof=1).sum(), row.variance, rtol=1e-6)
    rng = np.random.default_rng(38291 + step + M)
    boot = np.array(
        [G[rng.integers(R, size=R)].var(0, ddof=1).sum() for _ in range(200)]
    )
    d.loc[i, "variance_bootstrap_lo"] = np.quantile(boot, 0.025)
    d.loc[i, "variance_bootstrap_hi"] = np.quantile(boot, 0.975)
    d.loc[i, "bootstrap_resamples"] = 200
assert len(d) == 24
d.to_csv(H / "variance_M_rosenbrock_hier_k5_bootstrap95.csv", index=False)
print("Recomputed 24 intervals; variance point estimates verified unchanged.")
