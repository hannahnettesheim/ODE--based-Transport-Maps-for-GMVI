"""dopri5 tolerance sweep, K=50 hier mixture: -ELBO learning curves (log y) and
NFE vs tolerance."""
import sys as _sys, pathlib as _pl

_ROOT = next(
    _q for _q in _pl.Path(__file__).resolve().parents if (_q / "gmvi").is_dir()
)
_sys.path[:0] = [str(_ROOT), str(_ROOT / "Visualizations")]
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Visualisations"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import _style as S

    S.setup()
    FIG = S.figsize(1.0, 0.42)
    saver = lambda fig, n: S.save(fig, n, HERE)
except Exception:
    FIG = (11, 4.4)

    def saver(fig, n):
        for e in ("pdf", "png"):
            fig.savefig(os.path.join(HERE, f"{n}.{e}"), bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  wrote {n}.pdf {n}.png")


HERE = os.path.dirname(__file__)
BASE = "dopri_hier50"
st = pd.read_csv(os.path.join(HERE, f"{BASE}_steps.csv"))
ck = pd.read_csv(os.path.join(HERE, f"{BASE}_checkpoints.csv"))
TOLS = sorted(st.solver_param.unique())
C = plt.cm.viridis(np.linspace(0.05, 0.95, len(TOLS)))
COL = {t: C[i] for i, t in enumerate(TOLS)}

fig, (a0, a1) = plt.subplots(1, 2, figsize=FIG)

# learning curves: -ELBO, log y
for t in TOLS:
    g = st[st.solver_param == t].groupby("step").elbo
    med = (-g.median()).rolling(15, center=True, min_periods=1).median()
    lo = (-g.quantile(0.9)).rolling(15, center=True, min_periods=1).median()
    hi = (-g.quantile(0.1)).rolling(15, center=True, min_periods=1).median()
    a0.plot(med.index, med.values, color=COL[t], lw=1.0, label=f"{t:g}")
    a0.fill_between(med.index, lo.values, hi.values, color=COL[t], alpha=0.10, lw=0)
a0.set_xscale("symlog")
a0.set_yscale("log")
a0.set_xlabel("step")
a0.set_ylabel(S.objective_label("hierarchical_mixture") + " (median / seeds)")
d = st[st.step >= 900].groupby("seed").elbo.mean()
a0.set_ylim(
    0.6 * float(-d.min()), float(-st[st.step.between(20, 40)].elbo.median()) * 1.3
)
a0.set_title(S.target_name("hier_k50", short=True) + ", dopri5 tol sweep")
a0.legend(title="rtol=atol", fontsize=6, ncol=2, frameon=False)

# NFE vs tolerance
f1 = ck[ck.step == 1000].groupby("solver_param")[["nfe_mean", "nfe_max"]].mean()
a1.plot(f1.index, f1.nfe_mean, "o-", label="mean")
a1.plot(f1.index, f1.nfe_max, "x--", alpha=0.6, label="max")
a1.set_xscale("log")
a1.set_xlabel("dopri5 rtol=atol")
a1.set_ylabel("NFE / step")
a1.set_title("cost")
a1.legend(fontsize=7, frameon=False)

fig.subplots_adjust(wspace=0.32)
saver(fig, "dopri_hier50_curves")

print("\n=== D_KL(q||p) / NFE at step 1000 (mean / std over seeds) ===")
g = (
    ck[ck.step == 1000]
    .groupby("solver_param")
    .agg(
        neg_elbo=("elbo", lambda s: -s.mean()),
        sd=("elbo", "std"),
        nfe_mean=("nfe_mean", "mean"),
        nfe_max=("nfe_max", "mean"),
    )
)
print(g.round(3).to_string())
