"""Plot the K=50 hierarchical-mixture learning-rate/floor sweep."""
import sys as _sys, pathlib as _pl

_ROOT = next(
    _q for _q in _pl.Path(__file__).resolve().parents if (_q / "gmvi").is_dir()
)
_sys.path[:0] = [str(_ROOT), str(_ROOT / "Visualizations")]
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Visualisations"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
BASE = "hier50_lr_anneal_floor"
NSTEPS = 2000

try:
    import _style as S

    S.setup()
    FIG = S.figsize(1.0, 0.42)
    save_figure = lambda fig, name: S.save(fig, name, HERE)
except Exception:
    FIG = (11, 4.4)

    def save_figure(fig, name):
        for ext in ("pdf", "png"):
            fig.savefig(
                os.path.join(HERE, f"{name}.{ext}"), bbox_inches="tight", dpi=150
            )
        plt.close(fig)
        print(f"  wrote {name}.pdf {name}.png")


st = pd.read_csv(os.path.join(HERE, f"{BASE}_steps.csv"))
ck = pd.read_csv(os.path.join(HERE, f"{BASE}_checkpoints.csv"))
lrs = sorted(st.lr.unique())
colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(lrs)))

fig, axes = plt.subplots(1, 2, figsize=FIG, sharey=True)
for ax, (schedule, title) in zip(
    axes,
    [
        ("strict", "strict (constant) LR"),
        ("anneal_floor_0.1", r"cosine anneal, $\eta_{min}=0.1\eta_0$"),
    ],
):
    data = st[st.schedule == schedule]
    for lr, color in zip(lrs, colors):
        curve = -data[data.lr == lr].set_index("step").elbo
        curve = curve.rolling(21, center=True, min_periods=1).median()
        ax.plot(curve.index, curve.values, color=color, lw=1.0, label=f"{lr:g}")
    ax.set_xscale("symlog")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_title(title)
axes[0].set_ylabel(S.objective_label("hierarchical_mixture") + " (one seed)")
axes[0].legend(title=r"$\eta_0$", fontsize=7, frameon=False)
save_figure(fig, f"{BASE}_curves")

# High-resolution checkpoint ELBO shows whether rankings change during training.
fig, axes = plt.subplots(1, 2, figsize=FIG, sharey=True)
for ax, (schedule, title) in zip(
    axes,
    [
        ("strict", "strict (constant) LR"),
        ("anneal_floor_0.1", r"cosine anneal, $\eta_{min}=0.1\eta_0$"),
    ],
):
    data = ck[ck.schedule == schedule]
    for lr, color in zip(lrs, colors):
        d = data[data.lr == lr].sort_values("step")
        ax.plot(d.step, -d.elbo, "o-", color=color, ms=3, lw=1.0, label=f"{lr:g}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("checkpoint step")
    ax.set_title(title)
axes[0].set_ylabel("high-resolution " + S.objective_label("hierarchical_mixture"))
axes[0].legend(title=r"$\eta_0$", fontsize=7, frameon=False)
save_figure(fig, f"{BASE}_checkpoints")

print(f"\n=== D_KL(q||p) at step {NSTEPS} (one seed) ===")
final = ck[ck.step == NSTEPS].copy()
final["minus_elbo"] = -final.elbo
print(
    final.pivot(index="schedule", columns="lr", values="minus_elbo")
    .round(3)
    .to_string()
)
