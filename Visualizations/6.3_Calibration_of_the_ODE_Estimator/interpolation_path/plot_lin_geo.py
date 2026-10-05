"""linear vs geometric: -ELBO learning curves per regime, + final -ELBO / NFE."""
import sys as _sys, pathlib as _pl

_ROOT = next(
    _q for _q in _pl.Path(__file__).resolve().parents if (_q / "gmvi").is_dir()
)
_sys.path[:0] = [str(_ROOT), str(_ROOT / "Visualizations")]
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import _style as S

# Use the shared target names with this plot's local typography.
plt.rcdefaults()
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    }
)
FIG = (10, 6)


def save_figure(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  wrote {name}.pdf {name}.png")


HERE = os.path.dirname(__file__)
BASE = "lin_geo"
st = pd.read_csv(os.path.join(HERE, f"{BASE}_steps.csv"))
ck = pd.read_csv(os.path.join(HERE, f"{BASE}_checkpoints.csv"))
REG = ["banana", "rosenbrock_extreme", "hier_k5", "hier_k50"]
PCOL = {"linear": "#2A9D8F", "geometric": "#457B9D"}

fig, axes = plt.subplots(2, 2, figsize=FIG, sharex=True)
for ax, rk in zip(axes.flat, REG):
    d = st[st.regime == rk]
    for pth in ("linear", "geometric"):
        g = d[d.path == pth].groupby("step").elbo
        med = (-g.median()).rolling(15, center=True, min_periods=1).median()
        lo = (-g.quantile(0.9)).rolling(15, center=True, min_periods=1).median()
        hi = (-g.quantile(0.1)).rolling(15, center=True, min_periods=1).median()
        ax.plot(med.index, med.values, color=PCOL[pth], lw=1.1, label=pth)
        ax.fill_between(
            med.index, lo.values, hi.values, color=PCOL[pth], alpha=0.12, lw=0
        )
    ax.set_title(S.target_name(rk))
    ax.set_xscale("symlog")
    ax.set_yscale("log")
    fin = -d[d.step >= 900].groupby("seed").elbo.mean()
    ax.set_ylim(
        0.6 * float(fin.min()), float(-d[d.step.between(20, 40)].elbo.median()) * 1.3
    )
for ax in axes[-1]:
    ax.set_xlabel("step")
for ax in axes[:, 0]:
    ax.set_ylabel(S.objective_label())
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, -0.01),
)
fig.tight_layout(rect=(0, 0.07, 1, 1))
save_figure(fig, "lin_geo_curves")

# Paired relative difference. Pairing by seed and step removes much of the
# between-seed variation that obscures the comparison in two overlaid curves.
# Positive values mean geometric has the larger (worse) -ELBO.
fig, axes = plt.subplots(2, 2, figsize=FIG, sharex=True)
for ax, rk in zip(axes.flat, REG):
    wide = (
        st[st.regime == rk]
        .pivot(index=["seed", "step"], columns="path", values="elbo")
        .dropna(subset=["linear", "geometric"])
    )
    loss_lin = -wide["linear"]
    loss_geo = -wide["geometric"]
    wide["relative_diff"] = loss_geo - loss_lin

    # Smooth within each paired seed before summarising across seeds.
    diff = (
        wide["relative_diff"]
        .unstack("seed")
        .rolling(15, center=True, min_periods=1)
        .median()
    )
    med = diff.median(axis=1)
    lo = diff.quantile(0.1, axis=1)
    hi = diff.quantile(0.9, axis=1)
    ax.fill_between(med.index, lo, hi, color="#6C757D", alpha=0.18, lw=0)
    ax.plot(med.index, med, color="#343A40", lw=1.1)
    ax.axhline(0, color="#D1495B", lw=0.8, ls="--")
    ax.set_title(S.target_name(rk))
    ax.set_xscale("symlog")
for ax in axes[-1]:
    ax.set_xlabel("step")
for ax in axes[:, 0]:
    ax.set_ylabel(r"relative $\Delta D_{\mathrm{KL}}(q\,\|\,p)$ [\%]")
fig.suptitle("Geometric minus linear (paired by seed)", fontsize=11)
fig.legend(
    handles=[
        Line2D([], [], color="#343A40", lw=1.1, label="paired median"),
        Patch(facecolor="#6C757D", alpha=0.18, label="10th--90th percentile"),
        Line2D([], [], color="#D1495B", lw=0.8, ls="--", label="no difference"),
    ],
    loc="lower center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, -0.01),
)
fig.tight_layout(rect=(0, 0.07, 1, 0.96))
save_figure(fig, "lin_geo_relative_difference")

fin = ck[ck.step == 1000]
forward = st.groupby(["regime", "path", "seed"], as_index=False).wall_s.mean()
fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(FIG[0], FIG[1] * 0.45))
x = np.arange(len(REG))
w = 0.35
for k, pth in enumerate(("linear", "geometric")):
    g = fin[fin.path == pth].groupby("regime")
    m = (-g.elbo.mean()).reindex(REG)
    sd = g.elbo.std().reindex(REG)
    a0.bar(
        x + (k - 0.5) * w,
        m.values,
        w,
        yerr=sd.values,
        capsize=2,
        color=PCOL[pth],
        label=pth,
    )
    n = g.nfe_mean.mean().reindex(REG)
    a1.bar(x + (k - 0.5) * w, n.values, w, color=PCOL[pth], label=pth)
    timing = forward[forward.path == pth].groupby("regime").wall_s
    timing_mean = (1000 * timing.mean()).reindex(REG)
    timing_sd = (1000 * timing.std()).reindex(REG)
    a2.bar(
        x + (k - 0.5) * w,
        timing_mean.values,
        w,
        yerr=timing_sd.values,
        capsize=2,
        color=PCOL[pth],
        label=pth,
    )
a0.set_yscale("log")
a0.set_ylabel(r"final $D_{\mathrm{KL}}(q\,\|\,p)$")
a1.set_ylabel(r"NFE of $v_t^\theta$ / step")
a2.set_ylabel("forward-pass time / step [ms]")
for a in (a0, a1, a2):
    a.set_xticks(x)
    a.set_xticklabels(
        [S.target_name(r, short=True) for r in REG], rotation=20, ha="right", fontsize=7
    )
handles, labels = a0.get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, 1.02),
)
fig.tight_layout(rect=(0, 0, 1, 0.92))
save_figure(fig, "lin_geo_final")

print("\n=== D_KL(q||p) at step 1000 (mean / std) ===")
print(
    fin.pivot_table(
        index="regime",
        columns="path",
        values="elbo",
        aggfunc=[lambda s: -s.mean(), "std"],
    )
    .round(3)
    .to_string()
)
print("\n=== NFE / step ===")
print(
    fin.pivot_table(index="regime", columns="path", values="nfe_mean", aggfunc="mean")
    .round(1)
    .to_string()
)
print("\n=== forward-pass time / step [ms] (mean / std over seeds) ===")
forward["wall_ms"] = 1000 * forward.wall_s
print(
    forward.pivot_table(
        index="regime", columns="path", values="wall_ms", aggfunc=["mean", "std"]
    )
    .round(1)
    .to_string()
)
