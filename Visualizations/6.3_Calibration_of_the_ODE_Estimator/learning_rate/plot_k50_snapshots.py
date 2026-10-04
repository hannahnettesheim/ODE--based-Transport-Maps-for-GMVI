"""Final fitted q(z) for the K=50 hier-mixture runs, via
gmvi.utils.visualization.plot_2d_approximation. Rebuilds each model from the
saved *_final_params.csv (flattened state_dict) and renders a grid.

  python plot_k50_snapshots.py                # M sweep + dopri tol sweep
"""
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
import torch
import matplotlib.pyplot as plt
import _style as S

from gmvi.models.generalized_mixture import GeneralizedMixture
from gmvi.models.reference_distributions import make_reference
from gmvi.targets.distributions import make_target
from gmvi.utils.visualization import _make_grid  # existing grid helper

HERE = os.path.dirname(__file__)
K, DIM, LIM = 50, 2, (-6.5, 6.5)
target = make_target("hierarchical_mixture")
_XX, _YY, _GRID = _make_grid(LIM, LIM, 220)
with torch.no_grad():
    _LP = target.log_prob(_GRID).numpy().reshape(_XX.shape)
_PT = np.exp(_LP - _LP.max())


def _panel(ax, model=None):
    """target mode field (grey); q samples (blue) + weighted centres (red x)."""
    ax.contourf(_XX, _YY, _PT, levels=20, cmap="Greys", alpha=0.55, zorder=0)
    ax.set_aspect("equal")
    ax.set_xlim(LIM)
    ax.set_ylim(LIM)
    ax.set_xticks([])
    ax.set_yticks([])
    if model is None:
        return
    with torch.no_grad():
        z, _ = model.sample(3000)
        z = z.numpy()
        w = model.weights.numpy()
        cen = model.means.numpy()
    ax.scatter(z[:, 0], z[:, 1], s=3, alpha=0.25, color="#2166AC", lw=0, zorder=1)
    live = w > 0.005
    ax.scatter(
        cen[live, 0],
        cen[live, 1],
        s=300 * w[live] + 6,
        marker="x",
        color="#B2182B",
        lw=1.1,
        zorder=2,
    )


def rebuild(rows):
    """rows: DataFrame with columns param, idx, value for ONE run."""
    m = GeneralizedMixture(
        n_components=K,
        dim=DIM,
        reference=make_reference("normal", dim=DIM),
        param_type="cholesky",
    )
    sd = m.state_dict()
    rows = rows.drop_duplicates(["param", "idx"])
    for name, g in rows.groupby("param"):
        vals = g.sort_values("idx")["value"].to_numpy()
        sd[name] = torch.tensor(vals, dtype=sd[name].dtype).reshape(sd[name].shape)
    m.load_state_dict(sd)
    m.eval()
    return m


def grid_figure(fp, key_cols, cells, seed, title, outname):
    """cells: list of (label, filter-dict). One column per cell + target col."""
    n = len(cells)
    fig, axes = plt.subplots(1, n + 1, figsize=(2.5 * (n + 1), 2.8), squeeze=False)
    axes = axes[0]
    _panel(axes[0])
    axes[0].set_title("target p(z)")
    for j, (lab, filt) in enumerate(cells):
        ax = axes[j + 1]
        sub = fp
        for k, v in filt.items():
            sub = (
                sub[np.isclose(sub[k], v)]
                if not isinstance(v, str)
                else sub[sub[k] == v]
            )
        sub = sub[sub.seed == seed]
        if sub.empty:
            _panel(ax)
            ax.set_title(f"{lab}\n(missing)")
            continue
        m = rebuild(sub)
        with torch.no_grad():
            neff = float(
                np.exp(-(m.weights.numpy() * np.log(m.weights.numpy() + 1e-12)).sum())
            )
        _panel(ax, m)
        ax.set_title(f"{lab}\n$n_{{eff}}$={neff:.0f}/50")
    fig.suptitle(title, y=1.06)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"{outname}.{e}"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outname}.pdf / .png")


p = os.path.join(HERE, "M_sweep_final_params.csv")
if os.path.exists(p):
    fp = pd.read_csv(p)
    fp = fp[fp.regime == "hier_k50"]
    Ms = sorted(fp.M.unique())
    grid_figure(
        fp,
        ["M"],
        [(f"M={m}", {"M": m}) for m in Ms],
        seed=1,
        title=S.target_name("hier_k50") + ", dopri5 rtol=1e-7, final q(z)  (seed 1)",
        outname="k50_snapshots_M_sweep",
    )


p = os.path.join(HERE, "dopri_hier50_final_params.csv")
if os.path.exists(p):
    fp = pd.read_csv(p)
    tols = [
        t
        for t in (1e-1, 1e-2, 1e-3, 1e-6, 1e-9)
        if np.isclose(fp.solver_param, t).any()
    ]
    grid_figure(
        fp,
        ["solver_param"],
        [(f"rtol={t:g}", {"solver_param": t}) for t in tols],
        seed=1,
        title=S.target_name("hier_k50")
        + ", M=2048, dopri5 tol sweep, final q(z)  (seed 1)",
        outname="k50_snapshots_dopri_tol",
    )


p = os.path.join(HERE, "hier50_steps_anneal_final_params.csv")
if os.path.exists(p):
    fp = pd.read_csv(p)
    for sc in ("none", "cosine"):
        s = fp[fp.sched == sc]
        ns = sorted(s.n_steps.unique())
        grid_figure(
            s,
            ["n_steps"],
            [(f"{n} steps", {"n_steps": n}) for n in ns],
            seed=1,
            title=S.target_name("hier_k50")
            + f", M=1024, dopri5 1e-6, {sc} LR, final q(z)  (seed 1)",
            outname=f"k50_snapshots_steps_{sc}",
        )


p = os.path.join(HERE, "hier50_lr_anneal_floor_final_params.csv")
if os.path.exists(p):
    fp = pd.read_csv(p)
    schedule_titles = {
        "strict": "strict (constant) LR",
        "anneal_floor_0.1": r"cosine anneal, $\eta_{min}=0.1\eta_0$",
    }
    for schedule in ("strict", "anneal_floor_0.1"):
        s = fp[fp.schedule == schedule]
        lrs = sorted(s.lr.unique())
        grid_figure(
            s,
            ["lr"],
            [(rf"$\eta_0={lr:g}$", {"lr": lr}) for lr in lrs],
            seed=1,
            title=S.target_name("hier_k50")
            + f", M=2048, 2000 steps, dopri5 1e-5, "
            + schedule_titles[schedule]
            + ", final q(z)  (seed 1)",
            outname=f"k50_snapshots_lr_{schedule}",
        )
