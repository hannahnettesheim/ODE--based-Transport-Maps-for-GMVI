"""Plot fitted densities or Gaussian component ellipses from training snapshots.

Density panels use relative log density. Component panels show means, covariance
ellipses, and mixture weights and require a normal reference distribution.
Sweep arm selection is performed at the requested seed."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import glob
import json
import math
import os
import sys

import numpy as np
import torch

# Load the shared thesis plotting style from the repository when needed.
try:
    import _style as S
except ImportError:  # pragma: no cover
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for _cand in ("Visualisations",):
        _p = os.path.join(_root, _cand)
        if os.path.isdir(_p):
            sys.path.insert(0, _p)
            break
    import _style as S

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from gmvi.models.generalized_mixture import GeneralizedMixture
from gmvi.models.reference_distributions import make_reference


@dataclass
class Panel:
    """Fitted mixture parameters, configuration, and display metrics."""

    label: str
    state: Dict[str, torch.Tensor]
    config: Dict[str, Any]
    metrics: Dict[str, Any] = field(default_factory=dict)
    """e.g. {"kl": 1.455, "seed": 2, "step": 2000}. Rendered under the label."""

    def subtitle(self) -> str:
        bits = []
        if "seed" in self.metrics:
            bits.append(f"seed {self.metrics['seed']}")
        for k, fmt in (
            ("kl", "KL={:.3f}"),
            ("elbo", "ELBO={:.3f}"),
            ("sliced_w2", "$W_2$={:.3f}"),
        ):
            if k in self.metrics and self.metrics[k] == self.metrics[k]:
                bits.append(fmt.format(self.metrics[k]))
        return ", ".join(bits)


def build_model(config: Dict[str, Any], dim: int) -> GeneralizedMixture:
    ref = make_reference(config.get("reference", "normal"), dim=dim)
    return GeneralizedMixture(
        n_components=config["components"],
        dim=dim,
        reference=ref,
        param_type=config["chart"],
        init_scale=config.get("init_scale", 0.5),
    )


def _restore_panel_model(panel: Panel, dim: int) -> GeneralizedMixture:
    m = build_model(panel.config, dim)
    m.load_state_dict(panel.state)
    m.eval()
    return m


def _panel_dimension(panel: Panel) -> int:
    return int(panel.state["components.0.a"].shape[0])


def panels_from_result(
    res, step: Optional[int] = None, label: Optional[str] = None
) -> List[Panel]:
    """Panels from a single RunResult. `step` defaults to the last snapshot."""
    if not res.snapshots:
        raise ValueError("this run kept no snapshots; set probe_params=True")
    step = max(res.snapshots) if step is None else step
    cfg = res.config.__dict__
    met = {"seed": cfg.get("seed"), "step": step}
    ck = res.checkpoints
    if len(ck) and "step" in ck:
        row = ck[ck.step == step]
        for k in ("kl", "elbo", "sliced_w2"):
            if k in row and len(row):
                met[k] = float(row[k].iloc[0])
    return [
        Panel(
            label=label or cfg.get("estimator", "run"),
            state=res.snapshots[step],
            config=dict(cfg),
            metrics=met,
        )
    ]


def panels_from_sweep(
    sweep_dir: str,
    seed: int = 1,
    select: Optional[Sequence[str]] = None,
    pick: Optional[str] = None,
    metric: str = "kl",
    lower_is_better: bool = True,
    step: Optional[int] = None,
    label_by: Optional[Sequence[str]] = None,
) -> List[Panel]:
    """Load finished cells from a sweep directory.

    seed        which seed's cell to load. Always fixed, never mixed.
    select      explicit cell_id substrings, in the order you want them drawn.
    pick        "worst,mid,best" (any subset, any order) chosen by `metric`
                AT THIS SEED. Ignored when `select` is given.
    label_by    config keys to build the panel label from; defaults to whatever
                differs between the loaded cells.
    """
    cells = []
    for d in sorted(glob.glob(os.path.join(sweep_dir, "cells", "*"))):
        if not os.path.exists(os.path.join(d, "DONE")):
            continue
        mp, sp = os.path.join(d, "meta.json"), os.path.join(d, "snapshots.pt")
        if not (os.path.exists(mp) and os.path.exists(sp)):
            continue
        with open(mp) as fh:
            meta = json.load(fh)
        if int(meta["config"].get("seed", -1)) != seed:
            continue
        cells.append((d, meta))
    if not cells:
        raise FileNotFoundError(
            f"no finished cells with seed={seed} under {sweep_dir}/cells. "
            f"Run the sweep, or pass a different seed."
        )

    scored = []
    for d, meta in cells:
        blob = torch.load(os.path.join(d, "snapshots.pt"), weights_only=False)
        snaps = blob["snapshots"]
        st = max(snaps) if step is None else step
        score = _cell_metric(d, metric, st)
        scored.append(
            dict(
                dir=d,
                meta=meta,
                state=snaps[st],
                step=st,
                score=score,
                cell_id=meta["cell_id"],
            )
        )

    if select is not None:
        chosen = [c for key in select for c in scored if key in c["cell_id"]]
    elif pick:
        ok = [c for c in scored if c["score"] == c["score"]]
        if not ok:
            raise ValueError(f"no cell has a finite {metric!r} to rank by")
        ok.sort(key=lambda c: c["score"], reverse=not lower_is_better)
        idx = {"best": 0, "mid": len(ok) // 2, "worst": len(ok) - 1}
        chosen = [ok[idx[w.strip()]] for w in pick.split(",")]
    else:
        chosen = scored

    varying = label_by or _varying_keys([c["meta"]["config"] for c in chosen])
    out = []
    for c in chosen:
        cfg = c["meta"]["config"]
        lab = _config_label(cfg, varying)
        met = {"seed": seed, "step": c["step"], metric: c["score"]}
        out.append(Panel(label=lab, state=c["state"], config=cfg, metrics=met))
    return out


def panels_from_param_csv(
    path: str,
    target: Optional[str] = None,
    seed: int = 1,
    group_by: Sequence[str] = ("solver_kind", "solver_param"),
    chart: str = "cholesky",
    components: Optional[int] = None,
    order: Optional[Sequence] = None,
    label_fmt: Optional[str] = None,
    extra_config: Optional[Dict[str, Any]] = None,
    scores: Optional[Dict[Any, float]] = None,
    metric: str = "kl",
) -> List[Panel]:
    """Panels from a LONG-FORMAT parameter dump.

    One row per scalar:  target, <group_by...>, seed, param, idx, value
    which is what a sweep writes when it flattens final state_dicts rather than
    pickling them. Reassembled into state_dicts here.

    The chart is NOT recoverable from the file -- 'chol_raw' names the tensor
    but nothing records that it means A = LL^T -- so `chart` must be passed and
    must match the run that produced it. Loading a cholesky dump as
    matrixexponential would succeed silently and draw the wrong ellipses.

    scores: optional {group_key: value} to print under each panel, since a
    parameter dump carries no KL of its own. Without it the panels are labelled
    by their group only, and nothing is ranked.
    """
    import pandas as pd

    d = pd.read_csv(path)
    if target is not None:
        d = d[d.target == target]
    d = d[d.seed == seed]
    if not len(d):
        raise ValueError(f"no rows for target={target!r}, seed={seed}")

    K = components or (d[d.param == "log_weights"].idx.max() + 1)
    keys = list(group_by)
    groups = list(d.groupby(keys, sort=False))
    if order is not None:
        rank = {
            tuple(o) if isinstance(o, (list, tuple)) else (o,): i
            for i, o in enumerate(order)
        }
        groups.sort(
            key=lambda kv: rank.get(
                kv[0] if isinstance(kv[0], tuple) else (kv[0],), 10**6
            )
        )

    out = []
    for key, g in groups:
        key_t = key if isinstance(key, tuple) else (key,)
        state = {}
        for pname, gp in g.groupby("param"):
            gp = gp.sort_values("idx")
            state[pname] = torch.tensor(gp.value.to_numpy(), dtype=torch.float32)
        cfg = dict(
            components=int(K),
            chart=chart,
            reference="normal",
            target=target,
            dim=2,
            seed=seed,
            **(extra_config or {}),
        )
        cfg.update(dict(zip(keys, key_t)))
        lab = (
            label_fmt.format(**dict(zip(keys, key_t)))
            if label_fmt
            else ", ".join(f"{k}={_format_config_value(v)}" for k, v in zip(keys, key_t))
        )
        met = {"seed": seed}
        if scores and key_t in scores:
            met[metric] = scores[key_t]
        out.append(Panel(label=lab, state=state, config=cfg, metrics=met))
    return out


def _format_config_value(v):
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _cell_metric(cell_dir: str, metric: str, step: int) -> float:
    import pandas as pd

    p = os.path.join(cell_dir, "checkpoints.csv")
    if not os.path.exists(p):
        return float("nan")
    d = pd.read_csv(p)
    if metric not in d.columns or "step" not in d.columns:
        return float("nan")
    row = d[d.step == step]
    return float(row[metric].iloc[0]) if len(row) else float("nan")


def _varying_keys(cfgs: List[Dict[str, Any]]) -> List[str]:
    if len(cfgs) < 2:
        return ["estimator"]
    keys = [
        k
        for k in cfgs[0]
        if len({json.dumps(c.get(k), sort_keys=True, default=str) for c in cfgs}) > 1
    ]
    drop = {"seed", "tag", "verbose", "log_every"}
    return [k for k in keys if k not in drop] or ["estimator"]


_PRETTY_EST = {
    "exact_marginalization": "DM",
    "score_function": "SF",
    "gumbel_softmax": "ST",
    "ode_transport": "OTR",
}


def _config_label(cfg: Dict[str, Any], keys: Sequence[str]) -> str:
    bits = []
    for k in keys:
        v = cfg.get(k)
        if k == "estimator":
            bits.append(_PRETTY_EST.get(v, str(v)))
        elif k == "estimator_kwargs" and isinstance(v, dict):
            bits.append(
                " ".join(
                    f"{kk}={vv}"
                    for kk, vv in sorted(v.items())
                    if kk in ("path", "ode_solver", "ode_steps", "temperature")
                )
            )
        else:
            bits.append(f"{k}={v}")
    return ", ".join(b for b in bits if b)


@torch.no_grad()
def _marginal_params(model: GeneralizedMixture, dims: Tuple[int, int]):
    """(weights, means, covariances) of the exact 2-d marginal, per component."""
    d0, d1 = dims
    w = torch.softmax(model.log_weights, 0).numpy()
    mu, cov = [], []
    for comp, A in zip(model.components, model.get_matrices()):
        a = comp.a.detach()
        C = (A @ A.T).detach()
        mu.append([float(a[d0]), float(a[d1])])
        cov.append(
            [[float(C[d0, d0]), float(C[d0, d1])], [float(C[d1, d0]), float(C[d1, d1])]]
        )
    return w, np.array(mu), np.array(cov)


def _log_density_grid(w, mu, cov, X, Y):
    """log of the 2-d Gaussian-mixture marginal on a grid."""
    P = np.stack([X.ravel(), Y.ravel()], 1)  # (G, 2)
    comps = []
    for wi, m, C in zip(w, mu, cov):
        Ci = np.linalg.inv(C)
        d = P - m
        q = np.einsum("gi,ij,gj->g", d, Ci, d)
        ld = -0.5 * (q + np.log(np.linalg.det(C)) + 2 * math.log(2 * math.pi))
        comps.append(ld + math.log(max(wi, 1e-300)))
    L = np.stack(comps)  # (K, G)
    mx = L.max(0)
    return (mx + np.log(np.exp(L - mx).sum(0))).reshape(X.shape)


def _draw_density(ax, Z, X, Y, floor, levels, cmap):
    Zf = np.clip(Z - Z.max(), floor, 0.0)
    cs = ax.contourf(
        X, Y, Zf, levels=np.linspace(floor, 0.0, levels), cmap=cmap, extend="neither"
    )
    ax.contour(
        X,
        Y,
        Zf,
        levels=np.linspace(floor, 0.0, levels),
        colors="white",
        linewidths=0.18,
        alpha=0.5,
    )
    return cs


def _draw_components(ax, w, mu, cov, color, wmax, size=(6.0, 90.0), n_ellipse=128):
    """Means sized by weight, with the 1-sigma ellipse of each component."""
    t = np.linspace(0, 2 * math.pi, n_ellipse)
    circle = np.stack([np.cos(t), np.sin(t)])  # (2, n)
    order = np.argsort(w)  # faint ones underneath
    for i in order:
        L = np.linalg.cholesky(cov[i] + 1e-12 * np.eye(2))
        e = mu[i][:, None] + L @ circle
        rel = float(w[i] / max(wmax, 1e-300))
        ax.plot(e[0], e[1], color=color, lw=0.9, alpha=0.25 + 0.65 * rel, zorder=2)
        ax.fill(e[0], e[1], color=color, alpha=0.06 + 0.14 * rel, lw=0, zorder=1)
    s = size[0] + (size[1] - size[0]) * (w / max(w.max(), 1e-300))
    ax.scatter(
        mu[:, 0], mu[:, 1], s=s, color=color, zorder=4, edgecolor="white", linewidth=0.5
    )


def plot_snapshots(
    panels: Sequence[Panel],
    target=None,
    view: str = "density",
    dims: Tuple[int, int] = (0, 1),
    name: str = "snapshots",
    figdir: str = "figures",
    lim: Optional[Tuple[float, float]] = None,
    grid: int = 260,
    floor: float = -6.0,
    levels: int = 13,
    cmap: str = "viridis",
    ncols: Optional[int] = None,
    title: Optional[str] = None,
    show_target_outline: bool = True,
    frac: float = 1.0,
    aspect: Optional[float] = None,
):
    """Draw every panel side by side, target first when it can be drawn.

    view="density"     log q, floored at `floor` below each panel's own max
    view="components"  means, 1-sigma ellipses, marker area proportional to
                       weight
    """
    if view not in ("density", "components"):
        raise ValueError('view must be "density" or "components"')
    if not panels:
        raise ValueError("no panels to draw")
    S.setup()

    D = _panel_dimension(panels[0])
    models = [_restore_panel_model(p, D) for p in panels]
    params = [_marginal_params(m, dims) for m in models]

    if lim is None:
        pts = np.concatenate(
            [mu + 2.5 * np.sqrt(np.diagonal(cov, 0, 1, 2)) for _, mu, cov in params]
            + [mu - 2.5 * np.sqrt(np.diagonal(cov, 0, 1, 2)) for _, mu, cov in params]
        )
        pad = 0.12 * (pts.max() - pts.min())
        lim = (float(pts.min() - pad), float(pts.max() + pad))
    gx = np.linspace(lim[0], lim[1], grid)
    X, Y = np.meshgrid(gx, gx)

    draw_target = target is not None and D == 2
    Zt = None
    if draw_target:
        with torch.no_grad():
            P = torch.tensor(np.stack([X.ravel(), Y.ravel()], 1), dtype=torch.float32)
            Zt = target.log_prob(P).numpy().reshape(X.shape)

    n = len(panels) + int(draw_target)
    ncols = ncols or n
    nrows = math.ceil(n / ncols)

    # Square density panels need additional height for titles and metrics.
    n_title_lines = 2 if any(p.subtitle() for p in panels) else 1
    W = S.TEXTWIDTH_IN * frac
    ax_w = W / ncols  # ignores margins; corrected by save()
    # Reduce title size and widen gutters as the column count increases.
    tfs = 8.0 if ncols <= 4 else (7.0 if ncols <= 6 else 6.2)
    wspace = 0.10 if ncols <= 4 else (0.18 if ncols <= 6 else 0.24)
    title_in = 0.105 * n_title_lines + 0.05
    head_in = 0.24 if (title or _shared_title(panels)) else 0.0
    foot_in = 0.30  # x label + ticks on the bottom row
    if aspect is None:
        H = nrows * (ax_w + title_in) + head_in + foot_in
    else:
        H = W * aspect
    fig, axes = plt.subplots(nrows, ncols, figsize=(W, H), squeeze=False)
    axes = axes.flat
    cs = None

    idx = 0
    if draw_target:
        ax = next(axes)
        idx += 1
        if view == "density":
            cs = _draw_density(ax, Zt, X, Y, floor, levels, cmap)
        else:
            ax.contour(
                X,
                Y,
                np.clip(Zt - Zt.max(), floor, 0.0),
                levels=np.linspace(floor, 0.0, 7),
                colors=S.SCHEME["target"],
                linewidths=0.5,
                alpha=0.8,
            )
        ax.set_title("True target", fontsize=tfs, pad=3)
        _format_density_axes(ax, lim, first=True)

    for p, (w, mu, cov) in zip(panels, params):
        ax = next(axes)
        first_in_row = (idx % ncols) == 0
        idx += 1
        if view == "density":
            Z = _log_density_grid(w, mu, cov, X, Y)
            cs = _draw_density(ax, Z, X, Y, floor, levels, cmap)
        else:
            if draw_target and show_target_outline:
                ax.contour(
                    X,
                    Y,
                    np.clip(Zt - Zt.max(), floor, 0.0),
                    levels=np.linspace(floor, 0.0, 7),
                    colors="0.65",
                    linewidths=0.4,
                    alpha=0.9,
                    zorder=0,
                )
            _draw_components(
                ax, w, mu, cov, S.SCHEME["ode_transport"], wmax=float(w.max())
            )
        sub = p.subtitle()
        ax.set_title(p.label + (f"\n{sub}" if sub else ""), fontsize=tfs, pad=3)
        _format_density_axes(ax, lim, first=first_in_row)

    for ax in axes:  # unused cells in a ragged grid
        ax.axis("off")

    # Reserve title space in figure coordinates; hspace is relative to axis height.
    sup = title or _shared_title(panels)
    top = 1.0 - head_in / H
    fig.subplots_adjust(
        wspace=wspace, top=top, bottom=foot_in / H, hspace=title_in / max(ax_w, 1e-6)
    )
    if sup:
        fig.suptitle(sup, fontsize=8.5, y=1.0 - 0.30 * head_in / H, va="top")
    if view == "density" and cs is not None:
        cb = fig.colorbar(cs, ax=fig.axes, fraction=0.016, pad=0.012)
        cb.set_label(
            r"$\log q - \max \log q$ (nats), floored at " f"{floor:.0f}", fontsize=7
        )
        cb.ax.tick_params(labelsize=6.5)
    else:
        fig.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    color=S.SCHEME["ode_transport"],
                    marker="o",
                    lw=1.0,
                    ms=4,
                    label=r"component mean, $1\sigma$ ellipse; "
                    r"marker area $\propto$ weight",
                ),
                Line2D([], [], color="0.65", lw=0.6, label="target"),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=2,
            frameon=False,
            fontsize=7,
        )

    S.save(fig, name, figdir, frac=frac)
    return fig


def _format_density_axes(ax, lim, first):
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6.5)
    ax.set_xlabel(r"$z_1$", fontsize=7.5, labelpad=1)
    if first:
        ax.set_ylabel(r"$z_2$", fontsize=7.5, labelpad=1)
    else:
        ax.set_yticklabels([])


def _shared_title(panels: Sequence[Panel]) -> str:
    """Format configuration fields shared by all panels."""
    c = panels[0].config
    same = lambda k: all(p.config.get(k) == c.get(k) for p in panels)
    bits = []
    if same("target"):
        t = S.target_name(c.get("target", "?"), dim=c.get("dim"))
        d = c.get("dim")
        bits.append(t + (rf", $n={d}$" if d else ""))
    for k, fmt in (
        ("components", "$K={}$"),
        ("chart", "{}"),
        ("mc_samples", "$M={}$"),
        ("n_steps", "{} steps"),
        ("lr", "lr={}"),
    ):
        if same(k) and c.get(k) is not None:
            bits.append(fmt.format(c[k]))
    steps = {p.metrics.get("step") for p in panels}
    if len(steps) == 1 and None not in steps:
        bits.append(f"snapshot at step {steps.pop()}")
    return ",  ".join(str(b) for b in bits).replace("_", r"\_")
