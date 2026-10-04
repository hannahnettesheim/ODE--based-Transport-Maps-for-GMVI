"""Final linear/geometric fits as a directly comparable 2 x 4 snapshot grid."""
import sys as _sys, pathlib as _pl

_ROOT = next(
    _q for _q in _pl.Path(__file__).resolve().parents if (_q / "gmvi").is_dir()
)
_sys.path[:0] = [str(_ROOT), str(_ROOT / "Visualizations")]
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch

import _style as S
from gmvi.models.generalized_mixture import GeneralizedMixture
from gmvi.models.reference_distributions import make_reference
from gmvi.targets.distributions import make_target

REGIMES = {
    "banana": ("banana", {}),
    "rosenbrock_extreme": ("rosenbrock", {"a": 0.05, "b": 5.0}),
    "hier_k5": ("hierarchical_mixture", {}),
    "hier_k50": ("hierarchical_mixture", {}),
}
PATHS = ("linear", "geometric")
SEED = 1
Q_COLOR = "#457B9D"
MEAN_COLOR = "#D1495B"


def rebuild(rows):
    """Reconstruct one matrix-exponential mixture from long-form parameters."""
    k = int(rows[rows.param == "log_weights"].idx.max() + 1)
    model = GeneralizedMixture(
        n_components=k,
        dim=2,
        reference=make_reference("normal", dim=2),
        param_type="matrixexponential",
    )
    state = model.state_dict()
    for name, group in rows.drop_duplicates(["param", "idx"]).groupby("param"):
        values = group.sort_values("idx").value.to_numpy()
        state[name] = torch.tensor(values, dtype=state[name].dtype).reshape(
            state[name].shape
        )
    model.load_state_dict(state)
    model.eval()
    return model


def sample_q(model, n, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        return model.sample(n)[0].numpy()


def main():
    params = pd.read_csv(os.path.join(HERE, "lin_geo_final_params.csv"))
    checkpoints = pd.read_csv(os.path.join(HERE, "lin_geo_checkpoints.csv"))
    checkpoints = checkpoints[(checkpoints.seed == SEED) & (checkpoints.step == 1000)]

    models = {}
    samples = {}
    targets = {}
    limits = {}
    for col, (regime, (target_name, target_kwargs)) in enumerate(REGIMES.items()):
        targets[regime] = make_target(target_name, **target_kwargs)
        q_samples = []
        for row, path in enumerate(PATHS):
            subset = params[
                (params.regime == regime)
                & (params.path == path)
                & (params.seed == SEED)
            ]
            models[regime, path] = rebuild(subset)
            samples[regime, path] = sample_q(
                models[regime, path], 4000, seed=1000 + col
            )
            q_samples.append(samples[regime, path])
        # Use one common robust viewing range for the two fitted q's. Including
        # target draws makes the extreme Rosenbrock tail dominate the axes and
        # hides precisely the linear/geometric comparison this figure is for.
        cloud = np.concatenate(q_samples, axis=0)
        lo, hi = np.quantile(cloud, [0.005, 0.995])
        pad = 0.08 * (hi - lo)
        limits[regime] = (float(lo - pad), float(hi + pad))

    plt.rcdefaults()
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharex="col", sharey="col")
    for col, regime in enumerate(REGIMES):
        lo, hi = limits[regime]
        grid = np.linspace(lo, hi, 220)
        xx, yy = np.meshgrid(grid, grid)
        points = torch.tensor(
            np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32
        )
        with torch.no_grad():
            logp = targets[regime].log_prob(points).numpy().reshape(xx.shape)
        density = np.exp(np.clip(logp - np.nanmax(logp), -8, 0))

        for row, path in enumerate(PATHS):
            ax = axes[row, col]
            ax.contourf(xx, yy, density, levels=18, cmap="Greys", alpha=0.5)
            q = samples[regime, path]
            model = models[regime, path]
            with torch.no_grad():
                weights = model.weights.numpy()
                means = model.means.numpy()
            ax.scatter(
                q[:2500, 0], q[:2500, 1], s=2.5, color=Q_COLOR, alpha=0.22, linewidths=0
            )
            live = weights > 0.005
            ax.scatter(
                means[live, 0],
                means[live, 1],
                s=260 * weights[live] + 7,
                marker="x",
                color=MEAN_COLOR,
                linewidths=1.0,
            )
            score = checkpoints[
                (checkpoints.regime == regime) & (checkpoints.path == path)
            ].kl
            if len(score) and np.isfinite(score.iloc[0]):
                ax.text(
                    0.97,
                    0.04,
                    rf"$D_{{\rm KL}}={score.iloc[0]:.2f}$",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.5),
                )
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.set_title(S.target_name(regime), fontsize=11)
            if col == 0:
                ax.set_ylabel(
                    ("Linear path" if path == "linear" else "Geometric path")
                    + "\n$z_2$",
                    fontsize=10,
                )
            if row == 1:
                ax.set_xlabel("$z_1$", fontsize=10)

    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                color=Q_COLOR,
                marker="o",
                linestyle="none",
                markersize=4,
                alpha=0.5,
                label=r"samples from $q$",
            ),
            Line2D(
                [],
                [],
                color=MEAN_COLOR,
                marker="x",
                linestyle="none",
                markersize=5,
                label="component means",
            ),
            PatchProxy("0.65", "target density"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(f"Final approximations, seed {SEED}", fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    for ext in ("pdf", "png"):
        fig.savefig(
            os.path.join(HERE, f"lin_geo_snapshots_2x4.{ext}"),
            bbox_inches="tight",
            dpi=180,
        )
    plt.close(fig)
    print("  wrote lin_geo_snapshots_2x4.pdf / .png")


def PatchProxy(color, label):
    """Small legend proxy without importing the full patches namespace."""
    from matplotlib.patches import Patch

    return Patch(facecolor=color, edgecolor="none", alpha=0.5, label=label)


if __name__ == "__main__":
    main()
