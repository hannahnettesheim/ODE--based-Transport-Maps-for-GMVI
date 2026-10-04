from pathlib import Path
import sys
import os
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BENCHMARK = (
    HERE.parent / "6.4_Comparison_with_Benchmark_Estimators" / "benchmark_training"
)
sys.path[:0] = [str(ROOT), str(BENCHMARK)]
os.environ["MPLCONFIGDIR"] = str(BENCHMARK / "mpl_cache")
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from worker import config
from gmvi.targets.distributions import make_target


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    floor = -6.0
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 4, figsize=(418 / 72.27, 1.75), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.01, wspace=0.02)
    records = []
    for ax, tag in zip(axes.flat, ["banana", "hier_k5", "funnel", "rosenbrock"]):
        cfg = config(("comparison_2d", tag, "exact_marginalization", 1))
        kwargs = dict(cfg.target_kwargs)
        if tag == "funnel":
            kwargs["dim"] = cfg.dim
        target = make_target(cfg.target, **kwargs)
        # Bound the displayed six-nat contour rather than rare sample extrema.
        if tag == "rosenbrock":
            radius = np.sqrt(-floor / target.a)
            xlim = (target.mu - radius - 0.5, target.mu + radius + 0.5)
            ylim = (-3.0, max(abs(xlim[0]), abs(xlim[1])) ** 2 + 3.0)
            peak = -np.log(2 * np.pi * target.s1 * target.s2)
            title = "Rosenbrock"
        elif tag == "funnel":
            sigma = target.sigma_v
            center = -(sigma**2) / 2
            radius = np.sqrt(-2 * floor) * sigma
            v = np.linspace(center - radius, center + radius, 10000)
            envelope = np.sqrt(
                2
                * np.exp(v)
                * np.maximum(0.0, -floor - (v - center) ** 2 / (2 * sigma**2))
            )
            width = envelope.max() * 1.08
            xlim, ylim = (center - radius - 0.5, center + radius + 0.5), (-width, width)
            peak = float(target.log_prob(torch.tensor([[center, 0.0]]))[0])
            title = "Neal's Funnel"
        elif tag == "banana":
            xlim, ylim = (-8.0, 8.0), (-6.0, 30.0)
            peak = None
            title = "Banana"
        else:
            samples = target.sample(20000).numpy()
            lo, hi = samples.min(axis=0), samples.max(axis=0)
            pad = (hi - lo) * 0.08
            xlim, ylim = (lo[0] - pad[0], hi[0] + pad[0]), (
                lo[1] - pad[1],
                hi[1] + pad[1],
            )
            peak = None
            title = "Hierarchical Mixture\n(20 modes)"
        x = np.linspace(*xlim, 1400)
        y = np.linspace(*ylim, 1400)
        xx, yy = np.meshgrid(x, y)
        grid = torch.tensor(
            np.column_stack([xx.ravel(), yy.ravel()]), dtype=torch.float32
        )
        with torch.no_grad():
            lp = (
                torch.cat([target.log_prob(chunk) for chunk in grid.split(16384)])
                .numpy()
                .reshape(xx.shape)
            )
        if peak is None:
            peak = float(lp.max())
        relative = lp - peak
        im = ax.pcolormesh(
            x,
            y,
            np.clip(relative, floor, 0),
            cmap="viridis",
            vmin=floor,
            vmax=0,
            shading="auto",
            rasterized=True,
        )
        ax.contour(
            x,
            y,
            relative,
            levels=[-5, -3, -1],
            colors="white",
            linewidths=0.35,
            alpha=0.5,
        )
        ax.set(title=title, xlabel=r"$z_1$", ylabel=r"$z_2$")
        ax.set_box_aspect(1)
        records.append(
            dict(
                tag=tag,
                target=cfg.target,
                benchmark_kwargs=kwargs,
                parameters={
                    k: v for k, v in vars(target).items() if isinstance(v, (int, float))
                },
                xlim=list(map(float, xlim)),
                ylim=list(map(float, ylim)),
            )
        )
    cb = fig.colorbar(im, ax=axes, location="right", shrink=0.52, pad=0.015, aspect=13)
    cb.set_label("Rel. log density (nats)", fontsize=7)
    cb.set_ticks([-6, -5, -4, -3, -2, -1, 0])
    for ext in ["png", "pdf"]:
        fig.savefig(HERE / f"target_gallery.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    (HERE / "target_gallery_config.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
