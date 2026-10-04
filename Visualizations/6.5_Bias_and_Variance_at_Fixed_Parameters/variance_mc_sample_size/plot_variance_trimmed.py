"""Variance-only subset of variance_bias_M, using saved measurements."""
from pathlib import Path
import os

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / "matplotlib_cache"))
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

d = pd.read_csv(HERE / "variance_M_rosenbrock_hier_k5_bootstrap95.csv")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "pdf.fonttype": 42,
    }
)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
for ax, regime, title in zip(
    axes,
    ["rosenbrock_extreme", "hier_k5"],
    ["Rosenbrock", r"Hierarchical mixture, $K=5$"],
):
    sub = d[d.regime == regime]
    assert len(sub) == 12
    for step, g in sub.groupby("step"):
        g = g.sort_values("M")
        (line,) = ax.loglog(g.M, g.variance, "o-", markersize=4, label=f"Step {step}")
        ax.fill_between(
            g.M,
            g.variance_bootstrap_lo,
            g.variance_bootstrap_hi,
            color=line.get_color(),
            alpha=0.15,
        )
        ax.loglog(
            g.M,
            g.variance.iloc[0] * g.M.iloc[0] / g.M,
            ":",
            color=line.get_color(),
            alpha=0.7,
        )
    ax.set_title(title)
    ax.set_xlabel(r"Samples $M$")
    ax.set_ylabel(r"Estimated total variance $\widehat{V}_M$")
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
handles, labels = axes[0].get_legend_handles_labels()
handles.append(Line2D([0], [0], color="gray", linestyle=":"))
labels.append(r"$1/M$ guide")
fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.065),
    ncol=4,
    frameon=False,
)
fig.suptitle("OTR gradient variance at frozen parameters")
fig.text(
    0.5,
    0.015,
    "Shading: approximate 95% percentile-bootstrap CIs (200 resamples).",
    ha="center",
    fontsize=8,
)
fig.tight_layout(rect=(0, 0.17, 1, 0.95))
for ext in ["pdf", "png"]:
    fig.savefig(
        HERE / f"variance_M_rosenbrock_hier_k5.{ext}", dpi=240, bbox_inches="tight"
    )
plt.close(fig)
