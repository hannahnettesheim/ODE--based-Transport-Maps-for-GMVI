from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
rows = [json.loads(p.read_text()) for p in HERE.glob("*/step_*/result.json")]
d = (
    pd.DataFrame(rows) if rows else pd.read_csv(HERE / "dimension_comparison.csv")
).sort_values(["step", "estimator", "dimension"])
assert len(d) == 36 and not d.duplicated(["dimension", "step", "estimator"]).any()
d.to_csv(HERE / "dimension_comparison.csv", index=False)
styles = {
    "exact_marginalization": ("DM", "#264653", "o"),
    "score_function": ("SF (no baseline)", "#E63946", "s"),
    "gumbel_softmax": ("ST", "#457B9D", "^"),
    "ode_transport": ("OTR", "#2A9D8F", "D"),
}
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "pdf.fonttype": 42,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 180,
    }
)
fig, axs = plt.subplots(
    1, 3, figsize=(12, 4.2), sharex=True, sharey=True, squeeze=False
)
for col, step in enumerate([50, 250, 1000]):
    for estimator, (label, color, marker) in styles.items():
        g = d[(d.step == step) & (d.estimator == estimator)].sort_values("dimension")
        for row in [0]:
            scale = g.parameter_count.to_numpy() if row else np.ones(len(g))
            y = g.variance.to_numpy() / scale
            lo = g.variance_bootstrap_lo.to_numpy() / scale
            hi = g.variance_bootstrap_hi.to_numpy() / scale
            ax = axs[row, col]
            ax.plot(
                g.dimension,
                y,
                color=color,
                marker=marker,
                label=label,
                ls={
                    "exact_marginalization": "-",
                    "score_function": "--",
                    "gumbel_softmax": "-.",
                    "ode_transport": "-",
                }[estimator],
                lw=1.5,
                ms=5,
            )
            ax.vlines(g.dimension, lo, hi, color=color, lw=1.3, alpha=0.85)
            ax.fill_between(g.dimension, lo, hi, color=color, alpha=0.1)
            ax.set_yscale("log")
            ax.set_xticks([5, 10, 20])
            ax.grid(alpha=0.2, which="major")
    axs[0, col].set_title(f"Training snapshot: step {step}")
    axs[0, col].set_xlabel("Data dimension n")
axs[0, 0].set_ylabel(r"Estimated total variance $\widehat{V}_M$")
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.025))
fig.suptitle(
    "Rosenbrock: gradient variance versus dimension", fontsize=15, y=0.99, va="top"
)
fig.text(
    0.5,
    0.885,
    "5 components · M = 1,024 · 5,000 gradient draws · OTR rtol = atol = 10⁻⁷",
    ha="center",
    fontsize=10,
)
fig.text(
    0.5,
    0.006,
    "Shading: approximate 95% replicate-bootstrap intervals. One training trajectory per dimension.",
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.14, 1, 0.83))
for ext in ["png", "pdf"]:
    fig.savefig(HERE / f"dimension_variance.{ext}", bbox_inches="tight")
plt.close(fig)
