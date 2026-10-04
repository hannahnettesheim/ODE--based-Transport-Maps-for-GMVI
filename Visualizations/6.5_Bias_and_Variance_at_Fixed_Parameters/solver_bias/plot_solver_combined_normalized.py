from pathlib import Path
import os

H = Path(__file__).resolve().parent
os.environ["MPLCONFIGDIR"] = str(H / "mpl_cache")
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8,
        "axes.titlesize": 9,
        "pdf.fonttype": 42,
    }
)
fig, axs = plt.subplots(2, 2, figsize=(418 / 72.27, 4.9), sharey=True)
all_rows = []
for row, (folder, title) in enumerate(
    [
        ("solver_rosenbrock", "Rosenbrock"),
        ("solver_hier_k5_20260922", "Hierarchical mixture, K=5"),
    ]
):
    p = H / folder
    d = pd.read_csv(p / "solver_bias_dm_reference.csv")
    assert len(d) == 27 and (d.replicates == 5000).all()
    noise = {}
    for step in [50, 250, 1000]:
        C = np.load(p / f"dm_reference_{step}.npz")["chunks"]
        assert len(C) == 40
        noise[step] = np.sqrt(C.var(axis=0, ddof=1).sum() / 40)
    d["dm_reference_rms_error"] = d.step.map(noise)
    d["normalized_bias"] = d.bias / d.dm_reference_rms_error
    d["target"] = title
    all_rows.append(d)
    for ax, kind in zip(axs[row], ["dopri5", "rk4"]):
        for step, g in d[d.solver == kind].groupby("step"):
            g = g.sort_values("setting")
            ax.loglog(
                g.setting, g.normalized_bias, "o-", markersize=4, label=f"Step {step}"
            )
        ax.axhline(1, color="0.35", lw=0.8, ls=":", zorder=0)
        ax.set_title(title + " — " + ("DOPRI5" if kind == "dopri5" else "RK4"))
        ax.set_xlabel("rtol = atol" if kind == "dopri5" else "RK4 steps")
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axs[row, 0].set_ylabel(r"Noise-normalized bias $\widetilde{B}_M$")
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0.065, 1, 1), h_pad=1.5, w_pad=1)
for ext in ["pdf", "png"]:
    fig.savefig(
        H / f"solver_bias_combined_dm_reference_noise_normalized.{ext}",
        dpi=240,
        bbox_inches="tight",
    )
pd.concat(all_rows).to_csv(
    H / "solver_bias_combined_dm_reference_noise_normalized.csv", index=False
)
