"""Locate the upper 5% and upper 1% of per-sample gradient norms in draw 890."""
from pathlib import Path
import importlib.util
import json
import os
import argparse
import numpy as np
import torch
import pandas as pd

H = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument(
    "--reference",
    action="store_true",
    help="Plot the same selected samples at their reference positions.",
)
parser.add_argument(
    "--include-p80",
    action="store_true",
    help="Add the 80th–95th percentile as small blue dots in a separate plot.",
)
parser.add_argument(
    "--include-p999",
    action="store_true",
    help="Highlight samples at or above the 99.9th percentile separately.",
)
parser.add_argument(
    "--tight-zoom",
    action="store_true",
    help="Zoom further into the central strip and S0 in the right panel.",
)
args = parser.parse_args()
suffix = ("_p80" if args.include_p80 else "") + ("_p999" if args.include_p999 else "")
suffix += "_tight_zoom" if args.tight_zoom else ""
os.environ.setdefault("MPLCONFIGDIR", str(H / "mpl_cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = H.parent
spec = importlib.util.spec_from_file_location("study", B / "otr_study.py")
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
v = torch.load(H.parent / "replay_inputs.pt", weights_only=False)
cfg = s.cfg_for("rosenbrock_extreme")
m = s.build_model(cfg, 2, reseed=False).double()
m.load_state_dict(v["state"])
x = v["x0"].double()
norms = np.load(H / "per_sample_elbo_gradients.npz")["norms"]
q80, q95, q99 = np.percentile(norms, [80, 95, 99], method="linear")
q999 = float(np.percentile(norms, 99.9, method="linear"))
with torch.no_grad():
    if args.reference:
        z = x
    else:
        z = s.ot.integrate_ode(
            x, m, path="linear", ode_solver="dopri5", rtol=1e-10, atol=1e-10
        )
        terms = m.log_prob(z) - s.build_target(cfg).log_prob(z)
        expected = np.load(H / "native_reverse.npz")["negative_elbo_terms"]
        np.testing.assert_allclose(terms.numpy(), expected, rtol=1e-11, atol=1e-11)
    means = np.stack([c.a.numpy() for c in m.components])
z = z.numpy()
mid = (norms >= q95) & (norms < q99)
low = (norms >= q80) & (norms < q95)
high = norms >= q99
extreme = norms >= q999
high_display = high & ~extreme if args.include_p999 else high
if not args.reference:
    np.savez(H / "transported_samples.npz", reference=x.numpy(), transported=z)
pd.DataFrame(
    dict(
        sample_index=np.arange(len(norms)),
        x1=z[:, 0],
        x2=z[:, 1],
        gradient_norm=norms,
        at_or_above_p95=norms >= q95,
        at_or_above_p99=high,
    )
).to_csv(
    H
    / ("reference_sample_locations.csv" if args.reference else "sample_locations.csv"),
    index=False,
)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 10,
        "pdf.fonttype": 42,
    }
)
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.5))
lo, hi = z.min(0), z.max(0)
pad = 0.06 * (hi - lo)
corelo, corehi = np.percentile(z, [1, 99], axis=0)
corepad = 0.08 * (corehi - corelo)
if args.tight_zoom:
    corelo = np.array([-0.85, -0.15]) if args.reference else np.array([-0.9, 0.35])
    corehi = np.array([0.35, 2.05]) if args.reference else np.array([1.05, 2.05])
    corepad = np.zeros(2)
for ax, lower, upper, title in zip(
    axes,
    [lo - pad, corelo - corepad],
    [hi + pad, corehi + corepad],
    [
        "Full reference range" if args.reference else "All transported samples",
        "Central region (zoom)",
    ],
):
    gx, gy = np.linspace(lower[0], upper[0], 350), np.linspace(lower[1], upper[1], 350)
    xx, yy = np.meshgrid(gx, gy)
    with torch.no_grad():
        log_density = m.reference.log_prob if args.reference else m.log_prob
        density = (
            log_density(torch.tensor(np.c_[xx.ravel(), yy.ravel()]))
            .exp()
            .numpy()
            .reshape(xx.shape)
        )
    levels = density.max() * np.array([0.005, 0.02, 0.05, 0.1, 0.2, 0.4, 0.65, 0.85])
    ax.contourf(
        xx,
        yy,
        density,
        levels=np.r_[0, levels, density.max() * 1.001],
        cmap="Greys",
        alpha=0.35,
    )
    ax.contour(
        xx, yy, density, levels=levels, colors="0.55", linewidths=0.55, alpha=0.7
    )
    if args.include_p80:
        ax.scatter(
            z[low, 0],
            z[low, 1],
            s=5,
            c="#0072B2",
            linewidths=0,
            alpha=0.5,
            label=f"80th–95th percentile (n={low.sum()})",
            zorder=2,
        )
    ax.scatter(
        z[mid, 0],
        z[mid, 1],
        s=20,
        facecolors="none",
        edgecolors="#D08700",
        linewidths=0.75,
        alpha=0.8,
        label=f"95th–99th percentile (n={mid.sum()})",
        zorder=3,
    )
    high_label = (
        f"99th–99.9th percentile (n={high_display.sum()})"
        if args.include_p999
        else f"≥99th percentile (n={high.sum()})"
    )
    ax.scatter(
        z[high_display, 0],
        z[high_display, 1],
        s=27,
        c="#C62828",
        edgecolors="white",
        linewidths=0.3,
        alpha=0.9,
        label=high_label,
        zorder=4,
    )
    if args.include_p999:
        ax.scatter(
            z[extreme, 0],
            z[extreme, 1],
            s=60,
            marker="D",
            c="#7B2CBF",
            edgecolors="white",
            linewidths=0.7,
            label=f"≥99.9th percentile (n={extreme.sum()})",
            zorder=5,
        )
    if not args.reference:
        ax.scatter(
            means[:, 0],
            means[:, 1],
            marker="x",
            color="0.2",
            s=35,
            linewidths=1,
            zorder=5,
        )
        for i, mean in enumerate(means):
            ax.annotate(
                f"C{i+1}",
                mean,
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                color="0.2",
            )
    ax.scatter(
        *z[5999],
        marker="*",
        s=180,
        c="#0072B2",
        edgecolors="black",
        linewidths=0.7,
        label=r"Problematic $S_0$",
        zorder=6,
    )
    ax.annotate(
        r"$S_0$",
        z[5999],
        xytext=(8, -15),
        textcoords="offset points",
        color="#0072B2",
        fontsize=11,
    )
    ax.set(
        xlim=(lower[0], upper[0]),
        ylim=(lower[1], upper[1]),
        title=title,
        xlabel=r"Reference coordinate $z_1$"
        if args.reference
        else r"Transported coordinate $x_1$",
        ylabel=r"Reference coordinate $z_2$"
        if args.reference
        else r"Transported coordinate $x_2$",
    )
    ax.set_aspect("equal", adjustable="box")
    # Keep fixed-aspect panels next to each other instead of centering each
    # inside its wider subplot slot, which leaves a large empty middle gap.
    if args.tight_zoom:
        ax.set_anchor("E" if ax is axes[0] else "W")
    ax.spines[["top", "right"]].set_visible(False)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 0),
    ncol=2 if (args.include_p80 or args.include_p999) else 3,
    frameon=False,
)
fig.tight_layout(
    rect=(0, 0.12 if (args.include_p80 or args.include_p999) else 0.07, 1, 1)
)
for ext in ["pdf", "png"]:
    background = "reference" if args.reference else "mixture"
    fig.savefig(
        H / f"gradient_tail_samples_over_{background}{suffix}.{ext}",
        dpi=220,
        bbox_inches="tight",
    )
plt.close(fig)
(
    H
    / (
        ("reference_tail_plot_metadata" if args.reference else "tail_plot_metadata")
        + suffix
        + ".json"
    )
).write_text(
    json.dumps(
        dict(
            p80=float(q80),
            include_p80=args.include_p80,
            count_p80_to_p95=int(low.sum()),
            p95=float(q95),
            p99=float(q99),
            count_at_or_above_p95=int((norms >= q95).sum()),
            count_at_or_above_p99=int(high.sum()),
            p999=q999,
            include_p999=args.include_p999,
            count_at_or_above_p999=int(extreme.sum()),
            s0_position=z[5999].tolist(),
            background="reference density"
            if args.reference
            else "learned unperturbed mixture density",
            selection="individual sample gradient norm within draw 890; linear percentile interpolation",
        ),
        indent=2,
    )
)
print("Saved plot; counts:", mid.sum(), high.sum(), "S0:", z[5999], flush=True)
