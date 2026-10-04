from pathlib import Path
import os

H = Path(__file__).resolve().parent
os.environ["MPLCONFIGDIR"] = str(H / "mpl_cache")
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, NullLocator

STYLES = [
    ("exact_marginalization", "DM", "#264653", "-"),
    ("score_function", "SF", "#E63946", "--"),
    ("gumbel_softmax", "ST", "#457B9D", "-."),
    ("ode_transport", "OTR", "#2A9D8F", "-"),
]


def plot(group):
    p = H / group
    files = sorted(
        f for f in p.glob("cells/*/steps.csv") if (f.parent / "result.json").exists()
    )
    # Archived targets may have only aggregate data. Replace matching runs with
    # completed cell data while retaining every other target and seed.
    aggregate = p / "training_steps.csv"
    d = pd.read_csv(aggregate) if aggregate.exists() else pd.DataFrame()
    if files:
        fresh = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        keys = ["tag", "estimator", "seed"]
        if not d.empty:
            replaced = pd.MultiIndex.from_frame(fresh[keys].drop_duplicates())
            d = d.loc[~pd.MultiIndex.from_frame(d[keys]).isin(replaced)]
        d = pd.concat([d, fresh], ignore_index=True)
        d.to_csv(aggregate, index=False)
    if d.empty:
        return
    tags = {
        "dimension": ["n2", "n5", "n10", "n20"],
        "comparison_2d": ["banana", "rosenbrock", "funnel", "hier_k5", "hier_k50"],
        "logZ_minus50": ["rosenbrock"],
    }[group]
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
        }
    )
    cols = 1 if len(tags) == 1 else 2
    rows = (len(tags) + cols - 1) // cols
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(
            418 / 72.27,
            {"comparison_2d": 6.2, "dimension": 4.8, "logZ_minus50": 3.0}[group],
        ),
        squeeze=False,
    )
    for ax, tag in zip(axes.flat, tags):
        for est, label, color, ls in STYLES:
            s = d[(d.tag == tag) & (d.estimator == est)]
            if s.empty:
                continue
            values = -s.elbo + (-50 if group == "logZ_minus50" else 0)
            g = values.groupby(s.step)
            med = g.median().rolling(15, center=True, min_periods=1).median()
            lo = g.quantile(0.1).rolling(15, center=True, min_periods=1).median()
            hi = g.quantile(0.9).rolling(15, center=True, min_periods=1).median()
            ax.plot(med.index, med, color=color, ls=ls, label=label, lw=1.1)
            ax.fill_between(
                med.index, lo, hi, where=g.count().eq(5), color=color, alpha=0.14
            )
        titles = {
            "banana": "Banana, K=5",
            "funnel": "Funnel, K=5",
            "rosenbrock": "Rosenbrock, K=5",
            "hier_k5": "Hierarchical mixture, K=5",
            "hier_k50": "Hierarchical mixture, K=50",
        }
        ax.set_title(
            f"Rosenbrock, n={tag[1:]}" if group == "dimension" else titles[tag]
        )
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xlim(0, 1000)
        ax.set_xticks([0, 10, 100, 1000])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_yscale("log" if ax.dataLim.ymin > 0 else "symlog")
        ax.set_xlabel("Optimization step")
        ax.set_ylabel(
            r"$\log Z - \widehat{\mathrm{ELBO}}(\theta)$"
            if group == "logZ_minus50"
            else r"$\mathrm{KL}(\mathbb{Q}_{\theta} \,\|\, \mathbb{P}_{\star})$"
        )
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    handles = [Line2D([0], [0], color=c, ls=ls, label=l) for e, l, c, ls in STYLES]
    for ax in list(axes.flat)[len(tags) :]:
        ax.axis("off")
    if group == "comparison_2d":
        axes.flat[-1].legend(handles=handles, loc="center", ncol=2, frameon=False)
        fig.tight_layout(h_pad=1.2, w_pad=0.8)
    else:
        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
        fig.tight_layout(rect=(0, 0.09, 1, 1), h_pad=1.2, w_pad=0.8)
    for ext in ["pdf", "png"]:
        output = H / (
            {
                "comparison_2d": "training_comparison_grid",
                "dimension": "training_comparison_grid_dimensions",
                "logZ_minus50": "unnorm_rosenbrock_logZ_m50_kl",
            }[group]
            + "."
            + ext
        )
        fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    for group in ["comparison_2d", "dimension", "logZ_minus50"]:
        plot(group)
