"""Fresh, reproducible normalized and unnormalized extreme Rosenbrock comparisons."""
from pathlib import Path
import os
import sys
import json
import math
from dataclasses import asdict

HERE = Path(__file__).resolve().parent
os.environ["MPLCONFIGDIR"] = str(HERE / "mpl_cache")
ROOT = next((parent for parent in HERE.parents if (parent / "gmvi").is_dir()))
sys.path.insert(0, str(ROOT))
ESTIMATORS = [
    "exact_marginalization",
    "score_function",
    "gumbel_softmax",
    "ode_transport",
]


def worker(group, estimator, seed):
    import torch
    from gmvi.experiments.trainer import RunConfig, run

    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    kwargs = {
        "exact_marginalization": {},
        "score_function": {"baseline": "none"},
        "gumbel_softmax": dict(
            mode="straight_through",
            temperature=1.0,
            anneal_rate=0.9995,
            min_temperature=0.3,
        ),
        "ode_transport": dict(
            path="linear", ode_solver="dopri5", rtol=1e-05, atol=1e-05
        ),
    }[estimator]
    cfg = RunConfig(
        target="rosenbrock",
        target_kwargs=dict(mu=1.0, a=0.05, b=5.0),
        target_Z=1.0 if group == "extreme_normalized" else math.exp(-50),
        dim=2,
        components=5,
        chart="cholesky",
        init_scale=0.5,
        estimator=estimator,
        estimator_kwargs=kwargs,
        mc_samples=8192,
        n_steps=1000,
        lr=0.02,
        optimizer="adam",
        scheduler="cosine",
        scheduler_eta_min_ratio=0.1,
        checkpoint_fracs=[0.05, 0.1, 0.25, 0.5, 1.0],
        record_every=1,
        var_every=0,
        probe_elbo_highres=True,
        probe_geometry=True,
        probe_wasserstein=False,
        probe_params=True,
        seed=seed,
        tag="rosenbrock",
        verbose=True,
        log_every=50,
    )
    cell = HERE / group / "cells" / f"{estimator}_{seed}"
    cell.mkdir(parents=True, exist_ok=True)
    (cell / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    def save(step, state, row):
        torch.save(state, cell / f"snapshot_{step}.pt")
        (cell / "progress.json").write_text(json.dumps({"step": step}))

    result = run(cfg, checkpoint_callback=save)
    result.steps.to_csv(cell / "steps.csv", index=False)
    result.checkpoints.to_csv(cell / "checkpoints.csv", index=False)
    (cell / "meta.json").write_text(json.dumps(result.meta, indent=2, default=str))
    completed = not result.meta["diverged"] and result.meta["steps_completed"] == 1000
    (cell / "result.json").write_text(
        json.dumps({"completed": completed, "steps": result.meta["steps_completed"]})
    )
    if not completed:
        raise RuntimeError(f"Incomplete run: {cell}")


def launch(job):
    from gmvi.experiments.execution import experiment_environment

    group, estimator, seed = job
    cell = HERE / group / "cells" / f"{estimator}_{seed}"
    cell.mkdir(parents=True, exist_ok=True)
    if not (cell / "result.json").exists():
        with experiment_environment():
            worker(*job)
    completed = json.loads((cell / "result.json").read_text())["completed"]
    return dict(job=job, exit_code=0 if completed else 1)


def plot():
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter, NullLocator

    styles = [
        ("exact_marginalization", "DM", "#264653", "-"),
        ("score_function", "SF", "#E63946", "--"),
        ("gumbel_softmax", "ST", "#457B9D", "-."),
        ("ode_transport", "OTR", "#2A9D8F", "-"),
    ]
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
    for group in ["extreme_normalized", "extreme_logZ_minus50"]:
        files = sorted((HERE / group / "cells").glob("*/steps.csv"))
        if not files:
            continue
        data = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        data.to_csv(HERE / group / "training_steps.csv", index=False)
        fig, ax = plt.subplots(figsize=(418 / 72.27, 3.0))
        for estimator, label, color, linestyle in styles:
            s = data[data.estimator == estimator]
            if s.empty:
                continue
            values = -s.elbo + (0 if group == "extreme_normalized" else -50)
            grouped = values.groupby(s.step)
            median = grouped.median().rolling(15, center=True, min_periods=1).median()
            lower = (
                grouped.quantile(0.1).rolling(15, center=True, min_periods=1).median()
            )
            upper = (
                grouped.quantile(0.9).rolling(15, center=True, min_periods=1).median()
            )
            ax.plot(
                median.index, median, color=color, ls=linestyle, label=label, lw=1.1
            )
            ax.fill_between(
                median.index,
                lower,
                upper,
                where=grouped.count().eq(5),
                color=color,
                alpha=0.14,
            )
        ax.set_title(
            "Rosenbrock (extreme), K=5"
            if group == "extreme_normalized"
            else "Rosenbrock (extreme), K=5, $\\log Z=-50$"
        )
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xlim(0, 1000)
        ax.set_xticks([0, 10, 100, 1000])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_yscale("log" if ax.dataLim.ymin > 0 else "symlog")
        ax.set_xlabel("Optimization step")
        ax.set_ylabel(
            "$\\mathrm{KL}(\\mathbb{Q}_{\\theta} \\,\\|\\, \\mathbb{P}_{\\star})$"
            if group == "extreme_normalized"
            else "$\\log Z - \\widehat{\\mathrm{ELBO}}(\\theta)$"
        )
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        fig.legend(loc="lower center", ncol=4, frameon=False)
        fig.tight_layout(rect=(0, 0.09, 1, 1))
        for ext in ["pdf", "png"]:
            fig.savefig(
                HERE / f"rosenbrock_{group}.{ext}", dpi=220, bbox_inches="tight"
            )
        plt.close(fig)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        worker(*json.loads(sys.argv[2]))
    elif "--plot" in sys.argv:
        plot()
    else:
        jobs = [
            (group, estimator, seed)
            for seed in range(1, 6)
            for estimator in ESTIMATORS
            for group in ["extreme_normalized", "extreme_logZ_minus50"]
        ]
        results = []
        for job in jobs:
            results.append(launch(job))
            print(json.dumps(results[-1]), flush=True)
            (HERE / "progress.json").write_text(
                json.dumps(
                    {"finished": len(results), "total": len(jobs), "results": results},
                    indent=2,
                )
            )
        plot()
        success = all((row["exit_code"] == 0 for row in results))
        (HERE / "completion.json").write_text(
            json.dumps({"success": success, "finished": len(results)}, indent=2)
        )
        sys.exit(0 if success else 1)
