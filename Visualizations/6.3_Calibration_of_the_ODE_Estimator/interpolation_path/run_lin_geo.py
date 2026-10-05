"""Compare linear and geometric transport paths across four target regimes."""
import sys as _sys
import pathlib as _pl

_ROOT = next(
    (_q for _q in _pl.Path(__file__).resolve().parents if (_q / "gmvi").is_dir())
)
_sys.path[:0] = [str(_ROOT), str(_ROOT / "Visualizations")]
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
import pandas as pd
from gmvi.experiments.trainer import RunConfig, run

HERE = os.path.dirname(__file__)
BASE = "lin_geo"
CKPTS = [0, 10, 50, 100, 250, 500, 1000]
SEEDS = [1, 2, 3, 4, 5]
M = 1024
TOL = 1e-06
CHART = "matrixexponential"
REGIMES = {
    "banana": dict(target="banana", components=5, target_kwargs={}),
    "rosenbrock_extreme": dict(
        target="rosenbrock", components=5, target_kwargs=dict(a=0.05, b=5.0)
    ),
    "hier_k5": dict(target="hierarchical_mixture", components=5, target_kwargs={}),
    "hier_k50": dict(target="hierarchical_mixture", components=50, target_kwargs={}),
}
PATHS = ["linear", "geometric"]
JOBS = [(regime, path, s) for regime in REGIMES for path in PATHS for s in SEEDS]


def training_config(regime, path, seed):
    regime_config = REGIMES[regime]
    return RunConfig(
        target=regime_config["target"],
        target_kwargs=regime_config["target_kwargs"],
        components=regime_config["components"],
        chart=CHART,
        init_scale=0.5,
        estimator="ode_transport",
        estimator_kwargs=dict(path=path, ode_solver="dopri5", rtol=TOL, atol=TOL),
        mc_samples=M,
        n_steps=1000,
        lr=0.005,
        optimizer="adam",
        scheduler="none",
        checkpoint_fracs=[c / 1000 for c in CKPTS],
        probe_geometry=True,
        probe_elbo_highres=True,
        probe_wasserstein=False,
        probe_params=True,
        seed=seed,
        tag=f"{regime}|{path}",
        verbose=False,
    )


def run_training_job(regime, path, seed):
    result = run(training_config(regime, path, seed))
    steps, checkpoints = (result.steps, result.checkpoints)
    cols = [c for c in ("step", "nfe", "grad_norm") if c in steps.columns]
    checkpoints = checkpoints.merge(
        steps[steps.step.isin(CKPTS)][cols].drop_duplicates("step"),
        on="step",
        how="left",
    )
    for f, v in dict(regime=regime, path=path, seed=seed).items():
        checkpoints[f] = v
        steps[f] = v
    checkpoints["nfe_mean"] = (
        float(np.nanmean(steps["nfe"])) if "nfe" in steps else np.nan
    )
    checkpoints["nfe_max"] = (
        float(np.nanmax(steps["nfe"])) if "nfe" in steps else np.nan
    )
    parameters = [
        dict(regime=regime, path=path, seed=seed, param=name, idx=i, value=float(v))
        for name, tensor in result.snapshots.get(1000, {}).items()
        for i, v in enumerate(tensor.detach().cpu().numpy().ravel())
    ]
    return (checkpoints, steps, pd.DataFrame(parameters), result.meta)


def main():
    from gmvi.experiments.execution import run_training_jobs

    run_training_jobs(JOBS, run_training_job, HERE, BASE, config_for=training_config)


if __name__ == "__main__":
    main()
