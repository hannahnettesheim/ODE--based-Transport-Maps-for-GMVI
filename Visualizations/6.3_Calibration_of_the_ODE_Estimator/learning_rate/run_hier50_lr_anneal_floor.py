"""K=50 hierarchical-mixture learning-rate sweep."""
import sys as _sys
import pathlib as _pl

_ROOT = next(
    (_q for _q in _pl.Path(__file__).resolve().parents if (_q / "gmvi").is_dir())
)
_sys.path[:0] = [str(_ROOT), str(_ROOT / "Visualizations")]
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
from gmvi.experiments.trainer import RunConfig, run

HERE = os.path.dirname(__file__)
BASE = "hier50_lr_anneal_floor"
NSTEPS = 2000
CKPTS = [100, 500, 1000, 2000]
LRS = [0.001, 0.005, 0.01, 0.02, 0.05]
SCHEDS = ["none", "cosine"]
ETA_MIN_RATIO = 0.1
SEED = 1
TOL = 1e-05
JOBS = [(lr, schedule_name, SEED) for lr in LRS for schedule_name in SCHEDS]


def training_config(lr, schedule_name, seed):
    return RunConfig(
        target="hierarchical_mixture",
        target_kwargs={},
        components=50,
        chart="cholesky",
        init_scale=0.5,
        estimator="ode_transport",
        estimator_kwargs=dict(path="linear", ode_solver="dopri5", rtol=TOL, atol=TOL),
        mc_samples=2048,
        n_steps=NSTEPS,
        lr=lr,
        optimizer="adam",
        scheduler=schedule_name,
        scheduler_eta_min_ratio=ETA_MIN_RATIO,
        checkpoint_fracs=[step / NSTEPS for step in CKPTS],
        probe_geometry=True,
        probe_elbo_highres=True,
        probe_wasserstein=False,
        probe_params=True,
        seed=seed,
        tag=f"lr{lr:g}|{('floor0.1' if schedule_name == 'cosine' else 'strict')}",
        verbose=False,
    )


def run_training_job(lr, schedule_name, seed):
    result = run(training_config(lr, schedule_name, seed))
    steps, checkpoints = (result.steps, result.checkpoints)
    cols = [c for c in ("step", "nfe", "grad_norm") if c in steps.columns]
    checkpoints = checkpoints.merge(
        steps[steps.step.isin(CKPTS)][cols].drop_duplicates("step"),
        on="step",
        how="left",
    )
    schedule = "anneal_floor_0.1" if schedule_name == "cosine" else "strict"
    fields = dict(
        schedule=schedule,
        sched=schedule_name,
        eta_min_ratio=ETA_MIN_RATIO if schedule_name == "cosine" else 1.0,
        seed=seed,
    )
    for frame in (checkpoints, steps):
        for name, value in fields.items():
            frame[name] = value
    checkpoints["nfe_mean"] = (
        float(np.nanmean(steps["nfe"])) if "nfe" in steps else np.nan
    )
    parameters = [
        dict(
            lr=lr,
            schedule=schedule,
            sched=schedule_name,
            seed=seed,
            param=name,
            idx=i,
            value=float(value),
        )
        for name, tensor in result.snapshots.get(NSTEPS, {}).items()
        for i, value in enumerate(tensor.detach().cpu().numpy().ravel())
    ]
    meta = dict(
        result.meta,
        schedule=schedule,
        sched=schedule_name,
        eta_min_ratio=fields["eta_min_ratio"],
    )
    return (checkpoints, steps, pd.DataFrame(parameters), meta)


def main():
    from gmvi.experiments.execution import run_training_jobs

    run_training_jobs(JOBS, run_training_job, HERE, BASE, config_for=training_config)


if __name__ == "__main__":
    main()
