"""dopri5 tolerance sweep, K=50 mixture on the hierarchical target. OTR linear,"""
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
BASE = "dopri_hier50"
CKPTS = [0, 10, 50, 100, 250, 500, 1000]
SEEDS = [1, 2, 3, 4, 5]
TOLS = [10.0 ** (-k) for k in range(1, 10)]
JOBS = [(tensor, s) for tensor in TOLS for s in SEEDS]


def cfg_for(tol, seed):
    return RunConfig(
        target="hierarchical_mixture",
        target_kwargs={},
        components=50,
        chart="cholesky",
        init_scale=0.5,
        estimator="ode_transport",
        estimator_kwargs=dict(path="linear", ode_solver="dopri5", rtol=tol, atol=tol),
        mc_samples=2048,
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
        tag=f"dopri5|{tol:g}",
        verbose=False,
    )


def one_job(tol, seed):
    result = run(cfg_for(tol, seed))
    steps, checkpoints = (result.steps, result.checkpoints)
    cols = [c for c in ("step", "nfe", "grad_norm") if c in steps.columns]
    step_diagnostics = steps[steps.step.isin(CKPTS)][cols].drop_duplicates("step")
    checkpoints = checkpoints.merge(step_diagnostics, on="step", how="left")
    checkpoints["target"] = "hier_mixture"
    checkpoints["solver_kind"] = "dopri5"
    checkpoints["solver_param"] = tol
    checkpoints["seed"] = seed
    checkpoints["nfe_mean"] = (
        float(np.nanmean(steps["nfe"])) if "nfe" in steps else np.nan
    )
    checkpoints["nfe_max"] = (
        float(np.nanmax(steps["nfe"])) if "nfe" in steps else np.nan
    )
    steps = steps.assign(
        target="hier_mixture", solver_kind="dopri5", solver_param=tol, seed=seed
    )
    parameters = []
    for name, tensor in result.snapshots.get(1000, {}).items():
        values = tensor.detach().cpu().numpy().ravel()
        for i, val in enumerate(values):
            parameters.append(
                dict(
                    target="hier_mixture",
                    solver_param=tol,
                    seed=seed,
                    param=name,
                    idx=i,
                    value=float(val),
                )
            )
    return (checkpoints, steps, pd.DataFrame(parameters), result.meta)


def main():
    from gmvi.experiments.execution import run_training_jobs

    run_training_jobs(JOBS, one_job, HERE, BASE, config_for=cfg_for)


if __name__ == "__main__":
    main()
