"""MC-sample sweep. 4 regimes x M in {2^8,2^10,2^12,2^14} x 5 seeds. OTR linear,"""
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
BASE = "M_sweep"
CKPTS = [0, 10, 50, 100, 250, 500, 1000]
SEEDS = [1, 2, 3, 4, 5]
MS = [2**8, 2**10, 2**12, 2**14]
TOL = 1e-07
REGIMES = {
    "banana": dict(target="banana", components=5, target_kwargs={}),
    "rosenbrock_extreme": dict(
        target="rosenbrock", components=5, target_kwargs=dict(a=0.05, b=5.0)
    ),
    "hier_k5": dict(target="hierarchical_mixture", components=5, target_kwargs={}),
    "hier_k50": dict(target="hierarchical_mixture", components=50, target_kwargs={}),
}
JOBS = [(regime, m, s) for regime in REGIMES for m in MS for s in SEEDS]


def cfg_for(regime, m, seed):
    regime_config = REGIMES[regime]
    return RunConfig(
        target=regime_config["target"],
        target_kwargs=regime_config["target_kwargs"],
        components=regime_config["components"],
        chart="cholesky",
        init_scale=0.5,
        estimator="ode_transport",
        estimator_kwargs=dict(path="linear", ode_solver="dopri5", rtol=TOL, atol=TOL),
        mc_samples=m,
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
        tag=f"{regime}|M{m}",
        verbose=False,
    )


def one_job(regime, m, seed):
    result = run(cfg_for(regime, m, seed))
    steps, checkpoints = (result.steps, result.checkpoints)
    cols = [c for c in ("step", "nfe", "grad_norm") if c in steps.columns]
    checkpoints = checkpoints.merge(
        steps[steps.step.isin(CKPTS)][cols].drop_duplicates("step"),
        on="step",
        how="left",
    )
    checkpoints["regime"] = regime
    checkpoints["M"] = m
    checkpoints["seed"] = seed
    checkpoints["nfe_mean"] = (
        float(np.nanmean(steps["nfe"])) if "nfe" in steps else np.nan
    )
    steps = steps.assign(regime=regime, M=m, seed=seed)
    parameters = [
        dict(regime=regime, M=m, seed=seed, param=name, idx=i, value=float(v))
        for name, tensor in result.snapshots.get(1000, {}).items()
        for i, v in enumerate(tensor.detach().cpu().numpy().ravel())
    ]
    return (checkpoints, steps, pd.DataFrame(parameters), result.meta)


def main():
    from gmvi.experiments.execution import run_training_jobs

    run_training_jobs(JOBS, one_job, HERE, BASE, config_for=cfg_for)


if __name__ == "__main__":
    main()
