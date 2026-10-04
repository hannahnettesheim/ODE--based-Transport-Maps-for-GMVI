import sys as _sys
import pathlib as _pl

_ROOT = next(
    (_q for _q in _pl.Path(__file__).resolve().parents if (_q / "gmvi").is_dir())
)
_sys.path[:0] = [str(_ROOT), str(_ROOT / "Visualizations")]
from pathlib import Path
from dataclasses import asdict
import sys
import json
import time
import traceback
import math

HERE = Path(__file__).resolve().parent
import torch
import dimension_base as base
from gmvi.experiments.execution import experiment_environment
from gmvi.experiments.trainer import run


def config(job):
    group, tag, estimator, seed = job
    dim = int(tag[1:]) if group == "dimension" else 2
    cfg = base.config(dim, estimator, seed)
    cfg.tag = tag
    if group != "dimension":
        cfg.target = {
            "banana": "banana",
            "funnel": "funnel",
            "rosenbrock": "rosenbrock",
            "hier_k5": "hierarchical_mixture",
            "hier_k50": "hierarchical_mixture",
        }[tag]
        cfg.target_kwargs = dict(mu=1.0, a=0.05, b=5.0) if tag == "rosenbrock" else {}
        cfg.components = 50 if tag == "hier_k50" else 5
        cfg.checkpoint_fracs = [0.05, 0.1, 0.25, 0.5, 1.0]
        if estimator == "ode_transport":
            cfg.estimator_kwargs.update(rtol=1e-05, atol=1e-05)
    if group == "logZ_minus50":
        cfg.target_Z = math.exp(-50)
    assert cfg.chart == "cholesky"
    return cfg


def cell_path(job):
    return HERE / job[0] / "cells" / "_".join(map(str, job[1:]))


@experiment_environment()
def main(job):
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    cfg = config(job)
    cell = cell_path(job)
    cell.mkdir(parents=True, exist_ok=True)
    if (cell / "result.json").exists():
        return
    (cell / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    if cfg.estimator == "ode_transport" and (cfg.dim >= 10 or cfg.components == 50):
        base.checkpoint_velocity()

    def save(step, state, row):
        torch.save(state, cell / f"snapshot_{step}.pt")
        (cell / "progress.json").write_text(
            json.dumps({"step": step, "time": time.time()}, indent=2)
        )

    start = time.time()
    try:
        result = run(
            cfg,
            target=base.target_for(cfg.dim) if job[0] == "dimension" else None,
            checkpoint_callback=save,
        )
        result.steps.to_csv(cell / "steps.csv", index=False)
        result.checkpoints.to_csv(cell / "checkpoints.csv", index=False)
        (cell / "meta.json").write_text(json.dumps(result.meta, indent=2, default=str))
        row = {
            "job": job,
            "elapsed_s": time.time() - start,
            "completed": not result.meta["diverged"]
            and result.meta["steps_completed"] == 1000,
            "steps_completed": result.meta["steps_completed"],
        }
        (cell / "result.json").write_text(json.dumps(row, indent=2))
    except Exception:
        (cell / "error.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
