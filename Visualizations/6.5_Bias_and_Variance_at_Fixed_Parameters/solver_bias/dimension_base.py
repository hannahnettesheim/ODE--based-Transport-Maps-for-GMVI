"""Matched DM/SF/ST/OTR training configurations and checkpoint helpers."""
import sys as _sys
import pathlib as _pl

_ROOT = next(
    (_q for _q in _pl.Path(__file__).resolve().parents if (_q / "gmvi").is_dir())
)
_sys.path[:0] = [str(_ROOT), str(_ROOT / "Visualizations")]
from pathlib import Path
from dataclasses import asdict
import argparse
import json
import sys
import time
import traceback
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
from gmvi.experiments.execution import experiment_environment
from gmvi.experiments.trainer import RunConfig, run, build_model
from gmvi.targets.hybrid_rosenbrock import HybridRosenbrockTarget
from gmvi.targets.distributions import RosenbrockTarget
from gmvi.estimators.gradient_estimators import make_estimator
from gmvi.experiments.diagnostics import flat_grad
import gmvi.estimators.ode_transport as ot

ESTIMATORS = [
    "exact_marginalization",
    "score_function",
    "gumbel_softmax",
    "ode_transport",
]


def config(dim, estimator, seed):
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
            path="linear", ode_solver="dopri5", rtol=1e-07, atol=1e-07
        ),
    }[estimator]
    return RunConfig(
        target="rosenbrock",
        target_kwargs=dict(mu=1.0, a=0.05, b=5.0),
        dim=dim,
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
        checkpoint_fracs=[0, 0.05, 0.1, 0.25, 0.5, 1.0],
        record_every=1,
        var_every=0,
        probe_elbo_highres=True,
        probe_geometry=True,
        probe_wasserstein=False,
        probe_params=True,
        seed=seed,
        tag=f"n{dim}",
        verbose=True,
        log_every=50,
    )


def target_for(dim):
    return HybridRosenbrockTarget(dim=dim, n1=2, mu=1.0, a=0.05, b=5.0)


def checkpoint_velocity():
    from torch.utils.checkpoint import checkpoint

    original = ot._VelocityFunc.forward

    def forward(self, t, x):
        return checkpoint(
            lambda tt, xx: original(self, tt, xx), t, x, use_reentrant=False
        )

    ot._VelocityFunc.forward = forward
    return original


@experiment_environment()
def validate_setup():
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(9123)
    x = torch.randn(100, 2, dtype=torch.float64)
    torch.testing.assert_close(
        target_for(2).log_prob(x),
        RosenbrockTarget(a=0.05, b=5.0).log_prob(x),
        rtol=1e-07,
        atol=1e-07,
    )
    cfg = config(20, "ode_transport", 1)
    model = build_model(cfg, 20)
    target = target_for(20)

    def gradient(seed):
        torch.manual_seed(seed)
        model.zero_grad(set_to_none=True)
        est = make_estimator("ode_transport", MC_samples=64, **cfg.estimator_kwargs)
        loss, _ = est.loss(model, target.log_prob)
        loss.backward()
        return flat_grad(model).detach().clone()

    plain = gradient(41829)
    original = checkpoint_velocity()
    checked = gradient(41829)
    ot._VelocityFunc.forward = original
    torch.testing.assert_close(plain, checked, rtol=1e-05, atol=1e-06)
    timings = []
    for est_name in ESTIMATORS:
        cfg = config(20, est_name, 1)
        model = build_model(cfg, 20)
        original = checkpoint_velocity() if est_name == "ode_transport" else None
        started = time.time()
        estimator = make_estimator(
            est_name, MC_samples=cfg.mc_samples, **cfg.estimator_kwargs
        )
        loss, _ = estimator.loss(model, target.log_prob)
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(flat_grad(model)).all()
        timings.append(
            dict(
                estimator=est_name,
                dimension=20,
                M=8192,
                seconds=time.time() - started,
                loss=float(loss),
            )
        )
        if original is not None:
            ot._VelocityFunc.forward = original
    result = dict(
        passed=True,
        target_2d_matches=True,
        checkpoint_max_gradient_difference=float((plain - checked).abs().max()),
        initial_gradient_timings=timings,
    )
    (HERE / "validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


@experiment_environment()
def run_cell(dim, est_name, seed):
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    cell = HERE / "cells" / f"n{dim}_{est_name}_seed{seed}"
    cell.mkdir(parents=True, exist_ok=True)
    if (cell / "result.json").exists():
        return
    if (cell / "config.json").exists():
        raise RuntimeError(
            "Incomplete cell already exists; preserve it and use a fresh attempt directory."
        )
    cfg = config(dim, est_name, seed)
    (cell / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    if est_name == "ode_transport" and dim >= 10:
        checkpoint_velocity()

    def save(step, state, row):
        torch.save(state, cell / f"snapshot_{step}.pt")
        (cell / "progress.json").write_text(
            json.dumps(
                dict(step=step, time=time.time(), checkpoint=row), default=str, indent=2
            )
        )

    started = time.time()
    try:
        res = run(cfg, target=target_for(dim), checkpoint_callback=save)
        res.steps.to_csv(cell / "steps.csv", index=False)
        res.checkpoints.to_csv(cell / "checkpoints.csv", index=False)
        torch.save(
            dict(snapshots=res.snapshots, config=asdict(cfg)), cell / "snapshots.pt"
        )
        (cell / "meta.json").write_text(json.dumps(res.meta, indent=2, default=str))
        row = dict(
            dimension=dim,
            estimator=est_name,
            seed=seed,
            elapsed_s=time.time() - started,
            completed=not res.meta["diverged"] and res.meta["steps_completed"] == 1000,
            diverged=res.meta["diverged"],
            steps_completed=res.meta["steps_completed"],
        )
        (cell / "result.json").write_text(json.dumps(row, indent=2))
        print(json.dumps(row), flush=True)
    except Exception:
        (cell / "error.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--dim", type=int)
    ap.add_argument("--estimator")
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()
    if args.validate:
        validate_setup()
    else:
        run_cell(args.dim, args.estimator, args.seed)
