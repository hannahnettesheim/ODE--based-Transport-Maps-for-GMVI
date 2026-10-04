"""M-sweep variance and DM-reference discrepancy at common frozen snapshots."""
import sys as _sys
import pathlib as _pl

_ROOT = next(
    (_q for _q in _pl.Path(__file__).resolve().parents if (_q / "gmvi").is_dir())
)
_sys.path[:0] = [str(_ROOT), str(_ROOT / "Visualizations")]
from pathlib import Path
import argparse
import json
import os
import sys
import time
import traceback
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
from gmvi.experiments.trainer import RunConfig, run, build_model, build_target
from gmvi.experiments.diagnostics import flat_grad
from gmvi.experiments.fixed_theta import gradient_draw
from gmvi.experiments.execution import experiment_environment
from gmvi.estimators.gradient_estimators import make_estimator
import gmvi.estimators.ode_transport as ot

REGIMES = {
    f"n{d}": dict(target="rosenbrock", components=5, target_kwargs=dict(a=0.05, b=5.0))
    for d in [5, 10, 20]
}
MS = [1024]
EXPECTED_CELLS = 36
ARMS = ["exact_marginalization", "score_function", "gumbel_softmax", "ode_transport"]
TEMPERATURE = 1.0
STEPS = [50, 250, 1000]
R = 5000
ODE = dict(path="linear", ode_solver="dopri5", rtol=1e-07, atol=1e-07)
from gmvi.targets.hybrid_rosenbrock import HybridRosenbrockTarget


def build_target(cfg):
    return HybridRosenbrockTarget(dim=cfg.dim, n1=2, **cfg.target_kwargs)


def cfg_for(regime):
    return RunConfig(
        **REGIMES[regime],
        dim=int(regime[1:]),
        chart="cholesky",
        init_scale=0.5,
        estimator="ode_transport",
        estimator_kwargs=ODE,
        mc_samples=1024,
        n_steps=1000,
        lr=0.005,
        optimizer="adam",
        scheduler="none",
        checkpoint_fracs=[s / 1000 for s in [0, 10, 50, 100, 250, 500, 1000]],
        probe_geometry=True,
        probe_elbo_highres=True,
        probe_wasserstein=False,
        probe_params=True,
        seed=1,
        tag=regime,
        verbose=True,
    )


def enable_checkpointing():
    from torch.utils.checkpoint import checkpoint

    original = ot._VelocityFunc.forward

    def forward(self, t, x):
        return checkpoint(
            lambda tt, xx: original(self, tt, xx), t, x, use_reentrant=False
        )

    ot._VelocityFunc.forward = forward
    return original


def grad(model, target, estimator, mc_samples, seed):
    if estimator == "ode_transport":
        kwargs = ODE
    elif estimator == "gumbel_softmax":
        kwargs = dict(
            mode="straight_through",
            temperature=TEMPERATURE,
            anneal_rate=1.0,
            min_temperature=TEMPERATURE,
        )
    elif estimator == "score_function":
        kwargs = dict(baseline="none")
    else:
        kwargs = {}
    gradient, info = gradient_draw(
        model, target.log_prob, estimator, mc_samples, seed, kwargs, dtype=np.float64
    )
    return (gradient, info.get("nfe", np.nan))


def check():
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    cfg = cfg_for("n10")
    target = build_target(cfg)
    model = build_model(cfg, cfg.dim)
    a, _ = grad(model, target, "ode_transport", 256, 91919)
    original = enable_checkpointing()
    b, _ = grad(model, target, "ode_transport", 256, 91919)
    np.testing.assert_allclose(a, b, rtol=1e-05, atol=1e-06)
    ot._VelocityFunc.forward = original
    result = dict(
        passed=True,
        max_absolute_gradient_difference=float(np.max(np.abs(a - b))),
        mc=256,
        components=cfg.components,
    )
    (HERE / "checkpoint_validation.json").write_text(json.dumps(result, indent=2))
    print(result, flush=True)


def measure(regime):
    global TEMPERATURE
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    directory = HERE / regime
    directory.mkdir(exist_ok=True)
    if int(regime[1:]) >= 10:
        enable_checkpointing()
    cfg = cfg_for(regime)
    target = build_target(cfg)
    if not (directory / "training_meta.json").exists():

        def save(step, state, row):
            if step in STEPS:
                torch.save(state, directory / f"snapshot_{step}.pt")

        result = run(cfg, target=target, checkpoint_callback=save)
        result.steps.to_csv(directory / "training_steps.csv", index=False)
        result.checkpoints.to_csv(directory / "training_checkpoints.csv", index=False)
        (directory / "training_meta.json").write_text(
            json.dumps(result.meta, indent=2, default=str)
        )
    for step in STEPS:
        state_path = directory / f"snapshot_{step}.pt"
        if not state_path.exists():
            raise RuntimeError(f"Training did not reach snapshot {step}")
        model = build_model(cfg, cfg.dim, reseed=False)
        model.load_state_dict(torch.load(state_path))
        ref_path = directory / f"dm_reference_{step}.npz"
        if ref_path.exists():
            reference_chunks = np.load(ref_path)["chunks"]
        else:
            reference_chunks = np.stack(
                [
                    grad(
                        model,
                        target,
                        "exact_marginalization",
                        500,
                        900000 + step * 100 + i,
                    )[0]
                    for i in range(40)
                ]
            )
            np.savez(
                ref_path,
                chunks=reference_chunks,
                mean=reference_chunks.mean(0),
                samples_per_component=20000,
            )
        reference_mean = reference_chunks.mean(0)
        reference_variance = reference_chunks.var(0, ddof=1).sum() / len(
            reference_chunks
        )
        print(
            f"{regime} step={step}: DM reference RMS SE={np.sqrt(reference_variance):.5g}",
            flush=True,
        )
        for estimator in ARMS:
            mc_samples = 1024
            TEMPERATURE = max(0.3, 0.9995**step)
            cell = directory / f"step_{step}_{estimator}"
            cell.mkdir(exist_ok=True)
            if (cell / "result.json").exists():
                continue
            gradients = []
            nfes = []
            started = time.time()
            for saved in sorted(
                cell.glob("gradients_r*.npz"), key=lambda p: int(p.stem.split("_r")[1])
            ):
                with np.load(saved) as data:
                    gradients.extend(data["gradients"])
                    nfes.extend(data["nfe"])
            if len(gradients) > R:
                raise RuntimeError("Saved replicate count exceeds requested R")
            for r in range(len(gradients), R):
                g, nfe = grad(
                    model,
                    target,
                    estimator,
                    mc_samples,
                    15000000
                    + STEPS.index(step) * 100000
                    + ARMS.index(estimator) * 10000
                    + r,
                )
                gradients.append(g)
                nfes.append(nfe)
                if (r + 1) % 50 == 0:
                    np.savez(
                        cell / f"gradients_r{r + 1:03d}.npz",
                        gradients=np.stack(gradients[-50:]),
                        nfe=nfes[-50:],
                    )
                    print(
                        f"{regime} step={step} M={mc_samples}: {r + 1}/{R} replicates",
                        flush=True,
                    )
            gradient_samples = np.stack(gradients)
            variance = float(gradient_samples.var(0, ddof=1).sum())
            bias = float(np.linalg.norm(gradient_samples.mean(0) - reference_mean))
            floor = float(np.sqrt(variance / R + reference_variance))
            rng = np.random.default_rng(38291 + step + mc_samples)
            bootstrap_variances = np.array(
                [
                    gradient_samples[rng.integers(R, size=R)].var(0, ddof=1).sum()
                    for _ in range(200)
                ]
            )
            row = dict(
                estimator=estimator,
                dimension=cfg.dim,
                temperature=TEMPERATURE if estimator == "gumbel_softmax" else None,
                regime=regime,
                step=step,
                M=mc_samples,
                replicates=R,
                variance=variance,
                variance_bootstrap_lo=float(np.quantile(bootstrap_variances, 0.025)),
                variance_bootstrap_hi=float(np.quantile(bootstrap_variances, 0.975)),
                M_times_variance=mc_samples * variance,
                bias=bias,
                bias_rms_sampling_error=floor,
                bias_squared_corrected=bias * bias - floor * floor,
                reference_rms_se=float(np.sqrt(reference_variance)),
                otr_mean_rms_se=float(np.sqrt(variance / R)),
                reference_gradient_norm=float(np.linalg.norm(reference_mean)),
                dm_reference_samples_per_component=20000,
                reference_to_otr_se_ratio=float(
                    np.sqrt(reference_variance / (variance / R))
                )
                if variance > 0
                else None,
                nfe_mean=float(np.mean(nfes)),
                elapsed_s=time.time() - started,
            )
            squared = np.square(gradient_samples - gradient_samples.mean(0)).sum(1)
            row["largest_draw_variance_fraction"] = float(squared.max() / squared.sum())
            row["variance_first_half"] = float(
                gradient_samples[: R // 2].var(0, ddof=1).sum()
            )
            row["variance_second_half"] = float(
                gradient_samples[R // 2 :].var(0, ddof=1).sum()
            )
            row["parameter_count"] = int(gradient_samples.shape[1])
            row["variance_per_parameter"] = variance / gradient_samples.shape[1]
            (cell / "result.json").write_text(json.dumps(row, indent=2))
            print(json.dumps(row), flush=True)
    (directory / "completed.json").write_text(
        json.dumps(dict(regime=regime, state="completed"))
    )


def launch(regime):
    directory = HERE / regime
    directory.mkdir(exist_ok=True)
    try:
        with experiment_environment():
            measure(regime)
    except Exception:
        (directory / "error.txt").write_text(traceback.format_exc())
        raise
    return dict(
        regime=regime, exit_code=0, completed=(directory / "completed.json").exists()
    )


def summarize(statuses):
    rows = [json.loads(p.read_text()) for p in HERE.glob("*/step_*/result.json")]
    d = pd.DataFrame(rows)
    d.to_csv(HERE / "variance_results.csv", index=False)
    result = dict(
        statuses=statuses,
        completed_cells=len(d),
        expected_cells=EXPECTED_CELLS,
        all_complete=len(d) == EXPECTED_CELLS
        and all((s["completed"] for s in statuses)),
    )
    (HERE / "completion.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=list(REGIMES))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        with experiment_environment():
            check()
        return
    if args.regime:
        launch(args.regime)
        return
    statuses = []
    for regime in list(REGIMES):
        status = launch(regime)
        statuses.append(status)
        print(json.dumps(status), flush=True)
    summarize(statuses)


if __name__ == "__main__":
    main()
