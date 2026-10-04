import sys as _sys
import pathlib as _pl

_ROOT = next(
    (_q for _q in _pl.Path(__file__).resolve().parents if (_q / "gmvi").is_dir())
)
_sys.path[:0] = [str(_ROOT), str(_ROOT / "Visualizations")]
from pathlib import Path
import sys
import json
import os

HERE = Path(__file__).resolve().parent
os.environ["MPLCONFIGDIR"] = str(HERE / "mpl_cache")
import torch
import numpy as np
import pandas as pd
from gmvi.experiments.execution import experiment_environment
from gmvi.experiments.trainer import RunConfig, run, build_model, build_target
from gmvi.experiments.fixed_theta import gradient_draw
from gmvi.estimators.gradient_estimators import make_estimator
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = HERE / "solver_hier_k5_20260922"
OUTPUT_DIR.mkdir(exist_ok=True)
ARMS = [
    ("dopri5", t, dict(ode_solver="dopri5", rtol=t, atol=t))
    for t in [0.01, 0.001, 0.0001, 1e-05, 1e-06]
] + [("rk4", n, dict(ode_solver="rk4", ode_steps=n)) for n in [2, 8, 32, 128]]


@experiment_environment()
def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--step", type=int, choices=[50, 250, 1000])
    args = ap.parse_args(argv)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    from dimension_base import checkpoint_velocity

    checkpoint_velocity()
    cfg = RunConfig(
        target="hierarchical_mixture",
        target_kwargs={},
        dim=2,
        components=5,
        chart="cholesky",
        init_scale=0.5,
        estimator="exact_marginalization",
        mc_samples=1024,
        n_steps=1000,
        lr=0.005,
        scheduler="none",
        checkpoint_fracs=[0.05, 0.25, 1.0],
        probe_params=True,
        probe_elbo_highres=False,
        probe_geometry=False,
        probe_wasserstein=False,
        seed=1,
    )
    from dataclasses import asdict

    (OUTPUT_DIR / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    if (OUTPUT_DIR / "snapshots.pt").exists():
        snapshots = torch.load(OUTPUT_DIR / "snapshots.pt")
    else:
        print("Training DM snapshot trajectory: Cholesky, K=5, float64", flush=True)

        def save(step, state, row):
            torch.save(state, OUTPUT_DIR / f"snapshot_{step}.pt")
            print(f"DM snapshot step={step}", flush=True)

        res = run(cfg, checkpoint_callback=save)
        assert not res.meta["diverged"] and res.meta["steps_completed"] == 1000
        snapshots = res.snapshots
        torch.save(snapshots, OUTPUT_DIR / "snapshots.pt")
        res.steps.to_csv(OUTPUT_DIR / "training_steps.csv", index=False)
    if args.train_only:
        return
    target = build_target(cfg)
    rows = []
    for step, state in sorted(snapshots.items()):
        if args.step is not None and step != args.step:
            continue
        model = build_model(cfg, 2)
        model.load_state_dict(state)

        def grad(estimator, mc_samples, seed, estimator_kwargs):
            gradient, _ = gradient_draw(
                model,
                target.log_prob,
                estimator,
                mc_samples,
                seed,
                estimator_kwargs,
                set_to_none=True,
            )
            return gradient

        reference_path = OUTPUT_DIR / f"dm_reference_{step}.npz"
        if reference_path.exists():
            reference_chunks = np.load(reference_path)["chunks"]
        else:
            reference_chunks = np.stack(
                [
                    grad("exact_marginalization", 500, 900000 + step * 100 + i, {})
                    for i in range(40)
                ]
            )
            np.savez(
                reference_path, chunks=reference_chunks, mean=reference_chunks.mean(0)
            )
        for arm_index, (kind, setting, estimator_kwargs) in enumerate(ARMS):
            gradients_path = OUTPUT_DIR / f"gradients_{step}_{kind}_{setting:g}.npy"
            if gradients_path.exists():
                gradient_samples = np.load(gradients_path)
            else:
                chunks = []
                for start in range(0, 5000, 50):
                    chunk_path = (
                        OUTPUT_DIR
                        / f"gradients_{step}_{kind}_{setting:g}_r{start:05d}.npy"
                    )
                    if chunk_path.exists():
                        chunk = np.load(chunk_path)
                    else:
                        chunk = np.stack(
                            [
                                grad(
                                    "ode_transport",
                                    1024,
                                    15000000 + step * 100000 + arm_index * 10000 + r,
                                    dict(path="linear", **estimator_kwargs),
                                )
                                for r in range(start, start + 50)
                            ]
                        )
                        np.save(chunk_path, chunk)
                    chunks.append(chunk)
                    print(
                        f"step={step} {kind} {setting} replicates={start + 50}/5000",
                        flush=True,
                    )
                gradient_samples = np.concatenate(chunks)
                np.save(gradients_path, gradient_samples)
            variance = gradient_samples.var(0, ddof=1).sum()
            bias = np.linalg.norm(gradient_samples.mean(0) - reference_chunks.mean(0))
            rows.append(
                dict(
                    step=step,
                    solver=kind,
                    setting=setting,
                    bias=float(bias),
                    variance=float(variance),
                    bias_rms_sampling_error=float(
                        np.sqrt(
                            variance / 5000 + reference_chunks.var(0, ddof=1).sum() / 40
                        )
                    ),
                    replicates=5000,
                    mc=1024,
                    dm_samples_per_component=20000,
                )
            )
            pd.DataFrame(rows).to_csv(
                OUTPUT_DIR
                / (
                    f"solver_bias_step_{args.step}.csv"
                    if args.step is not None
                    else "solver_bias_dm_reference.csv"
                ),
                index=False,
            )
            print(rows[-1], flush=True)
    if args.step is not None:
        return
    import runpy

    runpy.run_path(
        str(HERE / "plot_solver_combined_normalized.py"), run_name="__main__"
    )
    (OUTPUT_DIR / "completion.json").write_text(
        json.dumps({"completed": len(rows) == 27})
    )


if __name__ == "__main__":
    main()
