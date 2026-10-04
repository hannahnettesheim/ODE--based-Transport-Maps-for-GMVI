from pathlib import Path
import sys
import importlib.util
import json
import time
import numpy as np
import torch

ROOT = next(
    (
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "gmvi").is_dir()
    )
)
import sys

sys.path.insert(0, str(ROOT))
from gmvi.experiments.execution import experiment_environment


@experiment_environment()
def main():
    HERE = Path(__file__).resolve().parent
    BASE = HERE.parent / "variance_mc_sample_size"
    spec = importlib.util.spec_from_file_location("study", BASE / "run_study.py")
    s = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s)
    torch.set_num_threads(1)
    D = BASE / "rosenbrock_extreme"
    cell = D / "step_1000_M_16384"
    G = np.concatenate(
        [
            np.load(p)["gradients"]
            for p in sorted(
                cell.glob("gradients_r*.npz"), key=lambda p: int(p.stem.split("_r")[1])
            )
        ]
    )
    sq = ((G - G.mean(0)) ** 2).sum(1)
    r = int(sq.argmax())
    seed = 2230000 + r
    cfg = s.cfg_for("rosenbrock_extreme")
    target = s.build_target(cfg)
    state = torch.load(D / "snapshot_1000.pt")
    torch.manual_seed(seed)
    model = s.build_model(cfg, 2, reseed=False)
    model.load_state_dict(state)
    torch.manual_seed(seed)
    x0 = model.reference.sample(16384).detach()
    torch.save(
        dict(state=state, x0=x0, seed=seed, replicate_index=r),
        HERE / "replay_inputs.pt",
    )
    meta = dict(
        seed=seed,
        replicate_index_zero_based=r,
        draw_number=r + 1,
        gradient_norm=float(np.linalg.norm(G[r])),
        variance_fraction=float(sq[r] / sq.sum()),
        source=str(cell),
    )
    (HERE / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(meta, flush=True)
    np.savez(HERE / "original_gradients.npz", outlier=G[r], mean=G.mean(0))
    rows = []
    arms = [
        ("float32", "dopri5", 1e-07, 0),
        ("float64", "dopri5", 1e-07, 0),
        ("float64", "dopri5", 1e-09, 0),
        ("float64", "dopri5", 1e-11, 0),
        ("float32", "dopri5", 1e-08, 0),
        ("float64", "rk4", 0, 128),
        ("float64", "rk4", 0, 256),
        ("float64", "rk4", 0, 512),
    ]
    for dtype, solver, tol, steps in arms:
        label = f"{dtype}_{solver}_{(tol if tol else steps)}"
        t = time.time()
        print("START", label, flush=True)
        model = s.build_model(cfg, 2, reseed=False)
        model.load_state_dict(state)
        if dtype == "float64":
            model = model.double()
        xx = x0.to(dtype=next(model.parameters()).dtype)
        original = None
        if solver == "rk4":
            original = s.enable_checkpointing()
        try:
            est = s.make_estimator(
                "ode_transport",
                MC_samples=16384,
                path="linear",
                ode_solver=solver,
                rtol=tol or 1e-07,
                atol=tol or 1e-07,
                ode_steps=steps or 20,
            )
            est._ref_sample = lambda n, m: xx
            loss, info = est.loss(model, target.log_prob)
            loss.backward()
            g = s.flat_grad(model).detach().numpy().astype(float)
            np.savez(HERE / (label + ".npz"), gradient=g)
            row = dict(
                label=label,
                loss=float(loss.detach()),
                gradient_norm=float(np.linalg.norm(g)),
                difference_from_saved=float(np.linalg.norm(g - G[r])),
                nfe=info.get("nfe"),
                elapsed_s=time.time() - t,
            )
        except Exception as e:
            row = dict(label=label, error=repr(e), elapsed_s=time.time() - t)
        finally:
            if original is not None:
                s.ot._VelocityFunc.forward = original
        rows.append(row)
        (HERE / "results.json").write_text(json.dumps(rows, indent=2))
        print(row, flush=True)
    (HERE / "completed.json").write_text(json.dumps(dict(completed=True)))


if __name__ == "__main__":
    main()
