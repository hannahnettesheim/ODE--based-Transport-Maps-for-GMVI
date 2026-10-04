"""Sequential experiment execution and resumable training-table exports."""

from contextlib import contextmanager
from dataclasses import asdict
from io import StringIO
from pathlib import Path
import hashlib
import json

import pandas as pd
import torch

from gmvi.experiments.safe_io import atomic_json_dump, save_versioned


@contextmanager
def experiment_environment():
    """Restore process-wide settings that formerly lived in separate workers."""
    import gmvi.estimators.ode_transport as transport

    dtype = torch.get_default_dtype()
    threads = torch.get_num_threads()
    velocity_forward = transport._VelocityFunc.forward
    try:
        yield
    finally:
        torch.set_default_dtype(dtype)
        torch.set_num_threads(threads)
        transport._VelocityFunc.forward = velocity_forward


def run_training_jobs(jobs, one_job, output_dir, prefix, config_for=None):
    """Run jobs in order and export their existing four-table schema.

    Each job returns checkpoints, steps, final parameters and metadata.
    Completed jobs are cached individually; interrupted jobs are rerun.
    """
    output_dir = Path(output_dir)
    cache_dir = output_dir / f".{prefix}_jobs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tables = {name: [] for name in ("checkpoints", "steps", "final_params", "meta")}

    for index, job in enumerate(jobs, start=1):
        key = json.dumps(job, sort_keys=True)
        config = asdict(config_for(*job)) if config_for is not None else None
        config = json.loads(json.dumps(config))
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        cache_path = cache_dir / f"{digest}.json"
        if cache_path.exists():
            payload = json.loads(cache_path.read_text())
            if payload["job"] != json.loads(key):
                raise ValueError(f"Cached job does not match {job}")
            if payload.get("config") != config:
                raise ValueError(
                    f"Cached configuration changed for {job}: {cache_path}"
                )
        else:
            print(f"Job {index}/{len(jobs)}: {job}", flush=True)
            with experiment_environment():
                checkpoints, steps, parameters, metadata = one_job(*job)
            frames = (checkpoints, steps, parameters, pd.DataFrame([metadata]))
            payload = {
                "job": json.loads(key),
                "config": config,
                "tables": {
                    name: frame.to_json(orient="split", double_precision=15)
                    for name, frame in zip(tables, frames)
                },
            }
            atomic_json_dump(payload, str(cache_path))
        for name in tables:
            frame = pd.read_json(StringIO(payload["tables"][name]), orient="split")
            tables[name].append(frame)

    for name, frames in tables.items():
        if frames:
            save_versioned(
                pd.concat(frames, ignore_index=True),
                str(output_dir / f"{prefix}_{name}.csv"),
            )
