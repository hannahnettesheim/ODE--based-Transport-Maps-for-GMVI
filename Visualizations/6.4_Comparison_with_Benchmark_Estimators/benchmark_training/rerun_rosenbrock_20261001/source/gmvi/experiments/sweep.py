"""
sweep.py -- run a grid of RunConfigs across independent shards.

A sweep is a base RunConfig plus a grid of overrides. expand() turns that into a
deterministic list of cells; each shard runs a disjoint subset in its own
process; merge() concatenates whatever has finished.

    from gmvi.experiments.sweep import Sweep
    from gmvi.experiments.trainer import RunConfig

    sweep = Sweep(
        name="funnel50_arms",
        base=RunConfig(target="funnel", dim=50, n_steps=2000, var_every=10),
        grid={"estimator": ["exact_marginalization", "score_function",
                            "gumbel_softmax", "ode_transport"]},
        seeds=[1, 2, 3, 4, 5],
    )                                            # -> 20 cells

    $ python -m gmvi.experiments.sweep run  my_sweeps.py:sweep --shard 0 --of 4
    $ python -m gmvi.experiments.sweep merge my_sweeps.py:sweep

ONE FILE PER CELL, not one per shard. The old sweeps wrote
`_shard0_of_5.json` and that ties the results to the shard layout: resuming
means redoing a whole shard, and changing --of invalidates everything. Here each
cell writes its own files under results/<name>/cells/<cell_id>/, so

  * a crash loses at most one cell,
  * a rerun skips whatever is already on disk,
  * --of can change between invocations, and
  * two shards can never write the same file, so no locking is needed.

DETERMINISTIC ASSIGNMENT. Cells are ordered by a sorted grid product and
assigned round-robin, cell i to shard i % n_shards. Round-robin rather than
contiguous blocks so that a partially finished sweep is spread evenly over the
grid instead of having one corner missing -- a half-done sweep is still
interpretable.

CELL IDS are a hash of the fields that actually vary, so they are stable when
the grid is reordered or extended, and a cell's identity does not change because
a neighbouring axis gained a value.

WHAT GETS WRITTEN, per cell:
    steps.csv        per-step rows
    checkpoints.csv  per-checkpoint rows
    meta.json        seed, timings, diverged, stop_reason, the resolved config
    snapshots.pt     {step: state_dict}, only if the config asks for params
    DONE             written last; a cell without it is treated as unfinished
"""
from dataclasses import dataclass, field, asdict, replace
from itertools import product
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback

import pandas as pd

from gmvi.experiments.safe_io import atomic_json_dump, save_versioned, timestamp
from gmvi.experiments.trainer import RunConfig, run


# ── the spec ─────────────────────────────────────────────────────────────────

@dataclass
class Sweep:
    """A base config, a grid of overrides, and a set of seeds."""

    name: str
    """Directory name under results/. Also the merged CSV's stem."""

    base: RunConfig = field(default_factory=RunConfig)
    """Everything the grid does not override."""

    grid: Dict[str, List[Any]] = field(default_factory=dict)
    """field name -> values. The Cartesian product is taken, so
    {"estimator": [4 values], "chart": [2]} is 8 cells per seed. Use `pairs`
    instead when the axes are not independent.

    A DOTTED KEY reaches inside a dict field, which is how the interesting
    estimator knobs are swept -- they are not RunConfig fields:

        {"estimator_kwargs.ode_steps": [4, 8, 16, 32]}
        {"estimator_kwargs.temperature": [0.1, 0.5, 1.0]}
        {"target_kwargs.kappa": [1, 100, 1000]}

    The base dict is copied and updated, never mutated, so cells do not leak
    into each other."""

    pairs: Optional[List[Dict[str, Any]]] = None
    """Explicit list of override dicts, used INSTEAD of the product when the
    combinations are not a full grid -- e.g. geometric only makes sense with
    matrixexponential, so a full product would generate illegal cells that
    RunConfig.__post_init__ rejects."""

    seeds: List[int] = field(default_factory=lambda: [1])

    results_root: str = "results"

    def __post_init__(self):
        allowed = set(RunConfig.__dataclass_fields__)

        def check(keys, what):
            bad = [k for k in keys if k.split(".", 1)[0] not in allowed]
            if bad:
                raise ValueError(
                    f"{what} keys are not RunConfig fields: {bad}. "
                    f"For estimator options use a dotted key, e.g. "
                    f"'estimator_kwargs.ode_steps'.")

        check(self.grid, "grid")
        for d in (self.pairs or []):
            check(d, "pairs")

    # -- expansion ------------------------------------------------------------

    def overrides(self) -> List[Dict[str, Any]]:
        """The per-cell override dicts, before seeds, in a stable order."""
        if self.pairs is not None:
            return [dict(d) for d in self.pairs]
        if not self.grid:
            return [{}]
        keys = sorted(self.grid)                      # sorted -> reorder-stable
        return [dict(zip(keys, vals))
                for vals in product(*(self.grid[k] for k in keys))]

    def cells(self) -> List["Cell"]:
        """Every (override, seed) pair, deterministically ordered."""
        out = []
        for ov in self.overrides():
            for seed in self.seeds:
                full = dict(ov, seed=seed)
                out.append(Cell(sweep_name=self.name, overrides=full,
                                config=apply_overrides(self.base, full)))
        return out

    def shard(self, index: int, n_shards: int) -> List["Cell"]:
        """Cells belonging to one shard. Round-robin, so a partly finished
        sweep is missing scattered cells rather than a whole corner."""
        if not 0 <= index < n_shards:
            raise ValueError(f"shard {index} out of range for {n_shards}")
        return [c for i, c in enumerate(self.cells()) if i % n_shards == index]

    def dir(self) -> str:
        return os.path.join(self.results_root, self.name)


def apply_overrides(base: RunConfig, overrides: Dict[str, Any]) -> RunConfig:
    """Build a config from `base`, honouring dotted keys.

    "estimator_kwargs.ode_steps" updates a COPY of base.estimator_kwargs, so
    sweeping one knob keeps the others from the base and no two cells share a
    mutable dict. RunConfig.__post_init__ then validates the result, so an
    illegal combination (geometric on a chart without a differentiable log A,
    say) fails at expansion time rather than hours into a shard.
    """
    flat: Dict[str, Any] = {}
    nested: Dict[str, Dict[str, Any]] = {}
    for k, v in overrides.items():
        if "." in k:
            outer, inner = k.split(".", 1)
            nested.setdefault(outer, {})[inner] = v
        else:
            flat[k] = v
    for outer, updates in nested.items():
        current = getattr(base, outer)
        if not isinstance(current, dict):
            raise TypeError(f"{outer!r} is not a dict field, so a dotted key "
                            f"cannot reach into it")
        flat[outer] = {**current, **flat.get(outer, {}), **updates}
    return replace(base, **flat)


@dataclass
class Cell:
    sweep_name: str
    overrides: Dict[str, Any]
    config: RunConfig

    @property
    def cell_id(self) -> str:
        """Stable, readable id: the varying fields, plus a hash for the rest."""
        parts = [f"{k}={_slug(v)}" for k, v in sorted(self.overrides.items())]
        stem = "_".join(parts) or "base"
        if len(stem) <= 80:
            return stem
        h = hashlib.sha256(stem.encode()).hexdigest()[:10]
        return stem[:70] + "_" + h

    def dir(self, root: str) -> str:
        return os.path.join(root, "cells", self.cell_id)


def _slug(v) -> str:
    s = str(v)
    for a, b in ((" ", ""), ("/", "-"), (":", "-"), (",", "-"),
                 ("{", ""), ("}", ""), ("'", ""), ('"', "")):
        s = s.replace(a, b)
    return s


# ── running ──────────────────────────────────────────────────────────────────

def is_done(cell: Cell, root: str) -> bool:
    return os.path.exists(os.path.join(cell.dir(root), "DONE"))


def run_cell(cell: Cell, root: str, force: bool = False) -> str:
    """Run one cell and write its outputs. Returns "done" | "skipped" | "failed".

    DONE is written last, so a cell interrupted mid-write is re-run rather than
    silently half-read at merge time.
    """
    d = cell.dir(root)
    if is_done(cell, root) and not force:
        return "skipped"
    os.makedirs(d, exist_ok=True)

    try:
        res = run(cell.config)
    except Exception:
        atomic_json_dump({"cell_id": cell.cell_id,
                          "overrides": _jsonable(cell.overrides),
                          "error": traceback.format_exc(),
                          "when": timestamp()},
                         os.path.join(d, "FAILED.json"), backup=False)
        return "failed"

    for name, df in (("steps", res.steps), ("checkpoints", res.checkpoints)):
        if len(df):
            df = df.copy()
            df.insert(0, "cell_id", cell.cell_id)
            df.to_csv(os.path.join(d, f"{name}.csv"), index=False)

    if res.snapshots:
        import torch
        torch.save({"snapshots": res.snapshots,
                    "config": asdict(cell.config)},
                   os.path.join(d, "snapshots.pt"))

    atomic_json_dump({"cell_id": cell.cell_id,
                      "overrides": _jsonable(cell.overrides),
                      "config": _jsonable(asdict(cell.config)),
                      "meta": _jsonable(res.meta)},
                     os.path.join(d, "meta.json"), backup=False)

    with open(os.path.join(d, "DONE"), "w") as fh:
        fh.write(timestamp() + "\n")
    if os.path.exists(os.path.join(d, "FAILED.json")):
        os.unlink(os.path.join(d, "FAILED.json"))
    return "done"


def run_shard(sweep: Sweep, index: int, n_shards: int,
              force: bool = False) -> Dict[str, int]:
    root = sweep.dir()
    os.makedirs(root, exist_ok=True)
    cells = sweep.shard(index, n_shards)
    tally = {"done": 0, "skipped": 0, "failed": 0}
    t0 = time.time()
    print(f"[shard {index}/{n_shards}] {len(cells)} cell(s) of "
          f"{len(sweep.cells())} in sweep {sweep.name!r}", flush=True)
    for i, cell in enumerate(cells, 1):
        print(f"\n[shard {index}] ({i}/{len(cells)}) {cell.cell_id}", flush=True)
        status = run_cell(cell, root, force=force)
        tally[status] += 1
        print(f"[shard {index}] -> {status}", flush=True)
    print(f"\n[shard {index}] {tally} in {time.time()-t0:.0f}s", flush=True)
    return tally


# ── merging ──────────────────────────────────────────────────────────────────

def merge(sweep: Sweep, write: bool = True) -> Dict[str, pd.DataFrame]:
    """Concatenate every FINISHED cell. Unfinished cells are reported, not
    silently dropped -- a merged frame that is quietly missing a third of the
    grid is how a sweep gets misread."""
    root = sweep.dir()
    all_cells = sweep.cells()
    done = [c for c in all_cells if is_done(c, root)]
    missing = [c for c in all_cells if not is_done(c, root)]
    failed = [c for c in all_cells
              if os.path.exists(os.path.join(c.dir(root), "FAILED.json"))]

    frames: Dict[str, List[pd.DataFrame]] = {"steps": [], "checkpoints": []}
    metas = []
    for c in done:
        for kind in frames:
            p = os.path.join(c.dir(root), f"{kind}.csv")
            if os.path.exists(p):
                frames[kind].append(pd.read_csv(p))
        p = os.path.join(c.dir(root), "meta.json")
        if os.path.exists(p):
            with open(p) as fh:
                j = json.load(fh)
            metas.append(dict(j["meta"], cell_id=j["cell_id"]))

    out = {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame())
           for k, v in frames.items()}
    out["meta"] = pd.DataFrame(metas)

    print(f"sweep {sweep.name!r}: {len(done)}/{len(all_cells)} cells complete"
          + (f", {len(failed)} FAILED" if failed else ""))
    for c in failed:
        print(f"  failed: {c.cell_id}")
    if missing:
        print(f"  {len(missing)} not finished, first few: "
              + ", ".join(c.cell_id for c in missing[:5]))
    if len(out["meta"]) and "diverged" in out["meta"]:
        n = int(out["meta"].diverged.sum())
        if n:
            print(f"  {n} completed cell(s) DIVERGED -- check meta.stop_reason "
                  f"before aggregating")

    if write:
        for kind, df in out.items():
            if len(df):
                save_versioned(df, os.path.join(root, f"{sweep.name}_{kind}.csv"))
    return out


def status(sweep: Sweep) -> pd.DataFrame:
    root = sweep.dir()
    rows = []
    for i, c in enumerate(sweep.cells()):
        d = c.dir(root)
        rows.append(dict(i=i, cell_id=c.cell_id,
                         done=os.path.exists(os.path.join(d, "DONE")),
                         failed=os.path.exists(os.path.join(d, "FAILED.json"))))
    return pd.DataFrame(rows)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return str(o)


# ── CLI ──────────────────────────────────────────────────────────────────────

def load_sweep(ref: str) -> Sweep:
    """Load a Sweep from "path/to/file.py:attribute_name"."""
    if ":" not in ref:
        raise ValueError('expected "file.py:name", got ' + repr(ref))
    path, attr = ref.rsplit(":", 1)
    spec = importlib.util.spec_from_file_location("_sweepdef", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_sweepdef"] = mod
    spec.loader.exec_module(mod)
    obj = getattr(mod, attr)
    return obj() if callable(obj) and not isinstance(obj, Sweep) else obj


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="run one shard")
    p.add_argument("spec", help='"file.py:sweep_name"')
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--of", type=int, default=1)
    p.add_argument("--force", action="store_true",
                   help="re-run cells that are already DONE")

    p = sub.add_parser("merge", help="concatenate finished cells")
    p.add_argument("spec")
    p.add_argument("--no-write", action="store_true")

    p = sub.add_parser("status", help="which cells are done")
    p.add_argument("spec")

    p = sub.add_parser("list", help="print the expanded cell list")
    p.add_argument("spec")
    p.add_argument("--of", type=int, default=1)

    a = ap.parse_args(argv)
    sweep = load_sweep(a.spec)

    if a.cmd == "run":
        t = run_shard(sweep, a.shard, a.of, force=a.force)
        return 1 if t["failed"] else 0
    if a.cmd == "merge":
        merge(sweep, write=not a.no_write)
        return 0
    if a.cmd == "status":
        df = status(sweep)
        print(df.to_string(index=False))
        print(f"\n{int(df.done.sum())}/{len(df)} done, "
              f"{int(df.failed.sum())} failed")
        return 0
    if a.cmd == "list":
        for i, c in enumerate(sweep.cells()):
            print(f"  shard {i % a.of}  {c.cell_id}")
        print(f"\n{len(sweep.cells())} cells over {a.of} shard(s)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
