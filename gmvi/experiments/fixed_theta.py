"""Measure estimator bias and variance at frozen parameters.

Bias is a discrepancy to a finite Monte Carlo DM reference; the reported sampling
error includes both reference uncertainty and replicate-mean uncertainty."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time

import numpy as np
import pandas as pd
import torch

from gmvi.models.generalized_mixture import GeneralizedMixture
from gmvi.estimators.gradient_estimators import make_estimator
from gmvi.experiments import diagnostics as dg
from gmvi.experiments.trainer import (
    RunConfig,
    build_model,
    build_target,
    checkpoint_steps,
    run,
)


@dataclass
class FixedThetaConfig:
    # arms to measure: (label, estimator name, kwargs)
    arms: List[Any] = field(default_factory=list)
    replicates: int = 200
    mc: int = 256
    gstar_mc: int = 16384
    probe_within_batch: bool = True  # the free per-batch variance, for cross-check
    var_chunk: int = 64
    probe_geometry: bool = True
    assert_gstar: bool = True  # fail loudly if DM's z is not ~0.5
    verbose: bool = True


def default_arms(mc: int, chart: str = "matrixexponential") -> List[Any]:
    arms = [
        ("DM", "exact_marginalization", dict(MC_samples=mc)),
        ("SF", "score_function", dict(MC_samples=mc)),
        (
            "ST",
            "gumbel_softmax",
            dict(
                MC_samples=mc,
                temperature=0.5,
                anneal_rate=1.0,
                min_temperature=0.5,
                mode="straight_through",
            ),
        ),
        (
            "OTR linear",
            "ode_transport",
            dict(MC_samples=mc, ode_steps=8, path="linear", ode_solver="rk4"),
        ),
    ]
    if chart == "matrixexponential":
        arms.append(
            (
                "OTR geometric",
                "ode_transport",
                dict(MC_samples=mc, ode_steps=8, path="geometric", ode_solver="rk4"),
            )
        )
    return arms


def gradient_draw(
    model,
    log_target,
    estimator,
    mc,
    seed,
    estimator_kwargs=None,
    *,
    set_to_none=False,
    dtype=None,
):
    """One seeded gradient at fixed parameters, plus estimator diagnostics."""
    torch.manual_seed(seed)
    model.zero_grad(set_to_none=set_to_none)
    estimator = make_estimator(estimator, MC_samples=mc, **(estimator_kwargs or {}))
    loss, info = estimator.loss(model, log_target)
    loss.backward()
    gradient = dg.flat_grad(model).detach().cpu().numpy().copy()
    if dtype is not None:
        gradient = gradient.astype(dtype, copy=False)
    if not np.isfinite(gradient).all():
        raise FloatingPointError("Nonfinite gradient")
    return gradient, info


def _one_grad(est, model, log_target) -> np.ndarray:
    model.zero_grad(set_to_none=False)
    loss, _ = est.loss(model, log_target)
    loss.backward()
    return dg.flat_grad(model).numpy()


def _g_star(cfg: FixedThetaConfig, make, state, log_target, seed):
    """DM at an effective MC of gstar_mc, accumulated in chunks of cfg.mc."""
    n_chunk = max(1, int(round(cfg.gstar_mc / cfg.mc)))
    torch.manual_seed(90_000 + seed)
    acc = None
    for _ in range(n_chunk):
        m = make()
        m.load_state_dict(state)
        est = make_estimator("exact_marginalization", MC_samples=cfg.mc)
        g = _one_grad(est, m, log_target)
        acc = g if acc is None else acc + g
    return acc / n_chunk, n_chunk


def measure_at(
    state: Dict,
    make_model,
    log_target,
    cfg: FixedThetaConfig,
    seed: int = 0,
    ident: Optional[Dict] = None,
) -> List[Dict]:
    """Bias and variance of every arm at one frozen theta.

    make_model: zero-arg callable returning a fresh model WITHOUT reseeding --
    reseeding would make all R replicates identical.
    """
    ident = dict(ident or {})
    R = cfg.replicates
    probe = make_model()
    probe.load_state_dict(state)
    geo = dg.geometry(probe) if cfg.probe_geometry else {}
    masks = dg.block_masks(probe)

    gs, n_chunk = _g_star(cfg, make_model, state, log_target, seed)
    gnorm = float(np.linalg.norm(gs))

    rows: List[Dict] = []
    var_dm = None
    for label, name, kwargs in cfg.arms:
        t0 = time.time()
        torch.manual_seed(70_000 + seed)
        G = []
        within = float("nan")
        for r in range(R):
            m = make_model()
            m.load_state_dict(state)
            est = make_estimator(name, **kwargs)
            if cfg.probe_within_batch and r == 0:
                m.zero_grad(set_to_none=False)
                loss, info = est.loss(
                    m, log_target, return_variance=True, chunk=cfg.var_chunk
                )
                loss.backward()
                within = info.get("grad_var", float("nan"))
                G.append(dg.flat_grad(m).numpy())
            else:
                G.append(_one_grad(est, m, log_target))
        G = np.stack(G)

        mean = G.mean(axis=0)
        per = G.var(axis=0, ddof=1)
        var = float(per.sum())
        if var_dm is None:
            if label != cfg.arms[0][0] or name != "exact_marginalization":
                raise ValueError(
                    "arms[0] must be DM: its variance sets the "
                    "g_star term of every noise floor"
                )
            var_dm = var

        bias = float(np.linalg.norm(mean - gs))
        floor2 = var / R + var_dm / n_chunk
        floor = float(np.sqrt(floor2))
        rows.append(
            dict(
                ident,
                seed=seed,
                estimator=label,
                replicates=R,
                mc=cfg.mc,
                gstar_mc=n_chunk * cfg.mc,
                bias=bias,
                bias_stderr=floor,
                bias_debiased=float(np.sqrt(max(bias**2 - floor2, 0.0))),
                z=bias / (2 * floor) if floor > 0 else float("nan"),
                var=var,
                var_w=float(per[masks["w"]].sum()),
                var_a=float(per[masks["a"]].sum()),
                var_A=float(per[masks["A"]].sum()),
                var_within_batch=within,
                mean_norm=float(np.linalg.norm(mean)),
                gnorm=gnorm,
                nsr=float(np.sqrt(var / cfg.mc) / gnorm) if gnorm > 0 else float("inf"),
                s_per_arm=time.time() - t0,
                **geo,
            )
        )

        if cfg.verbose:
            r = rows[-1]
            print(
                f"    {label:16s} var={var:10.4g}  bias={bias:9.4g} "
                f"(z={r['z']:.2f})  nsr={r['nsr']:.3f}",
                flush=True,
            )

    z_dm = rows[0]["z"]
    if cfg.assert_gstar and not 0.3 < z_dm < 0.8:
        raise AssertionError(
            f"DM's z is {z_dm:.2f}, expected ~0.50. DM is unbiased by "
            f"construction, so the noise floor or g_star is wrong; nothing in "
            f"this measurement is trustworthy. (z near 1.0 means the floor is "
            f"4x too small in variance; z near 0 means g_star is correlated "
            f"with the replicate mean.)"
        )
    return rows


def study(
    run_cfg: RunConfig,
    ft_cfg: Optional[FixedThetaConfig] = None,
    seeds: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Build snapshots by TRAINING WITH DM, then measure every arm at each.

    Snapshots are built with DM deliberately: the parameter path must not depend
    on the estimator being measured, or each arm is evaluated somewhere
    different and the comparison is meaningless.
    """
    if ft_cfg is None:
        ft_cfg = FixedThetaConfig(arms=default_arms(run_cfg.mc_samples, run_cfg.chart))
    if not ft_cfg.arms:
        ft_cfg.arms = default_arms(run_cfg.mc_samples, run_cfg.chart)
    seeds = seeds or [run_cfg.seed]

    target = build_target(run_cfg)
    D = target.dim
    out: List[Dict] = []

    for seed in seeds:
        snap_cfg = RunConfig(
            **{
                **run_cfg.__dict__,
                "estimator": "exact_marginalization",
                "estimator_kwargs": {},
                "seed": seed,
                "var_every": 0,
                "probe_params": True,
                "probe_wasserstein": False,
                "probe_elbo_highres": False,
            }
        )
        if run_cfg.verbose:
            print(
                f"\n=== seed {seed}: building snapshots (DM, "
                f"{run_cfg.n_steps} steps)",
                flush=True,
            )
        res = run(snap_cfg, target=target)

        for step, state in sorted(res.snapshots.items()):
            if run_cfg.verbose:
                print(f"  step {step}", flush=True)
            out += measure_at(
                state,
                make_model=lambda: build_model(run_cfg, D, reseed=False),
                log_target=target.log_prob,
                cfg=ft_cfg,
                seed=seed,
                ident=dict(run_cfg.id_fields(), step=step, snapshot_estimator="DM"),
            )

    df = pd.DataFrame(out)
    if run_cfg.verbose and len(df):
        print("\n=== tr Cov, median over seeds ===")
        print(
            df.pivot_table(
                index="estimator", columns="step", values="var", aggfunc="median"
            ).to_string()
        )
        print("\n=== bias, median over seeds ===")
        print(
            df.pivot_table(
                index="estimator", columns="step", values="bias", aggfunc="median"
            ).to_string()
        )
    return df
