"""
fixed_theta.py -- bias and variance of a gradient estimator at a FROZEN theta.

The companion to trainer.py. The trainer produces snapshots; this measures every
estimator at them. Nothing is trained here and no parameter is ever updated.

The two canonical quantities, as the discrete-gradient-estimator literature
(REBAR, RELAX, Gumbel-Softmax) defines them:

    bias      = || E[g_hat] - g* ||
    variance  = tr Cov(g_hat)

estimated from R independent gradient replicates at the same theta.

g* IS THE REFERENCE AND IT MATTERS. DM is unbiased for grad L, so E[g_DM] is
exactly grad L, and a high-MC DM draw is what every arm's bias is measured
against. g* is accumulated in chunks of `mc` rather than as one MC = gstar_mc
draw: identical variance, bounded memory.

THE NOISE FLOOR. Both terms in ||mean - g*|| are noisy, so the SQUARED estimate
is inflated by

    tr Cov(g_hat_est)/R  +  tr Cov(g_DM)/n_chunk,     n_chunk = gstar_mc/mc

and NOT by tr Cov(g_DM)/gstar_mc -- `var` is already the variance of one MC = mc
draw, so the mc factor must not be applied twice. Getting that wrong makes every
floor sqrt(mc) too small.

THE CHECK THAT CATCHES IT. DM is unbiased by construction, so its measured
||mean - g*|| must come out at the floor, i.e. z = bias/(2*sigma) = 0.5. Any
other value means the floor or g* is wrong and nothing in the run is
trustworthy. It is asserted, not printed.

TWO ROUTES TO THE VARIANCE, and they should agree:

    var          from R independent replicates (this file)
    var_within   from the spread of the M per-sample gradients inside ONE batch
                 (estimator.loss(..., return_variance=True))

Recorded side by side. A disagreement means samples are coupled somewhere.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time

import numpy as np
import pandas as pd
import torch

from gmvi.models.generalized_mixture import GeneralizedMixture
from gmvi.estimators.gradient_estimators import make_estimator
from gmvi.experiments import diagnostics as dg
from gmvi.experiments.trainer import (RunConfig, build_model, build_target,
                                      checkpoint_steps, run)


@dataclass
class FixedThetaConfig:
    # arms to measure: (label, estimator name, kwargs)
    arms: List[Any] = field(default_factory=list)
    replicates: int = 200
    mc: int = 256
    gstar_mc: int = 16384
    probe_within_batch: bool = True     # the free per-batch variance, for cross-check
    var_chunk: int = 64
    probe_geometry: bool = True
    assert_gstar: bool = True           # fail loudly if DM's z is not ~0.5
    verbose: bool = True


def default_arms(mc: int, chart: str = "matrixexponential") -> List[Any]:
    arms = [
        ("DM", "exact_marginalization", dict(MC_samples=mc)),
        ("SF", "score_function", dict(MC_samples=mc)),
        ("ST", "gumbel_softmax", dict(MC_samples=mc, temperature=0.5,
                                      anneal_rate=1.0, min_temperature=0.5,
                                      mode="straight_through")),
        ("OTR linear", "ode_transport", dict(MC_samples=mc, ode_steps=8,
                                             path="linear", ode_solver="rk4")),
    ]
    if chart == "matrixexponential":
        arms.append(("OTR geometric", "ode_transport",
                     dict(MC_samples=mc, ode_steps=8, path="geometric",
                          ode_solver="rk4")))
    return arms


# ── the measurement ──────────────────────────────────────────────────────────

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
        m = make(); m.load_state_dict(state)
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
    probe = make_model(); probe.load_state_dict(state)
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
            m = make_model(); m.load_state_dict(state)
            est = make_estimator(name, **kwargs)
            if cfg.probe_within_batch and r == 0:
                m.zero_grad(set_to_none=False)
                loss, info = est.loss(m, log_target, return_variance=True,
                                      chunk=cfg.var_chunk)
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
                raise ValueError("arms[0] must be DM: its variance sets the "
                                 "g_star term of every noise floor")
            var_dm = var

        bias = float(np.linalg.norm(mean - gs))
        floor2 = var / R + var_dm / n_chunk
        floor = float(np.sqrt(floor2))
        rows.append(dict(
            ident, seed=seed, estimator=label, replicates=R, mc=cfg.mc,
            gstar_mc=n_chunk * cfg.mc,
            bias=bias,
            bias_stderr=floor,
            bias_debiased=float(np.sqrt(max(bias ** 2 - floor2, 0.0))),
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
            **geo))

        if cfg.verbose:
            r = rows[-1]
            print(f"    {label:16s} var={var:10.4g}  bias={bias:9.4g} "
                  f"(z={r['z']:.2f})  nsr={r['nsr']:.3f}", flush=True)

    z_dm = rows[0]["z"]
    if cfg.assert_gstar and not 0.3 < z_dm < 0.8:
        raise AssertionError(
            f"DM's z is {z_dm:.2f}, expected ~0.50. DM is unbiased by "
            f"construction, so the noise floor or g_star is wrong; nothing in "
            f"this measurement is trustworthy. (z near 1.0 means the floor is "
            f"4x too small in variance; z near 0 means g_star is correlated "
            f"with the replicate mean.)")
    return rows


# ── the usual whole study: train to get snapshots, then measure at each ──────

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
        ft_cfg = FixedThetaConfig(arms=default_arms(run_cfg.mc_samples,
                                                    run_cfg.chart))
    if not ft_cfg.arms:
        ft_cfg.arms = default_arms(run_cfg.mc_samples, run_cfg.chart)
    seeds = seeds or [run_cfg.seed]

    target = build_target(run_cfg)
    D = target.dim
    out: List[Dict] = []

    for seed in seeds:
        snap_cfg = RunConfig(**{**run_cfg.__dict__,
                                "estimator": "exact_marginalization",
                                "estimator_kwargs": {},
                                "seed": seed,
                                "var_every": 0,
                                "probe_params": True,
                                "probe_wasserstein": False,
                                "probe_elbo_highres": False})
        if run_cfg.verbose:
            print(f"\n=== seed {seed}: building snapshots (DM, "
                  f"{run_cfg.n_steps} steps)", flush=True)
        res = run(snap_cfg, target=target)

        for step, state in sorted(res.snapshots.items()):
            if run_cfg.verbose:
                print(f"  step {step}", flush=True)
            out += measure_at(
                state,
                make_model=lambda: build_model(run_cfg, D, reseed=False),
                log_target=target.log_prob,
                cfg=ft_cfg, seed=seed,
                ident=dict(run_cfg.id_fields(), step=step,
                           snapshot_estimator="DM"))

    df = pd.DataFrame(out)
    if run_cfg.verbose and len(df):
        print("\n=== tr Cov, median over seeds ===")
        print(df.pivot_table(index="estimator", columns="step", values="var",
                             aggfunc="median").to_string())
        print("\n=== bias, median over seeds ===")
        print(df.pivot_table(index="estimator", columns="step", values="bias",
                             aggfunc="median").to_string())
    return df
