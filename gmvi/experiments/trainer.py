"""Train a configured variational mixture and record diagnostics and parameter snapshots."""
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Any, get_args
import math
import platform
import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from gmvi.choices import (
    Chart,
    EstimatorName,
    GumbelMode,
    ODEPath,
    ODESolver,
    Optimizer,
    Reference,
    Scheduler,
    TargetName,
    validate,
)
from gmvi.models.generalized_mixture import GeneralizedMixture
from gmvi.models.reference_distributions import make_reference
from gmvi.estimators.gradient_estimators import make_estimator
from gmvi.experiments import diagnostics as dg


@dataclass
class RunConfig:
    """Everything one training run needs. Hover any field for its meaning;
    fields typed as a Literal will offer their allowed values as completions."""

    target: TargetName = "banana"
    """Which target posterior to fit."""

    target_kwargs: Dict[str, Any] = field(default_factory=dict)
    """Keyword arguments passed to the target constructor."""

    target_Z: float = 1.0
    """Known scale in ``tilde p_Z = target_Z * p`` for a normalised target.
    Checkpoint diagnostics then use ``kl = log(target_Z) - elbo``."""

    dim: Optional[int] = None
    """Ambient dimension. None means the target fixes its own (banana = 2,
    Rosenbrock = 2, Lotka-Volterra = 8)."""

    components: int = 5
    """Number of mixture components K."""

    chart: Chart = "matrixexponential"
    """Component scale parameterization; see gmvi.choices.Chart."""

    init_scale: float = 0.5
    """Scale of the initial A_i and the spread of the initial shifts a_i."""

    reference: Reference = "normal"
    """Reference distribution q_ref that the affine maps push forward."""

    estimator: EstimatorName = "ode_transport"
    """Which gradient estimator to train with."""

    estimator_kwargs: Dict[str, Any] = field(default_factory=dict)
    """Estimator-specific arguments, combined with mc_samples; see gmvi.choices."""

    mc_samples: int = 256
    """Monte Carlo samples M per gradient. Also the M the per-step variance is
    reported for: tr Cov(g_hat) scales as 1/M."""

    n_steps: int = 2000
    """Number of optimiser steps."""

    lr: float = 5e-3
    """Initial learning rate."""

    optimizer: Optimizer = "adam"
    momentum: float = 0.9
    """SGD only; ignored by Adam."""

    scheduler: Optional[Scheduler] = "cosine"
    """`cosine` anneals the learning rate to ~0 over n_steps. None or "none"
    keeps it flat."""

    scheduler_eta_min_ratio: float = 0.0
    """Final/initial learning-rate ratio for the cosine scheduler. The default
    of 0 preserves the original cosine-to-zero behaviour; it is ignored when
    the scheduler is flat."""

    record_every: int = 1
    """Write a row every k steps. Only cheap probes are per-step, so 1 is
    normally fine. Independent of var_every ON PURPOSE: these were one knob, and
    subsampling rows to save memory then silently subsampled the variance."""

    var_at_steps: Optional[List[int]] = None
    """Compute tr Cov(g_hat) at exactly these steps (in addition to var_every).
    Use it to put the variance probe on non-uniform checkpoints without paying
    for it every k steps."""

    var_every: int = 0
    """Compute tr Cov(g_hat) every k steps; 0 disables it. Costs
    ceil(mc_samples / var_chunk) extra backward passes, so roughly 4-5x a plain
    step at the defaults. 10 gives 200 measurements on a 2000-step run for about
    a third more runtime, which resolves any trajectory worth plotting.

    NOTE this is measured at whatever theta the run has reached, and each arm
    follows its own path -- so it is NOT a like-for-like comparison between
    estimators. Use fixed_theta.study() for that."""

    geometry_every: int = 0
    """SVD of every component every k steps; 0 disables. O(K D^3), so leave it
    off at D = 100 unless you need sigma_min resolved in time."""

    probe_elbo: bool = True
    """Free: the estimator already reports it."""

    probe_grad_norm: bool = True
    """Free: a norm over the flat gradient. Also records per-block norms
    (grad_norm_w / _a / _A) for weights, shifts and scales."""

    checkpoint_every: Optional[int] = None
    """Checkpoint every k steps. None falls back to n_checkpoints."""

    n_checkpoints: int = 5
    """Number of evenly spaced checkpoints, including step 0 and n_steps."""

    checkpoint_fracs: Optional[List[float]] = None
    """Explicit checkpoint positions as fractions of n_steps; overrides both of
    the above. [0, .025, .1, .5, 1] puts three of five inside the first
    quarter, which is where the early transient actually happens."""

    probe_elbo_highres: bool = True
    """ELBO from n_eval fresh samples, with a standard error. KL(q||p) too, but
    only if the target declares `log_Z`; otherwise kl is nan rather than a
    wrong number."""

    probe_wasserstein: bool = False
    """Sliced Wasserstein-2 against target.sample(). nan for targets that
    cannot be sampled (Lotka-Volterra, logreg)."""

    probe_geometry: bool = True
    """sigma_min / sigma_max / cond / logdet, per component and aggregated,
    plus alpha_max."""

    probe_params: bool = True
    """Keep a state_dict clone at each checkpoint. These are the snapshots
    fixed_theta.study() consumes."""

    n_eval: int = 20000
    """Samples for the high-resolution ELBO, drawn in chunks."""

    n_wasserstein: int = 4096
    """Samples per side for the sliced Wasserstein distance."""

    n_projections: int = 256
    """Random directions for the sliced Wasserstein distance."""

    var_chunk: int = 64
    """Rows per batched vector-Jacobian product when computing grad_var. Trades
    peak memory (chunk x P) against the number of backward passes."""

    seed: int = 0
    """Seeds model construction and the sampling stream."""

    tag: str = ""
    """Free-text label carried into every output row; use it to separate
    variants of a sweep that otherwise share their id fields."""

    verbose: bool = True
    log_every: int = 200
    """Steps between progress lines. Printing only; does not affect recording."""

    stop_on_nonfinite: bool = True
    """Stop at the first non-finite loss and set meta["diverged"]. Off, the run
    continues and the frame is full of nan."""

    def __post_init__(self):
        """Literal types are checked by the editor, not at runtime, so a value
        built from a string variable or a CLI argument still needs checking --
        and it should fail here rather than a thousand steps in."""
        for name in ("chart", "estimator", "target", "reference", "optimizer"):
            validate(name, getattr(self, name))
        validate("scheduler", self.scheduler, allow_none=True)
        if not 0.0 <= self.scheduler_eta_min_ratio <= 1.0:
            raise ValueError("scheduler_eta_min_ratio must be between 0 and 1")
        if not math.isfinite(self.target_Z) or self.target_Z <= 0.0:
            raise ValueError("target_Z must be finite and strictly positive")

        kw = self.estimator_kwargs
        if self.estimator == "ode_transport":
            p = validate("path", kw.get("path", "linear"))
            validate("ode_solver", kw.get("ode_solver", "rk4"))
            if p == "geometric" and self.chart != "matrixexponential":
                raise ValueError(
                    f"path='geometric' needs a differentiable log A, which only "
                    f"chart='matrixexponential' provides; got {self.chart!r}."
                )
        if self.estimator == "gumbel_softmax":
            validate("mode", kw.get("mode", "straight_through"))
        if self.estimator == "score_function":
            validate("baseline", kw.get("baseline", "none"))
            if kw.get("stl"):
                raise ValueError(
                    "stl is meaningless for score_function: its whole gradient "
                    "is the score term. Use baseline= instead."
                )
        if (
            kw.get("stl")
            and self.estimator == "gumbel_softmax"
            and kw.get("mode", "straight_through") == "soft"
        ):
            raise ValueError("stl needs z ~ q; use mode='straight_through'.")
        if self.mc_samples < 2 and self.var_every:
            raise ValueError(
                "var_every needs mc_samples >= 2 to have a spread " "to measure"
            )

    def id_fields(self) -> Dict[str, Any]:
        """The columns every output row carries, so frames can be concatenated
        across a sweep and still be identifiable."""
        return dict(
            target=self.target,
            target_Z=self.target_Z,
            dim=self.dim,
            chart=self.chart,
            components=self.components,
            estimator=self.estimator,
            mc=self.mc_samples,
            lr=self.lr,
            n_steps=self.n_steps,
            seed=self.seed,
            tag=self.tag,
        )


@dataclass
class RunResult:
    config: RunConfig
    steps: pd.DataFrame
    checkpoints: pd.DataFrame
    snapshots: Dict[int, Dict]
    meta: Dict

    def best_elbo(self) -> float:
        e = self.steps.get("elbo")
        return float(e.max()) if e is not None and len(e) else float("-inf")


def build_target(cfg: RunConfig):
    if cfg.target == "aniso":
        from gmvi.targets.anisotropic_mixture import AnisotropicMixtureTarget

        kw = dict(dim=cfg.dim or 20, n_modes=5, seed=0)
        kw.update(cfg.target_kwargs)
        target = AnisotropicMixtureTarget(**kw)
        return _scale_target(target, cfg.target_Z)
    from gmvi.targets.distributions import make_target

    kw = dict(cfg.target_kwargs)
    if cfg.dim is not None:
        kw.setdefault("dim", cfg.dim)
    try:
        target = make_target(cfg.target, **kw)
    except TypeError:  # target fixes its own dim
        kw.pop("dim", None)
        target = make_target(cfg.target, **kw)
    return _scale_target(target, cfg.target_Z)


class _ScaledTarget:
    """Multiply a normalised target density by a known positive constant."""

    def __init__(self, base, Z: float):
        self.base = base
        self.Z = float(Z)
        self.log_Z = math.log(self.Z)
        self.name = f"{getattr(base, 'name', 'target')}_Z{self.Z:g}"

    def log_prob(self, z):
        return self.base.log_prob(z) + self.log_Z

    def sample(self, n):
        return self.base.sample(n)

    @property
    def dim(self):
        return self.base.dim

    def __getattr__(self, name):
        return getattr(self.base, name)


def _scale_target(target, Z: float):
    if Z == 1.0:
        return target
    base_log_Z = dg.target_log_Z(target)
    if base_log_Z is None or not math.isclose(base_log_Z, 0.0, abs_tol=1e-12):
        raise ValueError("target_Z requires an exactly normalised base target")
    return _ScaledTarget(target, Z)


def build_model(cfg: RunConfig, dim: int, reseed: bool = True) -> GeneralizedMixture:
    if reseed:
        torch.manual_seed(cfg.seed)
    ref = make_reference(cfg.reference, dim=dim)
    return GeneralizedMixture(
        n_components=cfg.components,
        dim=dim,
        reference=ref,
        param_type=cfg.chart,
        init_scale=cfg.init_scale,
    )


def build_estimator(cfg: RunConfig):
    kw = dict(MC_samples=cfg.mc_samples)
    kw.update(cfg.estimator_kwargs)
    return make_estimator(cfg.estimator, **kw)


def checkpoint_steps(cfg: RunConfig) -> List[int]:
    if cfg.checkpoint_fracs is not None:
        return sorted({int(round(f * cfg.n_steps)) for f in cfg.checkpoint_fracs})
    if cfg.checkpoint_every:
        s = list(range(0, cfg.n_steps + 1, cfg.checkpoint_every))
        return sorted(set(s + [cfg.n_steps]))
    k = max(cfg.n_checkpoints - 1, 1)
    return sorted({int(round(i * cfg.n_steps / k)) for i in range(k + 1)})


def run(
    cfg: RunConfig,
    model: Optional[GeneralizedMixture] = None,
    target=None,
    checkpoint_callback: Optional[Callable[[int, Dict, Dict], None]] = None,
) -> RunResult:
    """Train once. Pass model/target to reuse them; otherwise they are built
    from the config, which is what a sweep should do."""
    t_start = time.time()
    if target is None:
        target = build_target(cfg)
    D = target.dim
    if model is None:
        model = build_model(cfg, D)
    est = build_estimator(cfg)
    log_Z = dg.target_log_Z(target)

    if cfg.optimizer == "adam":
        opt = optim.Adam(model.parameters(), lr=cfg.lr)
    elif cfg.optimizer == "sgd":
        opt = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    else:
        raise ValueError(f"unknown optimizer {cfg.optimizer!r}")
    sched = (
        optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.n_steps, eta_min=cfg.lr * cfg.scheduler_eta_min_ratio
        )
        if cfg.scheduler == "cosine"
        else None
    )

    ckpts = set(checkpoint_steps(cfg))
    ident = cfg.id_fields()
    step_rows: List[Dict] = []
    ckpt_rows: List[Dict] = []
    snapshots: Dict[int, Dict] = {}
    masks = dg.block_masks(model)
    stop_reason = "completed"
    diverged = False

    def take_checkpoint(step: int):
        row = dict(ident, step=step)
        if cfg.probe_elbo_highres:
            row.update(
                dg.elbo_highres(model, target.log_prob, n=cfg.n_eval, log_Z=log_Z)
            )
        if cfg.probe_geometry:
            g = dg.geometry(model)
            row.update({k: v for k, v in g.items() if not k.endswith("_i")})
            # per-component lists kept whole; a sweep can explode them later
            row.update({k: v for k, v in g.items() if k.endswith("_i")})
        row.update(dg.weight_stats(model))
        if cfg.probe_wasserstein:
            row["sliced_w2"] = dg.sliced_wasserstein(
                model,
                target,
                n=cfg.n_wasserstein,
                n_projections=cfg.n_projections,
                seed=cfg.seed,
            )
        row["wall_s"] = time.time() - t_start
        ckpt_rows.append(row)
        state = {}
        if cfg.probe_params:
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            snapshots[step] = state
        if checkpoint_callback is not None:
            checkpoint_callback(step, state, row)

    for step in range(cfg.n_steps + 1):
        if step in ckpts:
            take_checkpoint(step)
        if step == cfg.n_steps:
            break

        t0 = time.time()
        opt.zero_grad(set_to_none=False)
        want_var = (cfg.var_every and step % cfg.var_every == 0) or (
            cfg.var_at_steps is not None and step in cfg.var_at_steps
        )
        want_row = (step % cfg.record_every == 0) or (step == cfg.n_steps - 1)
        loss, info = est.loss(
            model, target.log_prob, return_variance=bool(want_var), chunk=cfg.var_chunk
        )

        if not torch.isfinite(loss):
            diverged = True
            stop_reason = f"nonfinite loss at step {step}"
            if cfg.stop_on_nonfinite:
                break

        loss.backward()

        row = dict(ident, step=step, wall_s=time.time() - t0)
        if cfg.probe_elbo:
            row["elbo"] = info.get("elbo", float("nan"))
            row["elbo_std"] = info.get("elbo_std", float("nan"))
        for k in ("temperature", "nfe", "baseline_value"):
            if k in info:
                row[k] = info[k]
        if want_var:
            row["grad_var"] = info.get("grad_var", float("nan"))
            row["grad_var_per_sample"] = info.get("grad_var_per_sample", float("nan"))

        if cfg.probe_grad_norm and want_row:
            g = dg.flat_grad(model)
            row["grad_norm"] = float(g.norm())
            # per-block norms: the late variance migration into the weights was
            # only ever seen at five snapshots. This resolves WHEN it happens.
            for b, m in masks.items():
                row[f"grad_norm_{b}"] = float(g[torch.as_tensor(m)].norm())
        if cfg.geometry_every and (step % cfg.geometry_every == 0):
            row.update(
                {
                    k: v
                    for k, v in dg.geometry(model, probe=0).items()
                    if not k.endswith("_i")
                }
            )

        if want_row:
            step_rows.append(row)

        opt.step()
        if sched is not None:
            sched.step()

        if cfg.verbose and step % cfg.log_every == 0:
            v = f" var={row['grad_var']:.3g}" if "grad_var" in row else ""
            g = f" |g|={row['grad_norm']:.3g}" if "grad_norm" in row else ""
            print(
                f"[{cfg.estimator:20s} seed {cfg.seed}] {step:5d}/{cfg.n_steps} "
                f"elbo={row.get('elbo', float('nan')):9.3f}{g}{v}",
                flush=True,
            )

    if diverged and cfg.stop_on_nonfinite and cfg.n_steps not in snapshots:
        pass  # the final checkpoint is deliberately absent; `diverged` says why

    meta = dict(
        ident,
        diverged=diverged,
        stop_reason=stop_reason,
        steps_completed=len(step_rows),
        total_s=time.time() - t_start,
        torch=torch.__version__,
        host=platform.node(),
        log_Z=log_Z,
    )
    if cfg.verbose:
        print(f"  done in {meta['total_s']:.1f}s -- {stop_reason}", flush=True)

    return RunResult(
        config=cfg,
        steps=pd.DataFrame(step_rows),
        checkpoints=pd.DataFrame(ckpt_rows),
        snapshots=snapshots,
        meta=meta,
    )


def run_many(configs: List[RunConfig]) -> Dict[str, pd.DataFrame]:
    """Run a list of configs and return concatenated frames plus the snapshots.
    This is all a sweep script should need."""
    steps, ckpts, metas, snaps = [], [], [], {}
    for cfg in configs:
        r = run(cfg)
        steps.append(r.steps)
        ckpts.append(r.checkpoints)
        metas.append(r.meta)
        snaps[(cfg.estimator, cfg.seed, cfg.tag)] = r.snapshots
    return {
        "steps": pd.concat(steps, ignore_index=True) if steps else pd.DataFrame(),
        "checkpoints": pd.concat(ckpts, ignore_index=True) if ckpts else pd.DataFrame(),
        "meta": pd.DataFrame(metas),
        "snapshots": snaps,
    }
