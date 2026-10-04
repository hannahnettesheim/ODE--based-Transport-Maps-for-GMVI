import torch
from torch import Tensor
from typing import Callable, Dict, Tuple
import torch.distributions as dist

from gmvi.choices import Baseline, SampleMode, validate
from gmvi.estimators.baseEstimator import GradientEstimator
from gmvi.models.generalized_mixture import GeneralizedMixture


class ScoreFunctionEstimator(GradientEstimator):
    """
    REINFORCE / log-derivative trick, with an optional control variate.

        g = -E[ (elbo(z) - b) * grad log q(z) ],     z ~ q

    Every baseline here is unbiased, because E_q[grad log q] = 0 and each b is
    independent of the sample it multiplies. Subtracting b therefore changes
    the variance and nothing else. Note this is the SAME identity that makes a
    constant offset in log p invisible to the pathwise estimators (DM, ST, OTR)
    -- a baseline and a log-Z offset are the same operation with opposite sign,
    which is why no baseline in this file does anything for those three.

    baseline = "none"
        No control variate. Variance picks up c^2 E[||grad log q||^2] for any
        constant offset c in log p, so on an unnormalised target it grows
        quadratically in log Z. This is the default so that every sweep run
        before this option existed reproduces exactly.

    baseline = "ema"      (Williams 1992; the usual "moving average" baseline)
        b = exponential moving average of the batch-mean ELBO over PREVIOUS
        steps, Adam-style bias-corrected. One scalar, no extra evaluations,
        and it cancels a constant offset almost exactly.

    baseline = "optimal"  (Weaver & Tao 2001; Greensmith, Bartlett & Baxter 2004)
        b* = E[f ||grad log q||^2] / E[||grad log q||^2], the score-weighted
        mean of the ELBO -- the constant that actually minimises tr Cov, as
        opposed to the plain mean, which only removes the offset. Needs
        per-sample score norms, so it is estimated on a subsample of
        `optimal_probe` draws every `optimal_every` steps and carried as a
        running average; the current step reads the STALE value.

    baseline = "rloo"     (Mnih & Rezende 2016; Kool et al. 2019)
        b_i = mean of the other M-1 samples. Free, and the strongest of the
        three, but it COUPLES the batch: term i now depends on every draw, not
        just draw i. That breaks the i.i.d. assumption behind
        `GradientEstimator.loss(return_variance=True)`, so under this option
        grad_var is reported as nan rather than as a wrong number. Measure the
        variance of the RLOO estimator from independent replicates
        (fixed_theta.measure_at) instead.

    STATE. "ema" and "optimal" carry a running average ACROSS steps, so a
    fixed-theta measurement that builds a fresh estimator per replicate (as
    fixed_theta.measure_at does) will read b = 0 and make them behave exactly
    like "none". Warm the instance up with a few loss() calls at that theta and
    reuse it, or measure those two arms with test_sf_baselines.py.

    Which options keep the within-batch variance probe valid:

        none / ema / optimal    yes -- b depends only on past batches
        rloo                    NO  -- grad_var comes back nan

    Args:
        MC_samples:     MC samples per gradient estimate
        baseline:       control variate, see above
        baseline_decay: EMA decay for "ema" and "optimal"
        optimal_every:  refresh period, in steps, for "optimal"
        optimal_probe:  how many samples the "optimal" refresh differentiates
        optimal_chunk:  batched-VJP chunk size for that refresh
    """

    name = "score_function"

    def __init__(
        self,
        MC_samples: int = 64,
        sample_mode: SampleMode = "mc",
        baseline: Baseline = "none",
        baseline_decay: float = 0.99,
        optimal_every: int = 25,
        optimal_probe: int = 256,
        optimal_chunk: int = 64,
        stl: bool = False,
    ):
        if stl:
            raise ValueError(
                "stl=True is meaningless for the score-function estimator: "
                "its entire gradient IS the score term grad_theta log q, so "
                "freezing theta there gives identically zero. Use a baseline "
                "instead -- that is the score-function counterpart of STL."
            )
        super().__init__(
            MC_samples=MC_samples,
            sample_mode=sample_mode,
        )
        self.baseline = validate("baseline", baseline)
        if not 0.0 <= baseline_decay < 1.0:
            raise ValueError("baseline_decay must lie in [0, 1)")
        self.baseline_decay = baseline_decay
        self.optimal_every = max(1, int(optimal_every))
        self.optimal_probe = int(optimal_probe)
        self.optimal_chunk = int(optimal_chunk)

        self._ema = 0.0  # EMA accumulator, bias-corrected on read
        self._ema_t = 0
        self._num = 0.0  # EMA of E[f ||grad log q||^2]
        self._den = 0.0  # EMA of E[||grad log q||^2]
        self._step = 0

    @property
    def couples_samples(self) -> bool:
        """True when the per-sample terms are not independent, so the
        within-batch variance estimate does not apply."""
        return self.baseline == "rloo"

    def loss_terms(
        self,
        model: GeneralizedMixture,
        log_target: Callable[[Tensor], Tensor],
    ) -> Tuple[Tensor, Dict]:
        N = self.MC_samples

        # Sample from q — no gradient through sampling
        if self.sample_mode == "mc":
            z, _ = model.sample(N)
            z = z.detach()
        else:
            with torch.no_grad():
                k = dist.Categorical(probs=model.weights).sample((N,))
                x = self._ref_sample(N, model)
                z = torch.zeros(N, model.D, device=x.device)
                for i, comp in enumerate(model.components):
                    mask = k == i
                    if mask.any():
                        z[mask] = comp.forward(x[mask])

        # ELBO values (detached — used as signal, not differentiated)
        with torch.no_grad():
            elbo = self._elbo_samples(model, log_target, z)  # (N,)

        # log q(z; θ) — this carries gradients w.r.t. θ
        log_q = model.log_prob(z)  # (N,)

        b, b_info = self._compute_baseline(elbo, log_q, model)

        # REINFORCE surrogate, per sample: -(elbo_m - b) * log q(z_m).
        # grad of the mean = -E[(elbo - b) * grad log q(z)] = -grad ELBO,
        # since E[b * grad log q] = 0 for every b independent of its own draw.
        terms = -((elbo - b) * log_q)

        self._step += 1
        return terms, {
            "elbo": elbo.mean().item(),
            "elbo_std": elbo.std().item(),
            "log_q_mean": log_q.mean().item(),
            **b_info,
        }

    def loss(self, model, log_target, return_variance: bool = False, chunk: int = 64):
        """As the base class, except that RLOO's within-batch variance is
        reported as nan: the leave-one-out baseline couples the samples, so the
        spread of the per-sample gradients no longer estimates tr Cov."""
        loss, info = super().loss(
            model, log_target, return_variance=return_variance, chunk=chunk
        )
        if return_variance and self.couples_samples:
            info["grad_var"] = float("nan")
            info["grad_var_per_sample"] = float("nan")
            info["grad_var_invalid"] = "rloo couples the batch"
        return loss, info

    def _compute_baseline(self, elbo: Tensor, log_q: Tensor, model: GeneralizedMixture):
        if self.baseline == "none":
            return elbo.new_zeros(()), {"baseline_value": 0.0}

        if self.baseline == "rloo":
            N = elbo.numel()
            if N < 2:
                raise ValueError("rloo needs MC_samples >= 2")
            b = (elbo.sum() - elbo) / (N - 1)  # (N,), no graph
            return b, {"baseline_value": float(b.mean())}

        if self.baseline == "ema":
            b_val = self._ema_read()  # PAST batches only
            self._ema_update(float(elbo.mean()))
            return elbo.new_full((), b_val), {"baseline_value": b_val}

        if self.baseline == "optimal":
            b_val = self._num / self._den if self._den > 0.0 else 0.0
            if (self._step % self.optimal_every) == 0:
                self._refresh_optimal(elbo, log_q, model)
            return elbo.new_full((), b_val), {"baseline_value": b_val}

        raise ValueError(f"unknown baseline {self.baseline!r}")

    def _ema_read(self) -> float:
        if self._ema_t == 0:
            return 0.0
        return self._ema / (1.0 - self.baseline_decay**self._ema_t)

    def _ema_update(self, value: float) -> None:
        d = self.baseline_decay
        self._ema = d * self._ema + (1.0 - d) * value
        self._ema_t += 1

    def _refresh_optimal(
        self, elbo: Tensor, log_q: Tensor, model: GeneralizedMixture
    ) -> None:
        """Running estimate of b* = E[f ||g||^2] / E[||g||^2].

        The per-sample score norms come from the same batched vector-Jacobian
        trick as baseEstimator._grad_variance -- rows of the identity as
        grad_outputs, one backward per chunk rather than one per sample -- on a
        subsample, because doing it at full MC every step would cost more than
        the estimator itself. Accumulated in float64.
        """
        m = min(self.optimal_probe, elbo.numel())
        if m < 2:
            return
        params = [p for p in model.parameters() if p.requires_grad]
        lq = log_q[:m]
        eye = torch.eye(m, device=lq.device, dtype=lq.dtype)
        s2 = torch.zeros(m, dtype=torch.float64, device=lq.device)
        for i in range(0, m, self.optimal_chunk):
            rows = eye[i : i + self.optimal_chunk]
            gs = torch.autograd.grad(
                lq,
                params,
                grad_outputs=rows,
                is_grads_batched=True,
                retain_graph=True,
                allow_unused=True,
            )
            G = torch.cat(
                [
                    (
                        torch.zeros(rows.shape[0], p.numel(), device=lq.device)
                        if g is None
                        else g.reshape(rows.shape[0], -1)
                    ).double()
                    for g, p in zip(gs, params)
                ],
                dim=1,
            )  # (C, P)
            s2[i : i + rows.shape[0]] = (G * G).sum(dim=1)

        f = elbo[:m].detach().double()
        d = self.baseline_decay
        self._num = d * self._num + (1.0 - d) * float((f * s2).mean())
        self._den = d * self._den + (1.0 - d) * float(s2.mean())
