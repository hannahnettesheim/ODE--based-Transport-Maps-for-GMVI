import copy
from abc import ABC, abstractmethod
from typing import Callable, Dict, Literal, Tuple

import torch
from torch import Tensor
from torch.quasirandom import SobolEngine
from typing import Callable, Dict, Literal, Tuple
from torch.quasirandom import SobolEngine

from gmvi.choices import SampleMode
from gmvi.models.generalized_mixture import GeneralizedMixture


class GradientEstimator(ABC):
    """Base class for all gradient estimators."""

    name: str = "base"

    def __init__(
        self,
        MC_samples: int = 64,
        sample_mode: SampleMode = "mc",
        stl: bool = False,
    ):
        self.MC_samples = MC_samples
        self.sample_mode = sample_mode
        self.stl = bool(stl)
        self._frozen = None       # cached parameter-frozen copy for STL

    @staticmethod
    def _qmc_normal_sample(
        n: int,
        d: int,
        device,
        scramble: bool,
    ) -> Tensor:
        """Low-discrepancy normal samples via Sobol + inverse normal CDF."""
        engine = SobolEngine(dimension=d, scramble=scramble)
        u = engine.draw(n).to(device).clamp(1e-6, 1 - 1e-6)
        return torch.erfinv(2.0 * u - 1.0) * (2.0 ** 0.5)

    def _ref_sample(
        self,
        n: int,
        model: GeneralizedMixture,
    ) -> Tensor:
        """Draw reference samples using this estimator's sampling mode."""

        if self.sample_mode == "mc":
            return model.reference.sample(n)

        return self._qmc_normal_sample(
            n,
            model.D,
            model.log_weights.device,
            scramble=(self.sample_mode == "rqmc"),
        )

    def _elbo_samples(
        self,
        model: GeneralizedMixture,
        log_target: Callable[[Tensor], Tensor],
        z: Tensor,
    ) -> Tensor:
        """Per-sample ELBO: log p(z) - log q(z), the entropy term routed
        through _log_q so that STL applies wherever it is used."""
        return log_target(z) - self._log_q(model, z)

    # ── sticking the landing ─────────────────────────────────────────────────

    def _log_q(self, model: GeneralizedMixture, z: Tensor) -> Tensor:
        """log q(z), with the parameters frozen when when the streight through estimator is chosen by the user.
        """
        if not self.stl:
            return model.log_prob(z)
        f = self._frozen
        if (f is None or f.K != model.K or f.D != model.D
                or f.param_type != model.param_type):
            f = copy.deepcopy(model).requires_grad_(False)
            self._frozen = f
        # values only; requires_grad stays False, so no gradient reaches theta
        f.load_state_dict(model.state_dict())
        return f.log_prob(z)

    @abstractmethod
    def loss_terms(
        self,
        model: GeneralizedMixture,
        log_target: Callable[[Tensor], Tensor],
    ) -> Tuple[Tensor, Dict]:
        """Per-sample surrogate terms l_1..l_M, shape (M,), with mean = loss,
        i.e. individual sample elbo.
        """
        raise NotImplementedError

    def loss(
        self,
        model: GeneralizedMixture,
        log_target: Callable[[Tensor], Tensor],
        return_variance: bool = False,
        chunk: int = 64,
    ) -> Tuple[Tensor, Dict]:
        """Scalar training loss (negative elbo), optionally with thr follwoing reported quantities:

        With return_variance=True,

            grad_var   tr Cov(g_hat), the variance of THIS step's gradient
            grad_var_per_sample   tr Cov(g_m), the per-draw variance
            grad_norm  ||g_hat||

        estimated from the M per-sample gradients of the current batch.
        """
        terms, info = self.loss_terms(model, log_target)
        loss = terms.mean()
        if return_variance:
            info.update(self._grad_variance(terms, model, chunk=chunk))
        return loss, info

    @staticmethod
    def _grad_variance(
        terms: Tensor,
        model: GeneralizedMixture,
        chunk: int = 64,
    ) -> Dict:
        """tr Cov of the gradient, from the per-sample gradients of `terms`.
        """
        M = terms.numel()
        if M < 2:
            raise ValueError("need at least 2 samples to estimate a variance")
        params = [p for p in model.parameters() if p.requires_grad]
        eye = torch.eye(M, device=terms.device, dtype=terms.dtype)

        s1 = None            # sum_m g_m           (P,)
        s2 = 0.0             # sum_m ||g_m||^2     scalar
        for i in range(0, M, chunk):
            rows = eye[i:i + chunk]                       # (C, M)
            gs = torch.autograd.grad(
                terms, params, grad_outputs=rows,
                is_grads_batched=True, retain_graph=True, allow_unused=True)
            G = torch.cat([
                (torch.zeros(rows.shape[0], p.numel(), device=terms.device)
                 if g is None else g.reshape(rows.shape[0], -1)).double()
                for g, p in zip(gs, params)], dim=1)      # (C, P)
            s1 = G.sum(0) if s1 is None else s1 + G.sum(0)
            s2 = s2 + (G * G).sum()

        mean = s1 / M                                     # g_hat
        # unbiased: sum ||g_m - g_bar||^2 = sum ||g_m||^2 - M ||g_bar||^2
        ss = torch.clamp(s2 - M * (mean @ mean), min=0.0)
        var_per_sample = ss / (M - 1)
        return {
            "grad_var": float(var_per_sample / M),
            "grad_var_per_sample": float(var_per_sample),
            "grad_norm": float(mean.norm()),
        }