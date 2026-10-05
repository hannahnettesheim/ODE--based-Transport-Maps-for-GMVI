import torch
from torch import Tensor
from typing import Callable, Dict, Literal, Tuple
import torch.nn.functional as F

from gmvi.estimators.baseEstimator import GradientEstimator, SampleMode
from gmvi.models.generalized_mixture import GeneralizedMixture
from gmvi.choices import GumbelMode


class GumbelSoftmaxEstimator(GradientEstimator):
    """
    Gumbel-Softmax (Concrete) relaxation.

    Args:
        n_samples:       MC samples per estimate
        temperature:     Gumbel-Softmax temperature τ (lower = harder)
        anneal_rate:     multiplicative decay per step
        min_temperature: floor for annealing
        mode:            'straight_through' (hard fwd/soft bwd, ELBO at
                         z_hard) or 'soft' (plain relaxation, ELBO at z_soft
                         via the true mixture density)
        stl:             sticking the landing. Valid in straight_through mode:
                         z_hard is an exact categorical draw by the Gumbel-max
                         trick, so the dropped score term has mean zero. It
                         removes a noise term, NOT the relaxation bias -- ST
                         stays biased either way. In 'soft' mode z_soft is not
                         a q-draw at all, so the identity does not apply and
                         stl is refused.
    """

    name = "gumbel_softmax"

    def __init__(
        self,
        MC_samples: int = 64,
        temperature: float = 1.0,
        anneal_rate: float = 0.9995,
        min_temperature: float = 0.3,
        sample_mode: SampleMode = "mc",
        mode: GumbelMode = "straight_through",
        stl: bool = False,
    ):
        super().__init__(
            MC_samples=MC_samples,
            sample_mode=sample_mode,
            stl=stl,
        )

        self.temperature = temperature
        self.anneal_rate = anneal_rate
        self.min_temperature = min_temperature

        if mode not in ("straight_through", "soft"):
            raise ValueError(f"Unknown Gumbel-Softmax mode: {mode}")
        if stl and mode == "soft":
            raise ValueError(
                "stl=True needs z ~ q_theta, which holds for the hard "
                "categorical draw in mode='straight_through' but not for the "
                "interpolated z_soft. Use mode='straight_through'."
            )

        self.mode = mode
        self._step = 0

    def _annealed_temperature(self) -> float:
        return max(
            self.min_temperature, self.temperature * (self.anneal_rate**self._step)
        )

    def loss_terms(
        self,
        model: GeneralizedMixture,
        log_target: Callable[[Tensor], Tensor],
    ) -> Tuple[Tensor, Dict]:
        tau = self._annealed_temperature()
        self._step += 1

        N, K = self.MC_samples, model.K

        U = torch.rand(N, K, device=model.log_weights.device).clamp(1e-6, 1 - 1e-6)
        gumbel = -torch.log(-torch.log(U))

        soft_w = F.softmax((model.log_weights + gumbel) / tau, dim=-1)  # (N, K)

        x_per_comp = torch.stack(
            [self._ref_sample(N, model) for _ in range(K)], dim=1
        )  # (N, K, D)

        z_per_comp = torch.stack(
            [model.components[k].forward(x_per_comp[:, k, :]) for k in range(K)],
            dim=1,
        )  # (N, K, D)

        z_soft = (soft_w.unsqueeze(-1) * z_per_comp).sum(dim=1)  # (N, D)

        if self.mode == "straight_through":
            hard_w = F.one_hot(soft_w.argmax(dim=-1), K).float()  # (N, K)
            z_hard = (hard_w.unsqueeze(-1) * z_per_comp).sum(dim=1)  # (N, D)
            z = z_soft + (z_hard - z_soft).detach()
            # ELBO evaluated at z_hard (value-wise); gradients flow through z_soft
            elbo = self._elbo_samples(model, log_target, z)
        else:
            # Relaxed samples need not follow q; evaluating the mixture density
            # between components can produce large, biased gradients.
            log_p = log_target(z_soft)
            log_q = self._log_q(model, z_soft)
            elbo = log_p - log_q

        terms = -elbo

        return terms, {
            "elbo": elbo.mean().item(),
            "elbo_std": elbo.std().item(),
            "temperature": tau,
        }
