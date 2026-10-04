import torch
from torch import Tensor
from typing import Callable, Dict, Literal, Tuple

from gmvi.models.generalized_mixture import GeneralizedMixture
from gmvi.estimators.baseEstimator import GradientEstimator, SampleMode


class ExactMarginalizationEstimator(GradientEstimator):
    """
    Exact marginalization over the discrete mixture component.

    Instead of sampling k ~ Categorical(pi), we enumerate all components:

        ELBO = sum_k pi_k E_{x ~ p0}[
            log p(T_k(x)) - log q(T_k(x))
        ]

    This gives an unbiased Monte Carlo estimate of the original mixture ELBO,
    with the discrete expectation computed exactly. The remaining Monte Carlo
    noise comes only from the continuous base samples x.

    Cost: O(K) component forward passes per estimate, and potentially O(K^2)
    if evaluating log q(z) itself loops over all mixture components.

    stl: sticking the landing. Valid here because the w_k-weighted component
    draws reconstruct an expectation under q, so the dropped score term has
    mean sum_k w_k E_{q_k}[grad log q] = E_q[grad log q] = 0. Only the entropy
    evaluation is frozen -- the w_k prefactor keeps its gradient, which is
    where DM's weight signal lives.
    """

    name = "exact_marginalization"

    def __init__(
        self,
        MC_samples: int = 64,
        sample_mode: SampleMode = "mc",
        stl: bool = False,
    ):
        super().__init__(
            MC_samples=MC_samples,
            sample_mode=sample_mode,
            stl=stl,
        )

    def loss_terms(
        self,
        model: GeneralizedMixture,
        log_target: Callable[[Tensor], Tensor],
    ) -> Tuple[Tensor, Dict]:
        N, K = self.MC_samples, model.K

        weights = torch.softmax(model.log_weights, dim=0)  # (K,)

        elbo_per_component = []

        for k in range(K):
            # Sample from the base distribution
            x_k = self._ref_sample(N, model)  # (N, D)

            # Push through component k
            z_k = model.components[k].forward(x_k)  # (N, D)

            # Evaluate the true mixture ELBO at z_k:
            # log p(z_k) - log q_theta(z_k)
            elbo_k = self._elbo_samples(model, log_target, z_k)  # (N,)

            elbo_per_component.append(elbo_k)

        # (N, K) -- the per-draw structure is KEPT rather than meaned away here,
        # so that the per-sample gradients are available. The component sum is
        # linear and each component draws its own independent batch, so
        #     grad[ sum_k w_k (1/N) sum_n e_kn ] = (1/N) sum_n grad[ sum_k w_k e_kn ]
        # and term n depends only on draw n. The i.i.d. unit is the K-TUPLE
        # (x_1[n], ..., x_K[n]), not a single sample; the variance estimated
        # from these terms is still the variance of this estimator at MC = N.
        elbo_per_component = torch.stack(elbo_per_component, dim=1)  # (N, K)

        # Exact sum over mixture components, per draw
        elbo = (weights.unsqueeze(0) * elbo_per_component).sum(dim=1)  # (N,)

        terms = -elbo

        return terms, {
            "elbo": elbo.mean().item(),
            # (K,) as before: the per-component ELBO, now meaned over draws here
            # rather than inside the loop.
            "component_elbos": elbo_per_component.mean(dim=0).detach().cpu(),
            "weights": weights.detach().cpu(),
        }
