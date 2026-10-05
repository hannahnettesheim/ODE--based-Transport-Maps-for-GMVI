"""Diagonal Gaussian mixture with trainable logits, means, and log scales."""

import torch
import torch.nn as nn
import torch.distributions as dist
from torch import Tensor
from typing import Tuple


class GaussianMixture(nn.Module):
    """Variational Gaussian mixture with diagonal component covariances."""

    def __init__(self, n_components: int, latent_dim: int, init_scale: float = 1.0):
        super().__init__()
        self.K = n_components
        self.D = latent_dim

        self.log_weights = nn.Parameter(torch.zeros(n_components))
        self.means = nn.Parameter(torch.randn(n_components, latent_dim) * init_scale)
        self.log_scales = nn.Parameter(
            torch.full(
                (n_components, latent_dim),
                fill_value=torch.log(torch.tensor(init_scale)),
            )
        )

    @property
    def weights(self) -> Tensor:
        return torch.softmax(self.log_weights, dim=0)

    @property
    def scales(self) -> Tensor:
        return torch.exp(self.log_scales)

    def log_prob(self, z: Tensor) -> Tensor:
        """Log density for z of shape (N, D); returns (N,)."""
        # z: (N, D) -> (N, 1, D) for broadcasting with (K, D)
        z_exp = z.unsqueeze(1)  # (N, 1, D)
        means = self.means.unsqueeze(0)  # (1, K, D)
        scales = self.scales.unsqueeze(0)  # (1, K, D)

        component_log_prob = dist.Normal(means, scales).log_prob(z_exp).sum(-1)

        # log pi_k + log N(z | mu_k, sigma_k)
        log_w = torch.log_softmax(self.log_weights, dim=0)  # (K,)
        log_mix = log_w.unsqueeze(0) + component_log_prob  # (N, K)

        return torch.logsumexp(log_mix, dim=1)  # (N,)

    def sample(self, n: int) -> Tuple[Tensor, Tensor]:
        """Return samples (N, D) and component indices (N,)."""
        k = dist.Categorical(probs=self.weights).sample((n,))  # (N,)

        selected_means = self.means[k]  # (N, D)
        selected_scales = self.scales[k]  # (N, D)
        eps = torch.randn_like(selected_means)
        z = selected_means + selected_scales * eps
        return z, k

    def rsample(self, n: int) -> Tuple[Tensor, Tensor]:
        """Reparameterized samples (N, D) and hard component indices (N,).
        Component selection is not differentiable."""
        k = dist.Categorical(probs=self.weights.detach()).sample((n,))
        selected_means = self.means[k]
        selected_scales = self.scales[k]
        eps = torch.randn_like(selected_means)
        z = selected_means + selected_scales * eps
        return z, k

    def entropy_lb(self) -> Tensor:
        """Monte Carlo entropy estimate using 1,000 mixture samples."""
        with torch.no_grad():
            z, _ = self.sample(1000)
        return -self.log_prob(z).mean()

    def extra_repr(self) -> str:
        return f"K={self.K}, D={self.D}"
