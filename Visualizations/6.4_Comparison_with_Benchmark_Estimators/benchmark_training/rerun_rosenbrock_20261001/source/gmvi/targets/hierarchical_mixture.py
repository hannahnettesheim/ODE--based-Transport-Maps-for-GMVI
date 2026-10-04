"""
Hierarchical Gaussian mixture target for the component-count study.

Structure
---------
`n_groups` well-separated GROUPS, each containing `per_group` closely spaced
MODES, so there are M = n_groups * per_group modes at two spatial scales:

    group centres   on a circle of radius R_outer
    modes within    on a circle of radius R_inner  (R_inner << R_outer)

Group weights decay geometrically, w_group ~ rho^j, so the modes carry
*unequal* mass with a known profile. Within a group the mass is split evenly.

Why two scales
--------------
This turns the component-count sweep into a question with a right answer
rather than "more is better". With k components the variational mixture must:

    k <  n_groups   cover only the heaviest groups -- it must CHOOSE, and the
                    correct choice is by mass, so the weight gradient decides it
    k =  n_groups   one broad component per group: a coarse fit, each component
                    roughly sqrt(R_inner^2/2 + sigma^2) wide instead of sigma
    k =  M          exact recovery is possible; the test is whether the fitted
                    weights match the known profile
    k >  M          over-parameterized: do the surplus components collapse onto
                    existing modes, or split them?

So KL against k should show a plateau near k = n_groups and a second drop at
k = M. That is a falsifiable prediction, and it also answers "how should I
choose k?" directly.

Radii are derived from a requested separation `min_sep_sigma` rather than fixed,
so every (n_groups, per_group) configuration is equally well separated and the
sweep varies the component count and nothing else.

Exactly normalized (it is a probability density by construction), so log Z = 0
and KL(q||p) = -ELBO with no quadrature.

Suggested configurations
------------------------
    n_groups=5,  per_group=1   ->  M=5    (flat: 5 modes, no hierarchy)
    n_groups=5,  per_group=2   ->  M=10
    n_groups=5,  per_group=4   ->  M=20
    n_groups=10, per_group=10  ->  M=100  (weights span 99x)
    n_groups=20, per_group=5   ->  M=100  (weights span 1.6e4 -- brutal)

Sweep k over {2, 5, 10, 20, 50, 100} against a FIXED target (M=20 or M=100) so
the curves are comparable.
"""

import math
from typing import Optional

import torch
import torch.distributions as dist
from torch import Tensor

try:
    from .distributions import Target
except ImportError:  # module run directly rather than imported as gmvi.targets.*
    import os
    import sys
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    from gmvi.targets.distributions import Target


class HierarchicalMixtureTarget(Target):
    """
    Args:
        n_groups:       number of well-separated groups
        per_group:      modes per group (1 = flat, no hierarchy)
        sigma:          standard deviation of each mode (isotropic)
        rho:            geometric decay of the group weights, in (0, 1]
        min_sep_sigma:  requested separation between neighbouring modes, in
                        units of sigma. Radii are solved for from this.
        twist:          per-group rotation of the inner ring, so groups are not
                        exact copies of one another
    """

    log_Z = 0.0
    """Exactly normalised, so KL(q||p) = -ELBO. Declared so that
    diagnostics.elbo_highres computes a KL instead of returning nan;
    a target that does NOT set this gets kl = nan rather than a wrong
    number (Lotka-Volterra and logreg are unnormalised)."""

    name = "hierarchical_mixture"

    def __init__(
        self,
        n_groups: int = 5,
        per_group: int = 4,
        sigma: float = 0.25,
        rho: float = 0.6,
        min_sep_sigma: float = 6.0,
        twist: float = 0.37,
        device: Optional[torch.device] = None,
    ):
        if n_groups < 2:
            raise ValueError("n_groups must be at least 2")
        if not 0.0 < rho <= 1.0:
            raise ValueError("rho must lie in (0, 1]")

        self.n_groups = n_groups
        self.per_group = per_group
        self.sigma = sigma
        self.rho = rho
        self.n_modes = n_groups * per_group

        # Radii from the requested separation.
        s = min_sep_sigma
        self.R_inner = (0.0 if per_group == 1
                        else s * sigma / (2.0 * math.sin(math.pi / per_group)))
        self.R_outer = ((2.0 * (self.R_inner + 3.0 * sigma) + s * sigma)
                        / (2.0 * math.sin(math.pi / n_groups)))

        means, weights = [], []
        for j in range(n_groups):
            ang = 2.0 * math.pi * j / n_groups
            centre = self.R_outer * torch.tensor([math.cos(ang), math.sin(ang)])
            group_w = rho ** j
            for m in range(per_group):
                phi = 2.0 * math.pi * m / per_group + j * twist
                offset = self.R_inner * torch.tensor([math.cos(phi), math.sin(phi)])
                means.append(centre + offset)
                weights.append(group_w / per_group)

        self.true_means = torch.stack(means).to(device)          # (M, 2)
        w = torch.tensor(weights, device=device)
        self.true_weights = w / w.sum()                          # (M,)
        self.true_log_weights = torch.log(self.true_weights)

        # Group-level weights, for checking a coarse (k = n_groups) fit.
        gw = torch.tensor([rho ** j for j in range(n_groups)], device=device)
        self.true_group_weights = gw / gw.sum()

    @property
    def dim(self) -> int:
        return 2

    def log_prob(self, z: Tensor) -> Tensor:
        """Exact normalized log density. z: (..., 2) -> (...,)."""
        # (..., 1, 2) against (M, 2) -> (..., M)
        comp = dist.Normal(self.true_means, self.sigma)
        lp = comp.log_prob(z.unsqueeze(-2)).sum(-1)
        return torch.logsumexp(lp + self.true_log_weights, dim=-1)

    def sample(self, n: int) -> Tensor:
        idx = torch.multinomial(self.true_weights, n, replacement=True)
        return self.true_means[idx] + self.sigma * torch.randn(n, 2)

    # ── diagnostics for the component-count study ────────────────────────────

    def coarse_scale(self) -> float:
        """
        Standard deviation a single component must adopt to cover one whole
        group, versus the true mode width sigma. The ratio is how badly a
        k = n_groups fit is forced to over-disperse.
        """
        return math.sqrt(self.R_inner ** 2 / 2.0 + self.sigma ** 2)

    def assign_to_modes(self, means: Tensor) -> Tensor:
        """Nearest true mode for each fitted component mean. means: (k, 2)."""
        d = torch.cdist(means, self.true_means)                  # (k, M)
        return d.argmin(dim=-1)

    def weight_recovery(self, means: Tensor, weights: Tensor) -> dict:
        """
        Compare a fitted mixture against the truth: how many distinct modes are
        covered, and how far the recovered mass profile is from the true one.
        """
        assign = self.assign_to_modes(means)
        recovered = torch.zeros_like(self.true_weights)
        recovered.index_add_(0, assign, weights)
        covered = int(torch.unique(assign).numel())
        tv = 0.5 * (recovered - self.true_weights).abs().sum()
        # mass the fit places on modes it never reaches
        missed = self.true_weights[recovered == 0].sum()
        return {
            "modes_covered": covered,
            "of_modes": self.n_modes,
            "total_variation": tv.item(),
            "missed_mass": missed.item(),
        }

    def extra_repr(self) -> str:
        return (f"groups={self.n_groups} per_group={self.per_group} "
                f"M={self.n_modes} sigma={self.sigma} rho={self.rho} "
                f"R_inner={self.R_inner:.2f} R_outer={self.R_outer:.2f}")
