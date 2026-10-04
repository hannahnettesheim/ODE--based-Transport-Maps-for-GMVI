"""
Anisotropic Gaussian mixture: a dimension-scaling target that does NOT
confound dimension with conditioning.

Motivation
----------
Neal's funnel confounds two variables. As n grows it also demands worse-
conditioned components: the conditional scale exp(v/2) spans ~90x for
sigma_v=1.5, so any mixture covering both neck and mouth must contain
components with sigma_min -> 0. The grad-blowup diagnostic showed that the
ODE transport estimator's cost is governed by exactly that quantity, since
for the linear path

    lambda_min(A_{i,t}) = (1 - t) + t * sigma_min(A_i),

so the (1-t) term regularizes the flow everywhere except at t = 1, where it
vanishes and ||A_{i,1}^{-1}|| = 1/sigma_min(A_i) hits in full.

A "does it work in high dimension" experiment therefore cannot use the
funnel: a failure there is not attributable to n.

Design
------
This target holds the conditioning demand FIXED at a chosen condition number
kappa while n varies freely, so the two factors can be swept independently.

    p(x) = sum_m w_m N(x; mu_m, Sigma),    Sigma = Q diag(s)^2 Q^T

  * s_j = kappa^{(j-1)/(n-1) - 1/2}, log-spaced. Then cond(Sigma^{1/2})
    = kappa exactly for every n, and prod_j s_j = 1, so det Sigma = 1
    independent of BOTH n and kappa. Volume is therefore not a hidden
    third variable.
  * Q is a fixed random rotation. Sigma is deliberately NOT axis-aligned:
    an axis-aligned target would hand the diagonal chart a free win and
    make the chart ablation meaningless.
  * mu_m lie on a circle of radius R inside a fixed 2-plane (rotated by Q),
    so the modal geometry -- number of modes, angular spacing, separation in
    Mahalanobis units -- is identical at every n. Only the number of
    "nuisance" dimensions changes.
  * w_m decay geometrically with ratio rho, so the modes carry unequal mass
    and the component weights carry real information.

Exactly normalized by construction, so log Z = 0 and KL(q||p) = -ELBO with
no quadrature -- same property as BananaTarget and RosenbrockTarget.

Why k = n_modes in the sweep
----------------------------
Run the dimension sweep with k equal to the number of modes. Then exact
recovery is attainable and the achievable KL is 0 at every n, so raw KL is
directly comparable across dimensions with no per-dimension normalization,
and any residual is measurable optimizer failure rather than a family limit.
Mode-selection (k < n_modes) is a different question; keep it in the
hierarchical-mixture experiment.

Suggested protocol
------------------
    factor 1:  n in {2, 5, 10, 20, 50, 100}   at kappa = 1
    factor 2:  kappa in {1, 3, 10, 30, 100}   at n = 20

Prediction, to state before running: KL is flat in n at fixed kappa, and
degrades sharply in kappa at fixed n, with the onset tracking 1/sigma_min.
If that holds, dimension per se is not the obstacle and the funnel result is
a conditioning result. If KL degrades in n at kappa = 1, there IS an
intrinsic dimensional cost and that is a finding in its own right.

Instrument sigma_min(A_i) and max_i ||A_{i,1}^{-1}|| per step alongside KL.
Those are the mechanism variables; without them the sweep produces a curve
but not an explanation.
"""

import math
from typing import Optional

import torch
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


class AnisotropicMixtureTarget(Target):
    """
    Args:
        dim:        ambient dimension n (>= 2)
        n_modes:    number of mixture components in the TARGET
        kappa:      condition number of Sigma^{1/2}; kappa = 1 is isotropic
        rho:        geometric decay of the mode weights, in (0, 1]
        sep:        minimum pairwise separation between modes, in Mahalanobis
                    units (i.e. after whitening by Sigma^{-1/2})
        rotate:     if False, Sigma is axis-aligned. Only for debugging --
                    an axis-aligned target biases the chart ablation.
        seed:       seed for the fixed rotation Q
    """

    log_Z = 0.0
    """Exactly normalised, so KL(q||p) = -ELBO. Declared so that
    diagnostics.elbo_highres computes a KL instead of returning nan;
    a target that does NOT set this gets kl = nan rather than a wrong
    number (Lotka-Volterra and logreg are unnormalised)."""

    name = "anisotropic_mixture"

    def __init__(
        self,
        dim: int = 20,
        n_modes: int = 5,
        kappa: float = 1.0,
        rho: float = 0.6,
        sep: float = 6.0,
        rotate: bool = True,
        seed: int = 0,
        device: Optional[torch.device] = None,
    ):
        if dim < 2:
            raise ValueError("dim must be at least 2")
        if n_modes < 2:
            raise ValueError("n_modes must be at least 2")
        if kappa < 1.0:
            raise ValueError("kappa must be >= 1")
        if not 0.0 < rho <= 1.0:
            raise ValueError("rho must lie in (0, 1]")

        self._dim = dim
        self.n_modes = n_modes
        self.kappa = float(kappa)
        self.rho = rho
        self.sep = sep

        dtype = torch.get_default_dtype()
        g = torch.Generator().manual_seed(seed)

        # Spectrum: log-spaced, geometric mean 1, so det(Sigma) = 1 for all
        # (n, kappa). cond(Sigma^{1/2}) = kappa exactly.
        if dim == 1 or kappa == 1.0:
            expo = torch.zeros(dim, dtype=dtype)
        else:
            expo = torch.linspace(0.0, 1.0, dim, dtype=dtype) - 0.5
        self.s = torch.pow(torch.tensor(self.kappa, dtype=dtype), expo).to(device)

        # Fixed random rotation (QR of a Gaussian matrix, sign-fixed so Q is
        # a genuine rotation and reproducible).
        if rotate:
            A = torch.randn(dim, dim, generator=g, dtype=dtype)
            Q, R = torch.linalg.qr(A)
            Q = Q * torch.sign(torch.diagonal(R))[None, :]
        else:
            Q = torch.eye(dim, dtype=dtype)
        self.Q = Q.to(device)

        # Modes on a circle in the (rotated) 2-plane spanned by the two
        # directions whose scale is closest to 1 -- NOT the first two. At the
        # edge of the spectrum s ~ kappa^{-1/2}, so at kappa = 1e5 the modes
        # would sit in a Euclidean ball of radius ~0.016. That is
        # geometrically fine (Mahalanobis separation is still `sep` by
        # construction) but numerically delicate in float32. In the middle
        # s ~ 1, so the Euclidean radius stays O(sep) at every kappa.
        j = int(torch.argmin(torch.log(self.s).abs()))
        j = min(j, dim - 2)
        self.mode_axes = (j, j + 1)
        s_eff = float(max(self.s[j].item(), self.s[j + 1].item()))
        self.R = sep * s_eff / (2.0 * math.sin(math.pi / n_modes))

        angles = 2.0 * math.pi * torch.arange(n_modes, dtype=dtype) / n_modes
        raw = torch.zeros(n_modes, dim, dtype=dtype)
        raw[:, j] = self.R * torch.cos(angles)
        raw[:, j + 1] = self.R * torch.sin(angles)
        self.true_means = (raw @ self.Q.T).to(device)             # (M, n)

        w = torch.tensor([rho ** m for m in range(n_modes)], dtype=dtype)
        self.true_weights = (w / w.sum()).to(device)
        self.true_log_weights = torch.log(self.true_weights)

        # log det Sigma^{1/2} = sum_j log s_j = 0 by construction, but compute
        # it rather than hard-coding 0 so the density stays correct if the
        # spectrum is ever changed.
        self._log_det_half = torch.log(self.s).sum()
        self._log_norm = -0.5 * dim * math.log(2.0 * math.pi) - self._log_det_half

        self.n_evals = 0

    @property
    def dim(self) -> int:
        return self._dim

    # ── whitening ────────────────────────────────────────────────────────────

    def _whiten(self, delta: Tensor) -> Tensor:
        """Sigma^{-1/2} delta, for delta of shape (..., n)."""
        return (delta @ self.Q) / self.s

    def log_prob(self, z: Tensor) -> Tensor:
        """Exact normalized log density. z: (..., n) -> (...,)."""
        self.n_evals += int(z.reshape(-1, self._dim).shape[0])
        delta = z.unsqueeze(-2) - self.true_means                 # (..., M, n)
        w = self._whiten(delta)
        quad = -0.5 * (w * w).sum(-1)                             # (..., M)
        return torch.logsumexp(quad + self._log_norm + self.true_log_weights, dim=-1)

    def sample(self, n: int) -> Tensor:
        idx = torch.multinomial(self.true_weights, n, replacement=True)
        eps = torch.randn(n, self._dim, dtype=self.s.dtype)
        return self.true_means[idx] + (eps * self.s) @ self.Q.T

    # ── diagnostics ──────────────────────────────────────────────────────────

    def min_separation(self) -> float:
        """Achieved minimum pairwise mode separation, in Mahalanobis units."""
        d = torch.cdist(self._whiten(self.true_means), self._whiten(self.true_means))
        d = d + torch.eye(self.n_modes, dtype=d.dtype) * 1e9
        return float(d.min())

    def required_condition_number(self) -> float:
        """cond(A_i) an exactly-fitting component must have. This is the
        quantity the ODE estimator's gradient magnitude is governed by."""
        return self.kappa

    def weight_recovery(self, means: Tensor, weights: Tensor) -> dict:
        assign = torch.cdist(means, self.true_means).argmin(dim=-1)
        recovered = torch.zeros_like(self.true_weights)
        recovered.index_add_(0, assign, weights)
        return {
            "modes_covered": int(torch.unique(assign).numel()),
            "of_modes": self.n_modes,
            "total_variation": float(0.5 * (recovered - self.true_weights).abs().sum()),
            "missed_mass": float(self.true_weights[recovered == 0].sum()),
        }

    def extra_repr(self) -> str:
        return (f"dim={self._dim} modes={self.n_modes} kappa={self.kappa} "
                f"rho={self.rho} R={self.R:.2f} axes={self.mode_axes} "
                f"sigma_min={float(self.s.min()):.4g} "
                f"min_sep={self.min_separation():.2f}")
