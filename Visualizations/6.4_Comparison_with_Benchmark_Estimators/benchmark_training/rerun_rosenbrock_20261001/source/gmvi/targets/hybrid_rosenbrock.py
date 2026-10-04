"""Normalized Hybrid Rosenbrock distribution of Pagani, Wiegand & Nadarajah.

Paper: https://arxiv.org/abs/1903.09556, Equation (12) and Algorithm 1.
One Gaussian root is shared by n2 conditionally independent chains, each
containing n1-1 additional variables. Total dimension: 1+n2*(n1-1).
Coordinates are flattened as [root, branch_1..., branch_2..., ...].
"""
import math
import torch
from .distributions import Target


class HybridRosenbrockTarget(Target):
    name = 'hybrid_rosenbrock'
    log_Z = 0.0  # log_prob includes the normalization constant

    def __init__(self, dim=None, *, n1=2, n2=None, mu=1.0, a=0.05, b=5.0):
        if not isinstance(n1, int) or isinstance(n1, bool) or n1 < 2:
            raise ValueError('n1 must be an integer >= 2')
        if dim is not None:
            if not isinstance(dim, int) or isinstance(dim, bool) or dim < 2:
                raise ValueError('dim must be an integer >= 2')
            if (dim-1) % (n1-1):
                raise ValueError('dim-1 must be divisible by n1-1')
            inferred = (dim-1)//(n1-1)
            if n2 is not None and n2 != inferred:
                raise ValueError('dim does not match 1+n2*(n1-1)')
            n2 = inferred
        if n2 is None:
            n2 = 1
        if not isinstance(n2, int) or isinstance(n2, bool) or n2 < 1:
            raise ValueError('n2 must be a positive integer')
        self.n1, self.n2 = n1, n2
        self.mu, self.a = float(mu), float(a)
        if not math.isfinite(self.mu) or not math.isfinite(self.a) or self.a <= 0:
            raise ValueError('mu must be finite and a finite and positive')
        coefficients = torch.as_tensor(b, dtype=torch.float64).detach().clone()
        try:
            self.b = torch.broadcast_to(coefficients, (n2, n1-1)).clone()
        except RuntimeError as exc:
            raise ValueError('b must broadcast to (n2, n1-1)') from exc
        if not torch.isfinite(self.b).all() or not (self.b > 0).all():
            raise ValueError('Every b coefficient must be finite and positive')

    @property
    def dim(self):
        return 1 + self.n2*(self.n1-1)

    @property
    def log_kernel_normalizer(self):
        """Log multiplier converting the unnormalized exponential to a density."""
        return .5*(math.log(self.a)+self.b.log().sum().item()-self.dim*math.log(math.pi))

    def log_prob(self, z):
        if z.ndim < 1 or z.shape[-1] != self.dim:
            raise ValueError(f'Expected trailing dimension {self.dim}')
        branches = z[..., 1:].reshape(*z.shape[:-1], self.n2, self.n1-1)
        root = z[..., 0, None, None].expand(*z.shape[:-1], self.n2, 1)
        parents = torch.cat([root, branches[..., :-1]], dim=-1)
        residual = branches-parents.square()
        b = self.b.to(device=z.device, dtype=z.dtype)
        return (self.log_kernel_normalizer-self.a*(z[..., 0]-self.mu).square()
                -(b*residual.square()).sum(dim=(-2, -1)))

    def sample(self, n, *, device=None, dtype=None, generator=None):
        if not isinstance(n, int) or n < 0:
            raise ValueError('n must be a nonnegative integer')
        dtype = dtype or torch.get_default_dtype()
        b = self.b.to(device=device, dtype=dtype)
        root = self.mu + torch.randn(n, device=device, dtype=dtype, generator=generator)/(2*self.a)**.5
        branches = []
        for j in range(self.n2):
            parent = root
            values = []
            for i in range(self.n1-1):
                child = parent.square()+torch.randn(n, device=device, dtype=dtype, generator=generator)/torch.sqrt(2*b[j,i])
                values.append(child)
                parent = child
            branches.append(torch.stack(values, dim=-1))
        return torch.cat([root[:,None]]+branches, dim=-1)
