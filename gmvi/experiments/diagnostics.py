"""Shared model diagnostics for training and frozen-parameter experiments.

KL is reported only when target.log_Z is known; sliced Wasserstein distance
requires target samples. Geometry includes per-component scale singular values."""
from typing import Callable, Dict, List, Optional

import math
import numpy as np
import torch
from torch import Tensor

from gmvi.models.generalized_mixture import GeneralizedMixture


@torch.no_grad()
def geometry(model: GeneralizedMixture, probe: int = 2048) -> Dict:
    """Conditioning of every component's scale matrix, and how hard the
    component assignment is.

    sigma_min / sigma_max / cond / logdet are returned per component AND
    aggregated. alpha_max is the median largest responsibility over the model's
    own samples: 1.0 means each sample is explained by exactly one component,
    i.e. the assignment has saturated.
    """
    sv = [torch.linalg.svdvals(A) for A in model.get_matrices()]
    smin = np.array([float(s[-1]) for s in sv])
    smax = np.array([float(s[0]) for s in sv])
    logdet = np.array([float(torch.log(s).sum()) for s in sv])

    out = {
        "sigma_min_i": smin.tolist(),  # per component, length K
        "sigma_max_i": smax.tolist(),
        "cond_i": (smax / np.maximum(smin, 1e-300)).tolist(),
        "logdet_i": logdet.tolist(),
        "sigma_min": float(smin.min()),
        "sigma_max": float(smax.max()),
        "cond": float((smax / np.maximum(smin, 1e-300)).max()),
        "logdet": float(logdet.mean()),
    }
    if probe:
        z, _ = model.sample(probe)
        lw = torch.log_softmax(model.log_weights, 0)
        alpha = torch.softmax(lw + model.component_log_probs(z), dim=1)
        out["alpha_max"] = float(alpha.max(1).values.median())
    return out


@torch.no_grad()
def weight_stats(model: GeneralizedMixture) -> Dict:
    """Mixture-weight health. n_eff = exp(H) is the effective number of live
    components: it falls towards 1 when the mixture collapses onto one mode,
    which is a failure the ELBO alone can hide."""
    w = torch.softmax(model.log_weights, 0)
    H = float(-(w * torch.log(w.clamp_min(1e-300))).sum())
    return {
        "weights": w.tolist(),
        "weight_entropy": H,
        "n_eff_components": float(math.exp(H)),
        "weight_max": float(w.max()),
        "weight_min": float(w.min()),
    }


@torch.no_grad()
def elbo_highres(
    model: GeneralizedMixture,
    log_target: Callable[[Tensor], Tensor],
    n: int = 20000,
    chunk: int = 4096,
    log_Z: Optional[float] = None,
) -> Dict:
    """ELBO with a standard error, from n fresh samples, in chunks so that
    n = 20000 at D = 100 does not blow up memory.

    log_Z: pass the target's log normaliser if it is known. KL(q||p) is then
    log_Z - ELBO. Left as None, kl is nan -- an unnormalised target has no KL
    that can be computed this way, and reporting -ELBO as if it were one is the
    single easiest mistake to make here.
    """
    tot = tot2 = 0.0
    n_done = 0
    for i in range(0, n, chunk):
        m = min(chunk, n - i)
        z, _ = model.sample(m)
        e = (log_target(z) - model.log_prob(z)).double()
        tot += float(e.sum())
        tot2 += float((e * e).sum())
        n_done += m
    mean = tot / n_done
    var = max(tot2 / n_done - mean * mean, 0.0) * n_done / max(n_done - 1, 1)
    out = {
        "elbo": mean,
        "elbo_std": math.sqrt(var),
        "elbo_se": math.sqrt(var / n_done),
        "elbo_n": n_done,
        "kl": float("nan") if log_Z is None else log_Z - mean,
    }
    return out


def target_log_Z(target) -> Optional[float]:
    """Read a target's declared log normaliser, or None. Opt-in on purpose:
    a target that has not declared one gets kl = nan rather than a wrong KL."""
    z = getattr(target, "log_Z", None)
    return None if z is None else float(z)


@torch.no_grad()
def sliced_wasserstein(
    model: GeneralizedMixture,
    target,
    n: int = 4096,
    n_projections: int = 256,
    p: int = 2,
    seed: Optional[int] = None,
) -> float:
    """Sliced W_p between q's samples and the target's, or nan if the target
    cannot be sampled (Lotka-Volterra, logistic regression).

    Unequal sample counts are not supported -- both sets are drawn at size n so
    the sorted-quantile shortcut is exact per projection.
    """
    sampler = getattr(target, "sample", None)
    if sampler is None:
        return float("nan")
    try:
        y = sampler(n)
    except NotImplementedError:
        return float("nan")
    if y is None:
        return float("nan")
    y = torch.as_tensor(y, dtype=torch.float32)

    x, _ = model.sample(n)
    x = x.detach()
    if y.shape[0] != n:  # target may cap its own sample size
        n = min(n, y.shape[0])
        x, y = x[:n], y[:n]

    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(seed)
    D = x.shape[1]
    theta = torch.randn(D, n_projections, generator=g)
    theta = theta / theta.norm(dim=0, keepdim=True)

    xp = (x @ theta).sort(dim=0).values  # (n, P)
    yp = (y @ theta).sort(dim=0).values
    d = (xp - yp).abs() ** p
    return float((d.mean(dim=0) ** (1.0 / p)).mean())


def flat_grad(model: GeneralizedMixture) -> Tensor:
    return torch.cat(
        [
            (torch.zeros(p.numel()) if p.grad is None else p.grad.detach().flatten())
            for p in model.parameters()
        ]
    )


def block_masks(model: GeneralizedMixture) -> Dict[str, np.ndarray]:
    """Boolean masks over the flat parameter vector, for the three blocks the
    variance decomposition uses: mixture weights, component shifts, component
    scale parameters (whatever the chart calls them)."""
    w, a, A = [], [], []
    for name, p in model.named_parameters():
        n = p.numel()
        is_w = name == "log_weights"
        is_a = name.endswith(".a")
        w.append(np.full(n, is_w))
        a.append(np.full(n, is_a))
        A.append(np.full(n, not (is_w or is_a)))
    return {"w": np.concatenate(w), "a": np.concatenate(a), "A": np.concatenate(A)}
