import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable, Dict, Literal, Tuple, Optional
from torchdiffeq import odeint

from gmvi.models.generalized_mixture import GeneralizedMixture
from gmvi.estimators.baseEstimator import GradientEstimator, SampleMode
from gmvi.choices import ODEPath, ODESolver


# this is the simpler case: we are looking for diagonal ODES
def _velocity_diagonal(
    x: Tensor,  # (N, D)
    t: float,
    log_weights: Tensor,  # (K,) unnormalized
    means: Tensor,  # (K, D)
    diagonal_entries: Tensor,  # (K, D)  diagonal entries of A_j = diag(scales_j)
    ref_log_prob: Callable,
) -> Tensor:
    N, D = x.shape

    # A_{j,t} diagonal: t * sigma_j + (1 - t)
    Ajt_diag = t * diagonal_entries + (1.0 - t)  # (K, D)

    # Pre-image: z_j = A_{j,t}^{-1} (x - t a_j)
    x_shift = x[:, None, :] - t * means[None, :, :]  # (N, K, D)
    z = x_shift / Ajt_diag[None, :, :]  # (N, K, D)

    # log rho_{j,t}(x) = log rho_ref(z_j) - log|det A_{j,t}|
    N, K, D = z.shape
    log_ref = ref_log_prob(z.reshape(N * K, D)).reshape(N, K)  # (N, K)
    logdet_Ajt = torch.log(Ajt_diag).sum(dim=-1)  # (K,)
    log_rho_jt = log_ref - logdet_Ajt[None, :]  # (N, K)

    # Responsibilities: gamma_j = softmax_j(log w_j + log rho_{j,t})
    log_w = torch.log_softmax(log_weights, dim=0)  # (K,)
    gamma = torch.softmax(log_w[None, :] + log_rho_jt, dim=1)  # (N, K)

    # v_{j,t}(x) = a_j + (sigma_j - 1) * z_j
    v_jt = means[None, :, :] + (diagonal_entries - 1.0)[None, :, :] * z  # (N, K, D)

    return (gamma[:, :, None] * v_jt).sum(dim=1)  # (N, D)


def _velocity_general(
    x: Tensor,  # (N, D)
    t: float,
    log_weights: Tensor,  # (K,)
    components,  # list of AffineComponent
    ref_log_prob: Callable,
) -> Tensor:
    """
    General velocity field supporting tril and full A_j.
    Loops over components, uses matrix ops.
    """
    N, D = x.shape
    K = len(components)
    I = torch.eye(D, device=x.device, dtype=x.dtype)

    log_rho_jt = torch.zeros(N, K, device=x.device, dtype=x.dtype)
    v_jt = torch.zeros(N, K, D, device=x.device, dtype=x.dtype)

    for j, comp in enumerate(components):
        a_j = comp.a  # (D,)
        A_j = comp.get_A()  # (D, D)

        A_jt = t * A_j + (1.0 - t) * I  # (D, D)
        A_jt_inv = torch.linalg.inv(A_jt)  # (D, D)
        _, logdet = torch.linalg.slogdet(A_jt)  # scalar

        z = (x - t * a_j) @ A_jt_inv.T  # (N, D)

        log_rho_jt[:, j] = ref_log_prob(z) - logdet  # (N,)
        v_jt[:, j, :] = a_j + z @ (A_j - I).T  # (N, D)

    log_w = torch.log_softmax(log_weights, dim=0)  # (K,)
    gamma = torch.softmax(log_w[None, :] + log_rho_jt, dim=1)  # (N, K)

    return (gamma[:, :, None] * v_jt).sum(dim=1)  # (N, D)


def _velocity_geometric_diagonal(
    x: Tensor,  # (N, D)
    t: float,
    log_weights: Tensor,  # (K,) unnormalized
    means: Tensor,  # (K, D)
    log_diag_stack: Tensor,  # (K, D) — log_diag of each component
    ref_log_prob: Callable,
) -> Tensor:
    """
    Fast geometric velocity for diagonal A_j = diag(exp(s_j)).

    A_{j,t} = diag(exp(t s_j)),  v_{j,t}(x) = a_j + s_j ⊙ (x - t a_j).
    Fully differentiable w.r.t. log_diag parameters.
    """
    N, D = x.shape

    x_shift = x[:, None, :] - t * means[None, :, :]  # (N, K, D)
    z = x_shift * torch.exp(-t * log_diag_stack)[None, :, :]  # (N, K, D)

    N, K, D = z.shape
    log_ref = ref_log_prob(z.reshape(N * K, D)).reshape(N, K)  # (N, K)
    logdet = t * log_diag_stack.sum(dim=-1)  # (K,)
    log_rho_jt = log_ref - logdet[None, :]  # (N, K)

    log_w = torch.log_softmax(log_weights, dim=0)  # (K,)
    gamma = torch.softmax(log_w[None, :] + log_rho_jt, dim=1)  # (N, K)

    v_jt = means[None, :, :] + log_diag_stack[None, :, :] * x_shift  # (N, K, D)

    return (gamma[:, :, None] * v_jt).sum(dim=1)  # (N, D)


def _velocity_geometric(
    x: Tensor,  # (N, D)
    t: float,
    log_weights: Tensor,  # (K,)
    components,  # list of AffineComponent
    ref_log_prob: Callable,
) -> Tensor:
    """
    Geometric velocity field: A_{j,t} = A_j^t = exp(t log A_j).

    v_{j,t}(x) = a_j + log(A_j)(x - t a_j)
    rho_{j,t}  = |det A_j|^{-t} rho_ref(A_j^{-t}(x - t a_j))

    log(A_j) is exact and differentiable for all supported param_types
    (diagonal, eigenvaluedecomp, matrixexponential).
    """
    N, D = x.shape
    K = len(components)

    log_rho_jt = torch.zeros(N, K, device=x.device, dtype=x.dtype)
    v_jt = torch.zeros(N, K, D, device=x.device, dtype=x.dtype)

    for j, comp in enumerate(components):
        a_j = comp.a  # (D,)
        log_A_j = comp.get_log_A()  # (D, D)

        A_jt_inv = torch.linalg.matrix_exp(-t * log_A_j)  # (D, D)
        logdet = t * comp.log_abs_det()  # scalar

        z = (x - t * a_j) @ A_jt_inv.T  # (N, D)

        log_rho_jt[:, j] = ref_log_prob(z) - logdet
        v_jt[:, j, :] = a_j + (x - t * a_j) @ log_A_j.T  # (N, D)

    log_w = torch.log_softmax(log_weights, dim=0)  # (K,)
    gamma = torch.softmax(log_w[None, :] + log_rho_jt, dim=1)  # (N, K)

    return (gamma[:, :, None] * v_jt).sum(dim=1)  # (N, D)


def _velocity_general_cached(
    x: Tensor,  # (N, D)
    t: float,
    log_weights: Tensor,  # (K,)
    a_stack: Tensor,  # (K, D)   -- static, precomputed once per grad step
    A_stack: Tensor,  # (K, D, D) -- static, precomputed once per grad step
    ref_log_prob: Callable,
) -> Tensor:
    """
    Same math as _velocity_general, but takes the per-component a_j/A_j as
    precomputed (K, ...) stacks instead of re-deriving them from `components`
    (i.e. re-running get_A(), which for matrixexponential/eigenvaluedecomp is
    a matrix_exp / Cayley-map call) on every invocation. The caller is
    expected to compute a_stack/A_stack ONCE per gradient step -- A_j is
    t-independent, so recomputing it at every one of the ode_steps x
    stages-per-step calls to the velocity field is pure waste. Vectorized
    over K via batched matmul instead of a Python loop.
    """
    N, D = x.shape
    K = a_stack.shape[0]
    I = torch.eye(D, device=x.device, dtype=x.dtype)

    A_jt = t * A_stack + (1.0 - t) * I  # (K, D, D)
    A_jt_inv = torch.linalg.inv(A_jt)  # (K, D, D)
    logdet = torch.linalg.slogdet(A_jt)[1]  # (K,)

    x_shift = x[:, None, :] - t * a_stack[None, :, :]  # (N, K, D)
    z = torch.einsum("nkd,ked->nke", x_shift, A_jt_inv)  # (N, K, D)

    log_ref = ref_log_prob(z.reshape(N * K, D)).reshape(N, K)
    log_rho_jt = log_ref - logdet[None, :]

    log_w = torch.log_softmax(log_weights, dim=0)
    gamma = torch.softmax(log_w[None, :] + log_rho_jt, dim=1)  # (N, K)

    v_jt = a_stack[None, :, :] + torch.einsum("nkd,ked->nke", z, A_stack - I)

    return (gamma[:, :, None] * v_jt).sum(dim=1)


def _velocity_geometric_cached(
    x: Tensor,  # (N, D)
    t: float,
    log_weights: Tensor,  # (K,)
    a_stack: Tensor,  # (K, D)    -- static
    log_A_stack: Tensor,  # (K, D, D) -- static, = log(A_j), precomputed once
    logdet_A_stack: Tensor,  # (K,)      -- static, = log|det A_j|, precomputed once
    ref_log_prob: Callable,
) -> Tensor:
    """Cached/vectorized analogue of _velocity_geometric -- see
    _velocity_general_cached's docstring for the rationale. The
    t-dependent matrix_exp(-t * log_A_j) itself is irreducibly per-stage
    (it genuinely depends on t), but log_A_j and log|det A_j| no longer are."""
    N, D = x.shape
    K = a_stack.shape[0]

    A_jt_inv = torch.linalg.matrix_exp(-t * log_A_stack)  # (K, D, D)
    logdet = t * logdet_A_stack  # (K,)

    x_shift = x[:, None, :] - t * a_stack[None, :, :]  # (N, K, D)
    z = torch.einsum("nkd,ked->nke", x_shift, A_jt_inv)  # (N, K, D)

    log_ref = ref_log_prob(z.reshape(N * K, D)).reshape(N, K)
    log_rho_jt = log_ref - logdet[None, :]

    log_w = torch.log_softmax(log_weights, dim=0)
    gamma = torch.softmax(log_w[None, :] + log_rho_jt, dim=1)

    v_jt = a_stack[None, :, :] + torch.einsum("nkd,ked->nke", x_shift, log_A_stack)

    return (gamma[:, :, None] * v_jt).sum(dim=1)


def velocity(
    x: Tensor,
    t: float,
    model: GeneralizedMixture,
    path: ODEPath = "linear",
) -> Tensor:
    """
    Velocity field v_t(x). Dispatches to fast diagonal path when possible.

    path='linear':    A_{j,t} = t A_j + (1-t) I
    path='geometric': A_{j,t} = A_j^t = exp(t log A_j)
    """
    if path == "geometric":
        if model.param_type == "diagonal":
            log_diags = torch.stack([c.log_diag for c in model.components])
            return _velocity_geometric_diagonal(
                x,
                t,
                model.log_weights,
                model.means,
                log_diags,
                model.reference.log_prob,
            )
        else:
            return _velocity_geometric(
                x,
                t,
                model.log_weights,
                model.components,
                model.reference.log_prob,
            )
    else:
        if model.param_type == "diagonal":
            scales = torch.stack([torch.exp(c.log_diag) for c in model.components])
            means = model.means
            return _velocity_diagonal(
                x,
                t,
                model.log_weights,
                means,
                scales,
                model.reference.log_prob,
            )
        else:
            return _velocity_general(
                x,
                t,
                model.log_weights,
                model.components,
                model.reference.log_prob,
            )


class _VelocityFunc(nn.Module):
    """Wraps velocity() in the (t, x) -> dx signature torchdiffeq expects.

    Precomputes the per-component static matrices (A_j for the linear path;
    log_A_j and log|det A_j| for the geometric path) ONCE here, since
    integrate_ode() constructs a fresh _VelocityFunc per gradient step and
    this instance's forward() is then called once per RK stage per
    ode_step (e.g. 4x per step for rk4, more for dopri5). A_j is
    t-independent, so recomputing get_A()/get_log_A() -- a matrix_exp or
    Cayley-map call per component -- at every stage was pure waste; see
    BASELINE_SWEEP.md section 0.1. The diagonal param_type path is already
    cheap (no matrix ops) and is left as before.
    """

    def __init__(self, model: GeneralizedMixture, path: ODEPath = "linear"):
        super().__init__()
        self.model = model
        self.path = path
        self.diagonal = model.param_type == "diagonal"
        self.nfe = 0  # velocity-field evaluations so far; implementation-independent compute axis

        if not self.diagonal:
            self.a_stack = torch.stack([c.a for c in model.components])  # (K, D)
            if path == "geometric":
                self.log_A_stack = torch.stack(
                    [c.get_log_A() for c in model.components]
                )  # (K, D, D)
                self.logdet_A_stack = torch.stack(
                    [c.log_abs_det() for c in model.components]
                )  # (K,)
            else:
                self.A_stack = torch.stack(
                    [c.get_A() for c in model.components]
                )  # (K, D, D)

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        self.nfe += 1
        # Adaptive initial step selection can depend on model parameters.
        # Keep time in the graph so those sensitivities propagate through v(t, x).
        t_val = t
        model = self.model

        if self.diagonal:
            if self.path == "geometric":
                log_diags = torch.stack([c.log_diag for c in model.components])
                return _velocity_geometric_diagonal(
                    x,
                    t_val,
                    model.log_weights,
                    model.means,
                    log_diags,
                    model.reference.log_prob,
                )
            else:
                scales = torch.stack([torch.exp(c.log_diag) for c in model.components])
                return _velocity_diagonal(
                    x,
                    t_val,
                    model.log_weights,
                    model.means,
                    scales,
                    model.reference.log_prob,
                )
        else:
            if self.path == "geometric":
                return _velocity_geometric_cached(
                    x,
                    t_val,
                    model.log_weights,
                    self.a_stack,
                    self.log_A_stack,
                    self.logdet_A_stack,
                    model.reference.log_prob,
                )
            else:
                return _velocity_general_cached(
                    x,
                    t_val,
                    model.log_weights,
                    self.a_stack,
                    self.A_stack,
                    model.reference.log_prob,
                )


ADAPTIVE_SOLVERS = {"dopri5", "dopri8", "bosh3", "fehlberg2", "adaptive_heun"}


def integrate_ode(
    x0: Tensor,
    model: GeneralizedMixture,
    ode_steps: int = 20,
    path: ODEPath = "linear",
    ode_solver: ODESolver = "rk4",
    rtol: float = 1e-5,
    atol: float = 1e-7,
    return_nfe: bool = False,
):
    func = _VelocityFunc(model, path=path)
    if ode_solver in ADAPTIVE_SOLVERS:
        # Adaptive solvers pick their own internal step size; ode_steps doesn't
        # apply. Only the two endpoints are requested, tolerance controls cost.
        t_span = torch.tensor([0.0, 1.0], device=x0.device, dtype=x0.dtype)
        x1 = odeint(func, x0, t_span, method=ode_solver, rtol=rtol, atol=atol)
    else:
        t_span = torch.linspace(
            0.0, 1.0, ode_steps + 1, device=x0.device, dtype=x0.dtype
        )
        x1 = odeint(func, x0, t_span, method=ode_solver)
    if return_nfe:
        return x1[-1], func.nfe
    return x1[-1]


class ODETransportEstimator(GradientEstimator):
    """
    ODE transport estimator based on Theorem 5.1.

    Samples x1 = T(x0) by integrating the theorem's velocity field,
    then computes the ELBO directly:

        ELBO = E_{x0 ~ rho_ref}[ log p(x1) - log q(x1) ]

    This works for any A_j parameterization (diagonal, tril, full).
    Gradients flow through x1 back to (a_j, A_j, w_j).

    Args:
        n_samples:   MC samples per training step
        ode_steps:   number of RK4 steps for the ODE integrator (20 default, 40 for accuracy).
                     Distinct from the number of training steps in TrainConfig.n_steps.
        stl:         sticking the landing. Valid here by theorem 5.1: the flow
                     pushes rho_ref to q_theta, so x1 ~ q_theta and the dropped
                     score term has mean zero -- up to the solver tolerance,
                     the same condition this estimator's unbiasedness already
                     rests on, so it costs no new assumption.
    """

    name = "ode_transport"

    def __init__(
        self,
        MC_samples: int = 128,
        ode_steps: int = 20,
        sample_mode: SampleMode = "mc",
        path: ODEPath = "linear",
        ode_solver: ODESolver = "rk4",
        rtol: float = 1e-5,
        atol: float = 1e-7,
        stl: bool = False,
    ):
        super().__init__(
            MC_samples=MC_samples,
            sample_mode=sample_mode,
            stl=stl,
        )
        self.ode_steps = ode_steps
        self.path = path
        self.ode_solver = ode_solver
        self.rtol = rtol
        self.atol = atol

    def loss_terms(
        self,
        model: GeneralizedMixture,
        log_target: Callable[[Tensor], Tensor],
    ) -> Tuple[Tensor, Dict]:
        N = self.MC_samples

        # Sample from reference and integrate ODE
        with torch.no_grad():
            x0 = self._ref_sample(N, model)

        x1, nfe = integrate_ode(
            x0,
            model,
            ode_steps=self.ode_steps,
            path=self.path,
            ode_solver=self.ode_solver,
            rtol=self.rtol,
            atol=self.atol,
            return_nfe=True,
        )

        # ELBO: both terms evaluated at x1
        log_p = log_target(x1)  # (N,)
        log_q = self._log_q(model, x1)  # (N,), exact GMM density

        elbo = log_p - log_q  # (N,)
        terms = -elbo

        return terms, {
            "elbo": elbo.mean().item(),
            "elbo_std": elbo.std().item(),
            "log_p": log_p.mean().item(),
            "log_q": log_q.mean().item(),
            "nfe": nfe,
        }

    def transport(
        self,
        model: GeneralizedMixture,
        n_samples: int = 1000,
        ode_steps: Optional[int] = None,
        # deprecated — use ode_steps:
        n_steps: Optional[int] = None,
    ) -> Tensor:
        """Draw samples by integrating the ODE (no grad)."""
        if n_steps is not None:
            import warnings

            warnings.warn(
                f"transport(): n_steps={n_steps} is deprecated, use ode_steps=.",
                DeprecationWarning,
                stacklevel=2,
            )
            ode_steps = n_steps
        steps = ode_steps or self.ode_steps
        with torch.no_grad():
            x0 = model.reference.sample(n_samples)
            x1 = integrate_ode(
                x0, model, ode_steps=steps, path=self.path, ode_solver=self.ode_solver
            )
        return x1
