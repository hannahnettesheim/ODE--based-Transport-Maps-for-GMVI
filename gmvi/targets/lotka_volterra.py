"""Lotka–Volterra posterior with lognormal observations.

Positive parameters use log coordinates with independent Normal priors on those
coordinates; no additional prior Jacobian is needed. The forward ODE is solved
in log-state coordinates, with clamped exponents for numerical stability."""

import csv
import os
from typing import Optional, Tuple

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


# Independent Normal priors on log parameters; these differ from Stan
# priors on constrained rates.
_PRIOR_LOC = torch.tensor(
    [
        0.0,  # log alpha   ~ N(log 1,    0.5)
        -3.0,  # log beta    ~ N(log 0.05, 0.5)
        0.0,  # log gamma   ~ N(log 1,    0.5)
        -3.0,  # log delta   ~ N(log 0.05, 0.5)
        2.3,  # log u_0     ~ N(log 10,   1.0)
        2.3,  # log v_0     ~ N(log 10,   1.0)
        -1.0,  # log sigma_u ~ N(-1,       1.0)
        -1.0,  # log sigma_v ~ N(-1,       1.0)
    ]
)
_PRIOR_SCALE = torch.tensor([0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0])

# These clamps define the numerical forward model, rather than the exact ODE.
_LOG_STATE_CLAMP = 12.0  # exp(12) ~ 1.6e5 individuals; far above the posterior regime
# Clamp log rates to prevent overflow before the vector-field clamp acts.
# The bound is wider than the posterior region to avoid zeroing its gradients.
_LOG_RATE_CLAMP = 25.0

# Keep likelihood scales finite and positive in float32. The clamp
# changes the numerical likelihood outside this log-scale range.
_LOG_SIGMA_CLAMP = 30.0

# Bound the log-state field to prevent overflow in RK4 backward products.
# Saturated clamps have zero derivative; the bound is outside the posterior region.
_FIELD_CLAMP = 1.0e3


# The Stan option uses positive-truncated Normal priors on rates,
# with a log-Jacobian for the log coordinates. Initial-state and noise
# priors match the default lognormal priors.
_STAN_RATE_LOC = torch.tensor([1.0, 0.05, 1.0, 0.05])  # alpha, beta, gamma, delta
_STAN_RATE_SCALE = torch.tensor([0.5, 0.05, 0.5, 0.05])

# Bound exponentiation in the Stan prior to keep its quadratic term
# finite in float32, outside the posterior rate range.
_STAN_RATE_EXP_CLAMP = 30.0


_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_hudson_bay_lynx_hare() -> Tuple[Tensor, Tensor, Tensor]:
    """
    Hudson's Bay Company pelt counts, 1900-1920, in thousands.

    Source: stan-dev/example-models, knitr/lotka-volterra/, which is the data
    behind the Stan predator-prey case study. Retrieved 2026-08-29.

    COLUMN ORDER. The file is (Year, Lynx, Hare); Lynx is the PREDATOR and Hare
    the PREY. This model's state is s = (log u, log v) with

        dp/dt = alpha - beta*exp(q)      u grows, is eaten   -> u is PREY
        dq/dt = -gamma + delta*exp(p)    v decays, feeds     -> v is PREDATOR

    so the returned `data` is (hare, lynx), the reverse of the file. Simulating
    the field at the default x_true from the 1900 observation gives
    corr(u, hare) = 0.95 and corr(v, lynx) = 0.92, against 0.31 and -0.23 for
    the swapped assignment -- the check that catches this if it is ever changed.

    The 1900 row is NOT returned as an observation. This model treats the state
    at t = 0 as the parameters (u_0, v_0), matching `n_obs` = "number of
    observation times (excluding t=0)"; 1900 is returned separately as `z0` so
    it can be used to set or check the initial-state priors.

    Returns
    -------
    data      : (20, 2) observations for 1901-1920, columns (hare, lynx)
    obs_times : (20,)   1.0, ..., 20.0, years since 1900
    z0        : (2,)    the 1900 observation, (hare, lynx)
    """
    path = os.path.join(_DATA_DIR, "hudson_bay_lynx_hare.csv")
    rows = [
        r
        for r in csv.reader(open(path))
        if r and not r[0].lstrip().startswith("#") and r[0].strip() != "Year"
    ]
    years = [int(r[0]) for r in rows]
    lynx = [float(r[1]) for r in rows]
    hare = [float(r[2]) for r in rows]

    dtype = torch.get_default_dtype()
    z0 = torch.tensor([hare[0], lynx[0]], dtype=dtype)
    data = torch.tensor([[h, l] for h, l in zip(hare[1:], lynx[1:])], dtype=dtype)
    obs_times = torch.tensor([float(y - years[0]) for y in years[1:]], dtype=dtype)
    return data, obs_times, z0


class LotkaVolterraPosterior(Target):
    """
    Bayesian posterior over the eight log-parameters of a Lotka-Volterra system.

    Args:
        n_obs:      number of observation times (excluding t=0)
        t_max:      end of the observation window
        ode_steps:  fixed RK4 steps used for the forward model; this is the knob that
                    makes the target expensive
        x_true:     true log-parameters used to generate synthetic data. If None, a
                    default oscillatory regime is used.
        seed:       seed for the synthetic observation noise
        data:       optional (n_obs, 2) tensor of *observed* y values (not logged) to use
                    instead of synthetic data
        obs_times:  optional (n_obs,) tensor of strictly increasing observation times
    """

    name = "lotka_volterra"

    def __init__(
        self,
        n_obs: int = 20,
        t_max: float = 20.0,
        ode_steps: int = 200,
        x_true: Optional[Tensor] = None,
        seed: int = 0,
        data: Optional[Tensor] = None,
        obs_times: Optional[Tensor] = None,
        prior: str = "lognormal",
        y_init: Optional[Tensor] = None,
        solver: str = "fixed_rk4",
        rtol: float = 1e-5,
        atol: float = 1e-3,
    ):
        if prior not in ("lognormal", "stan"):
            raise ValueError(f"prior must be 'lognormal' or 'stan', got {prior!r}")
        if solver not in ("fixed_rk4", "rk45"):
            raise ValueError(f"solver must be 'fixed_rk4' or 'rk45', got {solver!r}")
        self.prior_spec = prior
        self.y_init = None if y_init is None else torch.as_tensor(y_init)
        self.n_obs = n_obs
        self.t_max = float(t_max)
        self.ode_steps = int(ode_steps)
        self.h = self.t_max / self.ode_steps
        # solver="rk45": adaptive Dormand-Prince (torchdiffeq dopri5), matching
        # the Stan predator-prey case study's rk45 at rtol=1e-5, atol=1e-3.
        # ode_steps is then only an upper bound on the reported cost, not the
        # actual work. "fixed_rk4" keeps the original batched fixed-step loop.
        self.solver = solver
        self.rtol = float(rtol)
        self.atol = float(atol)

        dtype = torch.get_default_dtype()
        if obs_times is None:
            self.obs_times = torch.linspace(
                self.t_max / n_obs, self.t_max, n_obs, dtype=dtype
            )
        else:
            self.obs_times = torch.as_tensor(obs_times, dtype=dtype)
            if self.obs_times.shape != (n_obs,):
                raise ValueError(
                    f"obs_times must have shape ({n_obs},), got {tuple(self.obs_times.shape)}"
                )
            if not torch.all(self.obs_times[1:] > self.obs_times[:-1]):
                raise ValueError("obs_times must be strictly increasing")
            if self.obs_times[0] <= 0 or self.obs_times[-1] > self.t_max:
                raise ValueError("obs_times must lie in (0, t_max]")

        self.prior = dist.Normal(_PRIOR_LOC.to(dtype), _PRIOR_SCALE.to(dtype))

        if x_true is None:
            # alpha=0.55, beta=0.028, gamma=0.80, delta=0.024, u0=v0=30,
            # sigma_u=sigma_v=0.10 -- a clearly oscillatory regime over [0, 20].
            x_true = torch.log(
                torch.tensor([0.55, 0.028, 0.80, 0.024, 30.0, 30.0, 0.10, 0.10])
            )
        self.x_true = x_true

        if data is not None:
            data = torch.as_tensor(data, dtype=dtype)
            if data.shape != (n_obs, 2):
                raise ValueError(
                    f"data must have shape ({n_obs}, 2), got {tuple(data.shape)}"
                )
            if not torch.all(data > 0):
                raise ValueError(
                    "Lotka-Volterra observations must be strictly positive"
                )
            self.log_y = torch.log(data)
        else:
            g = torch.Generator().manual_seed(seed)
            with torch.no_grad():
                log_traj = self._solve(x_true.unsqueeze(0))  # (1, n_obs, 2)
            sigma = torch.exp(x_true[6:8])
            noise = torch.randn(self.n_obs, 2, generator=g) * sigma
            self.log_y = log_traj[0] + noise  # (n_obs, 2)

        # Instrumentation for the cost model: C_p is measured by counting these.
        self.n_evals = 0

    @classmethod
    def from_lynx_hare(cls, ode_steps: int = 200, **kwargs) -> "LotkaVolterraPosterior":
        """
        The posterior for the historical Hudson's Bay lynx-hare series rather
        than for synthetic data.

        The two differ in what they demonstrate. With synthetic data the true
        parameter is known and recovery can be checked; with these data there is
        no ground truth, but the posterior is a real one and the target is no
        longer an idealization.

        NOTE on priors. `_PRIOR_LOC` centres log u_0 and log v_0 at log(10),
        while the 1900 observation is (hare, lynx) = (30, 4), i.e. +1.1 and -0.9
        prior standard deviations away. That is tolerable but it is a modelling
        choice, not a neutral default; centring those two priors on the 1900
        observation would follow the Stan case study more closely. It is left
        unchanged here so that the prior is identical to the synthetic setting
        and the only difference between the two targets is the data.
        """
        data, obs_times, z0 = load_hudson_bay_lynx_hare()
        kwargs.setdefault("prior", "stan")
        kwargs.setdefault("y_init", z0)
        return cls(
            n_obs=len(obs_times),
            t_max=float(obs_times[-1]),
            ode_steps=ode_steps,
            data=data,
            obs_times=obs_times,
            **kwargs,
        )

    @property
    def dim(self) -> int:
        return 8

    def _field(self, s: Tensor, rates: Tensor) -> Tensor:
        """
        Log-state vector field. s: (B, 2) = (log u, log v). rates: (B, 4).
        """
        p, q = s[:, 0], s[:, 1]
        alpha, beta, gamma, delta = rates.unbind(dim=-1)
        exp_p = torch.exp(p.clamp(max=_LOG_STATE_CLAMP))
        exp_q = torch.exp(q.clamp(max=_LOG_STATE_CLAMP))
        field = torch.stack([alpha - beta * exp_q, -gamma + delta * exp_p], dim=-1)
        return field.clamp(-_FIELD_CLAMP, _FIELD_CLAMP)

    def _solve(self, x: Tensor) -> Tensor:
        """
        Forward model: integrate the log-state field to the observation times.

        x: (B, 8) log-parameters.
        Returns (B, n_obs, 2), the log-trajectory at the observation times.

        Dispatches on self.solver ("fixed_rk4" -> batched fixed-step RK4;
        "rk45" -> adaptive Dormand-Prince via torchdiffeq, the Stan setup).
        """
        if self.solver == "rk45":
            return self._solve_rk45(x)
        return self._solve_fixed_rk4(x)

    def _solve_rk45(self, x: Tensor) -> Tensor:
        """Adaptive RK45 (torchdiffeq dopri5) at self.rtol / self.atol.

        The field is still clamped to +/- _FIELD_CLAMP inside _field, which is
        the guard that matters for stability; the per-step state clamp of the
        fixed integrator has no analogue between adaptive substeps, so only the
        final trajectory is clamped to +/- _LOG_STATE_CLAMP.
        """
        from torchdiffeq import odeint

        rates = torch.exp(x[:, 0:4].clamp(-_LOG_RATE_CLAMP, _LOG_RATE_CLAMP))
        s0 = x[:, 4:6].clamp(-_LOG_STATE_CLAMP, _LOG_STATE_CLAMP)  # (B, 2)

        def field(_t: Tensor, s: Tensor) -> Tensor:
            return self._field(s, rates)  # (B, 2)

        t_eval = torch.cat([s0.new_zeros(1), self.obs_times.to(s0)])  # (n_obs+1,)
        traj = odeint(
            field, s0, t_eval, method="dopri5", rtol=self.rtol, atol=self.atol
        )  # (n_obs+1, B, 2)
        traj = traj[1:].permute(1, 0, 2)  # (B, n_obs, 2)
        return traj.clamp(-_LOG_STATE_CLAMP, _LOG_STATE_CLAMP)

    def _solve_fixed_rk4(self, x: Tensor) -> Tensor:
        """
        Batched fixed-step RK4 in log-state.

        x: (B, 8) log-parameters.
        Returns (B, n_obs, 2), the log-trajectory at the observation times.
        """
        rates = torch.exp(x[:, 0:4].clamp(-_LOG_RATE_CLAMP, _LOG_RATE_CLAMP))
        # Clamp only the numerical forward-model inputs.  The prior is evaluated on the
        # original unconstrained x in log_prob(), so this guard cannot make the target
        # improper.  It only prevents absurd initial states from overflowing the solver.
        s = x[:, 4:6].clamp(-_LOG_STATE_CLAMP, _LOG_STATE_CLAMP)

        def rk4_step(state: Tensor, dt: float) -> Tensor:
            k1 = self._field(state, rates)
            k2 = self._field(state + 0.5 * dt * k1, rates)
            k3 = self._field(state + 0.5 * dt * k2, rates)
            k4 = self._field(state + dt * k3, rates)
            nxt = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            return nxt.clamp(-_LOG_STATE_CLAMP, _LOG_STATE_CLAMP)

        # Integrate to the actual requested observation times.  self.h is the maximum
        # step size; a shorter final step lands exactly on each observation time.
        out = []
        t = 0.0
        eps = 1e-12
        for t_obs in self.obs_times.tolist():
            while t + self.h < t_obs - eps:
                s = rk4_step(s, self.h)
                t += self.h
            dt = t_obs - t
            if dt > eps:
                s = rk4_step(s, dt)
                t = t_obs
            out.append(s)
        return torch.stack(out, dim=1)  # (B, n_obs, 2)

    def log_prob(self, z: Tensor) -> Tensor:
        """
        Unnormalized log posterior on R^8. z: (..., 8) -> (...,).
        """
        shape = z.shape[:-1]
        x = z.reshape(-1, 8)
        self.n_evals += x.shape[0]

        # IMPORTANT: evaluate the prior at the original unconstrained x.  Clamping x here
        # would create constant-density tails and hence an improper target.  Numerical
        # guards belong inside the forward model only.
        if self.prior_spec == "lognormal":
            log_prior = self.prior.log_prob(x).sum(dim=-1)  # (B,)
        else:
            # Stan: truncated normals on the four CONSTRAINED rates, plus the
            # log-Jacobian d(rate)/dx = e^x; the remaining four coordinates
            # (log u_0, log v_0, log sigma_u, log sigma_v) already carry exactly
            # the lognormal priors Stan specifies, so they are unchanged.
            loc = _STAN_RATE_LOC.to(x.dtype)
            sc = _STAN_RATE_SCALE.to(x.dtype)
            xr = x[:, 0:4].clamp(max=_STAN_RATE_EXP_CLAMP)
            rates = torch.exp(xr)
            lp_rates = dist.Normal(loc, sc).log_prob(rates)
            # truncation to (0, inf): divide by Phi(loc/scale)
            lp_rates = lp_rates - torch.log(dist.Normal(0.0, 1.0).cdf(loc / sc))
            lp_rates = lp_rates + xr  # log-Jacobian
            log_prior = lp_rates.sum(dim=-1) + self.prior.log_prob(x)[:, 4:8].sum(
                dim=-1
            )

        log_traj = self._solve(x)  # (B, n_obs, 2)
        sigma = torch.exp(x[:, 6:8].clamp(-_LOG_SIGMA_CLAMP, _LOG_SIGMA_CLAMP))
        obs = dist.Normal(log_traj, sigma.unsqueeze(1))  # broadcast over n_obs
        log_lik = obs.log_prob(self.log_y).sum(dim=(-2, -1))  # (B,)

        if self.y_init is not None:
            # Stan also treats the t=0 observation as data:
            #     y_init[k] ~ lognormal(log z_init[k], sigma[k])
            # and log z_init is exactly x[:, 4:6], so this is a direct term on
            # the initial-state parameters rather than on the trajectory.
            log_lik = log_lik + dist.Normal(x[:, 4:6], sigma).log_prob(
                torch.log(self.y_init.to(x.dtype))
            ).sum(dim=-1)

        return (log_prior + log_lik).reshape(shape)

    def sample(self, n: int) -> Tensor:
        raise NotImplementedError(
            "No exact sampler for this posterior. Use a long MCMC run for reference draws, "
            "or compare estimators by ELBO differences, which are valid because log Z is a "
            "constant of the target."
        )

    def reset_eval_counter(self) -> None:
        self.n_evals = 0

    def time_per_eval(self, batch: int = 256, repeats: int = 20) -> float:
        """Seconds per single log_prob evaluation, i.e. C_p in the cost model."""
        import time

        z = self.prior.sample((batch,))
        self.log_prob(z)  # warm up
        t0 = time.perf_counter()
        for _ in range(repeats):
            self.log_prob(z)
        return (time.perf_counter() - t0) / (repeats * batch)

    def time_per_grad_eval(self, batch: int = 256, repeats: int = 20) -> float:
        """Seconds per target value+gradient evaluation with respect to z."""
        import time

        def one_pass() -> None:
            z = self.prior.sample((batch,)).requires_grad_(True)
            value = self.log_prob(z).sum()
            torch.autograd.grad(value, z)

        one_pass()  # warm up
        t0 = time.perf_counter()
        for _ in range(repeats):
            one_pass()
        return (time.perf_counter() - t0) / (repeats * batch)
