"""
choices.py -- every closed set of string options in the project, in one place.

Import the alias, never retype the strings:

    from gmvi.choices import Chart, ODESolver

    def build(chart: Chart = "matrixexponential", solver: ODESolver = "rk4"):
        ...

An editor then offers the allowed values as completions and underlines a typo.

WHY THIS FILE EXISTS. The same lists were written out in five places --
affine_component.py, baseEstimator.py, gumbel_softmax.py, ode_transport.py and
trainer.py -- and they had already drifted: ode_transport accepts `midpoint`,
and the copy in trainer.py did not list it. A config using it would have been
rejected before reaching the solver that supports it.

STATIC AND RUNTIME ARE DIFFERENT THINGS. Literal is checked by the editor only.
A value that arrives from argparse, a config file or an f-string is invisible to
it, so `validate()` does the same check at runtime and raises with the allowed
list. Use it at the edges -- CLI parsing, config construction.

DRIFT AGAINST THE REGISTRIES. A Literal cannot be built from a runtime dict, so
EstimatorName / TargetName / Reference are written by hand and could fall out of
step with ESTIMATOR_REGISTRY and the make_* functions. check_registries()
compares them and is asserted by tests/test_choices.py, so adding an estimator
without updating this file fails the test rather than surfacing later as a
mysterious ValueError.
"""
from typing import Dict, Literal, Sequence, Tuple, get_args

# ── the variational family ───────────────────────────────────────────────────

Chart = Literal["diagonal", "eigenvaluedecomp", "matrixexponential", "cholesky"]
"""Parameterisation of the component scale matrices A_i.

`diagonal`
    A_i = diag(exp(d)). Cheapest; no correlations.
`eigenvaluedecomp`
    A_i = Q diag(s) Q^T, Q from a Cayley map. Needs an eigendecomposition.
`matrixexponential`
    A_i = exp(S), S symmetric and unconstrained. The ONLY chart with a
    differentiable log A, so the geometric ODE path requires it.
`cholesky`
    A_i = L L^T with L_jj = exp(d_j). sigma_min can collapse here through
    off-diagonal growth, a route the other charts do not offer.
"""

Reference = Literal["normal", "student_t", "laplace", "logistic", "uniform"]
"""Reference distribution q_ref that the affine maps push forward.
Keys of the registry in gmvi.models.reference_distributions.make_reference."""


# ── estimators ───────────────────────────────────────────────────────────────

EstimatorName = Literal["exact_marginalization", "score_function",
                        "gumbel_softmax", "ode_transport"]
"""Keys of gmvi.estimators.gradient_estimators.ESTIMATOR_REGISTRY.

`exact_marginalization`
    DM. Sums over components analytically; unbiased for grad L, which is why it
    is the reference g* every bias is measured against.
`score_function`
    REINFORCE. Unbiased, high variance. Takes a `baseline` control variate; see
    the Baseline alias below. It defaults to "none" so that sweeps predating
    the option reproduce exactly, and "rloo" is the one that couples the batch
    and so invalidates the within-batch variance probe.
`gumbel_softmax`
    ST / soft relaxation. Biased, low variance.
`ode_transport`
    OTR.
"""

SampleMode = Literal["mc", "qmc", "rqmc"]
"""How reference samples are drawn.
`mc` plain Monte Carlo; `qmc` Sobol; `rqmc` scrambled Sobol.

NOTE the per-step variance probe assumes i.i.d. samples. Under `qmc` the points
are deliberately NOT independent, so tr Cov from the within-batch spread does
not estimate the estimator's variance. Use `mc` when measuring variance."""

Baseline = Literal["none", "ema", "optimal", "rloo"]
"""Control variate subtracted from the ELBO signal in ScoreFunctionEstimator.
All four are unbiased -- E_q[grad log q] = 0, and each b is independent of the
draw it multiplies.

`none`
    No control variate. The variance then carries the full c^2 penalty of any
    constant offset c in log p (an unnormalised target), which is exactly what
    the log-Z sweep measures.
`ema`
    Exponential moving average of the batch-mean ELBO over PAST steps, the
    textbook moving-average baseline. One scalar, no extra evaluations.
`optimal`
    b* = E[f ||grad log q||^2] / E[||grad log q||^2], the variance-minimising
    CONSTANT, which is the score-weighted mean rather than the plain mean.
    Needs per-sample score norms, so it is refreshed periodically on a
    subsample and carried as a running average.
`rloo`
    Leave-one-out: b_i is the mean over the other M-1 samples. Free and
    usually the strongest, but it couples the batch, so the within-batch
    variance probe is invalid under it and grad_var comes back nan. Measure
    its variance from independent replicates instead."""

GumbelMode = Literal["straight_through", "soft"]
"""`straight_through` evaluates the ELBO at the hard one-hot assignment and
passes gradients through the soft one. `soft` is the textbook relaxation,
evaluated at the interpolated point."""

ODEPath = Literal["linear", "geometric"]
"""Interpolation from I to A_i.
`linear` A_{i,t} = t A_i + (1-t) I.
`geometric` A_i^t = exp(t log A_i); needs a differentiable log A, hence
chart="matrixexponential"."""

ODESolver = Literal["euler", "midpoint", "rk4", "dopri5"]
"""Fixed-step: `euler`, `midpoint`, `rk4` (ode_steps sets the count).
Adaptive: `dopri5` (rtol/atol instead; see ADAPTIVE_SOLVERS in ode_transport).

Under a fixed-step method the time grid does not depend on the data, so
x1[n] is a function of x0[n] alone and the per-sample gradients are i.i.d.

Under an adaptive method the step size is chosen from an error norm over the
WHOLE batch, so the grid -- and hence every x1[n] -- depends on all N initial
conditions, and strictly the i.i.d. assumption behind the within-batch variance
probe does not hold. In practice the effect should be invisible: each
trajectory still approximates the same deterministic map to within rtol/atol,
so the induced dependence is of order the truncation error (1e-5 relative)
against a sampling spread of order 1. Not assumed either way -- run
test_grad_variance.py --ode-solver dopri5, whose test 4 compares the
within-batch estimate against independent replicates and would show it."""


# ── targets ──────────────────────────────────────────────────────────────────

TargetName = Literal["banana", "rosenbrock", "funnel", "ring", "two_moons",
                     "gmm", "random_gm", "logreg", "hierarchical_mixture",
                     "lotka_volterra", "aniso"]
"""Keys of gmvi.targets.distributions.make_target, plus `aniso`, which
trainer.build_target routes to AnisotropicMixtureTarget separately.

Cannot be sampled, so sliced Wasserstein returns nan: `lotka_volterra`,
`logreg`. Not exactly normalised, so KL is nan unless the target declares
log_Z: the same two."""


# ── optimisation ─────────────────────────────────────────────────────────────

Optimizer = Literal["adam", "sgd"]
Scheduler = Literal["cosine", "none"]
"""`cosine` anneals the learning rate to ~0 over n_steps."""


# ── runtime access ───────────────────────────────────────────────────────────

ALIASES: Dict[str, object] = {
    "chart": Chart,
    "reference": Reference,
    "estimator": EstimatorName,
    "sample_mode": SampleMode,
    "mode": GumbelMode,
    "baseline": Baseline,
    "path": ODEPath,
    "ode_solver": ODESolver,
    "target": TargetName,
    "optimizer": Optimizer,
    "scheduler": Scheduler,
}
"""Field name -> Literal alias, so validate() and any CLI can look them up."""


def allowed(name: str) -> Tuple[str, ...]:
    """The permitted values for a field name, e.g. allowed("chart")."""
    if name not in ALIASES:
        raise KeyError(f"no choice list called {name!r}. "
                       f"Known: {sorted(ALIASES)}")
    return get_args(ALIASES[name])


def validate(name: str, value, allow_none: bool = False):
    """Check a value at runtime and return it, or raise listing the options.

    Literal is invisible to the interpreter, so anything arriving from argparse
    or a config file needs this.
    """
    if allow_none and value is None:
        return value
    opts = allowed(name)
    if value not in opts:
        extra = " or None" if allow_none else ""
        raise ValueError(f"{name}={value!r} is not valid. "
                         f"Allowed: {list(opts)}{extra}")
    return value


def add_argument(parser, name: str, default=None, flag: str = None, **kw):
    """argparse with choices= and the docstring as help, taken from here so the
    CLI cannot drift from the type either."""
    opts = list(allowed(name))
    parser.add_argument(flag or f"--{name.replace('_', '-')}",
                        dest=name, default=default, choices=opts, **kw)
    return parser


def check_registries() -> Dict[str, Dict[str, list]]:
    """Compare the hand-written Literals against the actual registries.

    Returns {alias: {"missing_from_literal": [...], "missing_from_registry":
    [...]}} for anything that disagrees, and an empty dict when they match.
    Imports are local so this module stays dependency-free.
    """
    out: Dict[str, Dict[str, list]] = {}

    def cmp(label: str, literal_vals, registry_vals):
        a, b = set(literal_vals), set(registry_vals)
        if a != b:
            out[label] = {"missing_from_literal": sorted(b - a),
                          "missing_from_registry": sorted(a - b)}

    from gmvi.estimators.gradient_estimators import ESTIMATOR_REGISTRY
    cmp("EstimatorName", allowed("estimator"), ESTIMATOR_REGISTRY.keys())

    import inspect
    from gmvi.models import reference_distributions as rd
    src = inspect.getsource(rd.make_reference)
    cmp("Reference", allowed("reference"),
        _string_keys(src, after="registry = {"))

    from gmvi.targets import distributions as td
    src = inspect.getsource(td.make_target)
    keys = _string_keys(src, after="registry = {")
    # make_target special-cases these two before the registry lookup, and
    # trainer.build_target adds `aniso`.
    keys += ["hierarchical_mixture", "lotka_volterra", "aniso"]
    cmp("TargetName", allowed("target"), keys)

    from gmvi.models.affine_component import AffineParamType
    cmp("Chart", allowed("chart"), get_args(AffineParamType))

    return out


def _string_keys(src: str, after: str) -> list:
    """Pull the quoted dict keys out of a literal `registry = {...}` block."""
    import re
    if after not in src:
        return []
    body = src[src.index(after) + len(after):]
    body = body[:body.index("}")]
    return re.findall(r"""["']([\w_]+)["']\s*:""", body)
