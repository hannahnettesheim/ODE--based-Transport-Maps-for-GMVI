from pathlib import Path

ROOT = next(
    (
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "gmvi").is_dir()
    )
)
import sys

sys.path.insert(0, str(ROOT))
from gmvi.experiments.execution import experiment_environment


def measure():
    """Exact full-batch per-term Jacobian via forward AD, checked against reverse AD."""
    from pathlib import Path
    import importlib.util
    import json
    import time
    import numpy as np
    import torch

    H = Path(__file__).resolve().parent
    B = H.parent
    spec = importlib.util.spec_from_file_location("study", B / "otr_study.py")
    s = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    v = torch.load(H.parent / "replay_inputs.pt", weights_only=False)
    cfg = s.cfg_for("rosenbrock_extreme")
    m = s.build_model(cfg, 2, reseed=False).double()
    m.load_state_dict(v["state"])
    x = v["x0"].double()
    target = s.build_target(cfg)
    est = s.make_estimator(
        "ode_transport",
        MC_samples=len(x),
        path="linear",
        ode_solver="dopri5",
        rtol=1e-10,
        atol=1e-10,
    )
    est._ref_sample = lambda n, model: x
    terms, info = est.loss_terms(m, target.log_prob)
    params = list(m.parameters())

    def flatten(gs):
        return torch.cat([g.reshape(-1) for g in gs]).detach().numpy()

    batch = -flatten(torch.autograd.grad(terms.mean(), params, retain_graph=True))
    s0 = -flatten(torch.autograd.grad(terms[5999], params))
    loss_values = terms.detach().numpy().copy()
    del terms
    np.savez(
        H / "native_reverse.npz",
        batch_elbo_gradient=batch,
        s0_elbo_gradient=s0,
        negative_elbo_terms=loss_values,
    )
    print(
        "Native reverse AD",
        dict(
            batch_norm=float(np.linalg.norm(batch)),
            s0_norm=float(np.linalg.norm(s0)),
            **info,
        ),
        flush=True,
    )
    import torchdiffeq._impl.rk_common as rk
    import torchdiffeq._impl.misc as misc

    old_optimal = rk._optimal_step_size
    rk._optimal_step_size = lambda *a, **kw: old_optimal(*a, **kw).detach()

    class Assign:
        @staticmethod
        def apply(scratch, value, index):
            result = scratch.clone()
            result[index] = value
            return result

    rk._UncheckedAssign = Assign

    def nextafter(x1, x2):
        out = torch.nextafter(x1.detach(), x2.detach())
        return out + (x1 - x1.detach())

    misc._nextafter = nextafter

    class Terms(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self):
            return est.loss_terms(self.model, target.log_prob)[0]

    wrapper = Terms(m)
    named = dict(wrapper.named_parameters())
    names = [
        f"{name.removeprefix('model.')}[{i}]"
        for name, p in named.items()
        for i in range(p.numel())
    ]
    columns = []
    start = time.time()
    for coordinate in range(len(names)):
        cp = H / f"column_{coordinate:02d}.npy"
        if cp.exists():
            column = np.load(cp)
        else:
            with torch.no_grad(), torch.autograd.forward_ad.dual_level():
                duals = {}
                offset = 0
                for name, p in named.items():
                    tangent = torch.zeros_like(p)
                    if offset <= coordinate < offset + p.numel():
                        tangent.reshape(-1)[coordinate - offset] = 1
                    duals[name] = torch.autograd.forward_ad.make_dual(
                        p.detach(), tangent
                    )
                    offset += p.numel()
                result = torch.func.functional_call(wrapper, duals, ())
                primal, tangent = torch.autograd.forward_ad.unpack_dual(result)
                np.testing.assert_allclose(
                    primal.numpy(), loss_values, rtol=1e-11, atol=1e-11
                )
                column = -tangent.numpy().copy()
            np.save(cp, column)
        columns.append(column)
        print(
            f"Coordinate {coordinate + 1}/{len(names)}: {names[coordinate]}, elapsed {time.time() - start:.1f}s",
            flush=True,
        )
    g = np.stack(columns, axis=1)
    with torch.no_grad():
        endpoint = s.ot.integrate_ode(
            x, m, path="linear", ode_solver="dopri5", rtol=1e-10, atol=1e-10
        )
    native_cholesky_inverse = torch.cholesky_inverse

    def corrected_cholesky_inverse(factor, upper=False):
        primal, tangent = torch.autograd.forward_ad.unpack_dual(factor)
        inverse = native_cholesky_inverse(primal, upper=upper)
        if tangent is None:
            return inverse
        if upper:
            da = tangent.T @ primal + primal.T @ tangent
        else:
            da = tangent @ primal.T + primal @ tangent.T
        return torch.autograd.forward_ad.make_dual(inverse, -inverse @ da @ inverse)

    class Density(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self):
            return self.model.log_prob(endpoint)

    density = Density(m)
    for coordinate, label in enumerate(names):
        if "chol_raw" not in label:
            continue
        with torch.no_grad(), torch.autograd.forward_ad.dual_level():
            duals, offset = ({}, 0)
            for name, p in named.items():
                tangent = torch.zeros_like(p)
                if offset <= coordinate < offset + p.numel():
                    tangent.reshape(-1)[coordinate - offset] = 1
                duals[name] = torch.autograd.forward_ad.make_dual(p.detach(), tangent)
                offset += p.numel()
            old = torch.autograd.forward_ad.unpack_dual(
                torch.func.functional_call(density, duals, ())
            )[1].clone()
            torch.cholesky_inverse = corrected_cholesky_inverse
            try:
                new = torch.autograd.forward_ad.unpack_dual(
                    torch.func.functional_call(density, duals, ())
                )[1]
                g[:, coordinate] -= (new - old).numpy()
            finally:
                torch.cholesky_inverse = native_cholesky_inverse
    assert np.isfinite(g).all()
    np.testing.assert_allclose(g.mean(0), batch, rtol=2e-07, atol=2e-07)
    np.testing.assert_allclose(g[5999], s0, rtol=2e-07, atol=2e-05)
    norms = np.linalg.norm(g, axis=1)
    summary = dict(
        draw=890,
        seed=int(v["seed"]),
        snapshot_step=1000,
        mc=len(x),
        dtype="float64",
        solver="dopri5",
        rtol=1e-10,
        atol=1e-10,
        batch_gradient_norm=float(np.linalg.norm(batch)),
        mean_sample_gradient_norm=float(norms.mean()),
        median_sample_gradient_norm=float(np.median(norms)),
        s0_gradient_norm=float(norms[5999]),
        s0_logit3_elbo_derivative=float(g[5999, 2]),
        s0_batch_contribution_norm=float(norms[5999] / len(x)),
        s0_fraction_sum_norms=float(norms[5999] / norms.sum()),
        mean_sample_gradient_norm_without_s0=float(np.delete(norms, 5999).mean()),
        reverse_validation_batch_max_abs=float(np.abs(g.mean(0) - batch).max()),
        reverse_validation_s0_max_abs=float(np.abs(g[5999] - s0).max()),
    )
    np.savez(
        H / "per_sample_elbo_gradients.npz",
        gradients=g,
        norms=norms,
        parameter_names=np.array(names),
        batch_gradient=batch,
    )
    (H / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (H / "REPORT.md").write_text(
        f"# Estimator draw 890 — float64, tolerance 10⁻¹⁰\n\nRosenbrock, frozen step-1,000 snapshot; seed 2230889; M = 16,384.\nDOPRI5 with rtol = atol = 10⁻¹⁰. The original saved reference samples and\nsnapshot are converted to float64; no samples are redrawn and no training is run.\n\n| Quantity | Gradient norm |\n|---|---:|\n| Complete estimator draw, ‖(1/M) Σᵢ ∇ℓᵢ‖ | {summary['batch_gradient_norm']:.10g} |\n| Mean of individual sample norms, (1/M) Σᵢ ‖∇ℓᵢ‖ | {summary['mean_sample_gradient_norm']:.10g} |\n| Median of individual sample norms | {summary['median_sample_gradient_norm']:.10g} |\n| Problematic sample S₀ (sample 6,000) | {summary['s0_gradient_norm']:.10g} |\n| S₀ contribution after division by M | {summary['s0_batch_contribution_norm']:.10g} |\n\nThe mean and median refer to the 16,384 individual sample gradients **within this\none estimator draw**, before division by M. They are not statistics across\nestimator draws or training steps. Norms are Euclidean over all 30 original\nCholesky-model parameters. Derivatives use the ELBO sign convention.\n\nFor S₀, ∂ℓ(S₀)/∂log_weights[2] = {summary['s0_logit3_elbo_derivative']:.10g}.\nS₀ accounts for {100 * summary['s0_fraction_sum_norms']:.6f}% of the sum of sample\ngradient norms. Excluding S₀, the mean sample norm is\n{summary['mean_sample_gradient_norm_without_s0']:.10g}.\n\nValidation: all sample gradients use the original full batch and adaptive solver,\nwith forward-mode derivatives matching the native reverse-mode mean gradient\nand S₀ gradient (maximum absolute errors {summary['reverse_validation_batch_max_abs']:.3g}\nand {summary['reverse_validation_s0_max_abs']:.3g}, respectively). Solver controller\nstop-gradient behavior is preserved. This is a measurement at the requested\ntolerance, not a claim of convergence as the tolerance goes to zero.\nThe installed PyTorch's Cholesky-inverse forward derivative required an analytic\ncorrection; final vectors are validated against native reverse differentiation.\n\nReproduce: `python3.11 Implementation/Visualisations/M_sweep_2d/draw890_float64_tol1e-10/run.py`.\nFull vectors and norms are in `per_sample_elbo_gradients.npz`; precise summary\nvalues are in `summary.json`.\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


@experiment_environment()
def main():
    import torchdiffeq._impl.rk_common as rk
    import torchdiffeq._impl.misc as misc

    original_step_size = rk._optimal_step_size
    original_assign = rk._UncheckedAssign
    original_nextafter = misc._nextafter
    try:
        measure()
    finally:
        rk._optimal_step_size = original_step_size
        rk._UncheckedAssign = original_assign
        misc._nextafter = original_nextafter


if __name__ == "__main__":
    main()
