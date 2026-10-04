# Estimator draw 890 — float64, tolerance 10⁻¹⁰

Rosenbrock, frozen step-1,000 snapshot; seed 2230889; M = 16,384.
DOPRI5 with rtol = atol = 10⁻¹⁰. The original saved reference samples and
snapshot are converted to float64; no samples are redrawn and no training is run.

| Quantity | Gradient norm |
|---|---:|
| Complete estimator draw, ‖(1/M) Σᵢ ∇ℓᵢ‖ | 10494.37057 |
| Mean of individual sample norms, (1/M) Σᵢ ‖∇ℓᵢ‖ | 10503.91473 |
| Median of individual sample norms | 5.585231806 |
| Problematic sample S₀ (sample 6,000) | 171940553.8 |
| S₀ contribution after division by M | 10494.41857 |

The mean and median refer to the 16,384 individual sample gradients **within this
one estimator draw**, before division by M. They are not statistics across
estimator draws or training steps. Norms are Euclidean over all 30 original
Cholesky-model parameters. Derivatives use the ELBO sign convention.

For S₀, ∂ℓ(S₀)/∂log_weights[2] = -104888374.8.
S₀ accounts for 99.909594% of the sum of sample
gradient norms. Excluding S₀, the mean sample norm is
9.496744001.

Validation: all sample gradients use the original full batch and adaptive solver,
with forward-mode derivatives matching the native reverse-mode mean gradient
and S₀ gradient (maximum absolute errors 3.27e-11
and 1.49e-07, respectively). Solver controller
stop-gradient behavior is preserved. This is a measurement at the requested
tolerance, not a claim of convergence as the tolerance goes to zero.
The installed PyTorch's Cholesky-inverse forward derivative required an analytic
correction; final vectors are validated against native reverse differentiation.

Reproduce: `python3.11 Implementation/Visualisations/M_sweep_2d/draw890_float64_tol1e-10/run.py`.
Full vectors and norms are in `per_sample_elbo_gradients.npz`; precise summary
values are in `summary.json`.
