# Thesis figures

Figures, plotting scripts, and the experiments that generate their CSV data.

Run experiment scripts directly with Python. They execute sequentially using
`gmvi.experiments.trainer` and the gradient helpers in `fixed_theta`; no worker
processes or shard/merge step is required. Plot scripts remain separate.

Calibration runners cache completed jobs in `.<prefix>_jobs/`, resume after an
interruption, and write the same CSV names used by the plots. Existing shard
files are historical inputs, not resume caches for these runners. A changed
configuration requires a fresh cache directory. Benchmark and variance runners
retain their existing per-cell completion checks. Long runs are not started on
module import.

`benchmark_training/run_queue.py` runs the three training comparisons and then
the Rosenbrock solver study. Solver studies can also be run directly with `solver_bias/solver.py` and
`solver_bias/solver_hier_k5.py`; `parallel_hier_k5.py` is a compatibility alias
for the sequential hierarchical runner.

Frozen `source/` snapshots and saved results are historical records; maintained
runners use the current `Implementation/gmvi` package. Full studies can take hours.


| Section | Figure (as in `\includegraphics`) | Plot script | Data / experiment script |
|---|---|---|---|
| 3.1 Variational Inference | `kl_forward_vs_reverse.pdf` | `kl_forward_vs_reverse_viz.py` | – (analytic) |
| 4.3 Gradient Estimators | `control_variate_variance.png` | `control_variate_viz.py` | – |
| 4.3 Gradient Estimators | `softmax_temperature.png` | `softmax_temperature_viz.py` | – |
| 4.3 Gradient Estimators | `gumbel_softmax_placement_2d.png` | `gumbel_softmax_placement_2d_viz.py` | uses `gmvi.utils.visualization` |
| 5.1 Construction of the Velocity Field | `rho_t_all.png` | `theory_visualization.py` | – |
| 6.1 Experimental Setup | `target_gallery.png`, `target_gallery.pdf` | `6.1_Experimental_Setup/target_gallery.py` | current benchmark configurations |
| 6.3 Interpolation path | `lin_geo_relative_difference.png`, `lin_geo_final.pdf` | `interpolation_path/plot_lin_geo.py` | `run_lin_geo.py` → `lin_geo_*.csv` |
| 6.3 Interpolation path | `lin_geo_snapshots_2x4.png` | `interpolation_path/plot_lin_geo_snapshots.py` | same |
| 6.3 Integrator and step count | `dopri_hier50_curves.pdf` | `integrator_and_step_count/plot_dopri_hier50.py` | `run_dopri_sweep_hier50.py` |
| 6.3 MC sampling rate | `k50_snapshots_M_sweep.pdf` | `mc_sample_size/plot_k50_snapshots.py` | `run_M_sweep.py` → `M_sweep_final_params.csv` |
| 6.3 Step count and learning rate | `hier50_lr_anneal_floor_curves.pdf` | `learning_rate/plot_hier50_lr_anneal_floor.py` | `run_hier50_lr_anneal_floor.py` |
| 6.3 Step count and learning rate | `k50_snapshots_lr_anneal_floor_0.1.pdf` | `learning_rate/plot_k50_snapshots.py` | same |
| 6.4 Two-dimensional comparison | `training_comparison_grid.pdf` | `benchmark_training/plot.py` | `run_queue.py` + `worker.py` + `dimension_base.py` → `*/training_steps.csv` |
| 6.4 Two-dimensional comparison | `unnorm_rosenbrock_logZ_m50_kl.pdf` | `benchmark_training/plot.py` | same |
| 6.4 Higher-dimensional comparison | `training_comparison_grid_dimensions.pdf` | `benchmark_training/plot.py` | same |
| 6.5 Bias under solver accuracy | `solver_bias_combined_dm_reference_noise_normalized.pdf` | `solver_bias/plot_solver_combined_normalized.py` | `solver.py` (Rosenbrock), `solver_hier_k5.py` (hierarchical mixture) |
| 6.5 Variance under MC sample size | `variance_M_rosenbrock_hier_k5.pdf` | `variance_mc_sample_size/plot_variance_trimmed.py` | `run_study.py` → `bootstrap_variance_trimmed.py` |
| 6.5 Variance in higher dimensions | `dimension_variance.pdf` | `variance_dimension/plot_dimension_variance.py` | `run_study.py` |
| 6.5 Transport sensitivity | `grid_sample_paths.pdf` | `transport_sensitivity/grid_sample_paths/plot_grid_paths.py` | `run.py` → `paths.npz` |
| 6.5 Transport sensitivity | `gradient_tail_samples_over_reference_p999_tight_zoom.pdf` | `transport_sensitivity/gradient_tail/plot_tail_samples.py --reference --include-p999 --tight-zoom` | `run.py` → `per_sample_elbo_gradients.npz` |

