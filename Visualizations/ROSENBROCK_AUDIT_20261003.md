# Rosenbrock consistency audit — 2026-10-03

Scope: current scripts and associated data under Implementation/Visualizations,
plus all 16 figure files referenced by
Masterarbeit_Overleaf/Thesis/Chapters/04_Numerical_Experiment.tex.
Archived and superseded experiments were inspected for provenance, not treated
as current thesis results. This was an audit; no experiments or figures were changed.

## Conclusion

No conflicting Rosenbrock parameters were found in the active figure workflows.
They use mu=1, a=0.05, b=5. The dimension experiments use n1=2 and n2=n-1:
one Gaussian root and conditionally independent children centered at its square.
The normalized densities agree with the equations supplied for the thesis.
The explicitly unnormalized benchmark uses the same shape multiplied by exp(-50).

## Evidence

| Workflow | Evidence checked |
| --- | --- |
| Target gallery | target_gallery.py reads benchmark worker.config; target_gallery_config.json records mu=1, a=0.05, b=5. |
| Interpolation-path calibration | Current run_lin_geo.py and plot_lin_geo_snapshots.py specify a=0.05, b=5; mu defaults to 1. lin_geo_steps.csv and lin_geo_meta.csv match archived lin_geo_comparison files byte-for-byte; its original run script specifies the same parameters. |
| MC-sample calibration | Current and archived run_M_sweep.py specify the same extreme Rosenbrock parameters. M_sweep_meta.csv matches its archived source byte-for-byte. The displayed K=50 snapshot figure itself concerns the hierarchical mixture. |
| Normalized 2D benchmark | All 20 saved Rosenbrock cell configurations explicitly specify mu=1, a=0.05, b=5. All 20,000 aggregate Rosenbrock rows match the individual cell data. target_Z=1. |
| Unnormalized benchmark | All 20 saved cell configurations explicitly specify mu=1, a=0.05, b=5. All 20,000 aggregate rows match the cell data. target_Z=exp(-50). |
| Dimension benchmark | All 80 original cell configurations in Visualizations_old/_to_delete/Cholesky_overnight_20260914/dimension/cells specify mu=1, a=0.05, b=5. All 80,000 current aggregate rows match these cells. Current and original dimension_base.py construct HybridRosenbrockTarget with n1=2. target_Z=1. |
| Solver bias | solver_rosenbrock/config.json and solver.py specify a=0.05, b=5, with mu=1 by default. The result CSV matches the original Cholesky overnight experiment byte-for-byte. |
| MC variance | Current and original M_otr/run_study.py specify a=0.05, b=5, with mu=1 by default. variance_bias_results.csv matches the archived overnight5000 M_otr dataset byte-for-byte. Training metadata records log_Z=0. |
| Dimension variance | experiment.json explicitly records mu=1, a=0.05, b=5 and the shared-root structure. Current and original run_study.py use n1=2. dimension_comparison.csv and variance_results.csv match their original overnight5000 dimension datasets byte-for-byte. The original frozen hybrid target also defaults to mu=1. |
| Transport sensitivity and gradient tails | Current scripts build cfg_for('rosenbrock_extreme') from otr_study.py, which specifies a=0.05, b=5. replay_inputs.pt matches the original outlier_solver_replay_20260908_213151 input byte-for-byte; the original replay script also selects rosenbrock_extreme. |

All 16 numerical-experiments figures included from Thesis/figures match their
current Implementation/Visualizations counterparts byte-for-byte, including the
gallery, normalized/unnormalized comparisons, dimension comparison, calibration,
bias, variance, and transport-sensitivity figures.

## Density check

Numerically compared log_prob on 100 deterministic points in each dimension
n=2,5,10,20 against

    0.5*(log(a)+(n-1)*log(b)-n*log(pi))
    - a*(x1-mu)^2 - b*sum((xi-x1^2)^2, i=2,...,n).

HybridRosenbrockTarget matches at rtol=atol=1e-12. The 2D RosenbrockTarget
matches at rtol=atol=1e-7 (its Normal scale construction introduces an
approximately 1.6e-8 constant rounding difference in this check).

## Qualifications

- The generic RosenbrockTarget class in gmvi/targets/distributions.py still has
  defaults mu=1, a=0.2, b=2. Active Rosenbrock experiment scripts override a and b.
  An unrelated future call without parameters would therefore use the soft target.
- Old soft-target results remain in archive_before_extreme_integration_20261001
  and historical directories. They are not inputs to the current benchmark plots.
- Some older CSVs omit mu/a/b. Their parameter attribution relies on the matching
  original datasets, saved experiment metadata where available, and associated
  scripts; CSV measurements alone cannot prove which density generated them.
- This audit did not rerun long experiments, certify every historical file, or
  verify the bibliographic claims in the thesis paragraph.
