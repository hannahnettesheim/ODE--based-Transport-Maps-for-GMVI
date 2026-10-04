# Extreme Rosenbrock integration (2026-10-01)

The standard comparison_2d and logZ_minus50 workflows now use explicit Rosenbrock parameters mu=1, a=0.05, b=5. All other targets and training settings are unchanged. The normalized and unnormalized reruns (four estimators, five seeds, 1000 steps) have been imported into the standard cells layout from rerun_rosenbrock_20261001; that directory retains the frozen source and validation record.

plot.py merges completed cells into the existing aggregate by target tag, estimator and seed, preserving targets available only in the aggregate. Both standard figures are regenerated with the shared plot code. Earlier aggregates and figures are saved in archive_before_extreme_integration_20261001.

The general RosenbrockTarget defaults are unchanged; benchmark configurations specify the extreme parameters explicitly.
