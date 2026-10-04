# Dimension comparison

All 36 settings are complete: n=5,10,20; snapshots 50,250,1000; DM, SF without baseline, ST, OTR; 5000 replicates each at M=1024. Parameter counts are 105,330,1155. OTR uses dopri5 rtol=atol=1e-7. ST temperatures follow the existing schedule at the snapshot and are frozen during measurement.

The target is the shared-root member of the hybrid Rosenbrock family (n1=2, n2=n-1), with mu=1, a=0.05, b=5. Five Cholesky components are trained by OTR at learning rate .005. Separate training trajectories supply the frozen parameters at each dimension, so changes reflect dimension together with the resulting fitted state, not dimension alone.

Variance bands are approximate 95% percentile intervals from 200 replicate-bootstrap resamples. Extreme OTR draws can make these intervals unstable, particularly at step 1000. Do not interpret the fitted slopes between three dimensions as a verified scaling law. Per-parameter variance adjusts for parameter count but is not invariant to parameterization.

Bias curves report discrepancy from a finite DM reference (20000 base samples per component, accumulated in 40 independent chunks). Dotted curves show combined RMS sampling error, not confidence bounds. DM at M=1024 uses 1024 base samples per component; this comparison is not at equal computational cost.
