# Frozen-parameter M sweep

{
  "statuses": [
    {
      "regime": "hier_k50",
      "exit_code": 0,
      "completed": true
    },
    {
      "regime": "rosenbrock_extreme",
      "exit_code": 0,
      "completed": true
    },
    {
      "regime": "banana",
      "exit_code": 0,
      "completed": true
    },
    {
      "regime": "hier_k5",
      "exit_code": 0,
      "completed": true
    },
    {
      "regime": "funnel",
      "exit_code": 0,
      "completed": true
    }
  ],
  "expected_cells": 60,
  "completed_cells": 60,
  "all_complete": true
}

See variance_bias_results.csv and variance_scaling_slopes.csv. The DM reference is unbiased in expectation but finite-sample: 20,000 samples per component, accumulated in 40 independent chunks. Bias is a noisy discrepancy estimate, with combined reference and OTR-mean RMS sampling error reported. No noise floor is a confidence bound. Variance intervals are approximate replicate bootstrap intervals. One training seed per regime limits generalization. Snapshots come from OTR at M=1024; all M values share the same frozen parameters.
