# Extreme Rosenbrock rerun, 2026-10-01

Both experiments use mu=1, a=0.05, b=5 in two dimensions, as requested. The normalized experiment has log Z=0; the unnormalized experiment has log Z=-50. Only the Rosenbrock panel is rerun.

Four estimators (DM, SF without baseline, straight-through Gumbel-Softmax, OTR), seeds 1–5, K=5, M=8192, Cholesky chart, initial scale 0.5, 1000 Adam steps, learning rate 0.02 with cosine decay to 0.002, float32. OTR uses linear interpolation and DOPRI5 with rtol=atol=1e-5. ST uses temperature 1, anneal rate 0.9995, minimum 0.3. Two subprocess workers, each with one PyTorch thread.

The implementation is frozen in source/gmvi; source_hashes.json records SHA-256 hashes. Each cell saves the configuration, log, checkpoint states, step measurements, independent high-resolution checkpoint measurements, metadata and completion status. Previous experiment outputs are preserved.

Run: `/Users/Hannah/anaconda3/bin/python run.py`
Replot: `/Users/Hannah/anaconda3/bin/python run.py --plot`

Plots match the existing benchmark styling: serif fonts, DM/SF/ST/OTR colors and line styles, per-step medians and 10–90% quantiles over five seeds, centered 15-step median smoothing, symlog step axis (linear threshold 10), logarithmic vertical axis when positive. The normalized plot shows negative ELBO; the unnormalized plot shows log Z minus estimated ELBO. Figures are written as PDF and PNG. The ST arm remains straight-through; “soft” in the original request referred to the target parameters.
