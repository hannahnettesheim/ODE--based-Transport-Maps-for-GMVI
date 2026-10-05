"""Recompute the gradient of sample 6000 in its original full reference batch."""

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
SOURCE = (
    HERE.parent / "6.5_Bias_and_Variance_at_Fixed_Parameters"
    / "transport_sensitivity"
)


def export_table(result):
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Accuracy check of the individual-sample ELBO gradient norm "
        r"for $S_0=Z_{6000}$ at the frozen Rosenbrock snapshot $\theta_{1000}$. "
        r"Each float64 DOPRI5 solve uses the same complete batch of "
        r"$M=16{,}384$ reference samples (draw 890, seed 2230889). "
        r"The norm includes all 30 parameter coordinates before division by $M$. "
        r"The final column compares full sample-gradient vectors against the "
        r"$10^{-15}$ result; it is a numerical comparison, not a certified error bound.}",
        r"\label{tab:sample-gradient-accuracy}",
        r"\begin{tabular}{rrr}",
        r"\hline",
        r"$\mathrm{rtol}=\mathrm{atol}$ & $\mathrm{gradnorm}(S_0)$ & Relative vector difference \\",
        r"\hline",
    ]
    for row in result.itertuples():
        exponent = int(round(np.log10(row.rtol)))
        difference = row.relative_vector_difference_to_tightest
        lines.append(
            f"$10^{{{exponent}}}$ & ${row.gradient_norm / 1e8:.8f}\\times10^8$"
            f" & ${difference:.6g}$" + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    (HERE / "sample_gradient_accuracy.tex").write_text("\n".join(lines) + "\n")


def main():
    spec = importlib.util.spec_from_file_location("study", SOURCE / "otr_study.py")
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    rows = []
    gradients = []
    with study.experiment_environment():
        torch.set_default_dtype(torch.float64)
        torch.set_num_threads(1)
        saved = torch.load(SOURCE / "replay_inputs.pt", weights_only=False)
        config = study.cfg_for("rosenbrock_extreme")
        model = study.build_model(config, 2, reseed=False).double()
        model.load_state_dict(saved["state"])
        samples = saved["x0"].double()
        assert samples.shape == (16384, 2)
        target = study.build_target(config)
        study.enable_checkpointing()
        for exponent in range(10, 16):
            tolerance = 10.0 ** (-exponent)
            estimator = study.make_estimator(
                "ode_transport", MC_samples=len(samples), path="linear",
                ode_solver="dopri5", rtol=tolerance, atol=tolerance,
            )
            estimator._ref_sample = lambda n, model: samples
            started = time.perf_counter()
            terms, info = estimator.loss_terms(model, target.log_prob)
            derivatives = torch.autograd.grad(terms[5999], tuple(model.parameters()))
            gradient = -torch.cat([value.reshape(-1) for value in derivatives])
            gradient = gradient.detach().numpy().copy()
            assert np.isfinite(gradient).all()
            gradients.append(gradient)
            row = dict(
                rtol=tolerance, atol=tolerance, dtype="float64", sample_number=6000,
                mc=len(samples), gradient_norm=float(np.linalg.norm(gradient)),
                negative_elbo_term=float(terms[5999].detach()), nfe=info.get("nfe"),
                elapsed_seconds=time.perf_counter() - started,
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
            pd.DataFrame(rows).to_csv(HERE / "sample_gradient_accuracy.csv", index=False)
            np.savez(HERE / "sample_gradient_accuracy.npz", gradients=np.stack(gradients),
                     tolerances=np.array([item["rtol"] for item in rows]))
            del terms, derivatives
        reference = np.load(SOURCE / "gradient_tail/native_reverse.npz")["s0_elbo_gradient"]
        np.testing.assert_allclose(gradients[0], reference, rtol=2e-7, atol=2e-5)
        result = pd.DataFrame(rows)
        result["relative_vector_difference_to_tightest"] = [
            np.linalg.norm(gradient - gradients[-1]) / np.linalg.norm(gradients[-1])
            for gradient in gradients
        ]
        result.to_csv(HERE / "sample_gradient_accuracy.csv", index=False)
        export_table(result)


if __name__ == "__main__":
    main()
