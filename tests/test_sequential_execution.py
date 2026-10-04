"""Regression checks for sequential execution without regenerating thesis data."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
import torch

from gmvi.experiments.execution import experiment_environment, run_training_jobs
from gmvi.experiments.fixed_theta import gradient_draw
from gmvi.experiments.trainer import RunConfig, build_model, build_target
from gmvi.experiments.diagnostics import flat_grad
from gmvi.estimators.gradient_estimators import make_estimator
import gmvi.estimators.ode_transport as transport


def test_resume_retains_completed_jobs_after_interruption(tmp_path):
    calls = []

    def one_job(seed):
        calls.append(seed)
        if seed == 2:
            raise RuntimeError("interrupted")
        frame = pd.DataFrame({"seed": [seed], "value": [0.123456789123456]})
        return frame, frame, pd.DataFrame(), {"seed": seed}

    with pytest.raises(RuntimeError, match="interrupted"):
        run_training_jobs([(1,), (2,)], one_job, tmp_path, "example")
    assert calls == [1, 2]

    def finish(seed):
        calls.append(seed)
        frame = pd.DataFrame({"seed": [seed], "value": [0.5]})
        return frame, frame, pd.DataFrame(), {"seed": seed}

    run_training_jobs([(1,), (2,)], finish, tmp_path, "example")
    assert calls == [1, 2, 2]
    steps = pd.read_csv(tmp_path / "example_steps.csv")
    assert steps.seed.tolist() == [1, 2]
    assert steps.value.iloc[0] == pytest.approx(0.123456789123456, abs=1e-14)
    run_training_jobs([(1,), (2,)], finish, tmp_path, "example")
    assert calls == [1, 2, 2]
    pd.testing.assert_frame_equal(steps, pd.read_csv(tmp_path / "example_steps.csv"))


def test_environment_restored_on_failure():
    original_dtype = torch.get_default_dtype()
    original_forward = transport._VelocityFunc.forward
    original_threads = torch.get_num_threads()
    with pytest.raises(RuntimeError):
        with experiment_environment():
            torch.set_default_dtype(torch.float64)
            torch.set_num_threads(1)
            transport._VelocityFunc.forward = lambda *args: None
            raise RuntimeError("failed job")
    assert torch.get_default_dtype() == original_dtype
    assert torch.get_num_threads() == original_threads
    assert transport._VelocityFunc.forward is original_forward


@pytest.mark.parametrize(
    "estimator,kwargs",
    [
        ("exact_marginalization", {}),
        ("score_function", {"baseline": "none"}),
        (
            "gumbel_softmax",
            {
                "mode": "straight_through",
                "temperature": 0.7,
                "anneal_rate": 1.0,
                "min_temperature": 0.7,
            },
        ),
        ("ode_transport", {"path": "linear", "ode_solver": "rk4", "ode_steps": 4}),
    ],
)
def test_gradient_draw_preserves_seeded_estimator_gradient(estimator, kwargs):
    with experiment_environment():
        torch.set_num_threads(1)
        cfg = RunConfig(target="rosenbrock", dim=2, components=2, chart="cholesky")
        model = build_model(cfg, 2)
        target = build_target(cfg)
        state = {key: value.clone() for key, value in model.state_dict().items()}
        torch.manual_seed(1234)
        model.zero_grad(set_to_none=False)
        loss, old_info = make_estimator(estimator, MC_samples=16, **kwargs).loss(
            model, target.log_prob
        )
        loss.backward()
        expected = flat_grad(model).detach().numpy().astype(np.float64, copy=True)
        actual, info = gradient_draw(
            model, target.log_prob, estimator, 16, 1234, kwargs, dtype=np.float64
        )
        np.testing.assert_array_equal(actual, expected)
        assert info.get("nfe") == old_info.get("nfe")
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)


def test_changed_config_does_not_reuse_cached_results(tmp_path):
    from dataclasses import replace

    cfg = RunConfig(target="rosenbrock", dim=2)

    def job(seed):
        frame = pd.DataFrame({"seed": [seed]})
        return frame, frame, frame, {"seed": seed}

    run_training_jobs([(1,)], job, tmp_path, "configured", config_for=lambda seed: cfg)
    with pytest.raises(ValueError, match="configuration changed"):
        run_training_jobs(
            [(1,)],
            job,
            tmp_path,
            "configured",
            config_for=lambda seed: replace(cfg, lr=cfg.lr * 2),
        )


@pytest.mark.parametrize(
    "relative_path",
    [
        "variance_dimension/run_study.py",
        "variance_mc_sample_size/run_study.py",
        "transport_sensitivity/otr_study.py",
    ],
)
def test_study_launcher_visits_its_own_regimes(monkeypatch, relative_path):
    import importlib.util

    path = (
        Path(__file__).resolve().parents[1]
        / "Visualizations"
        / "6.5_Bias_and_Variance_at_Fixed_Parameters"
        / relative_path
    )
    spec = importlib.util.spec_from_file_location("study_test", path)
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    seen = []
    summaries = []

    def launch(regime):
        assert regime in study.REGIMES
        assert regime not in seen
        seen.append(regime)
        return {"regime": regime, "completed": True}

    monkeypatch.setattr(study, "launch", launch)
    monkeypatch.setattr(study, "summarize", summaries.append)
    monkeypatch.setattr(sys, "argv", [str(path)])
    study.main()
    assert set(seen) == set(study.REGIMES)
    assert len(summaries) == 1
    assert [row["regime"] for row in summaries[0]] == seen
