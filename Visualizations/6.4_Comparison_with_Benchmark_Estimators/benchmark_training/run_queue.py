"""Run the benchmark comparisons sequentially, then refresh their figures."""
from pathlib import Path
import json
import runpy
from worker import config, cell_path, main as run_cell
from gmvi.experiments.execution import experiment_environment
from gmvi.experiments.safe_io import atomic_json_dump

HERE = Path(__file__).resolve().parent
ESTIMATORS = [
    "exact_marginalization",
    "score_function",
    "gumbel_softmax",
    "ode_transport",
]
GROUPS = [
    ("comparison_2d", ["banana", "funnel", "rosenbrock", "hier_k5", "hier_k50"]),
    ("logZ_minus50", ["rosenbrock"]),
    ("dimension", ["n2", "n5", "n10", "n20"]),
]
JOBS = [
    (group, tag, estimator, seed)
    for seed in range(1, 6)
    for group, tags in GROUPS
    for estimator in ESTIMATORS
    for tag in tags
]


def launch(job):
    with experiment_environment():
        run_cell(job)
    result_path = cell_path(job) / "result.json"
    result = json.loads(result_path.read_text())
    return dict(job=job, exit_code=0 if result["completed"] else 1, result=result)


def main():
    from plot import plot

    results = []
    for job in JOBS:
        result = launch(job)
        results.append(result)
        print(json.dumps(result), flush=True)
        atomic_json_dump(
            dict(finished=len(results), total=len(JOBS), results=results),
            str(HERE / "progress.json"),
        )
        plot(job[0])
    solver_script = (
        HERE.parent.parent
        / "6.5_Bias_and_Variance_at_Fixed_Parameters"
        / "solver_bias"
        / "solver.py"
    )
    with experiment_environment():
        runpy.run_path(str(solver_script), run_name="__main__")
    atomic_json_dump(
        dict(
            training_finished=len(results),
            training_success=all((row["result"]["completed"] for row in results)),
            solver_exit_code=0,
        ),
        str(HERE / "completion.json"),
    )


if __name__ == "__main__":
    main()
