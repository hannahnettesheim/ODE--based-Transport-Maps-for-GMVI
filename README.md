# Variational Inference with Affine Mixture Distributions

Implementation accompanying the master's thesis **Going with the Flow: ODE-based Transport Maps for Reparameterization Gradients in Variational Inference with Affine Mixture Distributions**.

The code fits mixture distributions of affiely transformed reference distributions and compares four main gradient estimators:

- **DM:** direct Monte Carlo estimation with exact marginalization over the mixture components.
- **SF:** the score-function estimator.
- **ST:** the straight-through Gumbel–Softmax estimator.
- **OTR:** reparameterization through an ODE-based transport map.

The repository includes the model and estimator implementations, training and fixed-parameter measurement routines, and scripts and saved data for the thesis figures.

## Repository structure

```text
Implementation/
├── gmvi/
│   ├── choices.py       Configuration options and their documentation
│   ├── models/          Affine components, mixtures, and reference distributions
│   ├── targets/         Target probability distributions
│   ├── estimators/      Gradient estimators and ODE transport
│   ├── experiments/     Training, fixed-parameter diagnostics, and result handling
│   └── utils/           Visualization helpers
├── Visualizations/      Experiments, saved results, and plots by thesis section
└── tests/               Regression tests for sequential experiment execution
```

See [Visualizations/README.md](Visualizations/README.md) for the mapping between thesis figures, experiment scripts, and plotting scripts. Supported configuration options are documented in [gmvi/choices.py](gmvi/choices.py).

## Installation

Run the commands below from this directory: the folder containing this README, `gmvi/`, and `Visualizations/`. The package is imported directly from the working directory; an editable installation is not required.


The follwing versions of packages are needed: 

```bash
python -m pip install \
    torch==2.0.1 \
    torchdiffeq==0.2.5 \
    numpy==1.25.2 \
    pandas==1.5.3 \
    scipy==1.16.3 \
    matplotlib==3.10.8
```

For the regression tests, also install `pytest`:

```bash
python -m pip install pytest==9.0.2
```

## Minimal example

Save the following as `minimal_example.py` next to this README, then run `python minimal_example.py` from the same directory.

This example fits a five-component mixture to the two-dimensional Banana target using ODE transport. It performs 50 Adam steps and evaluates the ELBO and KL divergence at steps 0, 25, and 50. These reduced settings demonstrate the API; they are not the full thesis experiment settings.

```python
import matplotlib.pyplot as plt
import torch
from gmvi.experiments.trainer import RunConfig, run


torch.set_num_threads(1)

def plot_elbo_decay(result):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(result.steps["step"], -result.steps["elbo"], color="#2A9D8F")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Negative ELBO")
    ax.set_title("Banana target — ODE transport")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig("example_elbo_decay.png", dpi=200)
    plt.show()
    plt.close(fig)


cfg = RunConfig(
    target="banana",
    dim=2,
    components=5,
    chart="cholesky",
    estimator="ode_transport",
    estimator_kwargs={
        "path": "linear",
        "ode_solver": "dopri5",
        "rtol": 1e-5,
        "atol": 1e-5,
    },
    mc_samples=128,
    n_steps=50,
    lr=0.01,
    scheduler="none",
    checkpoint_fracs=[0.0, 0.5, 1.0],
    n_eval=2048,
    probe_geometry=False,
    seed=1,
)

result = run(cfg)

print(result.checkpoints[["step", "elbo", "kl"]])
print("Completed steps:", result.meta["steps_completed"])
plot_elbo_decay(result)
```

The output contains three checkpoint rows and reports `Completed steps: 50`. The Banana target is normalized, so its estimated KL divergence equals the negative estimated ELBO. Values are Monte Carlo estimates and may vary across environments.

`run` returns a `RunResult` containing:

- `steps`: a pandas DataFrame with measurements recorded during training.
- `checkpoints`: a pandas DataFrame with diagnostics at the selected checkpoints.
- `snapshots`: saved model parameter states indexed by optimization step.
- `meta`: run metadata, including completion and divergence information.

The example keeps its results in memory. To save the tables, append:

```python
result.steps.to_csv("example_steps.csv", index=False)
result.checkpoints.to_csv("example_checkpoints.csv", index=False)
```
