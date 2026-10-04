from pathlib import Path
import importlib.util, json, os
import numpy as np, torch, pandas as pd

H = Path(__file__).resolve().parent
B = H.parent
P = H.parent
V = H
spec = importlib.util.spec_from_file_location("study", B / "otr_study.py")
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
torch.set_num_threads(1)
v = torch.load(P / "replay_inputs.pt")
x = v["x0"].double()
j = 5999
cfg = s.cfg_for("rosenbrock_extreme")
h = 1e-7
# A deterministic 5x5 local reference-space grid, centered on the original sample.
offsets = torch.linspace(-1e-8, 1e-8, 5, dtype=torch.float64)
a, b = torch.meshgrid(offsets, offsets, indexing="ij")
delta = torch.stack([a.flatten(), b.flatten()], dim=1)
order = torch.argsort(torch.linalg.vector_norm(delta, dim=1), stable=True)
delta = delta[order]
xs = x[j] + delta
dist = torch.linalg.vector_norm(delta, dim=1)
indices = torch.arange(25)
indices[0] = j
t = torch.linspace(0, 1, 301, dtype=torch.float64)


def model(delta):
    m = s.build_model(cfg, 2, reseed=False)
    m.load_state_dict(v["state"])
    m = m.double()
    with torch.no_grad():
        m.log_weights[2] += delta
    return m


data = np.load(H / "paths.npz")
paths = data["paths"]
assign = data["assignments"]
responsibilities = data["responsibilities"]
xs = torch.from_numpy(data["reference_samples"])
models = [model(-h), model(0.0), model(h)]
rows = pd.read_csv(H / "assignments.csv").to_dict("records")
os.environ["MPLCONFIGDIR"] = str(H / "mpl_cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8,
        "pdf.fonttype": 42,
    }
)
colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]
fig, axs = plt.subplots(2, 3, figsize=(10, 6.3))
with torch.no_grad():
    gx = np.linspace(-2.3, 2.3, 220)
    gy = np.linspace(-0.6, 4, 220)
    xx, yy = np.meshgrid(gx, gy)
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float64)
    m = models[1]
    lp = torch.stack(
        [c.component_log_prob(grid, m.reference.log_prob) for c in m.components], dim=1
    ) + torch.log(m.weights)
    density = torch.logsumexp(lp, dim=1).exp().numpy().reshape(xx.shape)
for a, title in enumerate([r"$\ell_3-h$", r"Unperturbed $\ell_3$", r"$\ell_3+h$"]):
    ax = axs[0, a]
    for k, r in enumerate(rows):
        c = colors[assign[a, k]]
        ax.scatter(
            *((xs[k] - xs[0]).numpy() / 1e-8),
            c=c,
            s=40 if k else 85,
            marker="o" if k else "*",
            edgecolors="black" if k == 0 else "none",
        )
        if k == 0:
            ax.annotate(
                "S0", (0.0, 0.0), xytext=(3, 3), textcoords="offset points", fontsize=6
            )
    ax.set_title(title)
    ax.set_xlabel(r"Reference offset $(x_1-x_{1,S0})/10^{-8}$")
    ax.set_ylabel(r"Reference offset $(x_2-x_{2,S0})/10^{-8}$")
    ax.set_aspect("equal")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.grid(alpha=0.15)
    ax = axs[1, a]
    ax.contour(xx, yy, density, levels=9, colors="0.83", linewidths=0.5)
    for ci, comp in enumerate(models[a].components):
        mean = comp.a.detach().numpy()
        ax.scatter(*mean, c=colors[ci], marker="x", s=40)
        ax.annotate(
            f"C{ci+1}",
            mean,
            xytext=(3, -9),
            textcoords="offset points",
            color=colors[ci],
            fontsize=7,
        )
    for k, r in enumerate(rows):
        z = paths[a, :, k]
        c = colors[assign[a, k]]
        ax.plot(
            z[:, 0],
            z[:, 1],
            color=c,
            alpha=1 if k == 0 else 0.65,
            lw=1.8 if k == 0 else 0.8,
            ls="--" if k == 0 else "-",
        )
        ax.scatter(
            *z[-1],
            c=c,
            s=65 if k == 0 else 20,
            marker="*" if k == 0 else "o",
            edgecolors="black" if k == 0 else "none",
            zorder=5,
        )
        if k == 0:
            ax.annotate(
                r["label"], z[-1], xytext=(3, 3), textcoords="offset points", fontsize=6
            )
    ax.set_xlim(-2.3, 2.3)
    ax.set_ylim(-0.6, 4)
    ax.set_aspect("equal")
    ax.set_xlabel("Transported coordinate 1")
    ax.set_ylabel("Transported coordinate 2")
for ax in axs.flat:
    ax.spines[["top", "right"]].set_visible(False)
handles = [
    Line2D([0], [0], color=c, marker="o", ls="", label=f"Component {i+1}")
    for i, c in enumerate(colors)
] + [Line2D([0], [0], color="black", marker="*", ls="--", label="Original sample S0")]
fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=7)
fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=2.0, w_pad=1.5)
for ext in ["pdf", "png"]:
    fig.savefig(H / f"grid_sample_paths.{ext}", dpi=220, bbox_inches="tight")
print(
    pd.DataFrame(rows)[
        [
            "label",
            "minus_component",
            "unperturbed_component",
            "plus_component",
            "minus_plus_displacement",
        ]
    ].to_string(index=False),
    flush=True,
)
