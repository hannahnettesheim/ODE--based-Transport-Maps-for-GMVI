"""Compare relaxed and hard Gumbel-Softmax sample locations in two dimensions."""
import sys as _sys, pathlib as _pl

_ROOT = next(
    _q for _q in _pl.Path(__file__).resolve().parents if (_q / "gmvi").is_dir()
)
_sys.path[:0] = [str(_ROOT), str(_ROOT / "Visualizations")]

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gmvi.utils.visualization import plot_gumbel_softmax_placement_2d

fig = plot_gumbel_softmax_placement_2d(
    temperatures=(1.0, 0.5, 0.1, 0.0),
    n_samples=100,
    seed=0,
)

out_path = os.path.join(os.path.dirname(__file__), "gumbel_softmax_placement_2d.png")
fig.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"Saved: {out_path}")
