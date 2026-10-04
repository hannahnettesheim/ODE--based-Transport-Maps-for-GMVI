"""Shared figure sizing, estimator colors, labels, and export helpers."""

import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TEXTWIDTH_PT = 418.0
PT_PER_IN = 72.27
TEXTWIDTH_IN = TEXTWIDTH_PT / PT_PER_IN  # 5.784 in


def figsize(frac=1.0, aspect=0.62):
    """Width as a fraction of \\textwidth; height = width * aspect."""
    w = TEXTWIDTH_IN * frac
    return (w, w * aspect)


BG = "#ffffff"
SCHEME = {
    "score_function": "#E63946",
    "gumbel_softmax": "#457B9D",
    "ode_transport": "#2A9D8F",
    "target": "#264653",
    "samples": "#A8DADC",
}
# ColorBrewer RdBu-5, for heat maps / diverging quantities
DIVERGING = ["#2166AC", "#4393C3", "#878787", "#D6604D", "#B2182B"]

_T, _G, _S, _R = (
    SCHEME["target"],
    SCHEME["gumbel_softmax"],
    SCHEME["score_function"],
    SCHEME["ode_transport"],
)

# Shades of the OTR teal, so the variants read as one family but stay separable.
# Ordered dark -> light. #2A9D8F is the scheme colour and belongs to plain OTR;
# the clip that works (tau = n) gets the darkest shade so it reads as the result.
# Not darker than about #147, or it stops reading as teal and starts reading as
# the dark slate #264653 that DM already owns.
TEAL = {
    "dark": "#14746A",
    "mid": "#2A9D8F",  # the scheme colour
    "light": "#3FAEA1",
    "lighter": "#62C2B6",
    "pale": "#86D2C9",
    "palest": "#A8DADC",  # = SCHEME["samples"]
}

COLORS = {
    "DM_1024": _T,
    "DM_256": _T,
    "DM_clip": _T,
    "ST": _G,
    "SF": _S,
    "SF_clip": _S,
    # Plain OTR keeps the scheme colour. Among the interventions, shade tracks
    # how well the arm did (dark = recovered, pale = did not), which is the
    # same ordering as their final KL: 3.3, 6.4, 24, 40, 91.
    "OTR": TEAL["mid"],
    "OTR_clip_n": TEAL["dark"],
    "OTR_clip_2n": TEAL["light"],
    "OTR_clip": TEAL["lighter"],
    "OTR_dopri5": TEAL["pale"],
    "OTR_mc2048": TEAL["palest"],
    "OTR_geo": TEAL["lighter"],
}
MARKERS = {
    "DM_1024": "o",
    "DM_256": "s",
    "DM_clip": "P",
    "ST": "D",
    "SF": "^",
    "SF_clip": "v",
    "OTR": "o",
    "OTR_clip": "s",
    "OTR_clip_n": "h",
    "OTR_clip_2n": "p",
    "OTR_dopri5": "*",
    "OTR_mc2048": "<",
    "OTR_geo": "X",
}
# within a family, the dash pattern separates variants
DASHES = {
    "DM_1024": (None, None),
    "DM_256": (3, 1.4),
    "DM_clip": (1, 1.4),
    "ST": (None, None),
    "SF": (None, None),
    "SF_clip": (3, 1.4),
    "OTR": (None, None),
    "OTR_clip": (4, 1.5),
    "OTR_clip_n": (1, 1.2),
    "OTR_clip_2n": (5, 1.5, 1, 1.5),
    "OTR_dopri5": (2, 1.2),
    "OTR_mc2048": (6, 1.6),
    "OTR_geo": (1, 1.6, 4, 1.6),
}
# Labels must render under BOTH usetex and the mathtext fallback, so: no
# \textsc, no \%, and only math that mathtext also understands.
PRETTY = {
    "DM_1024": r"DM ($M=1024$)",
    "DM_256": r"DM ($M=256$)",
    "DM_clip": r"DM + clip",
    "ST": r"ST",
    "SF": r"SF",
    "SF_clip": r"SF + clip",
    "OTR": r"OTR",
    "OTR_clip": r"OTR + clip (adaptive)",
    "OTR_clip_n": r"OTR + clip $\tau=n$",
    "OTR_clip_2n": r"OTR + clip $\tau=2n$",
    "OTR_dopri5": r"OTR dopri5",
    "OTR_mc2048": r"OTR ($M=2048$)",
    "OTR_geo": r"OTR geometric",
}
# compact forms, for tick labels in a 418pt-wide figure
SHORT = {
    "DM_1024": r"DM",
    "DM_256": r"DM",
    "DM_clip": r"DM+clip",
    "ST": r"ST",
    "SF": r"SF",
    "SF_clip": r"SF+clip",
    "OTR": r"OTR",
    "OTR_clip": r"+clip adap.",
    "OTR_clip_n": r"+clip $n$",
    "OTR_clip_2n": r"+clip $2n$",
    "OTR_dopri5": r"dopri5",
    "OTR_mc2048": r"$M=2048$",
    "OTR_geo": r"OTR geo.",
}
ORDER = [
    "DM_1024",
    "DM_256",
    "DM_clip",
    "ST",
    "SF",
    "SF_clip",
    "OTR",
    "OTR_clip",
    "OTR_clip_2n",
    "OTR_clip_n",
    "OTR_dopri5",
    "OTR_mc2048",
    "OTR_geo",
]


def canon(label: str):
    # Idempotent: callers routinely pass a key that is already canonical
    # (df["k"] holds canon output, and style()/short() are then called on it).
    # Without this, "OTR_clip_n" re-enters the regex below, fails the \b after
    # "clip_n", and silently degrades to "OTR_clip" -- i.e. the fixed clip gets
    # drawn and labelled as the adaptive one.
    if label in COLORS:
        return label
    s = label.lower()
    if s in ("st", "dm", "sf", "otr"):
        return {"st": "ST", "dm": "DM_256", "sf": "SF", "otr": "OTR"}[s]
    if "exact marg" in s or s.startswith("dm"):
        if "clip" in s:
            return "DM_clip"
        return "DM_1024" if "1024" in s else "DM_256"
    if "score function" in s or s.startswith("sf"):
        return "SF_clip" if "clip" in s else "SF"
    if "straight" in s or s.startswith("st "):
        return "ST"
    if "ode" in s or "otr" in s:
        if "clip" in s:
            if re.search(r"clip\s*2n", s):
                return "OTR_clip_2n"
            if re.search(r"clip\s*n\b", s):
                return "OTR_clip_n"
            return "OTR_clip"
        if "dopri" in s:
            return "OTR_dopri5"
        if "2048" in s:
            return "OTR_mc2048"
        if "geometric" in s:
            return "OTR_geo"
        return "OTR"
    return None


def style(label):
    """dict of plot kwargs (color, marker, dashes, label) for a file label."""
    k = canon(label)
    if k is None:
        return dict(color="#999999", marker=".", label=label)
    d = dict(color=COLORS[k], marker=MARKERS[k], label=PRETTY[k])
    dash = DASHES[k]
    if dash[0] is not None:
        d["dashes"] = dash
    return d


def short(label):
    k = canon(label)
    return SHORT.get(k, label)


def sort_key(label):
    k = canon(label)
    return ORDER.index(k) if k in ORDER else len(ORDER)


def shared_legend(fig, ax, ncol=4, y=-0.02):
    """One legend under both panels. At 418pt a per-axes legend eats the plot."""
    h, l = ax.get_legend_handles_labels()
    fig.legend(
        h,
        l,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.4,
        handletextpad=0.5,
    )


def _probe(fmt):
    """Try one real render under usetex. Cached beside this file, because the
    probe costs about a second and every figure script calls setup()."""
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), f".texprobe_{fmt}")
    if os.path.exists(cache):
        return open(cache).read().strip() == "1"
    import io

    saved = dict(plt.rcParams)
    ok = True
    try:
        plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
        f = plt.figure(figsize=(1, 1))
        f.text(0.5, 0.5, r"$\kappa$ 1\%")
        f.savefig(io.BytesIO(), format=fmt)
        plt.close(f)
    except Exception as e:
        ok = False
        print(
            f"  [_style] usetex {fmt} render failed: "
            f"{str(e).strip().splitlines()[-1][:90]}"
        )
    finally:
        plt.rcParams.update(saved)
    with open(cache, "w") as fh:
        fh.write("1" if ok else "0")
    return ok


def _tex_works():
    """A `latex` on PATH is not enough. matplotlib's usetex preamble pulls in
    type1cm/type1ec (the cm-super package), which minimal TeX installs omit,
    and the PNG path additionally needs dvipng. The PDF is what goes in the
    thesis, so PDF is what decides; PNG is a preview and may be skipped."""
    from shutil import which

    if which("latex") is None:
        return False
    return _probe("pdf")


def setup(usetex=True):
    """Reset plotting typography and layout to Matplotlib's native defaults.

    ``usetex`` is retained for API compatibility with existing scripts but is
    intentionally ignored.  Project helpers such as :func:`figsize`, colours,
    target names, and :func:`save` remain available without imposing a global
    thesis font/style preset.
    """
    plt.rcdefaults()
    global PCT
    PCT = "%"
    return False


PCT = "%"


TARGET_NAMES = {
    "banana": "Banana",
    "rosenbrock": "Rosenbrock",
    "rosenbrock_extreme": "Rosenbrock",
}


def target_name(target, dim=None, short=False, k=None):
    """Return a publication-ready target name from an internal identifier.

    Recognised parameterised forms include ``hier_k5``,
    ``hierarchical_mixture`` (optionally with an explicit ``k``),
    and funnel identifiers such as ``funnel_dim10`` or ``neal_funnel_10``.
    Unknown names are made readable instead of being returned with underscores.
    """
    key = str(target).strip().lower()

    match = re.fullmatch(r"(?:hier|hierarchical(?:_mixture)?)_k?(\d+)", key)
    if match:
        k = int(match.group(1))
        prefix = "Hier. Mixture" if short else "Hierarchical Mixture"
        return rf"{prefix}, $k={k}$"
    if key in ("hier_mixture", "hierarchical_mixture"):
        prefix = "Hier. Mixture" if short else "Hierarchical Mixture"
        return prefix if k is None else rf"{prefix}, $k={int(k)}$"

    if "funnel" in key:
        match = re.search(r"(?:dim[_-]?)?(\d+)$", key)
        funnel_dim = int(match.group(1)) if match else dim
        return (
            "Neal's Funnel"
            if funnel_dim is None
            else rf"Neal's Funnel, $\mathrm{{dim}}={int(funnel_dim)}$"
        )

    return TARGET_NAMES.get(key, str(target).replace("_", " ").title())


def objective_label(target=None):
    """Objective label for plots of ``-elbo`` values.

    The implemented benchmark targets are normalised, so ``-ELBO`` is exactly
    ``KL(q || p)``.  Lotka--Volterra is only known up to its normalising
    constant and must therefore retain the honest ``-ELBO`` label.  Passing
    ``None`` is appropriate for a figure containing only normalised targets.
    """
    key = "" if target is None else str(target).lower().replace("-", "_")
    if "lotka" in key or key in ("lv", "lotka_volterra"):
        return r"$-\mathrm{ELBO}$"
    return r"$D_{\mathrm{KL}}(q\,\|\,p)$"


def _pdf_width_pt(path):
    import re

    m = re.search(rb"/MediaBox \[([^\]]*)\]", open(path, "rb").read())
    return float(m.group(1).split()[2]) if m else None


def save(fig, name, figdir, frac=1.0, tol=1.0, passes=5):
    """Write the figure at EXACTLY `frac` x \\textwidth, then include it with a
    bare \\includegraphics{...} and no width argument.

    `bbox_inches="tight"` trims whitespace, so the file that lands on disk is
    not the figsize that was asked for -- it came out 371-422pt for a 418pt
    request. Included at \\textwidth LaTeX then rescales it, and rescaling a
    figure rescales its type, which is the whole thing this module exists to
    prevent. So: write, read the MediaBox back, correct the canvas width, and
    repeat until the PDF measures 418pt to within half a point.

    PNG is a preview only. Under usetex it needs dvipng; if that is missing the
    PNG is skipped with a warning rather than silently."""
    os.makedirs(figdir, exist_ok=True)
    target = TEXTWIDTH_PT * frac
    pdf = os.path.join(figdir, f"{name}.pdf")

    for _ in range(passes):
        fig.savefig(pdf, bbox_inches="tight", facecolor=BG)
        got = _pdf_width_pt(pdf)
        if got is None or abs(got - target) < tol:
            break
        w, h = fig.get_size_inches()
        fig.set_size_inches(w + (target - got) / PT_PER_IN, h)
    got = _pdf_width_pt(pdf)
    flag = "" if got is None or abs(got - target) < tol else "   <-- OFF TARGET"
    out = [pdf]

    if not plt.rcParams["text.usetex"] or _probe("png"):
        png = os.path.join(figdir, f"{name}.png")
        fig.savefig(png, dpi=300, bbox_inches="tight", facecolor=BG)
        out.append(png)
    else:
        print("  [_style] no PNG preview (usetex needs dvipng); PDF is fine")

    plt.close(fig)
    print(
        f"  wrote {'  '.join(os.path.basename(p) for p in out)}"
        f"   [{got:.1f}pt / {target:.0f}pt]{flag}"
    )
    return out
