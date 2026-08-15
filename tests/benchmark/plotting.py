"""Plotting for the benchmark suite.

The figures on the docs' benchmarks page are all the same two-panel shape --
runtime above, memory below, shared log-log x axis -- differing only in which
curves appear, the asymptotic-scaling annotations, and the axis limits. Those
differences live in :data:`SPECS` as data, so a whole figure is one call:

    fig, axes = make_benchmark_figure("llh", cpu, gpu_data=gpu)

and regenerating everything that has fresh results is one loop (see
``run_benchmark.py --plot``).

Previously this lived in ``plots-benchmark.ipynb`` as copy-pasted cells; the
notebook is kept for exploratory work, but the deployed figures should be
produced from here so they stay consistent.
"""

from __future__ import annotations

import os

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------
# Curve styling. One entry per benchmarked implementation:
#   SSM/QSM/GP    -- smolgp state space, tinygp quasisep, tinygp dense (CPU)
#   pSSM/pQSM     -- the parallel (associative-scan) solvers, run on GPU
# --------------------------------------------------------------------------
colors = {
    "SSM": "#1f77b4",
    "QSM": "#ff7f0e",
    "GP": "#2ca02c",
    "pQSM": "#d62728",
    "pSSM": "#6A0E95",
}
markers = {"SSM": "o", "QSM": "s", "GP": "D", "pQSM": "v", "pSSM": "*"}
markersize = {"SSM": 8, "QSM": 6, "GP": 6, "pQSM": 8, "pSSM": 10}

#: Which curves come from the GPU run rather than the CPU run.
GPU_CURVES = ("pSSM", "pQSM")

#: Default label per curve when both machines are shown.
MACHINE_LABELS = {
    "SSM": "SSM (CPU)",
    "QSM": "QSM (CPU)",
    "GP": "GP (CPU)",
    "pSSM": "SSM (GPU)",
    "pQSM": "QSM (GPU)",
}

#: Where the docs look for these figures (docs/_static/benchmarks/).
STATIC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "docs",
    "_static",
    "benchmarks",
)


def use_paper_style(usetex: bool = True) -> None:
    """Match the styling of the deployed figures.

    ``usetex=False`` falls back to mathtext, for machines without a LaTeX
    install (the labels still render, just less prettily).
    """
    if usetex:
        mpl.rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
    else:
        mpl.rc("font", family="serif")
    mpl.rc("text", usetex=usetex)
    mpl.rcParams["axes.formatter.useoffset"] = False
    mpl.rcParams["lines.markeredgewidth"] = 1.5
    mpl.rcParams["lines.markersize"] = 8
    mpl.rcParams["lines.markeredgecolor"] = "k"
    mpl.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def symlogticks(vmin, vmax, linthresh=1e-16, spacing=1):
    """Generate ticks for a symlog axis."""
    thresh = int(jnp.log10(linthresh))
    negticks = -(10.0 ** np.arange(thresh, vmin, spacing)[::-1])
    posticks = 10.0 ** jnp.arange(thresh, vmax, spacing)
    return jnp.concatenate([negticks, posticks])


def scale_nans(runtime_array, Ns, power=1):
    """Extrapolate past the last measured point, for curves that were cut off.

    A run that exceeded its memory/time cutoff leaves NaNs; those are filled in
    by continuing the known asymptotic power law from the last valid point, and
    drawn dotted, so the reader can see where a method *would* have gone.
    """
    last_valid_idx = jnp.where(~jnp.isnan(runtime_array[:, 0]))[0][-1]
    last_valid_runtime = runtime_array[last_valid_idx, 0]
    last_valid_N = Ns[last_valid_idx]

    for i in range(last_valid_idx + 1, len(runtime_array)):
        runtime_array = runtime_array.at[i, 0].set(
            last_valid_runtime * (Ns[i] / last_valid_N) ** power
        )
        runtime_array = runtime_array.at[i, 1].set(0)  # std is meaningless here
    return runtime_array


def scale_ext(Next, Ns, runtime_arr, ref=-1, power=1):
    """The same power-law continuation, evaluated on an arbitrary grid
    ``Next`` -- used to draw a guide line beyond the plotted range."""
    runtimes = np.array([rt[0] for rt in runtime_arr])
    cutnan = ~np.isnan(runtimes)
    runtimes = runtimes[cutnan]
    last_runtime = runtimes[ref]
    last_N = np.array(Ns)[cutnan][ref]
    return last_runtime * (Next / last_N) ** power


def hide_yticklabels(ax, which="both"):
    labels = ax.get_yticklabels()
    for i, lbl in enumerate(labels):
        if (which == "even" and i % 2 == 0) or (which == "odd" and i % 2 == 1):
            lbl.set_visible(False)


def merge_machines(cpu_data, gpu_data=None, curves=GPU_CURVES):
    """Splice the GPU-only curves into the CPU results.

    The two machines are benchmarked separately, so each result file holds only
    the curves that machine ran. Missing curves are skipped rather than raising,
    so a figure can be regenerated before every run has finished.
    """
    merged = {
        key: dict(cpu_data[key]) if isinstance(cpu_data[key], dict) else cpu_data[key]
        for key in cpu_data
    }
    if gpu_data is None:
        return merged
    for field in ("runtime", "memory", "outputs"):
        if field not in merged or field not in gpu_data:
            continue
        for name in curves:
            if name in gpu_data[field]:
                merged[field][name] = gpu_data[field][name]
    return merged


# --------------------------------------------------------------------------
# The two-panel figure
# --------------------------------------------------------------------------
def plot_benchmark(
    Ns,
    runtimes,
    ax=None,
    savefig=None,
    scale=True,
    powers=None,
    labels=None,
    xlabel="Number of data points",
    ylabel="Runtime [s]",
):
    """One panel: every curve in ``runtimes`` against ``Ns``, log-log.

    ``runtimes`` maps curve name -> sequence of ``(mean, std)`` per N.
    """
    if powers is None:
        powers = {"SSM": 1, "QSM": 1, "GP": 3, "pQSM": 1, "pSSM": 1}
    if labels is None:
        labels = {name: name for name in runtimes}

    if ax is None:
        _fig, ax = plt.subplots(1, 1, figsize=(6, 6), sharex=True)

    for name in sorted(runtimes):
        runtime_array = jnp.array(runtimes[name])
        mean_runtime = runtime_array[:, 0]
        std_runtime = runtime_array[:, 1]
        if scale:
            scaled = scale_nans(runtime_array, Ns, power=powers.get(name, 1))
            ax.errorbar(Ns, scaled[:, 0], scaled[:, 1], c=colors[name], ls=":")
        ax.errorbar(
            Ns,
            mean_runtime,
            std_runtime,
            c=colors[name],
            marker=markers[name],
            markersize=markersize[name],
            fmt="-",
            label=labels.get(name, name),
        )

    ax.legend()
    ax.set(xscale="log", yscale="log", xlabel=xlabel, ylabel=ylabel)
    tens = 10 ** jnp.arange(jnp.log10(Ns[0]), jnp.log10(Ns[-1]) + 1)
    ax.set_xticks(tens, labels=["" for _ in tens], minor=True)
    ax.grid(alpha=0.5, zorder=-10)
    ax.grid(alpha=0.5, zorder=-10, which="minor", axis="x")

    if savefig:
        plt.savefig(savefig, dpi=300, bbox_inches="tight")
    return ax


def _decade_locators(ax):
    ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_minor_locator(
        mpl.ticker.LogLocator(base=10.0, subs=jnp.arange(1, 10) * 0.1, numticks=100)
    )
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())


def benchmark_plot(
    Ns,
    runtime,
    memory,
    labels=None,
    title=None,
    savefig=None,
    xlabel="Number of data points",
    **kwargs,
):
    """Runtime (top) over memory (bottom), sharing a log x axis."""
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(5, 6.6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5], "hspace": 0.05},
    )

    ax1 = plot_benchmark(Ns, runtime, labels=labels, ax=ax1, **kwargs)
    ax1.legend(loc="upper left", title=title, fontsize=15)
    _decade_locators(ax1)
    ax1.set_xlabel("")

    ax2 = plot_benchmark(
        Ns,
        memory,
        ax=ax2,
        scale=False,
        labels=labels,
        xlabel=xlabel,
        ylabel="Memory",
    )
    ax2.get_legend().remove()
    _decade_locators(ax2)
    ax2.set_yticks(
        [1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12],
        labels=["1 MB", "", "", "1 GB", "", "", "1 TB"],
    )
    ax2.grid(lw=0.5, zorder=-1, which="major")
    ax2.set_xlim(left=Ns[0], right=Ns[-1])

    if savefig:
        plt.savefig(savefig, dpi=300, bbox_inches="tight")
    return fig, (ax1, ax2)


# --------------------------------------------------------------------------
# Per-figure specifications
#
# Everything that distinguishes one benchmarks-page figure from another lives
# here as data: the legend title, the asymptotic power law each curve follows
# (used to extrapolate past a cutoff), the "$N^3$"-style annotations, and the
# axis limits. Keyed by (kind, integrated).
#
# annotations: (text, (x, y), curve) -- placed in data coordinates, coloured
# to match the curve they label.
# guides: (curve, log10-range, power, ref-index) -- dotted power-law guide
# lines drawn on the memory panel beyond the measured points.
# --------------------------------------------------------------------------
SPECS = {
    ("llh", False): {
        "title": "Likelihood",
        "powers": {"SSM": 1, "QSM": 1, "GP": 3, "pSSM": 1, "pQSM": 1},
        "ylim1": (5e-5, 1e6),
        "ylim2": (None, 1e12),
        "annotations1": [
            ("$N^3$", (3e5, 4e3), "GP"),
            ("$N$", (3e6, 5e1), "SSM"),
            ("$N$", (2e6, 7e-2), "QSM"),
            (r"$\sim$$N/T$", (8e4, 3e-3), "pSSM"),
        ],
        "annotations2": [
            ("$N$", (3e6, 6e7), "SSM"),
            ("$N$", (2e4, 1e8), "QSM"),
            ("$N^2$", (1.2e5, 2e11), "GP"),
        ],
        "guides": [("GP", (4, 8), 2, -1)],
    },
    ("llh", True): {
        "title": "Likelihood",
        "powers": {"SSM": 1, "GP": 3, "pSSM": 1},
        "ylim1": (5e-5, 1e6),
        "ylim2": (None, 1e12),
        "annotations1": [
            ("$N^3$", (1.2e5, 2e5), "GP"),
            ("$N$", (3e6, 1e3), "SSM"),
            (r"$\sim$$N/T$", (9e5, 9e-1), "pSSM"),
        ],
        "annotations2": [
            ("$N$", (3e6, 4e8), "SSM"),
            ("$N$", (3e6, 6e10), "pSSM"),
            ("$N^2$", (2e4, 4e11), "GP"),
        ],
        "guides": [("GP", (4, 8), 2, -1), ("pSSM", (6, 8), 1, -1)],
    },
    ("cond", False): {
        "title": "Conditioning",
        "powers": {"SSM": 1, "QSM": 1, "GP": 3, "pSSM": 1, "pQSM": 1},
        "ylim1": (5e-5, 1e6),
        "ylim2": (1e5, 1e12),
        "annotations1": [
            ("$N^3$", (3e5, 1e4), "GP"),
            ("$N$", (3e6, 1e2), "SSM"),
            (r"$\sim$$N/T$", (8e5, 1e-1), "pSSM"),
            ("$N$", (2e5, 3e0), "QSM"),
        ],
        "annotations2": [
            ("$N$", (3e6, 3e8), "SSM"),
            ("$N$", (2e4, 2e8), "QSM"),
            ("$N$", (5e6, 1e11), "pSSM"),
            ("$N^2$", (1.2e5, 2e11), "GP"),
        ],
        "guides": [("GP", (4.3, 8), 2, 5)],
    },
    ("cond", True): {
        "title": "Conditioning",
        "powers": {"SSM": 1, "GP": 3, "pSSM": 1},
        "ylim1": (5e-5, 1e6),
        "ylim2": (1e5, 1e12),
        "annotations1": [
            ("$N^3$", (3e5, 3e4), "GP"),
            ("$N$", (3e6, 1e3), "SSM"),
            (r"$\sim$$N/T$", (9e5, 8e-1), "pSSM"),
        ],
        "annotations2": [
            ("$N$", (1e6, 6e11), "pSSM"),
            ("$N$", (2e6, 6e8), "SSM"),
            ("$N^2$", (8e3, 5e11), "GP"),
        ],
        "guides": [("GP", (4.3, 8), 2, 5), ("pSSM", (5.8, 8), 1, -1)],
    },
    ("pred", False): {
        "title": "Prediction",
        "powers": {"SSM": 1, "QSM": 2, "GP": 3},
        "secondary_axis": ("Number of prediction points", 100),
        "ylim1": (5e-5, 1e6),
        "ylim2": (None, 1e12),
        "xlim": (10, 1e7),
        "annotations1": [
            ("$N+M$", (4e5, 5e1), "SSM"),
            ("$NM$", (1e5, 1e4), "QSM"),
            ("$N^2M$", (1e4, 1e5), "GP"),
        ],
        "annotations2": [
            ("$N$", (5e5, 1e10), "SSM"),
            ("$NM$", (4e2, 5e11), "QSM"),
            ("$NM$", (1e4, 1e11), "GP"),
        ],
        "guides": [
            ("SSM", (4, 8), 1, -2),
            ("GP", (4, 8), 2, -2),
            ("QSM", (4, 8), 2, -2),
        ],
    },
    ("pred", True): {
        "title": "Prediction",
        "powers": {"SSM": 1, "GP": 3},
        "secondary_axis": ("Number of prediction points", 100),
        "ylim1": (5e-5, 1e6),
        "ylim2": (None, 1e12),
        "xlim": (10, 1e7),
        "annotations1": [("$N+M$", (4e5, 4e2), "SSM"), ("$N^2M$", (5e3, 1e5), "GP")],
        "annotations2": [("$N+M$", (1e5, 1e11), "SSM"), ("$NM$", (8e2, 5e11), "GP")],
        "guides": [("SSM", (4, 8), 1, -2), ("GP", (4, 8), 2, -2)],
    },
    # --- sampling -----------------------------------------------------------
    # Prior draws have no training data, so the only size parameter is M, the
    # number of coordinates the realization is drawn at.
    ("sample_prior", False): {
        "title": "Prior sample",
        "powers": {"SSM": 1, "QSM": 1, "GP": 3},
        "xlabel": "Number of sample points",
        "ylim1": (5e-5, 1e6),
        "ylim2": (None, 1e12),
        "annotations1": [
            ("$M$", (3e6, 5e1), "SSM"),
            ("$M$", (2e6, 7e-2), "QSM"),
            ("$M^3$", (3e5, 4e3), "GP"),
        ],
        "annotations2": [("$M$", (3e6, 6e7), "SSM"), ("$M^2$", (1.2e5, 2e11), "GP")],
        "guides": [("GP", (4, 8), 2, -1)],
    },
    ("sample_prior", True): {
        "title": "Prior sample",
        "powers": {"SSM": 1, "GP": 3},
        "xlabel": "Number of sample points",
        "ylim1": (5e-5, 1e6),
        "ylim2": (None, 1e12),
        "annotations1": [("$M$", (3e6, 1e3), "SSM"), ("$M^3$", (1.2e5, 2e5), "GP")],
        "annotations2": [("$M$", (3e6, 4e8), "SSM"), ("$M^2$", (2e4, 4e11), "GP")],
        "guides": [("GP", (4, 8), 2, -1)],
    },
    # Posterior draws mirror `pred`: N training points, M = 100N test points.
    ("sample_post", False): {
        "title": "Posterior sample",
        "powers": {"SSM": 1, "QSM": 2, "GP": 3},
        "secondary_axis": ("Number of sample points", 100),
        "ylim1": (5e-5, 1e6),
        "ylim2": (None, 1e12),
        "xlim": (10, 1e7),
        "annotations1": [
            ("$N+M$", (4e5, 5e1), "SSM"),
            ("$NM$", (1e5, 1e4), "QSM"),
            ("$N^2M$", (1e4, 1e5), "GP"),
        ],
        "annotations2": [("$N$", (5e5, 1e10), "SSM"), ("$NM$", (1e4, 1e11), "GP")],
        "guides": [("SSM", (4, 8), 1, -2), ("GP", (4, 8), 2, -2)],
    },
    ("sample_post", True): {
        "title": "Posterior sample",
        "powers": {"SSM": 1, "GP": 3},
        "secondary_axis": ("Number of sample points", 100),
        "ylim1": (5e-5, 1e6),
        "ylim2": (None, 1e12),
        "xlim": (10, 1e7),
        "annotations1": [("$N+M$", (4e5, 4e2), "SSM"), ("$N^2M$", (5e3, 1e5), "GP")],
        "annotations2": [("$N+M$", (1e5, 1e11), "SSM"), ("$NM$", (8e2, 5e11), "GP")],
        "guides": [("SSM", (4, 8), 1, -2), ("GP", (4, 8), 2, -2)],
    },
}


def make_benchmark_figure(
    kind, cpu_data, gpu_data=None, integrated=False, savefig=None, annotate=True
):
    """Build one benchmarks-page figure from its spec.

    Args:
        kind: one of ``llh``, ``cond``, ``pred``, ``sample_prior``, ``sample_post``.
        cpu_data: dict from :func:`~benchmark.load_benchmark_data`.
        gpu_data: optional second dict; its ``pSSM``/``pQSM`` curves are spliced
            in. Curves absent from it are simply omitted, so a figure can be
            regenerated before the GPU run exists.
        integrated: select the integrated variant of the spec.
        savefig: path to write; ``True`` writes the canonical
            ``docs/_static/<kind><_int>_benchmark.png``.

    Returns:
        ``(fig, (ax_runtime, ax_memory))``.
    """
    spec = SPECS[(kind, integrated)]
    data = merge_machines(cpu_data, gpu_data)
    Ns = data["Ns"]
    runtime, memory = data["runtime"], data["memory"]

    # Only plot curves we actually have results for.
    present = [n for n in runtime if n in colors]
    runtime = {n: runtime[n] for n in present}
    memory = {n: memory[n] for n in present if n in memory}
    Ns = Ns[: len(next(iter(runtime.values())))]

    if savefig is True:
        savefig = os.path.join(
            STATIC_DIR, f"{kind}{'_int' if integrated else ''}_benchmark.png"
        )

    fig, (ax1, ax2) = benchmark_plot(
        Ns,
        runtime,
        memory,
        labels={n: MACHINE_LABELS.get(n, n) for n in present},
        title=spec["title"],
        powers=spec["powers"],
        xlabel=spec.get("xlabel", "Number of data points"),
    )

    if "secondary_axis" in spec:
        label, factor = spec["secondary_axis"]
        ax_top = ax1.secondary_xaxis(
            "top", functions=(lambda x: factor * x, lambda x: x / factor)
        )
        ax_top.set_xlabel(label)

    if annotate:
        for ax, key in ((ax1, "annotations1"), (ax2, "annotations2")):
            for text, xy, curve in spec.get(key, []):
                if curve not in present:
                    continue
                ax.annotate(
                    text,
                    xy=xy,
                    color=colors[curve],
                    ha="left",
                    va="top",
                    xycoords="data",
                    fontsize=16,
                )

    for curve, (lo, hi), power, ref in spec.get("guides", []):
        if curve not in memory:
            continue
        Next = jnp.logspace(lo, hi, 10)
        ax2.plot(
            Next,
            scale_ext(Next, Ns, memory[curve], ref=ref, power=power),
            ls=":",
            color=colors[curve],
        )

    if spec.get("ylim1"):
        ax1.set_ylim(*spec["ylim1"])
    if spec.get("ylim2"):
        ax2.set_ylim(*spec["ylim2"])
    if spec.get("xlim"):
        ax2.set_xlim(*spec["xlim"])
    ax1.grid(lw=0.5, zorder=-1, which="major")

    if savefig:
        os.makedirs(os.path.dirname(savefig), exist_ok=True)
        plt.savefig(savefig, dpi=300, bbox_inches="tight")
        print(f"  wrote {savefig}")
    return fig, (ax1, ax2)
