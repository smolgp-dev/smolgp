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
    """Match the styling of the paper figures.

    ``usetex=False`` falls back to mathtext, for machines without a LaTeX
    install (the labels still render, just less prettily).
    """
    if usetex:
        mpl.rc("font", **{"family": "serif", "serif": ["Computer Modern"], "size": 16})
    else:
        mpl.rc("font", family="serif", size=16)
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
    show_errors=False,
    hollow=None,
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

    # Extrapolation is drawn later, by make_benchmark_figure, once the axis
    # limits are final -- the dotted lines have to reach the right edge, and
    # that edge is not known until xlim is set. The legend is built there too,
    # so it can pair CPU and GPU entries across both panels.
    #
    # Line and markers are drawn separately so that individual points can be
    # suppressed: the memory panel replaces unresolvable measurements with
    # theory and redraws those as hollow markers, which only works if the
    # filled ones are not already sitting underneath.
    for name in sorted(runtimes):
        arr = jnp.array(runtimes[name])
        # Result files can be ragged: an interrupted sweep, or one machine's
        # curves spliced into another's, leaves a curve shorter or longer than
        # Ns. Plot the overlap rather than refusing to draw the figure.
        n = min(len(Ns), arr.shape[0])
        x = np.asarray(Ns)[:n]
        y = np.asarray(arr[:n, 0])
        ax.plot(x, y, c=colors[name], ls="-", label=labels.get(name, name))
        if show_errors:
            ax.errorbar(x, y, np.asarray(arr[:n, 1]), c=colors[name], fmt="none")
        hide = np.zeros(n, dtype=bool)
        if hollow is not None and name in hollow:
            hide = np.asarray(hollow[name])[:n]
        ax.plot(
            x[~hide],
            y[~hide],
            ls="none",
            marker=markers[name],
            markersize=markersize[name],
            c=colors[name],
        )

    ax.set(xscale="log", yscale="log", xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=0.5, zorder=-10)
    ax.grid(alpha=0.5, zorder=-10, which="minor", axis="x")

    if savefig:
        plt.savefig(savefig, dpi=300, bbox_inches="tight")
    return ax



# --------------------------------------------------------------------------
# Curve geometry: measured points, power-law continuation, and label placing
# --------------------------------------------------------------------------
CPU_ORDER = ("GP", "QSM", "SSM")
#: CPU curve -> the GPU curve that is the same method on the other machine.
GPU_PARTNER = {"SSM": "pSSM", "QSM": "pQSM"}


def measured_points(Ns, series):
    """The ``(x, y)`` a curve actually has data for: finite and positive.

    Zeros are dropped alongside NaNs -- a zero came from a memory measurement
    below what RSS can resolve, and it has no place on a log axis.
    """
    x = np.asarray(Ns, dtype=float)
    y = np.array([p[0] for p in series], dtype=float)
    # A curve spliced in from the other machine's result file can be a
    # different length from the CPU curves that set `Ns`, so line them up on
    # whichever is shorter rather than letting the mask blow up.
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    ok = np.isfinite(y) & (y > 0)
    return x[ok], y[ok]


def continue_curve(Ns, series, power, xmax, n=64):
    """Measured points plus a power-law continuation out to ``xmax``.

    Returns ``(x, y, n_measured)``. Curves are cut off at different sizes --
    the dense GP runs out of memory decades before the state-space solvers run
    out of patience -- so without this each dotted line stopped wherever its
    curve happened to end, at a different x for every method.
    """
    x, y = measured_points(Ns, series)
    if len(x) == 0:
        return None
    n_meas = len(x)
    if x[-1] < xmax:
        xe = np.logspace(np.log10(x[-1]), np.log10(xmax), n)[1:]
        x = np.concatenate([x, xe])
        y = np.concatenate([y, y[n_meas - 1] * (xe / x[n_meas - 1]) ** power])
    return x, y, n_meas


def _to_frac(ax, x, y):
    """Data coords -> axes fraction, on log-log axes."""
    (lx0, lx1) = np.log10(ax.get_xlim())
    (ly0, ly1) = np.log10(ax.get_ylim())
    return (np.log10(x) - lx0) / (lx1 - lx0), (np.log10(y) - ly0) / (ly1 - ly0)


def _from_frac(ax, fx, fy):
    (lx0, lx1) = np.log10(ax.get_xlim())
    (ly0, ly1) = np.log10(ax.get_ylim())
    return 10 ** (fx * (lx1 - lx0) + lx0), 10 ** (fy * (ly1 - ly0) + ly0)


#: Clearance (axes fractions, box edge to box edge) past which a candidate spot
#: counts as simply "clear" and is scored only on how far right it sits.
_CLEAR_ENOUGH = 0.05
#: Below this the label is touching something. Such spots are still usable as a
#: last resort -- a crowded panel may offer nothing better -- but any readable
#: spot outranks all of them, however far left it sits.
_MIN_CLEAR = 0.025
#: Separation a label needs from *its own* curve, which is a much weaker demand
#: than from a curve it does not name -- sitting alongside its own line is the
#: point. It has to be weaker: a label as wide as "$N+M$" on a roughly unit
#: slope in the short memory panel grazes its own line at every offset that is
#: still close enough to read as belonging to it, so holding own clearance to
#: _MIN_CLEAR left nothing acceptable anywhere and the label fell back to a
#: poor spot on the left. This asks only that the text not print over the line.
_MIN_OWN = 0.002


def _OFFSETS(side):
    """Candidate (dx, dy) displacements from a point on the curve, best first.

    ``side`` is a horizontal step wide enough to clear the label's own box.
    """
    out = [(0.0, dy) for dy in (0.06, -0.06, 0.10, -0.10, 0.14, -0.14)]
    out += [(sx * side, 0.0) for sx in (1, -1)]
    out += [(sx * side, dy) for sx in (1, -1) for dy in (0.05, -0.05)]
    return out


def auto_annotate(ax, polys, texts, fontsize=16):
    """Place each curve's scaling label so it collides with as little as possible.

    Hand-tuned data coordinates go stale the moment a curve's reach changes --
    a new cutoff, a machine with more memory, a crashed point -- and silently
    end up on top of another curve. Instead, search a few candidate spots along
    each curve and keep whichever sits furthest from every *other* curve and
    from the labels already placed.

    Everything is scored in axes-fraction space so that vertical and horizontal
    distance are comparable despite the log scaling.

    Args:
        ax: the panel to annotate.
        polys: curve name -> ``(x, y)`` polyline, including any continuation.
        texts: curve name -> label string.
    """
    frac = {}
    for name, (x, y) in polys.items():
        fx, fy = _to_frac(ax, x, y)
        keep = np.isfinite(fx) & np.isfinite(fy)
        if keep.any():
            frac[name] = (fx[keep], fy[keep])

    # Labels are boxes, not points. Measuring each one lets the placer keep the
    # whole box inside the axes instead of only its centre -- otherwise a label
    # anchored at 0.95 hangs half of "$N^2M$" over the frame -- and lets it
    # reject positions where the box, rather than merely its centre, sits on a
    # curve.
    boxes = {name: _label_halfsize(ax, texts[name], fontsize) for name in texts}

    placed = []  # (cx, cy, hw, hh) of everything already committed
    # Longest curves first: they have the most candidate positions to give up.
    for name in sorted(texts, key=lambda n: -len(frac.get(n, ([], []))[0])):
        if name not in frac:
            continue
        fx, fy = frac[name]
        hw, hh = boxes[name]
        others = [v for k, v in frac.items() if k != name]
        best = None
        for ax_at in np.linspace(0.94, 0.20, 20):
            if ax_at < fx.min() or ax_at > fx.max():
                continue
            cy = float(np.interp(ax_at, fx, fy))
            # Sideways offsets as well as vertical ones. A steep curve -- a
            # dense-GP extrapolation is very nearly vertical -- cannot be
            # stepped off by moving up or down, because the line is still right
            # there at the new height; only a horizontal shift clears it.
            side = hw + 0.035
            for dx, dy in _OFFSETS(side):
                px, py = float(ax_at + dx), cy + dy
                # The whole box has to fit, with a hair of margin.
                if not (hw + 0.015 < px < 1 - hw - 0.015):
                    continue
                if not (hh + 0.015 < py < 1 - hh - 0.015):
                    continue
                # Clearance measured from the box edge, not its centre, so a
                # wide label is held further off a curve than a narrow one.
                clearance = 1.0
                for ox, oy in others:
                    d = np.maximum(np.abs(ox - px) - hw, 0.0)
                    e = np.maximum(np.abs(oy - py) - hh, 0.0)
                    clearance = min(clearance, float(np.min(np.hypot(d, e))))
                for qx, qy, qhw, qhh in placed:
                    d = max(abs(qx - px) - hw - qhw, 0.0)
                    e = max(abs(qy - py) - hh - qhh, 0.0)
                    clearance = min(clearance, float(np.hypot(d, e)))
                # Its own curve too: the label should sit beside the line it
                # names, not on top of it.
                dow = np.maximum(np.abs(fx - px) - hw, 0.0)
                eow = np.maximum(np.abs(fy - py) - hh, 0.0)
                own = float(np.min(np.hypot(dow, eow)))
                if own > 0.13:      # too far to read as belonging to it
                    continue
                # A label names the curve's *asymptotic* slope, so it belongs at
                # the high-N end where the curve has actually reached it -- at
                # small N every curve looks alike and the label misleads.
                # Clearance is therefore capped: once a spot is clear enough to
                # read, extra room buys nothing and the rightmost spot wins.
                score = (
                    (1.0 if clearance >= _MIN_CLEAR else 0.0)
                    + (0.5 if own >= _MIN_OWN else 0.0)
                    + min(clearance, _CLEAR_ENOUGH)
                    + 0.25 * px
                    - 0.10 * abs(dy)
                    - 0.10 * abs(dx)
                )
                if best is None or score > best[0]:
                    best = (score, px, py)
        if best is None:
            continue
        _, bx, by = best
        placed.append((bx, by, hw, hh))
        ax.annotate(
            texts[name],
            xy=_from_frac(ax, bx, by),
            color=colors[name],
            ha="center",
            va="center",
            xycoords="data",
            fontsize=fontsize,
            zorder=5,
        )


def _label_halfsize(ax, text, fontsize):
    """Half-width and half-height of a label, in axes fractions.

    Rendered once off-screen and measured, rather than guessed from the string
    length: these are LaTeX maths labels, so "$N^2M$" and "$N$" differ in width
    by far more than their character counts suggest.
    """
    fig = ax.figure
    try:
        renderer = fig.canvas.get_renderer()
    except AttributeError:  # some backends only build one at draw time
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    probe = ax.text(0.5, 0.5, text, fontsize=fontsize, transform=ax.transAxes,
                    ha="center", va="center", alpha=0.0)
    bb = probe.get_window_extent(renderer=renderer)
    probe.remove()
    box = ax.get_window_extent(renderer=renderer)
    return 0.5 * bb.width / box.width, 0.5 * bb.height / box.height


def _legend_proxy(name):
    return mpl.lines.Line2D(
        [],
        [],
        color=colors[name],
        marker=markers[name],
        markersize=markersize[name],
        markeredgecolor="k",
        markeredgewidth=1.5,
        lw=2,
    )


class _RowLabelHandler(mpl.legend_handler.HandlerBase):
    """Draws nothing and takes no width, so a legend entry can be pure text."""

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        handlebox.set_width(0)
        return mpl.patches.Rectangle((0, 0), 0, 0, visible=False)


class _RowLabel:
    """Marker type for the text-only entries that head each legend row."""


def paired_legend(ax, present, title=None, y=1.0, fontsize=15):
    """Legend above the panel: a CPU row, and a GPU row directly beneath it,
    with each method in its own column.

    The machine is named once per row rather than repeated on every entry --
    ``CPU:  -o- GP  -o- QSM  -o- SSM`` instead of three separate "(CPU)"
    suffixes -- which is what lets the whole thing sit on two short rows.

    Matplotlib fills legend columns top-to-bottom before moving right, so the
    handles are interleaved as (CPU, GPU) pairs to put each column's two rows on
    the same method. The leading column holds the row labels, and methods with
    no GPU curve get an invisible spacer so the remaining columns stay aligned.

    Sitting above the axes, the title reads as the panel's heading and nothing
    can collide with the curves.
    """
    base = [c for c in CPU_ORDER if c in present]
    if not base:
        return None
    partners = {c: GPU_PARTNER.get(c) for c in base}
    two_rows = any(g in present for g in partners.values() if g)

    def spacer():
        return mpl.lines.Line2D([], [], ls="none", marker="none")

    # Leading column: the row labels. Zero-width handles, so "CPU:" starts hard
    # against the legend's left edge rather than being indented by an empty
    # handle slot.
    handles, labels = [_RowLabel()], ["CPU:"]
    if two_rows:
        handles.append(_RowLabel())
        labels.append("GPU:")

    for c in base:
        handles.append(_legend_proxy(c))
        labels.append(c)
        if two_rows:
            g = partners.get(c)
            if g and g in present:
                handles.append(_legend_proxy(g))
                labels.append(c)
            else:
                handles.append(spacer())
                labels.append("")

    return ax.legend(
        handles,
        labels,
        handler_map={_RowLabel: _RowLabelHandler()},
        ncol=1 + len(base),
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        title=title,
        fontsize=fontsize,
        title_fontsize=fontsize + 2,
        columnspacing=1.4,
        handletextpad=0.4,
        borderaxespad=0.0,
    )



# --------------------------------------------------------------------------
# Theoretical memory model
#
# Bytes the implementation must hold, as a function of the plotted x variable.
# Only curves whose footprint is *derived* from what the code actually stores
# appear here -- see README.md for each derivation. Everything else is left to
# its measurement.
#
# These exist because both devices have a resolution floor: on CPU a few
# hundred kB is lost in the noise between two ~300 MB RSS readings, and on GPU
# XLA hands out allocator chunks so small sizes quantise to multiples of 16 MB.
# Below the floor the measurement carries no information, and the theory does.
# --------------------------------------------------------------------------
#: (kind, curve) -> bytes(x, m_per_n), x being the figure's x axis.
#:
#: Where a first-principles derivation exists it is used (the dense curves, and
#: SSM's 8*(d+9) law); otherwise the measured asymptote stands in. Both are
#: drawn hollow, so the figure never presents either as an observation.
THEORY_MEM = {
    ("llh", "GP"): lambda x, mpn: 16.0 * x**2,
    ("llh", "QSM"): lambda x, mpn: 184.0 * x,
    ("llh", "SSM"): lambda x, mpn: 88.0 * x,
    ("cond", "GP"): lambda x, mpn: 24.0 * x**2,
    ("cond", "QSM"): lambda x, mpn: 585.0 * x,
    ("cond", "SSM"): lambda x, mpn: 184.0 * x,
    ("pred", "GP"): lambda x, mpn: 24.0 * x * (mpn * x),
    ("pred", "QSM"): lambda x, mpn: 48.0 * x * (mpn * x),
    ("pred", "SSM"): lambda x, mpn: 445.0 * (mpn * x),
    ("sample_prior", "GP"): lambda x, mpn: 16.0 * x**2,
    ("sample_prior", "QSM"): lambda x, mpn: 584.0 * x,
    ("sample_prior", "SSM"): lambda x, mpn: 812.0 * x,
    # x is N here; the draw is quadratic in M = m_per_n * N.
    ("sample_post", "GP"): lambda x, mpn: 24.0 * (mpn * x) ** 2,
    ("sample_post", "QSM"): lambda x, mpn: 49.0 * (mpn * x) ** 2,
    ("sample_post", "SSM"): lambda x, mpn: 1066.0 * (mpn * x),
}

#: The same for exposure-integrated runs. These are NOT interchangeable with
#: the instantaneous constants -- the augmented state makes SSM up to 4x
#: heavier -- so an --int figure filled from the table above would draw its
#: sub-floor points a factor of four low, with a visible step where the
#: substituted points meet the measured ones. There is no QSM here; tinygp has
#: no integrated quasiseparable kernel.
THEORY_MEM_INT = {
    ("llh", "GP"): lambda x, mpn: 16.0 * x**2,
    ("llh", "SSM"): lambda x, mpn: 350.0 * x,
    ("cond", "GP"): lambda x, mpn: 72.0 * x**2,
    ("cond", "SSM"): lambda x, mpn: 906.0 * x,
    ("pred", "GP"): lambda x, mpn: 24.0 * x * (mpn * x),
    ("pred", "SSM"): lambda x, mpn: 1680.0 * (mpn * x),
    ("sample_prior", "GP"): lambda x, mpn: 16.0 * x**2,
    ("sample_prior", "SSM"): lambda x, mpn: 1228.0 * x,
    ("sample_post", "GP"): lambda x, mpn: 24.0 * (mpn * x) ** 2,
    ("sample_post", "SSM"): lambda x, mpn: 1936.0 * (mpn * x),
}

#: Below this, a CPU memory measurement is indistinguishable from allocator
#: noise. Points at or under it are replaced by THEORY_MEM where available.
MEM_FLOOR_BYTES = 2e6


def theory_curve(kind, curve, xs, m_per_n=100, integrated=False):
    """Theoretical bytes for ``curve`` at each x, or None if unknown."""
    table = THEORY_MEM_INT if integrated else THEORY_MEM
    fn = table.get((kind, curve))
    if fn is None:
        return None
    return fn(np.asarray(xs, dtype=float), m_per_n)


def substitute_below_floor(measured, theory, floor=MEM_FLOOR_BYTES):
    """Swap unresolvable measurements for theory.

    Returns ``(values, was_substituted)``. A point is substituted when it was
    measured but the measurement carries no information: non-positive, meaning
    RSS saw no change at all, or under ``floor``, where what it reports is
    allocator noise rather than the allocation.

    NaN is explicitly *not* substituted. A NaN means the point was never run --
    the size cutoff retired it, or it crashed -- and inventing a value there
    would draw a curve the sweep never measured, out to sizes the machine
    cannot reach. Where a curve *would* have gone is already shown, separately
    and dotted, by the power-law continuation.

    The mask comes back so the caller can draw those points differently; they
    are computed, not observed, and the figure should say so.
    """
    values, sub = [], []
    for i, m in enumerate(measured):
        m = float(m)
        bad = np.isfinite(m) and (m <= 0 or m < floor)
        if bad and theory is not None and np.isfinite(theory[i]):
            values.append(float(theory[i]))
            sub.append(True)
        else:
            values.append(m)
            sub.append(False)
    return np.array(values), np.array(sub)


def _decade_locators(ax):
    ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_minor_locator(
        mpl.ticker.LogLocator(base=10.0, subs=jnp.arange(1, 10) * 0.1, numticks=100)
    )
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    # x majors are decades, but matplotlib thins them to every other decade once
    # the axis spans more than a handful. The gaps are filled by minor ticks at
    # *every* decade -- as a locator rather than a fixed list, so they follow a
    # later set_xlim (the predict panels run to M = 1e7, well past the data).
    ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, numticks=100))
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())


def benchmark_plot(
    Ns,
    runtime,
    memory,
    labels=None,
    title=None,
    savefig=None,
    xlabel="Number of data points",
    mem_hollow=None,
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
        hollow=mem_hollow,
    )
    _decade_locators(ax2)
    # Down to 1 KB: with sub-floor points replaced by theory the curves now
    # reach into the hundreds of bytes (SSM at N=10 is 88*10 = 880 B), and an
    # unlabelled decade there is hard to read off.
    ax2.set_yticks(
        [1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12],
        labels=["1 KB", "", "", "1 MB", "", "", "1 GB", "", "", "1 TB"],
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
# powers / powers2: the asymptotic power law each curve follows in the runtime
# and memory panels respectively. Used to continue every curve past its last
# measured point, dotted, out to the right edge of the axes.
# annotations: (text, curve) -- the label for a curve's scaling. Placement is
# worked out at draw time by auto_annotate, so these carry no coordinates.
# --------------------------------------------------------------------------
SPECS = {
    ("llh", False): {
        "title": "Likelihood",
        "powers": {"SSM": 1, "QSM": 1, "GP": 3, "pSSM": 1, "pQSM": 1},
        "powers2": {"SSM": 1, "QSM": 1, "GP": 2, "pSSM": 1, "pQSM": 1},
        "ylim1": (5e-5, 1e6),
        # Floor set just below the smallest plotted point rather than left to
        # autoscale: the tick list is fixed, so autoscale either clips the
        # leftmost markers or leaves a decade of empty panel.
        "ylim2": (3e2, 1e12),
        "annotations1": [
            ("$N^3$", "GP"),
            ("$N$", "SSM"),
            ("$N$", "QSM"),
            (r"$\sim$$N/T$", "pSSM"),
        ],
        "annotations2": [
            ("$N$", "SSM"),
            ("$N$", "QSM"),
            ("$N^2$", "GP"),
        ],
    },
    ("llh", True): {
        "title": "Likelihood",
        "powers": {"SSM": 1, "GP": 3, "pSSM": 1},
        "powers2": {"SSM": 1, "GP": 2, "pSSM": 1},
        "ylim1": (5e-5, 1e6),
        # Floor set just below the smallest plotted point rather than left to
        # autoscale: the tick list is fixed, so autoscale either clips the
        # leftmost markers or leaves a decade of empty panel.
        "ylim2": (3e2, 1e12),
        "annotations1": [
            ("$N^3$", "GP"),
            ("$N$", "SSM"),
            (r"$\sim$$N/T$", "pSSM"),
        ],
        "annotations2": [
            ("$N$", "SSM"),
            ("$N$", "pSSM"),
            ("$N^2$", "GP"),
        ],
    },
    ("cond", False): {
        "title": "Conditioning",
        "powers": {"SSM": 1, "QSM": 1, "GP": 3, "pSSM": 1, "pQSM": 1},
        "powers2": {"SSM": 1, "QSM": 1, "GP": 2, "pSSM": 1, "pQSM": 1},
        "ylim1": (5e-5, 1e6),
        # Floor set just below the smallest plotted point rather than left to
        # autoscale: the tick list is fixed, so autoscale either clips the
        # leftmost markers or leaves a decade of empty panel.
        "ylim2": (3e2, 1e12),
        # The linear curves run in a tight parallel bundle here, so one "$N$"
        # labels the whole bundle; repeating it per curve only adds clutter.
        # pSSM keeps its own because $\sim$$N/T$ is a different statement.
        "annotations1": [
            ("$N^3$", "GP"),
            ("$N$", "SSM"),
            (r"$\sim$$N/T$", "pSSM"),
        ],
        "annotations2": [
            ("$N$", "SSM"),
            ("$N^2$", "GP"),
        ],
    },
    ("cond", True): {
        "title": "Conditioning",
        "powers": {"SSM": 1, "GP": 3, "pSSM": 1},
        "powers2": {"SSM": 1, "GP": 2, "pSSM": 1},
        "ylim1": (5e-5, 1e6),
        # Floor set just below the smallest plotted point rather than left to
        # autoscale: the tick list is fixed, so autoscale either clips the
        # leftmost markers or leaves a decade of empty panel.
        "ylim2": (3e2, 1e12),
        "annotations1": [
            ("$N^3$", "GP"),
            ("$N$", "SSM"),
            (r"$\sim$$N/T$", "pSSM"),
        ],
        "annotations2": [
            ("$N$", "pSSM"),
            ("$N$", "SSM"),
            ("$N^2$", "GP"),
        ],
    },
    ("pred", False): {
        "title": "Prediction",
        "powers": {"SSM": 1, "QSM": 2, "GP": 3},
        "powers2": {"SSM": 1, "QSM": 2, "GP": 2},
        "secondary_axis": ("Number of prediction points", 100),
        "ylim1": (5e-5, 1e6),
        # Floor set just below the smallest plotted point rather than left to
        # autoscale: the tick list is fixed, so autoscale either clips the
        # leftmost markers or leaves a decade of empty panel.
        "ylim2": (1e5, 1e12),
        "xlim": (10, 1e7),
        "annotations1": [
            ("$N+M$", "SSM"),
            ("$NM$", "QSM"),
            ("$N^2M$", "GP"),
        ],
        "annotations2": [
            ("$N$", "SSM"),
            ("$NM$", "QSM"),
            ("$NM$", "GP"),
        ],
    },
    ("pred", True): {
        "title": "Prediction",
        "powers": {"SSM": 1, "GP": 3},
        "powers2": {"SSM": 1, "GP": 2},
        "secondary_axis": ("Number of prediction points", 100),
        "ylim1": (5e-5, 1e6),
        # Floor set just below the smallest plotted point rather than left to
        # autoscale: the tick list is fixed, so autoscale either clips the
        # leftmost markers or leaves a decade of empty panel.
        "ylim2": (1e5, 1e12),
        "xlim": (10, 1e7),
        "annotations1": [("$N+M$", "SSM"), ("$N^2M$", "GP")],
        "annotations2": [("$N+M$", "SSM"), ("$NM$", "GP")],
    },
    # --- sampling -----------------------------------------------------------
    # Prior draws have no training data, so the only size parameter is M, the
    # number of coordinates the realization is drawn at.
    ("sample_prior", False): {
        "title": "Prior sample",
        "powers": {"SSM": 1, "QSM": 1, "GP": 3},
        "powers2": {"SSM": 1, "QSM": 1, "GP": 2},
        "xlabel": "Number of sample points",
        "ylim1": (5e-5, 1e6),
        # Prior draws start at M = 10, where GP needs only 16*10^2 = 1.6 KB, so
        # an autoscaled floor lands on top of the leftmost markers and clips
        # them. Pinned a half decade lower to leave them room.
        "ylim2": (3e2, 1e12),
        "annotations1": [
            ("$M$", "SSM"),
            ("$M$", "QSM"),
            ("$M^3$", "GP"),
        ],
        "annotations2": [("$M$", "SSM"), ("$M^2$", "GP")],
    },
    ("sample_prior", True): {
        "title": "Prior sample",
        "powers": {"SSM": 1, "GP": 3},
        "powers2": {"SSM": 1, "GP": 2},
        "xlabel": "Number of sample points",
        "ylim1": (5e-5, 1e6),
        # Prior draws start at M = 10, where GP needs only 16*10^2 = 1.6 KB, so
        # an autoscaled floor lands on top of the leftmost markers and clips
        # them. Pinned a half decade lower to leave them room.
        "ylim2": (3e2, 1e12),
        "annotations1": [("$M$", "SSM"), ("$M^3$", "GP")],
        "annotations2": [("$M$", "SSM"), ("$M^2$", "GP")],
    },
    # Posterior draws mirror `pred`: N training points, M = 100N test points.
    ("sample_post", False): {
        "title": "Posterior sample",
        "powers": {"SSM": 1, "QSM": 2, "GP": 3},
        "powers2": {"SSM": 1, "QSM": 2, "GP": 2},
        "secondary_axis": ("Number of sample points", 100),
        "ylim1": (5e-5, 1e6),
        # Floor set just below the smallest plotted point rather than left to
        # autoscale: the tick list is fixed, so autoscale either clips the
        # leftmost markers or leaves a decade of empty panel.
        "ylim2": (3e5, 1e12),
        "xlim": (10, 1e7),
        "annotations1": [
            ("$N+M$", "SSM"),
            ("$NM$", "QSM"),
            ("$N^2M$", "GP"),
        ],
        "annotations2": [("$N$", "SSM"), ("$NM$", "GP")],
    },
    ("sample_post", True): {
        "title": "Posterior sample",
        "powers": {"SSM": 1, "GP": 3},
        "powers2": {"SSM": 1, "GP": 2},
        "secondary_axis": ("Number of sample points", 100),
        "ylim1": (5e-5, 1e6),
        # Floor set just below the smallest plotted point rather than left to
        # autoscale: the tick list is fixed, so autoscale either clips the
        # leftmost markers or leaves a decade of empty panel.
        "ylim2": (3e5, 1e12),
        "xlim": (10, 1e7),
        "annotations1": [("$N+M$", "SSM"), ("$N^2M$", "GP")],
        "annotations2": [("$N+M$", "SSM"), ("$NM$", "GP")],
    },
}


def make_benchmark_figure(
    kind,
    cpu_data,
    gpu_data=None,
    integrated=False,
    savefig=None,
    annotate=True,
    substitute_theory=True,
    show_theory=False,
):
    """Build one benchmarks-page figure from its spec.

    Args:
        kind: one of ``llh``, ``cond``, ``pred``, ``sample_prior``, ``sample_post``.
        cpu_data: dict from :func:`~benchmark.load_benchmark_data`.
        gpu_data: optional second dict; its ``pSSM``/``pQSM`` curves are spliced
            in. Curves absent from it are simply omitted, so a figure can be
            regenerated before the GPU run exists.
        integrated: select the integrated variant of the spec.
        substitute_theory: replace memory points that fall below the
            measurement floor with THEORY_MEM, drawn as hollow markers so
            they stay distinguishable from measured ones.
        show_theory: additionally draw each derived theoretical curve as a
            thin line across the whole range, for checking how far the
            measurements track it. Independent of substitute_theory.
        savefig: path to write; ``True`` writes the canonical
            ``docs/_static/<kind><_int>_benchmark.png``.

    Returns:
        ``(fig, (ax_runtime, ax_memory))``.
    """
    spec = SPECS[(kind, integrated)]
    data = merge_machines(cpu_data, gpu_data)
    Ns = data["Ns"]
    runtime, memory = data["runtime"], data["memory"]

    # Curves a kind is *expected* to have, whether or not they produced data.
    # A curve that ran but errored, or has not been run yet, keeps its legend
    # entry with nothing plotted -- that gap is the point, it flags a figure
    # that is still missing a result rather than quietly dropping it.
    present = [n for n in runtime if n in colors]
    runtime = {n: runtime[n] for n in present}
    memory = {n: memory[n] for n in present if n in memory}
    Ns = Ns[: len(next(iter(runtime.values())))]

    if savefig is True:
        savefig = os.path.join(
            STATIC_DIR, f"{kind}{'_int' if integrated else ''}_benchmark.png"
        )

    # Swap unresolvable memory measurements for theory *before* anything is
    # drawn, so the substituted points can be given hollow markers instead of
    # ending up with a filled one underneath.
    m_per_n = spec.get("m_per_n", 100)
    mem_hollow = {}
    if substitute_theory:
        for name in list(memory):
            th = theory_curve(kind, name, Ns, m_per_n, integrated)
            if th is None:
                continue
            vals, sub = substitute_below_floor([p[0] for p in memory[name]], th)
            if sub.any():
                memory[name] = [(v, 0.0) for v in vals]
                mem_hollow[name] = sub

    fig, (ax1, ax2) = benchmark_plot(
        Ns,
        runtime,
        memory,
        labels={n: MACHINE_LABELS.get(n, n) for n in present},
        title=spec["title"],
        powers=spec["powers"],
        xlabel=spec.get("xlabel", "Number of data points"),
        mem_hollow=mem_hollow,
    )

    if "secondary_axis" in spec:
        label, factor = spec["secondary_axis"]
        ax_top = ax1.secondary_xaxis(
            "top", functions=(lambda x: factor * x, lambda x: x / factor)
        )
        ax_top.set_xlabel(label)

    # Limits first: every dotted continuation runs to the right edge, and the
    # label placer scores positions in axes fractions, so both need the final
    # axes before they can do anything sensible.
    if spec.get("ylim1"):
        ax1.set_ylim(*spec["ylim1"])
    if spec.get("ylim2"):
        ax2.set_ylim(*spec["ylim2"])
    if spec.get("xlim"):
        ax2.set_xlim(*spec["xlim"])
    ax1.grid(lw=0.5, zorder=-1, which="major")

    xmax = ax2.get_xlim()[1]
    panels = (
        (ax1, runtime, spec["powers"], spec.get("annotations1", [])),
        (ax2, memory, spec.get("powers2", {}), spec.get("annotations2", [])),
    )
    # Hollow markers for the substituted points, and optionally the full
    # theoretical curve for comparison.
    for name, sub in mem_hollow.items():
        vals = np.array([p[0] for p in memory[name]])
        ax2.plot(
            np.asarray(Ns)[sub],
            vals[sub],
            ls="none",
            marker=markers[name],
            markersize=markersize[name],
            markerfacecolor="none",
            markeredgecolor=colors[name],
            markeredgewidth=1.5,
            zorder=4,
        )
    if show_theory:
        for name in memory:
            th = theory_curve(kind, name, Ns, m_per_n, integrated)
            if th is not None:
                ax2.plot(Ns, th, ls="-", lw=0.8, alpha=0.55,
                         color=colors[name], zorder=0)

    for ax, series_by_curve, powers, annots in panels:
        polys, texts = {}, {}
        for name, series in series_by_curve.items():
            got = continue_curve(Ns, series, powers.get(name, 1), xmax)
            if got is None:
                continue
            x, y, n_meas = got
            polys[name] = (x, y)
            if n_meas < len(x):  # something to continue
                ax.plot(
                    x[n_meas - 1 :],
                    y[n_meas - 1 :],
                    ls=":",
                    color=colors[name],
                    zorder=1,
                )
        # Spec annotations carry a hand-tuned xy that is now ignored: the
        # placer works it out from where the curves actually ended up.
        for entry in annots:
            text, curve = (entry[0], entry[-1])
            if curve in polys:
                texts[curve] = text
        if annotate and texts:
            auto_annotate(ax, polys, texts)

    paired_legend(
        ax1,
        present,
        title=spec["title"],
        # Clear the secondary top axis and its label where there is one.
        y=1.18 if "secondary_axis" in spec else 1.02,
    )

    if savefig:
        os.makedirs(os.path.dirname(savefig), exist_ok=True)
        plt.savefig(savefig, dpi=300, bbox_inches="tight")
        print(f"  wrote {savefig}")
    return fig, (ax1, ax2)
