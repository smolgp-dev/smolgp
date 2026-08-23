#!/usr/bin/env python
"""Derive the runtime table in run/README.md from the measured results.

Each repeat of a point is a **fresh subprocess** (``profile_jax_function`` spawns
one per repeat), and each subprocess makes an untimed warm-up call before the
timed one (``tracer``). So a point of per-call time ``t`` given ``n`` repeats
costs

    n x (2t + c)

on the wall clock, not ``n x (t + c)``: the warm-up executes the same
computation, so the compute term is paid twice per repeat, and ``c`` is only the
genuinely fixed part -- fork, ``import jax``, unpickling the data, and XLA
compilation. Ignoring the warm-up under-predicts by ~2x at large N, which is
what the earlier ``n x (t + c)`` form did (it put GP@56234 at 950 s/call against
~1855 s of wall clock).

All three inputs are known:

* per-call times come from ``results/*.pkl`` -- real measurements, not models;
* repeats come from ``benchmark.NREPEAT_SCHEDULE``, applied per point exactly as
  the harness applies it;
* ``c`` is calibrated against completed sweeps -- see OVERHEAD_SECONDS and
  ``--calibrate``.

Re-run it whenever the timings move (a solver change, a new machine, a different
schedule) rather than editing the table by hand:

    uv run estimate_runtime.py            # print the markdown table
    uv run estimate_runtime.py --write    # splice it into run/README.md

    # re-derive c from a sweep whose wall clock you know, e.g. the llh-vg CPU
    # half of logs/llhvg_full_20260821_130405.log (13:04:06 -> 17:29:44, at the
    # then-current fixed nrepeat=7):
    uv run estimate_runtime.py \
        --calibrate results/cpu_llh_value_and_grad_benchmark.pkl,15938,7
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

from benchmark import load_benchmark_data

try:
    from benchmark import NREPEAT_SCHEDULE, repeats_for
except ImportError:  # benchmark.py predates the adaptive schedule
    NREPEAT_SCHEDULE = ((1.0, 7), (10.0, 5), (60.0, 3), (float("inf"), 1))

    def repeats_for(seconds):
        for limit, n in NREPEAT_SCHEDULE:
            if seconds < limit:
                return n
        return NREPEAT_SCHEDULE[-1][1]

    print("note: benchmark.py has no NREPEAT_SCHEDULE; using a built-in copy",
          file=sys.stderr)

#: Fixed cost per repeat, in seconds -- fork, ``import jax``, unpickling and
#: compilation, with the two executions of the computation itself already
#: accounted for by the 2t term. Calibrated by ``--calibrate``: solve
#:
#:     c = (wall - 2 * sum(n_p * t_p)) / sum(n_p)
#:
#: against a sweep whose wall clock is known.
#:
#: Keyed by ``(device, integrated)``, because the cost is dominated by different
#: things on the two devices. On CPU it is fork + ``import jax`` + XLA:CPU
#: compilation; on GPU each subprocess additionally creates a fresh CUDA context
#: and loads the cuBLAS/cuSOLVER kernels, which is why it is ~5x larger. The
#: measured split, from the two halves of logs/llhvg_full_20260821_130405.log:
#:
#:     cpu  c = 2.39 s   (compute was 95% of a 15938 s wall clock)
#:     gpu  c = 13.27 s  (compute was  6% of a  3266 s wall clock)
#:
#: The GPU number therefore carries almost the whole estimate for GPU sweeps,
#: while on CPU it is a rounding correction -- which is what UNCERTAINTY below
#: turns into the reported range.
#:
#: The CPU value is corroborated directly: profiling an N=10 likelihood, where
#: 2t is ~1 ms and so wall/n_repeat *is* c, measures 2.15 s (SSM), 2.09 s (QSM)
#: and 2.62 s (SSM value-and-grad, a bigger graph to compile) -- consistent with
#: the 2.39 s fitted across a whole sweep. That agreement is also what rules out
#: the earlier n x (t + c) form: fitting it to the same sweep forces c = 26.5 s,
#: which a 2 s fork-and-import cannot be.
#:
#: The ``integrated`` entries are measured the same way, from the two halves of
#: logs/llhvg_int_20260821_185101.log (CPU 18:51:01 -> 19:31:52 = 2451 s, GPU
#: 19:31:52 -> 20:10:45 = 2333 s, both on the adaptive schedule):
#:
#:     cpu  --int  c = 6.18 s   (compute was 64% of wall)
#:     gpu  --int  c = 22.58 s  (compute was  2% of wall)
#:
#: Integrated really is ~2.6x the instantaneous fixed cost, not the +1.9 s that
#: was assumed before there was data: the K = 2N state arrays are larger to
#: unpickle and their graphs larger to compile. Both figures are marginally
#: inflated -- each of those sweeps lost two subprocesses to OOM, which burned
#: wall clock while contributing no calls, worth an estimated 0.4 s on the CPU
#: number. Small enough to leave in rather than model.
OVERHEAD_SECONDS = {
    ("cpu", False): 2.39,
    ("cpu", True): 6.18,
    ("gpu", False): 13.27,
    ("gpu", True): 22.58,
}

#: Fractional uncertainty on c, applied to produce the low--high range in the
#: table. The compute term is measured, so this band only widens the part of the
#: estimate that is modelled: a CPU sweep comes out nearly a point estimate, a
#: GPU sweep comes out visibly bracketed. A wide range is information -- it says
#: the number is mostly model, not measurement.
UNCERTAINTY = 0.25

#: Per-call budget the Tier 2 long runs are launched with, matching
#: run_benchmark.LONG_RUN_MAX_SECONDS. Sets how far up the grid each band goes.
LONG_RUN_SECONDS = 1800.0

#: Tier 2 is for points that are individually expensive. Anything projected to
#: take less than this per call belongs to the production suite, which already
#: carries a 600 s per-call budget -- if such a point is missing from it, the
#: reason is its memory cutoff or a crash, and neither is fixed by running it
#: here. Matches run_benchmark.LONG_RUN_MIN_SECONDS.
LONG_RUN_MIN_SECONDS = 600.0

CPU_CURVES = ("SSM", "QSM", "GP")
GPU_CURVES = ("pSSM", "pQSM")

#: The full suite. Sampling has no parallel-solver implementation, so it is
#: CPU-only; everything else runs on both.
KINDS = ("llh", "llh_value_and_grad", "cond", "pred", "sample-prior", "sample-post")
CPU_ONLY = ("sample-prior", "sample-post")


def sweeps():
    for kind in KINDS:
        for integrated in (False, True):
            for device in ("cpu", "gpu"):
                if device == "gpu" and kind in CPU_ONLY:
                    continue
                yield kind, integrated, device


def path_for(kind, integrated, device):
    """results/<dev>_<kind>[_int]_benchmark.pkl, with _int before the vg tag."""
    stem = kind.replace("_value_and_grad", "")
    tag = "_int" if integrated else ""
    vg = "_value_and_grad" if kind.endswith("_value_and_grad") else ""
    return f"results/{device}_{stem}{tag}{vg}_benchmark.pkl"


def sweep_cost(f, curves, c, nrepeat=None):
    """(seconds, n_points) for one results file under a fixed per-repeat cost."""
    data = load_benchmark_data(f)
    total = 0.0
    pairs = 0
    for curve in curves:
        if curve not in data.get("runtime", {}):
            continue
        t = np.array([p[0] for p in data["runtime"][curve]], dtype=float)
        for value in t[np.isfinite(t)]:
            n = nrepeat if nrepeat is not None else repeats_for(float(value))
            pairs += 1
            total += n * (2 * value + c)
    return total, pairs


def estimate(kind, integrated, device, overhead):
    """(low_minutes, high_minutes, n_pairs) or None when the sweep never ran."""
    f = path_for(kind, integrated, device)
    if not os.path.exists(f):
        return None
    curves = GPU_CURVES if device == "gpu" else CPU_CURVES
    c = overhead[(device, integrated)]
    lo, pairs = sweep_cost(f, curves, c * (1 - UNCERTAINTY))
    hi, _ = sweep_cost(f, curves, c * (1 + UNCERTAINTY))
    if pairs == 0:
        return None
    return lo / 60, hi / 60, pairs


# ---------------------------------------------------------------------------
# Tier 2: the "long runs only" fill-in (run_benchmark.py --long-runs-only)
# ---------------------------------------------------------------------------
# A production sweep stops each curve at the first of its memory bound, its
# 600 s per-call bound, or the flat grid cap. Tier 2 buys back the band above
# that, at nrepeat=1 and an 1800 s per-call budget. Nothing in that band has
# ever been measured, so every number below is a *projection*: per-call times
# come from extrapolating each curve's own measured tail, not from measurement.
# Re-run this after the long suite finishes and the rows become measured.

M_PER_N = 100

#: Kinds that draw at M = 100N, so their N ladder stops two decades earlier to
#: reach the same largest array. Mirrors run_benchmark.py's `logmax -= 2`.
SCALES_WITH_M = ("pred", "sample-post")

N_SIZES = 17


def grid_for(kind):
    """The 17 log-spaced sizes a kind actually runs, in N.

    Not one shared ladder: `pred` and `sample-post` draw at M = 100N, so they
    stop at N = 1e5 to reach the same M = 1e7 the others reach in N. Assuming a
    single 1e1..1e7 grid put sizes in the table that those kinds never run --
    23713 and 56234 for `pred`, where the real neighbours are 17782 and 31622 --
    and truncated their bands two points early.
    """
    logmax = 5 if kind in SCALES_WITH_M else 7
    return sorted({int(x) for x in np.logspace(1, logmax, N_SIZES)})


def tail_exponent(Ns, ts, table_pow):
    """Local power-law exponent from a curve's last two measured points.

    Returns the table's nominal exponent when there are too few points to fit.
    """
    if len(Ns) < 2:
        return float(table_pow)
    (n0, t0), (n1, t1) = (Ns[-2], ts[-2]), (Ns[-1], ts[-1])
    if n1 <= n0 or t0 <= 0 or t1 <= 0:
        return float(table_pow)
    return float(np.log(t1 / t0) / np.log(n1 / n0))


def project_seconds(size, mNs, mts, cost, power):
    """Projected per-call time at ``size``: the larger of two extrapolations.

    Two ways to get it, each wrong in a different direction:

    * the curve's own **local slope**, anchored on its last measured point. Right
      when the curve is already in its asymptotic regime, but it under-predicts
      badly from inside a transition. Integrated GP measures N^1.59 across its
      last two points -- still dominated by building the dense kernel, an O(N^2)
      term with a large constant -- while the O(N^3) Cholesky that is about to
      take over is nowhere in that fit.
    * the **cost table's law**, a fixed coefficient and nominal exponent. Right
      asymptotically, but it over-predicts inside the same transition, and its
      coefficients are calibrated on the forward pass so it under-predicts a
      value-and-grad curve.

    Taking the max is deliberately conservative. This projection exists to
    decide whether a night is enough, and the failure that costs something is
    under-predicting: an over-estimate wastes a slot in the plan, an
    under-estimate wastes the night.
    """
    local = mts[-1] * (size / mNs[-1]) ** power
    if not cost:
        return local
    coeff, (pn, pm) = cost[1]
    table = coeff * size ** (pn + pm)
    return max(local, table)


def long_run_band(kind, integrated, device):
    """[(curve, [(size, projected_seconds), ...]), ...] for one sweep's Tier 2.

    Empty when the sweep has nothing above its production cutoff.
    """
    from benchmark import _COST, _COST_INT, existing_floors, ram_budget, size_cutoffs

    base = kind.replace("_value_and_grad", "")
    vg = kind.endswith("_value_and_grad")
    f = path_for(kind, integrated, device)
    if not os.path.exists(f):
        return []

    curves = GPU_CURVES if device == "gpu" else CPU_CURVES
    ratio = M_PER_N if base in SCALES_WITH_M else None
    cutoffs = size_cutoffs(
        ram_budget(device=device), base, max_seconds=LONG_RUN_SECONDS,
        gpu=(device == "gpu"), integrated=integrated, value_and_grad=vg,
    )
    floors = existing_floors(f, cutoffs, m_per_n=ratio)
    data = load_benchmark_data(f)
    stored = np.asarray(data.get("Ns", []), dtype=float)
    table = _COST_INT if integrated else _COST

    out = []
    for curve in curves:
        cut, floor = cutoffs.get(curve, 0.0), floors.get(curve, 0.0)
        if cut <= floor:
            continue
        points = data.get("runtime", {}).get(curve)
        if not points:
            continue
        t = np.array([q[0] for q in points], dtype=float)
        n = min(len(t), len(stored))
        good = np.isfinite(t[:n])
        if not good.any():
            continue
        mNs, mts = stored[:n][good], t[:n][good]
        cost = table.get((base, curve))
        table_pow = cost[1][1][0] + cost[1][1][1] if cost else 3.0
        power = tail_exponent(mNs, mts, table_pow)

        todo = []
        for size in grid_for(base):
            x = size * (ratio or 1)          # compare in the cutoff's variable
            if not (floor < x <= cut):
                continue
            proj = project_seconds(size, mNs, mts, cost, power)
            todo.append((size, proj))
            if proj > LONG_RUN_SECONDS:
                break                        # retires here; nothing beyond runs
        # Only bands whose slowest call clears the Tier 2 threshold. The rest
        # are production's business.
        if todo and max(t for _, t in todo) >= LONG_RUN_MIN_SECONDS:
            out.append((curve, power, todo))
    return out


def build_long_table():
    rows, total = [], 0.0
    for kind, integrated, device in sweeps():
        try:
            band = long_run_band(kind, integrated, device)
        except Exception as exc:                     # noqa: BLE001
            print(f"  ({label(kind, integrated)} {device}: {exc})", file=sys.stderr)
            continue
        if not band:
            continue
        c = OVERHEAD_SECONDS[(device, integrated)]
        for curve, power, todo in band:
            secs = sum(2 * t + c for _, t in todo)   # nrepeat = 1
            total += secs
            sizes = ", ".join(f"{s:,}" for s, _ in todo)
            slowest = max(t for _, t in todo)
            rows.append((secs, label(kind, integrated),
                         "🟦 CPU" if device == "cpu" else "🟪 GPU",
                         curve, sizes, power, slowest, secs))
    rows.sort(reverse=True)

    out = [
        "| sweep | device | curve | N to add | local slope | slowest call | est. runtime |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, lab, dev, curve, sizes, power, slowest, secs in rows:
        out.append(
            f"| {lab} | {dev} | {curve} | {sizes} | N^{power:.2f} | "
            f"{fmt_secs(slowest)} | {marker(secs / 60)} {fmt_secs(secs)} |"
        )
    out.append(
        f"| **TOTAL ({len(rows)} curve-bands)** | | | | | | "
        f"**{fmt_secs(total)}** |"
    )
    return "\n".join(out), total


def fmt_secs(x):
    if x < 90:
        return f"{x:.0f} s"
    if x < 5400:
        return f"{x / 60:.0f} min"
    return f"{x / 3600:.1f} h"


def calibrate(spec):
    """Solve a completed sweep's wall clock for the fixed per-repeat cost c.

    ``spec`` is ``pkl,wall_seconds[,nrepeat]``. Omit ``nrepeat`` to use the
    adaptive schedule; give it when the sweep predates the schedule or was run
    with an explicit ``--nrepeat``.

    wall = sum_p n_p * (2 t_p + c)  =>  c = (wall - 2 sum n_p t_p) / sum n_p
    """
    parts = spec.split(",")
    if len(parts) not in (2, 3):
        sys.exit(f"--calibrate wants PKL,WALL_SECONDS[,NREPEAT], got {spec!r}")
    f, wall = parts[0], float(parts[1])
    fixed = int(parts[2]) if len(parts) == 3 else None
    if not os.path.exists(f):
        sys.exit(f"{f} does not exist")

    data = load_benchmark_data(f)
    compute = 0.0
    calls = 0
    for curve, points in data.get("runtime", {}).items():
        t = np.array([q[0] for q in points], dtype=float)
        for value in t[np.isfinite(t)]:
            n = fixed if fixed is not None else repeats_for(float(value))
            compute += n * value
            calls += n
    if calls == 0:
        sys.exit(f"{f} has no finite timings to calibrate against")

    c = (wall - 2 * compute) / calls
    print(f"{f}")
    print(f"  wall clock            {wall:9.1f} s")
    print(f"  repeats (calls)       {calls:9d}"
          f"   ({'fixed nrepeat=%d' % fixed if fixed else 'adaptive schedule'})")
    print(f"  compute 2*sum(n*t)    {2 * compute:9.1f} s"
          f"   ({200 * compute / wall:.0f}% of wall)")
    print(f"  => c                  {c:9.2f} s per repeat")
    if c < 0:
        print("  WARNING: negative -- the 2t warm-up term already exceeds the"
              " wall clock, so either the wall clock or the repeat count is wrong")
    return c


def marker(minutes):
    return "✅" if minutes < 30 else ("⚠️" if minutes < 90 else "🛑")


def label(kind, integrated):
    name = "llh-vg" if kind.endswith("_value_and_grad") else kind
    return f"{name} `--int`" if integrated else name


def build_table(overhead):
    rows, missing = [], []
    for kind, integrated, device in sweeps():
        got = estimate(kind, integrated, device, overhead)
        dev = "🟦 CPU" if device == "cpu" else "🟪 GPU"
        if got is None:
            missing.append((label(kind, integrated), dev))
            continue
        lo, hi, pairs = got
        rows.append((hi, label(kind, integrated), dev, pairs, lo, hi))
    rows.sort(reverse=True)

    out = [
        "| sweep | device | pts | runtime | basis |",
        "|---|---|---|---|---|",
    ]
    for _, lab, dev, pairs, lo, hi in rows:
        out.append(f"| {lab} | {dev} | {pairs} | {marker(hi)} {lo:.0f} -- {hi:.0f} m | measured |")
    for lab, dev in missing:
        out.append(f"| {lab} | {dev} | — | ❌ — | never run |")
    tl = sum(r[4] for r in rows)
    th = sum(r[5] for r in rows)
    out.append(
        f"| **TOTAL ({len(rows)} of {len(rows) + len(missing)} sweeps)** | | | "
        f"**{tl:.0f} -- {th:.0f} m ≈ {tl / 60:.1f} -- {th / 60:.1f} h** | |"
    )
    return "\n".join(out), (tl, th, len(rows), len(missing))


START = "<!-- RUNTIME TABLE START -->"
END = "<!-- RUNTIME TABLE END -->"
LONG_START = "<!-- LONG RUN TABLE START -->"
LONG_END = "<!-- LONG RUN TABLE END -->"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help=f"splice the table into run/README.md between {START} and {END}")
    ap.add_argument("--overhead", default=None, metavar="CPU,GPU",
                    help="fixed seconds per repeat, overriding both non-int entries "
                         f"(default {OVERHEAD_SECONDS[('cpu', False)]},"
                         f"{OVERHEAD_SECONDS[('gpu', False)]}); the --int entries "
                         "keep their measured offset above these")
    ap.add_argument("--long-runs", action="store_true",
                    help="print the Tier 2 (--long-runs-only) projection instead "
                         "of the production table")
    ap.add_argument("--calibrate", action="append", default=None,
                    metavar="PKL,WALL_SECONDS[,NREPEAT]",
                    help="solve a finished sweep's wall clock for c and exit; "
                         "repeatable, one per sweep")
    args = ap.parse_args()

    if args.calibrate:
        cs = [calibrate(spec) for spec in args.calibrate]
        print(f"\nOVERHEAD_SECONDS = ({min(cs):.2f}, {max(cs):.2f})")
        return

    if args.long_runs:
        table, total = build_long_table()
        print(f"Tier 2 projection, nrepeat=1, {LONG_RUN_SECONDS:g}s per-call budget")
        print("per-call times are EXTRAPOLATED from each curve's measured tail\n")
        print(table)
        print(f"\ntotal {fmt_secs(total)}")
        if args.write:
            path = "run/README.md"
            text = open(path).read()
            if LONG_START not in text or LONG_END not in text:
                sys.exit(f"{path} has no {LONG_START} / {LONG_END} markers")
            text = re.sub(f"{re.escape(LONG_START)}.*?{re.escape(LONG_END)}",
                          f"{LONG_START}\n{table}\n{LONG_END}", text, flags=re.S)
            open(path, "w").write(text)
            print(f"\nwrote the Tier 2 table into {path}")
        return

    overhead = dict(OVERHEAD_SECONDS)
    if args.overhead:
        cpu, gpu = (float(x) for x in args.overhead.split(","))
        int_offset = {
            dev: OVERHEAD_SECONDS[(dev, True)] - OVERHEAD_SECONDS[(dev, False)]
            for dev in ("cpu", "gpu")
        }
        overhead = {("cpu", False): cpu, ("gpu", False): gpu,
                    ("cpu", True): cpu + int_offset["cpu"],
                    ("gpu", True): gpu + int_offset["gpu"]}

    table, (tl, th, n_have, n_miss) = build_table(overhead)
    sched = ", ".join(
        f"<{lim:g}s->{n}" if lim != float("inf") else f"else->{n}"
        for lim, n in NREPEAT_SCHEDULE
    )
    print(f"repeat schedule: {sched}")
    fixed = ", ".join(
        f"{dev}{'+int' if isint else ''}={overhead[(dev, isint)]:.2f}s"
        for dev in ("cpu", "gpu") for isint in (False, True)
    )
    print(f"fixed cost per repeat: {fixed}  (+/-{UNCERTAINTY:.0%})")
    print("model: n x (2t + c) per point -- 2t because each repeat is a fresh"
          " subprocess that warms up untimed before the timed call\n")
    print(table)
    print(f"\n{n_have} sweeps measured, {n_miss} never run; "
          f"total {tl / 60:.1f}--{th / 60:.1f} h")

    if args.write:
        p = "run/README.md"
        s = open(p).read()
        if START not in s or END not in s:
            sys.exit(f"{p} has no {START} / {END} markers; add them around the table first")
        s = re.sub(f"{re.escape(START)}.*?{re.escape(END)}",
                   f"{START}\n{table}\n{END}", s, flags=re.S)
        open(p, "w").write(s)
        print(f"\nwrote the table into {p}")


if __name__ == "__main__":
    main()
