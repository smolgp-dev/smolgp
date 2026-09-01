#!/usr/bin/env python
"""Re-derive the memory constants from the current result files and diff them.

Three families of hand-maintained constants are all "the measured asymptote":

* ``benchmark._COST`` / ``_COST_INT`` -- memory and time laws behind the size
  cutoffs, which decide how far each curve is allowed to run;
* ``benchmark._GRAD_MEM_FACTOR`` -- how much more memory reverse mode needs;
* ``plotting.THEORY_MEM`` / ``THEORY_MEM_INT`` -- what gets drawn in place of a
  sub-floor memory point, as a hollow marker.

Being hand-maintained, they drift, and a stale one is not cosmetic: the cutoffs
decide whether a sweep OOMs, and the theory constants are what the left-hand end
of every memory panel actually shows.

They also all predate the memory-profiler fix (see CPUMemorySampler), which
means any of them read off a point above ~30 GB was calibrated against a capped
figure and is low. This prints what the current results imply, beside what is in
the source, so the gap is visible rather than assumed.

    uv run calibrate_costs.py              # everything
    uv run calibrate_costs.py --grad       # just the reverse-mode factors
    uv run calibrate_costs.py --theory     # just the sub-floor theory constants
    uv run calibrate_costs.py --min-gb 1   # ignore points below 1 GB

Nothing is written. Edit the constants by hand, so that changing a number that
governs an OOM stays a deliberate act.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark import _COST, _COST_INT, _GRAD_MEM_FACTOR, load_benchmark_data
from plotting import THEORY_MEM, THEORY_MEM_INT

CPU_CURVES = ("SSM", "QSM", "GP")
M_PER_N = 100
SCALES_WITH_M = ("pred", "sample-post")
KINDS = ("llh", "cond", "pred", "sample-prior", "sample-post")


def path_for(kind, integrated, vg=False, device="cpu"):
    return (f"results/{device}_{kind}{'_int' if integrated else ''}"
            f"{'_value_and_grad' if vg else ''}_benchmark.pkl")


def measured(kind, integrated, curve, vg=False, min_bytes=0.0):
    """[(size, bytes)] for finite memory points above ``min_bytes``."""
    f = path_for(kind, integrated, vg)
    if not os.path.exists(f):
        return []
    d = load_benchmark_data(f)
    Ns = np.asarray(d.get("Ns", []), dtype=float)
    pts = d.get("memory", {}).get(curve)
    if not pts:
        return []
    m = np.array([q[0] for q in pts], dtype=float)
    n = min(len(Ns), len(m))
    return [(float(N), float(v)) for N, v in zip(Ns[:n], m[:n])
            if np.isfinite(v) and v > min_bytes]


def per_unit(kind, size, value, n_pow, m_pow):
    """value / (N^n_pow * M^m_pow), with M = 100N for the M-scaled kinds."""
    mpn = M_PER_N if kind in SCALES_WITH_M or kind == "pred" else 1
    N = size
    M = mpn * size if kind in SCALES_WITH_M or kind == "pred" else size
    denom = (N ** n_pow) * (M ** m_pow) if m_pow else N ** n_pow
    return value / denom if denom else float("nan")


def report_cost(min_bytes, verbose):
    print("=" * 78)
    print("_COST / _COST_INT memory coefficients: source vs measured")
    print("=" * 78)
    print(f"{'kind':14s} {'curve':5s} {'int':4s} {'source':>10s} "
          f"{'measured':>10s} {'ratio':>7s}   points used")
    for integrated in (False, True):
        table = _COST_INT if integrated else _COST
        for (kind, curve), ((coef, (pn, pm)), _t) in sorted(table.items()):
            pts = measured(kind, integrated, curve, min_bytes=min_bytes)
            if not pts:
                continue
            vals = [per_unit(kind, s, v, pn, pm) for s, v in pts]
            got = vals[-1]
            flag = "" if 0.8 <= got / coef <= 1.25 else "  <-- drifted"
            detail = ", ".join(f"{v:.0f}" for v in vals[-3:]) if verbose else \
                     f"n={len(vals)}"
            print(f"{kind:14s} {curve:5s} {str(integrated):4s} {coef:10.0f} "
                  f"{got:10.0f} {got / coef:7.2f}   {detail}{flag}")


def report_grad(min_bytes, verbose):
    print()
    print("=" * 78)
    print("_GRAD_MEM_FACTOR: source vs measured (value-and-grad / forward)")
    print("=" * 78)
    print(f"{'kind':14s} {'curve':5s} {'int':4s} {'source':>7s} "
          f"{'measured':>9s}   basis")
    for (kind, curve, integrated), factor in sorted(_GRAD_MEM_FACTOR.items()):
        table = _COST_INT if integrated else _COST
        cost = table.get((kind, curve))
        if not cost:
            continue
        (_c, (pn, pm)) = cost[0]
        fwd = dict(measured(kind, integrated, curve, vg=False, min_bytes=min_bytes))
        vg = dict(measured(kind, integrated, curve, vg=True, min_bytes=min_bytes))
        # Compare at the largest size where BOTH exist. Taking each series'
        # own last point instead would divide values from different N -- and
        # worse, the value-and-grad curve stops earlier precisely because it
        # needs more memory, so its last point is never the forward one's.
        shared = sorted(set(fwd) & set(vg))
        if not shared:
            print(f"{kind:14s} {curve:5s} {str(integrated):4s} {factor:7.1f} "
                  f"{'--':>9s}   no size measured both ways")
            continue
        N = shared[-1]
        f_unit = per_unit(kind, N, fwd[N], pn, pm)
        v_unit = per_unit(kind, N, vg[N], pn, pm)
        got = v_unit / f_unit if f_unit else float("nan")
        flag = "" if 0.75 <= got / factor <= 1.33 else "  <-- drifted"
        print(f"{kind:14s} {curve:5s} {str(integrated):4s} {factor:7.1f} "
              f"{got:9.1f}   at N={N:.0f}: fwd {f_unit:.0f}, vg {v_unit:.0f}"
              f"{flag}")


def report_theory(min_bytes, verbose):
    print()
    print("=" * 78)
    print("THEORY_MEM / THEORY_MEM_INT: source vs measured asymptote")
    print("=" * 78)
    print("(these are what a sub-floor point is drawn as, so a low one puts the")
    print(" hollow markers below where the real curve goes)")
    print(f"{'kind':14s} {'curve':5s} {'int':4s} {'source@N':>12s} "
          f"{'measured':>12s} {'ratio':>7s}")
    for integrated in (False, True):
        table = THEORY_MEM_INT if integrated else THEORY_MEM
        for (pkind, curve), fn in sorted(table.items()):
            kind = pkind.replace("_", "-") if pkind.startswith("sample") else pkind
            pts = measured(kind, integrated, curve, min_bytes=min_bytes)
            if not pts:
                continue
            size, value = pts[-1]
            want = fn(size, M_PER_N)
            print(f"{kind:14s} {curve:5s} {str(integrated):4s} "
                  f"{want / 1e9:11.2f}G {value / 1e9:11.2f}G "
                  f"{value / want:7.2f}" +
                  ("" if 0.8 <= value / want <= 1.25 else "  <-- drifted"))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grad", action="store_true", help="only the gradient factors")
    ap.add_argument("--theory", action="store_true", help="only the theory constants")
    ap.add_argument("--cost", action="store_true", help="only the cost coefficients")
    ap.add_argument("--min-gb", type=float, default=0.004,
                    help="ignore points below this (default 0.004 GB = the 4 MB "
                         "above which a reading is meaningfully above the plot floor)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="show the last three per-unit values rather than a count")
    a = ap.parse_args()
    mb = a.min_gb * 1e9
    everything = not (a.grad or a.theory or a.cost)
    if everything or a.cost:
        report_cost(mb, a.verbose)
    if everything or a.grad:
        report_grad(mb, a.verbose)
    if everything or a.theory:
        report_theory(mb, a.verbose)


if __name__ == "__main__":
    main()
