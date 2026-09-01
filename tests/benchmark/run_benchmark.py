import argparse
import logging
import math
import os

import jax
import jax.numpy as jnp
import psutil
import tinygp
from benchmark import (
    MACHINE_RAM_GB,
    make_data_files,
    rebuild_from_points,
    RESERVE_FRAC,
    RESERVE_GB,
    existing_floors,
    format_bytes,
    load_benchmark_data,
    ram_budget,
    run_benchmark,
    run_pred_benchmark,
    run_prior_sample_benchmark,
    save_benchmark_data,
    size_cutoffs,
)
from funcs import (
    gp_cond,
    gp_llh,
    gp_pred,
    gp_sample_post,
    gp_sample_prior,
    igp_cond,
    igp_llh,
    igp_pred,
    igp_sample_post,
    igp_sample_prior,
    ipss_cond,
    ipss_llh,
    iss_cond,
    iss_llh,
    iss_pred,
    iss_sample_post,
    iss_sample_prior,
    pqs_cond,
    pqs_llh,
    pss_cond,
    pss_llh,
    qs_cond,
    qs_llh,
    qs_pred,
    qs_sample_post,
    qs_sample_prior,
    ss_cond,
    ss_llh,
    ss_pred,
    ss_sample_post,
    ss_sample_prior,
    gp_llh_vg,
    igp_llh_vg,
    ipss_llh_vg,
    iss_llh_vg,
    pqs_llh_vg,
    pss_llh_vg,
    qs_llh_vg,
    ss_llh_vg,
)
from plotting import make_benchmark_figure, use_paper_style

import smolgp

key = jax.random.PRNGKey(0)
jax.config.update("jax_enable_x64", True)

#: Per-call budget for --long-runs-only, in seconds. These are the points a
#: production sweep declines precisely because they are slow, so the budget has
#: to be loose enough to be worth the trip: 30 min per call lets the dense O(N^3)
#: curves reach one more grid point, and the _retired retirement in
#: benchmark() still stops each curve the first time it goes over.
LONG_RUN_MAX_SECONDS = 1800.0

#: ...and the lower bound. Tier 2 exists for points that are individually
#: expensive; anything faster than this per call is the production suite's job,
#: which already runs a 600 s per-call budget. A cheap point missing from
#: production is missing because of its memory cutoff or because it crashed,
#: and re-attempting it here fixes neither -- it just re-runs known failures,
#: since a failed point is stored as NaN and so looks unmeasured.
LONG_RUN_MIN_SECONDS = 600.0

# Suppress only JAX XLA bridge warnings
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

######################################## MAIN ########################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark smolgp/tinygp")
    parser.add_argument(
        "func",
        type=str,
        help=(
            "What to benchmark: 'llh', 'cond', 'pred', 'sample-prior', or "
            "'sample-post'."
        ),
    )
    parser.add_argument("--gpu", action="store_true", help="Run on GPU (default: CPU).")
    parser.add_argument(
        "--int",
        action="store_true",
        help="Run with integrated data (default: instantaneous data).",
    )
    parser.add_argument(
        "--machine",
        choices=sorted(MACHINE_RAM_GB),
        default=None,
        help=(
            "Derive the memory budget from a preset instead of measuring this "
            "machine. Use it to preview the cutoffs for a box you are not "
            "sitting at. "
            + ", ".join(
                f"{name}: {d['cpu']} GB CPU / {d['gpu']} GB GPU"
                for name, d in sorted(MACHINE_RAM_GB.items())
            )
        ),
    )
    parser.add_argument(
        "--max-ram",
        type=float,
        default=None,
        help=(
            "Memory budget in GB, overriding both the detected RAM and "
            f"--machine. Taken literally: no {RESERVE_GB:g} GB reserve and no "
            "safety factor on the cost constants, so use it deliberately."
        ),
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help=(
            "Per-measurement runtime budget in sec (default: 600, or 5 if --quick)."
            "Used twice: up front, to derive size cutoffs from the runtime cost "
            "model, and during the run, to retire a curve for all larger sizes "
            "once it blows the budget (runtime is monotonic in size). Pass inf "
            "to disable both -- note the dense curves are memory-feasible well "
            "past the point where one call would take days."
        ),
    )
    parser.add_argument(
        "--gpu-serial",
        action="store_true",
        help=(
            "Also run the serial solvers (SSM/QSM/GP) on the GPU. Off by "
            "default: those curves are measured on the CPU, and a GPU run's "
            "copies of them are never plotted, so they cost hours for nothing. "
            "Use only for a deliberate one-off CPU-vs-GPU comparison."
        ),
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default=None,
        help=(
            "Re-run only these grid sizes, comma separated, e.g. --sizes 56234 "
            "or --sizes 23713,56234. Matched to the nearest grid point. The "
            "aggregate is NOT rewritten by a partial run -- the measurements "
            "land in results/individual/, then --rebuild folds them in."
        ),
    )
    parser.add_argument(
        "--indices",
        type=str,
        default=None,
        help=(
            "Re-run only these 1-based grid positions, comma separated, e.g. "
            "--indices 11 for the point the sweep logs as (11/17). Same "
            "partial-run rules as --sizes."
        ),
    )
    parser.add_argument(
        "--value-and-grad",
        action="store_true",
        help=(
            "Benchmark the likelihood as a hyperparameter fit calls it: value "
            "AND gradient with respect to the kernel's parameters. llh only. "
            "Writes to a separate '<kind>_value_and_grad' family of results, "
            "checkpoints and figures, so it cannot be confused with the "
            "forward-only numbers. _COST is calibrated on the forward pass, so "
            "the memory coefficients are scaled by _GRAD_MEM_FACTOR (2.6x to "
            "98x, measured per curve) to account for reverse mode keeping the "
            "forward intermediates -- without which the model badly "
            "over-estimates how large a dense curve can go."
        ),
    )
    parser.add_argument(
        "--nrepeat",
        type=int,
        default=None,
        help=(
            "Repeats per measured point. The default is adaptive, chosen per "
            "point from its first call (NREPEAT_SCHEDULE: 7 below 1 s/call, 5 "
            "below 10 s, 3 below 60 s, 1 above), which is what production runs "
            "should use -- scatter falls with runtime, so a slow point needs "
            "fewer samples to pin its mean well inside a plot pixel. --quick "
            "defaults to a fixed 3 and --long-runs-only to 1. An explicit "
            "value here is fixed at every size; lower it for a fast check that "
            "the sweep runs and the numbers land in the right place. Outlier "
            "rejection costs two samples, so it is skipped below 5 repeats."
        ),
    )
    parser.add_argument(
        "--curves",
        default=None,
        help=(
            "Re-run only these curves, comma separated, e.g. --curves SSM or "
            "--curves SSM,pSSM. Useful when a change touches one "
            "implementation and re-measuring the others (GP/QSM are tinygp) "
            "would cost hours for identical numbers. Same partial-run rules "
            "as --sizes: the aggregate is left alone and --rebuild folds the "
            "new points in, keeping the untouched curves' existing "
            "measurements."
        ),
    )
    parser.add_argument(
        "--make-data",
        action="store_true",
        help=(
            "Only build the data/*.npz inputs for this kind's grid, then stop. "
            "Combine with --sizes/--indices to repair individual files, and "
            "--overwrite-data to rebuild ones that already exist. Sizes that "
            "cannot be built are reported and skipped rather than aborting."
        ),
    )
    parser.add_argument(
        "--overwrite-data",
        action="store_true",
        help="With --make-data, rewrite datasets that already exist.",
    )
    parser.add_argument(
        "--max-n",
        type=float,
        default=None,
        help=(
            "Refuse sizes above this. Integrated data has a hard ceiling near "
            "N = 9.9e6: generate_integrated_data samples the truth on a 1 s "
            "grid across the whole baseline, so it needs N * cadence * 1.2 "
            "points, which overflows int32 dimensions above that."
        ),
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help=(
            "Skip profiling and rebuild the aggregate result file from the "
            "per-point checkpoints in results/individual/. Use after a "
            "--sizes/--indices run to fold the new points in, or to recover an "
            "aggregate that was lost. Combine with --plot to redraw."
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Abridged local pass: fewer sizes, smaller maximum, and a 5 s "
            "per-measurement budget unless --max-seconds says otherwise."
        ),
    )
    parser.add_argument(
        "--long-runs-only",
        action="store_true",
        help=(
            "Fill in only the points a production sweep declines. For each "
            "curve the production cutoff becomes a *floor*, so the run "
            "measures the band (production cutoff, long cutoff] and "
            "re-measures nothing that already has a number. The long cutoff "
            "comes from a relaxed budget: the full detected RAM taken "
            "literally, and a per-call budget of "
            f"{LONG_RUN_MAX_SECONDS:g}s instead of the production default. "
            "Repeats default to 1, since these points cost minutes to hours "
            "each and their scatter at that runtime is well under 1%. Follows "
            "the partial-run rules: the aggregate is left alone and --rebuild "
            "folds the new points in. Expect some to be OOM-killed -- that is "
            "the intent, and a dead subprocess is recorded as NaN."
        ),
    )
    parser.add_argument(
        "--xla-only",
        action="store_true",
        help=(
            "Do not profile. Compile each point and record only XLA's buffer "
            "accounting (temp + output + argument) -- the computation's working "
            "set, excluding the interpreter, the JAX runtime, allocator slack "
            "and the CUDA context. Static, so it needs no timed run: the whole "
            "suite takes minutes. The result is MERGED into the existing "
            "aggregate, filling the xla slot of each memory entry and leaving "
            "every measured value untouched."
        ),
    )
    parser.add_argument(
        "--absolute-only",
        action="store_true",
        help=(
            "Re-execute each point once and record only the ABSOLUTE peak "
            "memory -- the whole process high-water mark, including the Python "
            "interpreter and JAX runtime, i.e. what must actually be free to "
            "run it. Merged into the existing aggregate: the mean, std and xla "
            "slots are left exactly as they are, and no per-point checkpoint is "
            "written, so a single-shot run cannot degrade the averaged values. "
            "Implies --nrepeat 1, since a peak needs no averaging."
        ),
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="After running, write the formatted figure to docs/_static/.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip benchmarking; just replot from existing results/*.pkl.",
    )
    parser.add_argument(
        "--no-tex",
        action="store_true",
        help="Render plot labels with mathtext instead of LaTeX.",
    )
    args = parser.parse_args()

    # --quick trades resolution for wall clock, so a full four-run sweep on a
    # laptop finishes in minutes rather than hours.
    #
    # `logmax` bounds the largest array actually built, which is NOT always N:
    # `pred` and `sample-post` draw at M = 100N, so their N ladder has to stop
    # two decades earlier to reach the same M. Without this correction --quick
    # is a no-op for those kinds (it caps N at 1e5, i.e. M at 1e7 -- a hundred
    # times *larger* than the M-scaled kinds).
    M_PER_N = 100
    SCALES_WITH_M = ("pred", "sample-post")
    # Production default. Clears every point in the currently deployed figures
    # (the slowest are --int cond GP at 392 s and --int pred GP at 442 s) while
    # still bounding the dense O(N^3) curves, which are memory-feasible far
    # past the point where one call would run for days. Note this is per call
    # and each point is measured n_repeat times, so it bounds a *point* at
    # ~600*n_repeat seconds, not the sweep.
    DEFAULT_MAX_SECONDS = 600.0
    n_sizes = 17
    logmax = 7
    max_seconds = args.max_seconds
    if args.quick:
        n_sizes = 9
        logmax = 5
        if max_seconds is None:
            max_seconds = 5.0
    if max_seconds is None:
        max_seconds = (
            LONG_RUN_MAX_SECONDS if args.long_runs_only else DEFAULT_MAX_SECONDS
        )
    if args.func in SCALES_WITH_M:
        logmax -= round(math.log10(M_PER_N))
    if args.quick:
        biggest = "M" if args.func in SCALES_WITH_M else "N"
        print(
            f"Quick mode: {n_sizes} sizes, {biggest} up to 1e{logmax + (2 if args.func in SCALES_WITH_M else 0)}, "
            f"{max_seconds:g}s per-measurement budget"
        )

    # # Set device
    if args.gpu:
        print("Running benchmark on GPU")
        device = machine = "gpu"
    else:
        print("Running benchmark on CPU")
        device = machine = "cpu"

    # Cutoffs follow from the memory and time budgets rather than being
    # hardcoded per machine, so the same command works on either box. Each
    # curve stops at whichever bound binds first (see size_cutoffs).
    # Exposure geometry, needed here as well as for data generation: it sets the
    # ceiling on how large an integrated dataset can be built at all, and hence
    # how far any integrated curve can be measured.
    texp = 140.0 if args.int else 0.0
    readout = 40.0 if args.int else 0.0

    budget = ram_budget(machine=args.machine, device=device, max_ram_gb=args.max_ram)
    cutoffs, bounds = size_cutoffs(
        budget,
        args.func,
        max_seconds=max_seconds,
        gpu=args.gpu,
        gpu_serial=args.gpu_serial,
        integrated=args.int,
        # Only an explicit --max-n caps the grid now. The old automatic ceiling
        # existed because integrated data generation was O(N * cadence) and blew
        # past 2**31 elements at N = 1e7; it now draws straight from the
        # integrated kernel in O(N), so every grid size is buildable and nothing
        # needs to be retired on the data generator's behalf.
        data_ceiling=args.max_n,
        # An explicit --max-ram is a deliberate, hand-picked budget: honour it
        # literally rather than shaving it again with the calibration margin.
        safety=1.0 if args.max_ram is not None else None,
        # Reverse mode keeps the forward pass's intermediates, and _COST is
        # calibrated on the forward pass, so the memory coefficients need
        # scaling (see _GRAD_MEM_FACTOR). Without this the model allowed
        # integrated GP up to N = 9.8e4 against a real limit near 1e4.
        value_and_grad=args.value_and_grad,
        detail=True,
    )
    if args.max_ram is not None:
        source = f"--max-ram {args.max_ram:g} GB, taken literally"
    elif args.machine is not None:
        source = f"{args.machine} preset, less reserve"
    else:
        source = (
            f"detected on {os.uname().nodename.split('.')[0]}, less "
            f"{min(RESERVE_GB, RESERVE_FRAC * psutil.virtual_memory().total / 1e9):g} GB reserve"
        )
    budget_s = "unbounded" if max_seconds == float("inf") else f"{max_seconds:g}s"

    # --long-runs-only inverts the window: the cutoffs just computed (under a
    # deliberately loose budget) become the top of the band, and the *floor* is
    # read off the existing result file further down, once its name is known.
    # The floor deliberately comes from what has already been measured rather
    # than from re-deriving the production cutoff: a production sweep may have
    # been run with its own --max-ram (llh-vg uses 170 GB), so the model would
    # reconstruct the wrong boundary, whereas "the largest size this curve
    # already has a number for" is exact whatever flags produced it.
    floors = {}
    if args.long_runs_only:
        print(
            f"Long-run fill-in. Upper bounds at "
            f"{format_bytes(budget)} / {budget_s} per call ({source}):"
        )
        for name, c in sorted(cutoffs.items()):
            print(f"    {name:5s} {c:9.3g}   ({bounds[name]}-bound)")
    else:
        print(
            f"Size cutoffs for {device.upper()} "
            f"({format_bytes(budget)} / {budget_s} per call; {source}):"
        )
        for name, c in sorted(cutoffs.items()):
            print(f"    {name:5s} {c:9.3g}   ({bounds[name]}-bound)")

    # Pre-flight sanity check. The budget is a promise about a specific device,
    # and the two ways to get it wrong are both silent and both fatal: a CPU
    # budget handed to a --gpu run (490 GB targeted at a 48 GB card), or a
    # budget above what this machine can actually hand out. There is no swap
    # here, so overshooting is a hard kill, not a slowdown.
    if device == "gpu":
        capacity = MACHINE_RAM_GB[args.machine or "workstation"]["gpu"] * 1e9
        what = "this GPU's capacity"
    else:
        capacity = psutil.virtual_memory().available
        what = "the RAM currently available"
    if budget > capacity:
        print(
            f"  WARNING: budget {format_bytes(budget)} exceeds {what} "
            f"({format_bytes(capacity)}). Largest points may be OOM-killed."
        )
    elif budget > 0.95 * capacity:
        print(
            f"  NOTE: budget is {budget / capacity:.0%} of {what} "
            f"({format_bytes(capacity)}) -- only {format_bytes(capacity - budget)} "
            "of slack, and swap is off. Run this on an otherwise idle box."
        )

    ## Setup function dictionaries
    llh_funcs = [
        {"SSM": ss_llh, "QSM": qs_llh, "GP": gp_llh, "pQSM": pqs_llh, "pSSM": pss_llh},
        {"SSM": iss_llh, "GP": igp_llh, "pSSM": ipss_llh},
    ]
    cond_funcs = [
        {
            "SSM": ss_cond,
            "QSM": qs_cond,
            "GP": gp_cond,
            "pQSM": pqs_cond,
            "pSSM": pss_cond,
        },
        {"SSM": iss_cond, "GP": igp_cond, "pSSM": ipss_cond},
    ]
    pred_funcs = [
        {"SSM": ss_pred, "QSM": qs_pred, "GP": gp_pred},
        {"SSM": iss_pred, "GP": igp_pred},
    ]
    # Sampling is only optimized on the serial solvers (the parallel ones fall
    # back to a per-sample loop), so there are no pSSM/pQSM curves here.
    sample_prior_funcs = [
        {"SSM": ss_sample_prior, "QSM": qs_sample_prior, "GP": gp_sample_prior},
        {"SSM": iss_sample_prior, "GP": igp_sample_prior},
    ]
    sample_post_funcs = [
        {"SSM": ss_sample_post, "QSM": qs_sample_post, "GP": gp_sample_post},
        {"SSM": iss_sample_post, "GP": igp_sample_post},
    ]
    ################### True GP parameters ######################
    S = 2.36
    w = 0.0195
    Q = 7.63
    sigma = jnp.sqrt(S * w * Q)
    # The kernel the datasets are drawn from. Instantaneous data uses tinygp's
    # quasiseparable SHO, which the benchmark itself shows is the fastest way to
    # draw an instantaneous prior. Integrated data must come from an integrated
    # state-space kernel instead, so the exposure averaging is done by the model
    # (O(N)) rather than by quadrature over a dense realization (O(N * cadence),
    # which overflowed at N = 1e7). true_kernel is used only for data generation.
    if args.int:
        true_kernel = smolgp.kernels.IntegratedSHO(
            omega=w, quality=Q, sigma=sigma, num_insts=1
        )
    else:
        true_kernel = tinygp.kernels.quasisep.SHO(omega=w, quality=Q, sigma=sigma)
    ################# Which kernels to benchmark ##################
    if args.int:
        ssm_kernel = smolgp.kernels.integrated.IntegratedSHO(
            omega=w, quality=Q, sigma=sigma, num_insts=1
        )
        gp_kernel = smolgp.kernels.dense.IntegratedSHOKernel(w=w, Q=Q, S=S)
        kernels = {
            "SSM": ssm_kernel,
            "GP": gp_kernel,
            "pSSM": ssm_kernel,
        }
    else:
        qsm_kernel = tinygp.kernels.quasisep.SHO(omega=w, quality=Q, sigma=sigma)
        ssm_kernel = smolgp.kernels.SHO(omega=w, quality=Q, sigma=sigma)
        gp_kernel = smolgp.kernels.dense.SHOKernel(w=w, Q=Q, S=S)
        kernels = {
            "SSM": ssm_kernel,
            "QSM": qsm_kernel,
            "GP": gp_kernel,
            "pSSM": ssm_kernel,
            "pQSM": qsm_kernel,
        }
    ################ Data properties ####################
    yerr = 0.3
    if args.int:
        print("Using integrated data with texp =", texp, "and readout =", readout)
    ############################################################
    isinst = "_int" if args.int else ""
    # A --quick pass is an abridged grid with a 5 s budget -- not comparable to
    # a production sweep, and it must never land on the production filename.
    # (It used to: one `--quick` run would silently replace a multi-hour
    # result with nine truncated points.)
    isquick = "_quick" if args.quick else ""
    isvg = "_value_and_grad" if args.value_and_grad else ""
    if args.value_and_grad and args.func != "llh":
        raise SystemExit(
            "--value-and-grad is implemented for 'llh' only. The predict and "
            "sample benchmarks take (t_test, gp, y) rather than (data, kernel), "
            "so they need their own wrappers before they can be differentiated."
        )
    out_filename = f"results/{device}_{args.func}{isinst}{isvg}{isquick}_benchmark.pkl"

    if args.long_runs_only:
        # pred and sample-post store their x axis in N but express cutoffs in
        # M, so the stored sizes need scaling before they can be compared.
        # sample-prior already stores M, and llh/cond are in N throughout.
        floors = existing_floors(
            out_filename,
            cutoffs,
            m_per_n=M_PER_N if args.func in SCALES_WITH_M else None,
        )
        print("  Floors, from the largest size each curve already has:")
        empty = True
        for name in sorted(cutoffs):
            f, c = floors.get(name, 0.0), cutoffs[name]
            if c <= f:
                print(f"    {name:5s} {f:9.3g}  ->  {c:9.3g}   nothing to do")
            else:
                empty = False
                print(f"    {name:5s} {f:9.3g}  ->  {c:9.3g}   to measure")
        if empty:
            # A no-op, not a failure: a suite that loops over every kind will
            # hit this for the ones already complete, and exiting non-zero
            # there would fill the summary with false alarms.
            print(
                "  Every curve is already measured up to its long-run cutoff, "
                "so there is nothing to fill in. Raise --max-seconds or "
                "--max-ram to go further."
            )
            raise SystemExit(0)

    def _int_list(v):
        return [int(x) for x in v.split(",") if x.strip()] if v else None

    # Repeats per point. --quick is for "does this run at all", so it drops to
    # 3 by default; an explicit --nrepeat always wins.
    # None => adaptive (benchmark.NREPEAT_SCHEDULE picks per point from the
    # first call). --quick stays on a small fixed count: it exists to answer
    # "does this run at all", and adaptive would give it 7 repeats at every
    # size, since its whole grid is sub-second.
    n_repeat = args.nrepeat if args.nrepeat is not None else (3 if args.quick else None)
    if args.absolute_only:
        n_repeat = 1
    if n_repeat is None and args.long_runs_only:
        # Every point in a long run is minutes to hours per call, where the
        # measured repeat-to-repeat scatter is 0.4% -- far below a plot pixel.
        # Paying for 3 of them would triple a ~10 h sweep to buy nothing.
        n_repeat = 1
    if n_repeat is not None and n_repeat < 1:
        raise SystemExit(f"--nrepeat must be at least 1, got {n_repeat}")
    if n_repeat is not None:
        print(
            f"Note: fixed n_repeat = {n_repeat} at every size (default is "
            "adaptive: 7 / 5 / 3 / 1 by per-call time)."
        )

    only_sizes = _int_list(args.sizes)
    only_indices = _int_list(args.indices)
    only_curves = (
        [c.strip() for c in args.curves.split(",") if c.strip()]
        if args.curves
        else None
    )
    # A long run measures a disjoint band of sizes, so its result arrays are
    # NaN everywhere the production sweep has numbers. Writing that over the
    # aggregate would erase the sweep; treat it as partial and let --rebuild
    # merge the per-point checkpoints in.
    partial = bool(only_sizes or only_indices or only_curves or args.long_runs_only)

    # Naming sizes is a deliberate request for those points, exactly as
    # --long-runs-only is, so an over-budget result is the measurement rather
    # than an accident: retire the curve but keep the number. Without this a
    # re-measurement of a known-slow point silently throws itself away -- which
    # is what happened to llh-vg GP at N=56234, 949 s of compute discarded for
    # being over a 600 s budget it was never meant to respect.
    keep_over_budget = bool(args.long_runs_only or only_sizes or only_indices)

    if args.value_and_grad:
        # Same curves, differentiated. Only llh reaches here (checked above).
        llh_funcs = [
            {"SSM": ss_llh_vg, "QSM": qs_llh_vg, "GP": gp_llh_vg,
             "pQSM": pqs_llh_vg, "pSSM": pss_llh_vg},
            {"SSM": iss_llh_vg, "GP": igp_llh_vg, "pSSM": ipss_llh_vg},
        ]
        print("Benchmarking value AND gradient (hyperparameter-fit cost)")

    if only_curves:
        # Drop the unrequested curves from every kind's function dict. Done in
        # one place rather than per-dispatch so a new kind cannot forget it.
        _all = [llh_funcs, cond_funcs, pred_funcs, sample_prior_funcs, sample_post_funcs]
        known = {c for fl in _all for d in fl for c in d}
        unknown = sorted(set(only_curves) - known)
        if unknown:
            raise SystemExit(
                f"Unknown curve(s): {', '.join(unknown)}. "
                f"Known curves: {', '.join(sorted(known))}"
            )
        for fl in _all:
            for i, d in enumerate(fl):
                fl[i] = {k: v for k, v in d.items() if k in only_curves}
        print(f"Restricting to curves: {', '.join(only_curves)}")

    if args.make_data:
        print(f"Building data files for {args.func}{isinst or ''}...")
        _w, _s, _f = make_data_files(
            true_kernel, args.func, yerr=yerr,
            exposure_quantities=(texp, readout) if args.int else None,
            n_sizes=n_sizes, logmin=1,
            logmax=logmax + (round(math.log10(M_PER_N)) if args.func in SCALES_WITH_M else 0),
            m_per_n=M_PER_N, only_sizes=only_sizes, only_indices=only_indices,
            overwrite=args.overwrite_data, max_n=args.max_n,
        )
        raise SystemExit(1 if _f else 0)

    if args.rebuild:
        Ns, runtime, memory, outputs = rebuild_from_points(
            args.func + isvg, device, integrated=args.int, m_per_n=M_PER_N,
            n_sizes=n_sizes, logmin=1,
            logmax=logmax + (round(math.log10(M_PER_N)) if args.func in SCALES_WITH_M else 0),
            tag=isvg + isquick,
        )
    elif args.plot_only:
        pass  # nothing to run; jump straight to plotting below
    elif args.func in ["llh", "cond"]:
        if args.func == "llh":
            print("Benchmarking likelihood...")
            funcs = llh_funcs[int(args.int)]
            N_N = 17
            logN_min = 1
            logN_max = 7
        elif args.func == "cond":
            print("Benchmarking condition...")
            funcs = cond_funcs[int(args.int)]
            N_N = 17
            logN_min = 1
            logN_max = 7

        Ns, runtime, memory, outputs = run_benchmark(
            true_kernel,
            funcs,
            kernels,
            yerr=yerr,
            n_repeat=n_repeat,
            N_N=n_sizes,
            logN_min=logN_min,
            logN_max=logmax,
            cutoffs=cutoffs,
            floors=floors,
            keep_over_budget=keep_over_budget,
            min_seconds=LONG_RUN_MIN_SECONDS if args.long_runs_only else None,
            xla_only=args.xla_only,
            no_checkpoint=args.absolute_only,
            drop_outliers=True,
            use_gpu_profiler=args.gpu,
            exposure_quantities=(texp, readout) if args.int else None,
            max_seconds=max_seconds,
            tag=isvg + isquick,
            only_sizes=only_sizes,
            only_indices=only_indices,
        )
    elif args.func == "pred":
        print("Benchmarking prediction...")
        funcs = pred_funcs[int(args.int)]
        # cutoffs come from the memory budget above; M = 100N inside the runner
        Ns, runtime, memory, outputs = run_pred_benchmark(
            true_kernel,
            funcs,
            kernels,
            yerr=yerr,
            n_repeat=n_repeat,
            N_N=n_sizes,
            logN_min=1,
            logN_max=logmax,
            maxN=1e5,  # in N
            cutoffs=cutoffs,  # in M
            floors=floors,  # in M
            keep_over_budget=keep_over_budget,
            min_seconds=LONG_RUN_MIN_SECONDS if args.long_runs_only else None,
            xla_only=args.xla_only,
            no_checkpoint=args.absolute_only,
            use_gpu_profiler=args.gpu,
            exposure_quantities=(texp, readout) if args.int else None,
            max_seconds=max_seconds,
            tag=isvg + isquick,
            only_sizes=only_sizes,
            only_indices=only_indices,
        )
    elif args.func == "sample-prior":
        # Prior draws are conditioned on nothing, so M (the number of sample
        # coordinates) is the only size parameter and becomes the x axis.
        print("Benchmarking prior sampling (scaling with M)...")
        funcs = sample_prior_funcs[int(args.int)]
        Ns, runtime, memory, outputs = run_prior_sample_benchmark(
            funcs,
            kernels,
            n_repeat=n_repeat,
            N_N=n_sizes,
            logM_min=1,
            logM_max=logmax,
            cutoffs=cutoffs,
            floors=floors,
            keep_over_budget=keep_over_budget,
            min_seconds=LONG_RUN_MIN_SECONDS if args.long_runs_only else None,
            xla_only=args.xla_only,
            no_checkpoint=args.absolute_only,
            drop_outliers=True,
            use_gpu_profiler=args.gpu,
            exposure_quantities=(texp, readout) if args.int else None,
            max_seconds=max_seconds,
            tag=isvg + isquick,
            only_sizes=only_sizes,
            only_indices=only_indices,
        )
    elif args.func == "sample-post":
        # Posterior draws mirror `pred`: N training points, M = 100N sample
        # coordinates, so run_pred_benchmark drives them unchanged.
        print("Benchmarking posterior sampling (M = 100N)...")
        funcs = sample_post_funcs[int(args.int)]
        Ns, runtime, memory, outputs = run_pred_benchmark(
            true_kernel,
            funcs,
            kernels,
            yerr=yerr,
            n_repeat=n_repeat,
            N_N=n_sizes,
            logN_min=1,
            logN_max=logmax,
            maxN=1e5,
            cutoffs=cutoffs,
            floors=floors,
            keep_over_budget=keep_over_budget,
            min_seconds=LONG_RUN_MIN_SECONDS if args.long_runs_only else None,
            xla_only=args.xla_only,
            no_checkpoint=args.absolute_only,
            use_gpu_profiler=args.gpu,
            exposure_quantities=(texp, readout) if args.int else None,
            max_seconds=max_seconds,
            tag=isvg + isquick,
            only_sizes=only_sizes,
            only_indices=only_indices,
        )
    else:
        raise ValueError(
            "Argument must be one of 'llh', 'cond', 'pred', 'sample-prior', "
            "or 'sample-post'."
        )

    if args.xla_only or args.absolute_only:
        slot = 3 if args.xla_only else 2
        what = "xla" if args.xla_only else "absolute"
        # Merge, never overwrite. This pass produces only the xla figure, so it
        # fills slot [3] of each memory entry and leaves the measured mean, std
        # and absolute peak exactly as they were. Entries that predate the
        # widened tuple are padded to four fields on the way through.
        if not os.path.exists(out_filename):
            raise SystemExit(
                f"{out_filename} does not exist -- --{what}-only merges into an "
                "existing aggregate rather than creating one. Run the sweep first."
            )
        old_data = load_benchmark_data(out_filename)
        old_mem = old_data.get("memory", {})
        # Match by size, never by position. --sizes/--indices narrow the grid,
        # so this run's arrays are shorter than the aggregate's and a positional
        # merge silently writes each value onto the wrong N -- measuring
        # N = 749 and storing it as N = 23.
        fresh_by_n = {}
        for curve in memory:
            fresh_by_n[curve] = {
                int(n): e for n, e in zip(Ns, memory[curve])
            }
        merged, filled = {}, 0
        for curve, entries in old_mem.items():
            lookup = fresh_by_n.get(curve, {})
            out = []
            for n, e in zip(old_data["Ns"], entries):
                e = list(tuple(e) + (float("nan"),) * (4 - len(e)))
                got = lookup.get(int(n))
                x = got[slot] if got is not None and len(got) > slot else float("nan")
                if x == x:  # not NaN
                    filled += 1
                    e[slot] = x
                out.append(tuple(e))
            merged[curve] = out
        save_benchmark_data(
            out_filename, old_data["Ns"], old_data["runtime"], merged,
            old_data["outputs"],
        )
        print(f"  filled the {what} slot for {filled} points; other values "
              f"untouched\n  wrote {out_filename}")
    elif args.rebuild:
        print("Wrote results to", out_filename)
        save_benchmark_data(out_filename, Ns, runtime, memory, outputs)
    elif partial:
        # A --sizes/--indices run measured a handful of points. Writing those
        # to the aggregate would replace a full sweep with a two-point file,
        # so it is deliberately not written. The measurements are safe in
        # results/individual/; --rebuild merges them back.
        print(
            f"\n  Partial run: {out_filename} left untouched.\n"
            f"  The new points are checkpointed in results/individual/. Fold "
            f"them in with:\n"
            f"      uv run run_benchmark.py {args.func}"
            f"{' --int' if args.int else ''}{' --gpu' if args.gpu else ''}"
            f" --rebuild --plot"
        )
    elif not args.plot_only:
        print("Wrote results to", out_filename)
        save_benchmark_data(out_filename, Ns, runtime, memory, outputs)

    ############################## PLOTTING ##############################
    if args.plot or args.plot_only:
        kind = args.func.replace("-", "_")  # sample-prior -> sample_prior
        use_paper_style(usetex=not args.no_tex)
        # isquick matters here: an abridged run has its own aggregate and its own
        # figure, and must not read or overwrite the production ones.
        cpu_data = load_benchmark_data(
            f"results/cpu_{args.func}{isinst}{isvg}{isquick}_benchmark.pkl"
        )
        gpu_file = f"results/gpu_{args.func}{isinst}{isvg}{isquick}_benchmark.pkl"
        gpu_data = load_benchmark_data(gpu_file) if os.path.exists(gpu_file) else None
        if gpu_data is None:
            print(f"  (no {gpu_file}; plotting CPU curves only)")
        make_benchmark_figure(
            kind, cpu_data, gpu_data=gpu_data, integrated=args.int,
            tag=isvg,
            suffix=isquick,
            title_suffix=" + gradient" if args.value_and_grad else "",
            # An abridged run is for eyeballing, so its figure goes to
            # scratch_figures/ rather than the published docs directory.
            scratch=args.quick,
            savefig=True,
        )
