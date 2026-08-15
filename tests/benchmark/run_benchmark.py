import argparse
import logging
import math
import os

import jax
import jax.numpy as jnp
import tinygp
from benchmark import (
    MACHINE_RAM_GB,
    RAM_HEADROOM,
    load_benchmark_data,
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
)
from plotting import make_benchmark_figure, use_paper_style

import smolgp

key = jax.random.PRNGKey(0)
jax.config.update("jax_enable_x64", True)

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
        default="workstation",
        help=(
            "Which machine this is running on; selects the memory budget used "
            "to derive the size cutoffs (combined with --gpu). Defaults to the "
            "workstation, where the production figures are generated. "
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
            "Memory budget in GB, overriding --machine. Taken literally, "
            f"without the {RAM_HEADROOM:.0%} headroom applied to the presets."
        ),
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help=(
            "Per-measurement runtime budget in seconds. A curve that exceeds it "
            "is retired for all larger sizes (runtime is monotonic in size), "
            "which bounds the total wall clock. Useful for a quick local pass; "
            "leave unset for production runs."
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
    n_sizes = 17
    logmax = 7
    max_seconds = args.max_seconds
    if args.quick:
        n_sizes = 9
        logmax = 5
        if max_seconds is None:
            max_seconds = 5.0
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

    # Cutoffs follow from the memory budget rather than being hardcoded per
    # machine, so the same command works on either box (see size_cutoffs).
    max_ram = args.max_ram
    if max_ram is None:
        # Leave headroom for the OS; see RAM_HEADROOM.
        max_ram = MACHINE_RAM_GB[args.machine][device] * RAM_HEADROOM
    cutoffs = size_cutoffs(max_ram, args.func, gpu=args.gpu)
    print(f"Size cutoffs for {args.machine} {device.upper()} ({max_ram:g} GB):")
    for name, c in sorted(cutoffs.items()):
        print(f"    {name:5s} {c:.3g}")

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
    true_kernel = tinygp.kernels.quasisep.SHO(omega=w, quality=Q, sigma=sigma)
    ################# Which kernels to benchmark ##################
    if args.int:
        ssm_kernel = smolgp.kernels.integrated.IntegratedSHO(
            omega=w, quality=Q, sigma=sigma, num_inst=1
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
    texp = 140.0 if args.int else 0.0
    readout = 40.0 if args.int else 0.0
    if args.int:
        print("Using integrated data with texp =", texp, "and readout =", readout)
    ############################################################
    isinst = "_int" if args.int else ""
    out_filename = f"results/{device}_{args.func}{isinst}_benchmark.pkl"

    if args.plot_only:
        pass  # nothing to run; jump straight to plotting below
    elif args.func in ["llh", "cond"]:
        if args.func == "llh":
            print("Benchmarking likelihood...")
            funcs = llh_funcs[int(args.int)]
            n_repeat = 7
            N_N = 17
            logN_min = 1
            logN_max = 7
        elif args.func == "cond":
            print("Benchmarking condition...")
            funcs = cond_funcs[int(args.int)]
            n_repeat = 7
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
            drop_outliers=True,
            use_gpu_profiler=args.gpu,
            exposure_quantities=(texp, readout) if args.int else None,
            max_seconds=max_seconds,
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
            n_repeat=7,
            N_N=n_sizes,
            logN_min=1,
            logN_max=logmax,
            maxN=1e5,  # in N
            cutoffs=cutoffs,  # in M
            use_gpu_profiler=args.gpu,
            exposure_quantities=(texp, readout) if args.int else None,
            max_seconds=max_seconds,
        )
    elif args.func == "sample-prior":
        # Prior draws are conditioned on nothing, so M (the number of sample
        # coordinates) is the only size parameter and becomes the x axis.
        print("Benchmarking prior sampling (scaling with M)...")
        funcs = sample_prior_funcs[int(args.int)]
        Ns, runtime, memory, outputs = run_prior_sample_benchmark(
            funcs,
            kernels,
            n_repeat=7,
            N_N=n_sizes,
            logM_min=1,
            logM_max=logmax,
            cutoffs=cutoffs,
            drop_outliers=True,
            use_gpu_profiler=args.gpu,
            exposure_quantities=(texp, readout) if args.int else None,
            max_seconds=max_seconds,
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
            n_repeat=7,
            N_N=n_sizes,
            logN_min=1,
            logN_max=logmax,
            maxN=1e5,
            cutoffs=cutoffs,
            use_gpu_profiler=args.gpu,
            exposure_quantities=(texp, readout) if args.int else None,
            max_seconds=max_seconds,
        )
    else:
        raise ValueError(
            "Argument must be one of 'llh', 'cond', 'pred', 'sample-prior', "
            "or 'sample-post'."
        )

    if not args.plot_only:
        print("Wrote results to", out_filename)
        save_benchmark_data(out_filename, Ns, runtime, memory, outputs)

    ############################## PLOTTING ##############################
    if args.plot or args.plot_only:
        kind = args.func.replace("-", "_")  # sample-prior -> sample_prior
        use_paper_style(usetex=not args.no_tex)
        cpu_data = load_benchmark_data(f"results/cpu_{args.func}{isinst}_benchmark.pkl")
        gpu_file = f"results/gpu_{args.func}{isinst}_benchmark.pkl"
        gpu_data = load_benchmark_data(gpu_file) if os.path.exists(gpu_file) else None
        if gpu_data is None:
            print(f"  (no {gpu_file}; plotting CPU curves only)")
        make_benchmark_figure(
            kind, cpu_data, gpu_data=gpu_data, integrated=args.int, savefig=True
        )
