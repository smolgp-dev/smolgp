import logging

# Suppress only JAX XLA bridge warnings
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

import argparse
import jax
import jax.numpy as jnp

import tinygp
import smolgp
from benchmark import *

import sys
import kernels as testgp

key = jax.random.PRNGKey(0)
jax.config.update("jax_enable_x64", True)

from funcs import *

######################################## MAIN ########################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark smolgp/tinygp")
    parser.add_argument(
        "func", type=str, help="Function to benchmark: 'llh' or 'cond'."
    )
    parser.add_argument("--gpu", action="store_true", help="Run on GPU (default: CPU).")
    parser.add_argument("--int", action="store_true", help="Run with integrated data (default: instantaneous data).")
    args = parser.parse_args()

    # # Set device
    if args.gpu:
        # jax.config.update("jax_platform_name", "gpu")
        print("Running benchmark on GPU")
        machine = 'gpu'
        if args.int:
            cutoffs={"GP": 0, "SSM": 0, "QSM": 0, "pSSM": 1e6}
        else:
            cutoffs={"GP": 0, "SSM": 0, "QSM": 0, "pSSM": 1e7}
    else:
        # jax.config.update("jax_platform_name", "cpu")
        print("Running benchmark on CPU")
        machine = 'cpu'
        cutoffs={"GP": 6e4, "SSM": 1e7, "QSM": 1e7, "pSSM": 0}

    ## Setup function dictionaries
    llh_funcs = [{"SSM": ss_llh, "QSM": qs_llh, "GP": gp_llh, "pSSM": pss_llh},
                 {"SSM": iss_llh, "GP": igp_llh, "pSSM": ipss_llh}]
    cond_funcs = [{"SSM": ss_cond, "QSM": qs_cond, "GP": gp_cond, "pSSM": pss_cond},
                  {"SSM": iss_cond, "GP": igp_cond, "pSSM": ipss_cond}]
    pred_funcs = [{"SSM": ss_pred, "QSM": qs_pred, "GP": gp_pred},
                  {"SSM": iss_pred, "GP": igp_pred}]
    ################### True GP parameters ######################
    S = 2.36
    w = 0.0195
    Q = 7.63
    sigma = jnp.sqrt(S * w * Q)
    true_kernel = tinygp.kernels.quasisep.SHO(omega=w, quality=Q, sigma=sigma)
    ################# Which kernels to benchmark ##################
    if args.int:
        ssm_kernel = smolgp.kernels.integrated.IntegratedSHO(omega=w, quality=Q, sigma=sigma, num_inst=1)
        gp_kernel = testgp.IntegratedSHOKernel(w=w, Q=Q, S=S)
        kernels = {
            "SSM": ssm_kernel,
            "GP": gp_kernel,
            "pSSM": ssm_kernel,
        }
    else:
        qsm_kernel = tinygp.kernels.quasisep.SHO(omega=w, quality=Q, sigma=sigma)
        ssm_kernel = smolgp.kernels.SHO(omega=w, quality=Q, sigma=sigma)
        gp_kernel = testgp.SHOKernel(w=w, Q=Q, S=S)
        kernels = {
            "SSM": ssm_kernel,
            "QSM": qsm_kernel,
            "GP": gp_kernel,
            "pSSM": ssm_kernel,
        }
    ################ Data properties ####################
    yerr = 0.3
    texp = 140. if args.int else 0.
    readout = 40. if args.int else 0.
    if args.int:
        print("Using integrated data with texp =", texp, "and readout =", readout)
    ############################################################
    if args.func in ["llh", "cond"]:
        if args.func == "llh":
            print("Benchmarking likelihood...")
            funcs = llh_funcs[int(args.int)]
            n_repeat=7
            N_N=17
            logN_min=1
            logN_max=7
        elif args.func == "cond":
            print("Benchmarking condition...")
            funcs = cond_funcs[int(args.int)]
            n_repeat=7
            N_N=17
            logN_min=1
            logN_max=7

        Ns, runtime, memory, outputs = run_benchmark(
            true_kernel, funcs, kernels, yerr=yerr,
            n_repeat=n_repeat, N_N=N_N, logN_min=logN_min, logN_max=logN_max,
            cutoffs=cutoffs,
            drop_outliers=True,
            use_gpu_profiler=args.gpu,
            exposure_quantities=(texp, readout) if args.int else None
        )
    elif args.func == "pred":
        print("Benchmarking prediction...")
        funcs = pred_funcs[int(args.int)]
        if args.gpu:
            # cutoffs={"GP": 3e5, "SSM": 3e5, "QSM": 3e5, "pSSM": 3e5} # these are now cutoffs in M
            cutoffs={"GP": 0, "SSM": 1e6, "QSM": 0} # these are now cutoffs in M
        else:
            cutoffs={"GP": 1e6, "SSM": 1e6, "QSM": 1e6} # these are now cutoffs in M

        # M is set to be 100x N inside run_pred_benchmark
        Ns, runtime, memory, outputs = run_pred_benchmark(
            true_kernel,
            funcs,
            kernels,
            yerr=yerr,
            n_repeat=7,
            N_N=17,
            logN_min=1,
            logN_max=7,
            maxN=1e5, # in N
            cutoffs=cutoffs, # in M
            use_gpu_profiler=args.gpu,
            exposure_quantities=(texp, readout) if args.int else None
        )
    else:
        raise ValueError("Argument must be one of 'llh', 'cond', or 'pred'.")

    isinst = "_int" if args.int else ""
    out_filename = f"results/{machine}_{args.func}{isinst}_benchmark.pkl"
    print("Wrote results to", out_filename)
    save_benchmark_data(out_filename, Ns, runtime, memory, outputs)
