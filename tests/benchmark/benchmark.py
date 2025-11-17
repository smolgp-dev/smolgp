import time
import numpy as np

import jax
import jax.numpy as jnp
import tinygp
import smolgp

# import matplotlib as mpl
import matplotlib.pyplot as plt

import pickle
import os
import psutil
import multiprocessing as mp

mp.set_start_method("spawn", force=True)

key = jax.random.PRNGKey(0)

__all__ = [
    "save_benchmark_data",
    "load_benchmark_data",
    "run_benchmark",
    "plot_benchmark",
    "generate_data",
    "colors",
]

## Colors for benchmarking plots
colors = {"SSM": "#1f77b4", "QSM": "#ff7f0e", "GP": "#2ca02c", "pSSM": "#6A0E95"}


def generate_data(N, kernel, yerr=0.3):
    t_train = jnp.linspace(0, 86400, N)
    true_gp = tinygp.GaussianProcess(kernel, t_train)
    y_true = true_gp.sample(key=jax.random.PRNGKey(32))
    y_train = y_true + yerr * jax.random.normal(key, shape=(N,))
    return t_train, y_train


def save_benchmark_data(filename, Ns, runtime_llh, memory_llh, outputs):
    import pickle

    data = {
        "Ns": Ns,
        "runtime_llh": runtime_llh,
        "memory_llh": memory_llh,
        "outputs": outputs,
    }
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def load_benchmark_data(filename):
    import pickle

    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data["Ns"], data["runtime_llh"], data["memory_llh"], data["outputs"]


def scale_nans(runtime_array, Ns, power=1):
    # Find the last non-nan index
    last_valid_idx = jnp.where(~jnp.isnan(runtime_array[:, 0]))[0][-1]
    last_valid_runtime = runtime_array[last_valid_idx, 0]
    last_valid_N = Ns[last_valid_idx]

    # Scale the nans based on the last valid point
    for i in range(last_valid_idx + 1, len(runtime_array)):
        runtime_array = runtime_array.at[i, 0].set(
            last_valid_runtime * (Ns[i] / last_valid_N) ** power
        )
        runtime_array = runtime_array.at[i, 1].set(0)  # Set std to 0 for scaled points

    return runtime_array


def plot_benchmark(
    Ns,
    runtimes,
    ax=None,
    savefig=None,
    scale=True,
    powers={"SSM": 1, "QSM": 1, "GP": 3, "pSSM": 1},
):
    """
    runtimes should be a dict with values (mean, std) for each N
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), sharex=True)

    for name in runtimes:
        runtime_array = jnp.array(runtimes[name])
        mean_runtime = runtime_array[:, 0]
        std_runtime = runtime_array[:, 1]
        if scale:
            scaled = scale_nans(runtime_array, Ns, power=powers[name])
            ax.errorbar(Ns, scaled[:, 0], scaled[:, 1], c=colors[name], ls=":")
        ax.errorbar(Ns, mean_runtime, std_runtime, c=colors[name], fmt=".-", label=name)

    ax.legend()
    ax.set(
        xscale="log", yscale="log", xlabel="Number of data points", ylabel="Runtime [s]"
    )
    tens = 10 ** jnp.arange(jnp.log10(Ns[0]), jnp.log10(Ns[-1]) + 1)
    ax.set_xticks(tens, labels=["" for _ in tens], minor=True)
    ax.grid(alpha=0.5, zorder=-10)
    ax.grid(alpha=0.5, zorder=-10, which="minor", axis="x")
    # ax.set_ylim(top=jnp.nanmax(runtime_ss[:,0])*1e3)
    if savefig:
        plt.savefig(savefig, dpi=300, bbox_inches="tight")
    return ax


def _runner(fn_bytes, kernel_bytes, args_bytes, return_pipe):
    """
    Helper function to run a function in an isolated subprocess for memory profiling.
    """
    # Unpickle function and arguments inside isolated subprocess
    fn = pickle.loads(fn_bytes)
    kernel = pickle.loads(kernel_bytes)
    args = pickle.loads(args_bytes)

    proc = psutil.Process(os.getpid())

    # Create the jitted function here
    @jax.jit
    def fn_jit(*args):
        return fn(*args, kernel=kernel)

    # Warm up (JIT compilation)
    out = fn_jit(*args)
    if hasattr(out, "block_until_ready"):
        out.block_until_ready()

    # Peak memory tracking via polling
    peak_rss = 0
    baseline_rss = proc.memory_info().rss

    def track_memory():
        nonlocal peak_rss
        while True:
            try:
                m = proc.memory_info().rss
                peak_rss = max(peak_rss, m)
                time.sleep(0.01)
            except psutil.NoSuchProcess:
                break

    import threading

    t = threading.Thread(target=track_memory)
    t.daemon = True
    t.start()

    # Time the function with JAX block_until_ready
    start = time.perf_counter()
    out = fn_jit(*args)
    if hasattr(out, "block_until_ready"):
        out.block_until_ready()
    end = time.perf_counter()

    # Return both result (pickled) and stats
    return_pipe.send(
        {
            "output": pickle.dumps(out),
            "runtime": end - start,
            "peak_mem": peak_rss,
        }
    )
    return_pipe.close()


def profile_jax_function(fn, kernel, *args, n_repeat=5):
    """
    JAX profiler for time benchmarking and memory tracking using isolated subprocesses.
    """
    fn_bytes = pickle.dumps(fn)
    kernel_bytes = pickle.dumps(kernel)
    args_bytes = pickle.dumps(args)

    runtimes = []
    peaks = []
    output = None

    for _ in range(n_repeat):
        parent_conn, child_conn = mp.Pipe()
        p = mp.Process(
            target=_runner, args=(fn_bytes, kernel_bytes, args_bytes, child_conn)
        )
        p.start()
        result = parent_conn.recv()
        p.join()

        runtimes.append(result["runtime"])
        peaks.append(result["peak_mem"])
        output = pickle.loads(result["output"])

    return (
        (np.mean(runtimes), np.std(runtimes)),
        (np.mean(peaks), np.std(peaks)),
        output,
    )


def benchmark(funcs, kernels, data, n_repeat=3, cutoffs={}):
    """
    Given some (jitted) functions, benchmark their runtimes over a range of input sizes.

    Parameters
    ----------
    funcs : list of callables
        List of functions to benchmark. Each function should take a single input array.
    data : list of tuples
        List of data tuples (t_train, y_train, yerr) for each input size

    Returns
    -------
    Ns : list of input sizes
    runtime : dict
        Dictionary mapping function names to lists of runtimes (means and stds) for each input size.
    memory : dict
        Dictionary mapping function names to lists of memory usages (means and stds) for each input size.
    outputs : dict
        Dictionary mapping function names to lists of outputs for each input size.
    """

    runtime = {}
    memory = {}
    outputs = {}
    Ns = []
    for n in range(len(data)):
        N = data[n][0].shape[0]
        Ns.append(N)
        print(f"  ({n + 1}/{len(data)}):  N = {N}")
        for name in funcs:
            func = funcs[name]
            kernel = kernels[name]
            cutoff = cutoffs.get(name, 3e4)

            if N <= cutoff:
                t, mem, val = profile_jax_function(
                    func, kernel, data[n], n_repeat=n_repeat
                )
                basestr = f"    {name}: time = {t[0]:.4f} ± {t[1]:.4f} s"
                memstr = f", mem = {format_bytes(mem[0])} ± {format_bytes(mem[1])}"
                print(basestr + memstr)
            else:
                t, mem, val = (jnp.nan, jnp.nan), (jnp.nan, jnp.nan), jnp.nan
                print(f"    {name}: Skipped (N={N} > cutoff={cutoff})")

            if name not in runtime:
                runtime[name] = []
                memory[name] = []
                outputs[name] = []

            runtime[name].append(t)
            memory[name].append(mem)
            outputs[name].append(val)

    return Ns, runtime, memory, outputs


def run_benchmark(
    true_kernel,
    funcs,
    kernels,
    yerr=0.3,
    N_N=10,
    logN_min=1,
    logN_max=7,
    n_repeat=3,
    cutoffs={},
):
    """
    Generate data and benchmark the provided functions over a range of input sizes.
    """
    print("Generating data for benchmarking...")
    ## Generate all data ahead of time
    Ns = jnp.logspace(logN_min, logN_max, N_N).astype(int)
    data = []
    for N in Ns:
        t_train, y_train = generate_data(N, true_kernel, yerr=yerr)
        data.append(jnp.array([t_train, y_train, jnp.full_like(t_train, yerr)]))

    print("Running benchmark...")
    Ns, runtime, memory, outputs = benchmark(
        funcs, kernels, data, n_repeat=n_repeat, cutoffs=cutoffs
    )
    return Ns, runtime, memory, outputs


#### MOVE ALL THIS INTO RUN_BENCHMARK.PY
#################### PREDICT CONDITION ####################
def benchmark_prediction(
    ssm_kernel,
    qsm_kernel,
    gp_kernel=None,
    true_kernel=None,
    yerr=0.3,
    N_M=10,
    logM_min=1,
    logM_max=7,
    n_repeat=3,
    cutoffs={},
    N=1000,
):
    runtime = {}
    memory = {}
    outputs = {}

    def block(out):
        mu, var = out
        m = mu.block_until_ready()
        P = var.block_until_ready()
        return m, P

    kernel = qsm_kernel if true_kernel is None else true_kernel
    Ms = jnp.logspace(logM_min, logM_max, N_M).astype(int)

    for M in Ms:
        # t_train, y_train = generate_data(N, yerr, baseline_minutes=900)
        t_train, y_train = generate_data(N, kernel)
        # t_train, y_train = generate_data(M, kernel)
        dt = 0.1 * (t_train.max() - t_train.min())
        t_test = jnp.linspace(t_train.min() - dt, t_train.max() + dt, M)

        ## Prepare GP objects
        gp_qs = tinygp.GaussianProcess(qsm_kernel, t_train, diag=yerr**2)
        gp_gp = tinygp.GaussianProcess(gp_kernel, t_train, diag=yerr**2)
        gp_ss = smolgp.GaussianProcess(ssm_kernel, t_train, diag=yerr**2)
        _, condGPss = gp_ss.condition(y_train)

        ## Only time the actual prediction part
        @jax.jit
        def ss_pred(t_test):
            # _, condGPss = gp_ss.condition(y_train)
            return condGPss.predict(t_test, return_var=True)

        @jax.jit
        def qs_pred(t_test):
            return gp_qs.predict(y_train, t_test, return_var=True)

        @jax.jit
        def gp_pred(t_test):
            return gp_gp.predict(y_train, t_test, return_var=True)

        funcs = {"SSM": ss_pred, "QSM": qs_pred, "GP": gp_pred}

        _, t, mem, val = benchmark(funcs, [t_test], n_repeat=n_repeat, cutoffs=cutoffs)

        for name in funcs:
            if name not in runtime:
                runtime[name] = []
                memory[name] = []
                outputs[name] = []

            runtime[name].append(t)
            memory[name].append(mem)
            outputs[name].append(val)

    return Ms, runtime, memory, outputs


def format_bytes(n):
    if n == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(n)
    idx = 0
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024
        idx += 1
    return f"{size:.1f} {units[idx]}"
