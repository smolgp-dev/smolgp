import time
import numpy as np

# import matplotlib as mpl
import matplotlib.pyplot as plt

import pickle
import os
import psutil
import multiprocessing as mp
import threading, subprocess

mp.set_start_method("spawn", force=True)

import jax
import jax.numpy as jnp
import tinygp
import smolgp

key = jax.random.PRNGKey(0)

__all__ = [
    "save_benchmark_data",
    "load_benchmark_data",
    "run_benchmark",
    "run_pred_benchmark",
    "generate_data",
]

def generate_data(N, kernel, yerr=0.3):
    t_train = jnp.linspace(0, 86400, N)
    true_gp = tinygp.GaussianProcess(kernel, t_train)
    y_true = true_gp.sample(key=jax.random.PRNGKey(32))
    y_train = y_true + yerr * jax.random.normal(key, shape=(N,))
    return t_train, y_train

def save_benchmark_data(filename, Ns, runtime, memory, outputs):
    import pickle

    data = {
        "Ns": Ns,
        "runtime": runtime,
        "memory": memory,
        "outputs": outputs,
    }
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def load_benchmark_data(filename, unpack=False):
    import pickle

    with open(filename, "rb") as f:
        data = pickle.load(f)
    if unpack:
        return data["Ns"], data["runtime"], data["memory"], data["outputs"]
    else:
        return data

class CPUMemorySampler:
    def __init__(self, interval=0.005):  # 5 ms
        self.interval = interval
        self.running = False
        self.peak = 0
        self.baseline = 0
        self.proc = psutil.Process(os.getpid())

    def _sample(self):
        while self.running:
            mem = self.proc.memory_info().rss
            self.peak = max(self.peak, mem)
            time.sleep(self.interval)

    def start(self):
        self.running = True
        t = threading.Thread(target=self._sample)
        t.daemon = True
        t.start()

    def record_baseline(self):
        self.baseline = self.peak

    def stop(self):
        self.running = False


class GPUMemorySampler:
    def __init__(self, interval=0.005):  # 5 ms
        self.interval = interval
        self.running = False
        self.peak = 0
        self.baseline = 0

    def _sample(self):
        while self.running:
            mem = int(subprocess.check_output(
                "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits",
                shell=True
            ).decode().strip())
            self.peak = max(self.peak, mem)
            time.sleep(self.interval)

    def start(self):
        self.running = True
        t = threading.Thread(target=self._sample)
        t.daemon = True
        t.start()

    def record_baseline(self):
        self.baseline = self.peak

    def stop(self):
        self.running = False


def memthread(fn_bytes, dat_bytes, obj_bytes, args_bytes, return_pipe, machine):
    """
    Helper thread for memory profiling.
    """
    # Unpickle function and arguments inside isolated subprocess
    fn = pickle.loads(fn_bytes)
    dat = pickle.loads(dat_bytes) # this should be JAXArray
    obj = pickle.loads(obj_bytes) # this can be anything (e.g. kernel/gp)
    args = pickle.loads(args_bytes) # extra args for fn

    # Create the jitted function here
    @jax.jit
    def fn_jit(x):
        return fn(x, obj, *args)

    # Warm up (JIT compilation)
    out = fn_jit(dat)
    if hasattr(out, "block_until_ready"):
        out.block_until_ready()

    if machine == "gpu":
        sampler = GPUMemorySampler(interval=1e-5)
    elif machine == "cpu":
        sampler = CPUMemorySampler(interval=1e-5)
    else:
        raise ValueError(f"Unknown machine type: {machine}")

    sampler.start()
    time.sleep(0.1)  # give thread time to measure baseline memory
    sampler.record_baseline()
    # Time the function with JAX block_until_ready
    start = time.perf_counter()
    out = fn_jit(dat)
    if hasattr(out, "block_until_ready"):
        out.block_until_ready()
    end = time.perf_counter()
    sampler.stop()

    # Return both result (pickled) and stats
    return_pipe.send(
        {
            "output": pickle.dumps(out),
            "runtime": end - start,
            "peak_mem": sampler.peak - sampler.baseline,
        }
    )
    return_pipe.close()

def profile_jax_function(fn, data, obj, *args, n_repeat=5, machine="cpu", **kwargs):
    """
    JAX profiler for time benchmarking and memory tracing a function.
    """
    fn_bytes = pickle.dumps(fn)
    dat_bytes = pickle.dumps(data)
    obj_bytes = pickle.dumps(obj)
    args_bytes = pickle.dumps(args)

    runtimes = []
    peaks = []
    output = None

    for _ in range(n_repeat):
        parent_conn, child_conn = mp.Pipe()
        p = mp.Process(
            target=memthread, args=(fn_bytes, dat_bytes, obj_bytes, args_bytes, child_conn, machine)
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

def benchmark(funcs, data, objs, *args, n_repeat=3, cutoffs={}, use_gpu_profiler=False):
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
        N = data[n].shape[-1]
        Ns.append(N)
        print(f"  ({n + 1}/{len(data)}):  N = {N}")
        for name in funcs:
            func = funcs[name]
            obj = objs[name] # either kernel or gp
            cutoff = cutoffs.get(name, 3e4)

            if N <= cutoff:
                if use_gpu_profiler:
                    t, mem, val = profile_jax_function(
                        func, data[n], obj, *args, n_repeat=n_repeat, machine="gpu"
                    )
                else:
                    t, mem, val = profile_jax_function(
                        func, data[n], obj, *args, n_repeat=n_repeat, machine="cpu"
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

            # TODO: save per-iteration here and combine into table, skip if already done
            save_benchmark_data(f"results/individual/{func.__name__}_{N}.pkl", 
                                [N], {name: [t]}, {name: [mem]}, {name: [val]})

    return Ns, runtime, memory, outputs


def get_data(true_kernel, yerr=0.3, N=10, save=True):

    # Generate data of length N
    t_train, y_train = generate_data(N, true_kernel, yerr=yerr)
    data = jnp.array([t_train, y_train, jnp.full_like(t_train, yerr)])
    if save:
        jnp.savez(f'data/{N}.npz', data) 
    return data

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
    use_gpu_profiler=False,
):
    """
    Generate data and benchmark the provided functions over a range of input sizes.
    """
    print("Generating data for benchmarking...")
    ## Generate all data ahead of time
    Ns = jnp.logspace(logN_min, logN_max, N_N).astype(int)
    data = []
    for N in Ns:
        datafile = f'data/{N}.npz'
        if os.path.exists(datafile):
            d = jnp.load(datafile)['arr_0']
        else:
            d = get_data(true_kernel, yerr=yerr, N=N, save=True)
        data.append(d)

    print("Running benchmark...")
    Ns, runtime, memory, outputs = benchmark(
        funcs, data, kernels, n_repeat=n_repeat, cutoffs=cutoffs, use_gpu_profiler=use_gpu_profiler,
    )
    return Ns, runtime, memory, outputs

def run_pred_benchmark(
            true_kernel,
            funcs,
            kernels,
            yerr=0.3,
            N_N=10,
            maxN=1e5,
            logN_min=1,
            logN_max=5,
            n_repeat=3,
            cutoffs={},
            use_gpu_profiler=False,
            ):

    runtime = {}
    memory = {}
    outputs = {}

    # Data (N) and test (M) 
    Ns = jnp.logspace(logN_min, logN_max, N_N).astype(int)
    Ms = 100*Ns

    for i, (N, M) in enumerate(zip(Ns, Ms), 1):
        
        print(f"  ({i}/{N_N}):  N = {N}, M = {M}")
        ## Data to condition on/predict from
        datafile = f'data/{N}.npz'
        if os.path.exists(datafile):
            data = jnp.load(datafile)['arr_0']
        else:
            data = get_data(true_kernel, yerr=yerr, N=N, save=True)
        t_train = data[0, :]
        y_train = data[1, :]
        yerr_train = data[2, :]

        # Test grid to predict at
        print("Generating data for benchmarking...")
        dt = 0.1 * (t_train.max() - t_train.min()) # how much to predict/retrodict
        t_test = jnp.linspace(t_train.min() - dt, t_train.max() + dt, M)

        ## Prepare GP objects
        gp_qs = tinygp.GaussianProcess(kernels['QSM'], t_train, diag=yerr_train**2)
        gp_gp = tinygp.GaussianProcess(kernels['GP'],  t_train, diag=yerr_train**2)
        gp_ss = smolgp.GaussianProcess(kernels['SSM'], t_train, diag=yerr_train**2)
        ## Pre-condition those that are compatible with it
        # _, condGPss = gp_ss.condition(y_train)
        ## Pack dict
        # gp = {'SSM': condGPss, 'QSM': gp_qs, 'GP': gp_gp}
        gp = {'SSM': gp_ss, 'QSM': gp_qs, 'GP': gp_gp}

        _, t, m, o = benchmark(
            funcs, [t_test], gp, y_train, 
            n_repeat=n_repeat, cutoffs=cutoffs, 
            use_gpu_profiler=use_gpu_profiler,
        )
        for name in funcs:
            if name not in runtime:
                runtime[name] = []
                memory[name]  = []
                outputs[name] = []

            if N <= maxN:
                runtime[name].append(t[name][0])
                memory[name].append(m[name][0])
                outputs[name].append(o[name][0])
            else:
                runtime[name].append((jnp.nan, jnp.nan))
                memory[name].append((jnp.nan, jnp.nan))
                outputs[name].append(jnp.nan)

    return Ns, runtime, memory, outputs

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