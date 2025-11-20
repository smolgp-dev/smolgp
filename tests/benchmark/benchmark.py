from abc import abstractmethod
import time
import numpy as np
import pickle
import os

import re
import psutil
import threading, subprocess
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import jax
import jax.numpy as jnp
import tinygp
import smolgp
from funcs import unpack_data, unpack_idata

key = jax.random.PRNGKey(0)

__all__ = [
    "save_benchmark_data",
    "load_benchmark_data",
    "get_data",
    "run_benchmark",
    "run_pred_benchmark",
    "generate_data",
    "generate_integrated_data",
]

def generate_data(N, kernel, yerr=0.3, tmin=0, tmax=86400):
    t_train = jnp.linspace(tmin, tmax, N)
    true_gp = tinygp.GaussianProcess(kernel, t_train)
    y_true = true_gp.sample(key=key)
    y_train = y_true + yerr * jax.random.normal(key, shape=(N,))
    return t_train, y_train

def generate_integrated_data(N, kernel, texp=180, yerr=0.3, readout=40):
    # Generate true GP over baseline
    cadence = texp + readout
    tmin = 0
    tmax = N*cadence
    buffer = 0.1*(tmax-tmin)
    t = jnp.arange(tmin-buffer, tmax+buffer, 1)
    true_gp = tinygp.GaussianProcess(kernel, t)
    y = true_gp.sample(key=key) 

    # Generate synthetic observations
    @jax.jit
    def make_exposure(tmid, texp):
        t_in_exp = jnp.linspace(tmid-texp/2, tmid+texp/2, 50)
         # quickly slice region of interest
        idx = jnp.searchsorted(t, t_in_exp, side="right")
        idx = jnp.clip(idx, 0, t.size-2)
        # just interpolate that region
        y_in_exp = jnp.interp(t_in_exp, t[idx], y[idx])
        return jnp.mean(y_in_exp)
    texp_train = jnp.full(N, texp) # constant exposure time
    t_train = jnp.arange(tmin, tmax, cadence)
    y_true = jax.vmap(make_exposure)(t_train, texp_train)
    y_train = y_true + yerr * jax.random.normal(key, shape=(len(t_train),))
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

class MemorySampler:
    def __init__(self, interval=1e-3):
        self.interval = interval
        self.running = False
        self.peak = 0
        self.baseline = 0
        self.proc = psutil.Process(os.getpid())

    @abstractmethod
    def fetch_memory(self):
        raise NotImplementedError
    
    def _sample(self):
        while self.running:
            mem = self.fetch_memory()
            self.peak = max(self.peak, mem)
            time.sleep(self.interval)

    def start(self):
        self.running = True
        t = threading.Thread(target=self._sample)
        t.daemon = True
        t.start()

    def record_baseline(self, interval=0.1):
        mem = []
        for _ in range(int(interval / self.interval)):
            mem.append(self.fetch_memory())
            time.sleep(self.interval)
        self.baseline = np.mean(mem)

    def stop(self):
        self.running = False

class CPUMemorySampler(MemorySampler):
    def fetch_memory(self):
        return self.proc.memory_info().rss

def get_gpu_processes():
    """
    Parse the 'Processes' section of NVIDIA SMI
    Returns dict {pid: used_bytes}.
    """
    
    pid_re = re.compile(r'Process ID\s*:\s*(\d+)')
    mem_re = re.compile(r'Used GPU Memory\s*:\s*([\d\.]+)\s*MiB', re.IGNORECASE)
    name_re = re.compile(r'Name\s*:\s*(.*)')
    type_re = re.compile(r'Type\s*:\s*(\S)')

    # Get the "Processes" section of nvidia-smi
    cmd = "nvidia-smi --query"
    out = subprocess.check_output(cmd, shell=True).decode()
    lines = out.split("\n")
    for i, line in enumerate(lines):
        if line.lstrip().startswith("Processes"):
            break
        
    results = {}
    for line in lines[i:]:
        line = line.strip()

        # Process ID
        m_pid = pid_re.search(line)
        if m_pid:
            # Start a new record
            pid = int(m_pid.group(1))
            results[pid] = {}
            continue

        # Memory
        m_mem = mem_re.search(line)
        if m_mem:
            mem_mib = float(m_mem.group(1))
            mem = int(mem_mib * 1024 * 1024)
            results[pid]['used_memory'] = mem
            continue

        for reg in [name_re, type_re]:
            m = reg.search(line)
            if m:
                name = reg.pattern.split('\\')[0].lower()
                results[pid][name] = m.group(1)
                continue

    return results


class GPUMemorySampler(MemorySampler):
    def fetch_memory(self):
        smi = get_gpu_processes()
        tot_mem = 0
        for pid in smi:
            if 'smolgp' in smi[pid]['name']:
                tot_mem += smi[pid]['used_memory']
        return tot_mem

    def record_baseline(self, interval=0.1):
        return 0 

def tracer(fn_bytes, dat_bytes, obj_bytes, args_bytes, return_pipe, machine):
    """
    Time and trace memory of a JAX function inside an isolated subprocess.
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

    # Benchmarking/tracing
    peak_mem = 0
    interval = 1e-2 if machine == "gpu" else 1e-5
    while peak_mem == 0:
        if machine == "gpu":
            sampler = GPUMemorySampler(interval=interval)
        elif machine == "cpu":
            sampler = CPUMemorySampler(interval=interval)
        else:
            raise ValueError(f"Unknown machine type: {machine}")
        
        sampler.start()
        sampler.record_baseline(0.1)
        # Time the function with JAX block_until_ready
        start = time.perf_counter()
        out = fn_jit(dat)
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()
        end = time.perf_counter()
        time.sleep(0.1)
        sampler.stop()
        peak_mem = sampler.peak - sampler.baseline
        # Repeat if no memory usage detected 
        # (function ran faster than sampler.interval)
        # try a faster interval
        interval /= 10

    # Return both result (pickled) and stats
    return_pipe.send(
        {
            "output": pickle.dumps(out),
            "runtime": end - start,
            "peak_mem": peak_mem,
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
            target=tracer, args=(fn_bytes, dat_bytes, obj_bytes, args_bytes, child_conn, machine)
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
    Given some (to-be-jitted) functions, benchmark their runtimes over a range of input sizes.

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
    machine = "gpu" if use_gpu_profiler else "cpu"
    for n in range(len(data)):
        N = data[n].shape[-1]
        Ns.append(N)
        print(f"  ({n + 1}/{len(data)}):  N = {N}")
        for name in funcs:
            func = funcs[name]
            obj = objs[name] # either kernel or gp
            cutoff = cutoffs.get(name, 3e4)

            if N <= cutoff:
                t, mem, val = profile_jax_function(
                    func, data[n], obj, *args, n_repeat=n_repeat, machine=machine
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

            save_benchmark_data(f"results/individual/{func.__name__}_{N}.pkl", 
                                [N], {name: [t]}, {name: [mem]}, {name: [val]})

    return Ns, runtime, memory, outputs


def get_data(true_kernel, N, yerr=0.3, exposure_quantities=None, save=True):

    # Generate data of length N
    if exposure_quantities:
        texp, readout = exposure_quantities
        t_train, y_train = generate_integrated_data(N, true_kernel, texp=texp, readout=readout, yerr=yerr)
        texp_train = jnp.full_like(t_train, texp)
        yerr_train = jnp.full_like(t_train, yerr)
        instid = jnp.full_like(t_train, 0)
        data = jnp.array([t_train, y_train, yerr_train, texp_train, instid])
        savename = f'data/{N}_int.npz'
    else:
        t_train, y_train = generate_data(N, true_kernel, yerr=yerr)
        yerr_train = jnp.full_like(t_train, yerr)
        data = jnp.array([t_train, y_train, yerr_train])
        savename = f'data/{N}.npz'
    if save:
        jnp.savez(savename, data) 
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
    exposure_quantities=None,
):
    """
    Generate data and benchmark the provided functions over a range of input sizes.
    """
    print("Generating data for benchmarking...")
    ## Generate all data ahead of time
    isint = '_int' if exposure_quantities else ''
    Ns = jnp.logspace(logN_min, logN_max, N_N).astype(int)
    data = []
    for N in Ns:
        datafile = f'data/{N}{isint}.npz'
        if os.path.exists(datafile):
            d = jnp.load(datafile)['arr_0']
            print('  Loaded data from', datafile)
        else:
            print('  Generating data for N =', N)
            d = get_data(true_kernel, N, yerr=yerr,
                         exposure_quantities=exposure_quantities, save=True)
            print('  Generated and saved data to', datafile)
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
            exposure_quantities=None,
            ):

    runtime = {}
    memory = {}
    outputs = {}

    # Data (N) and test (M) 
    Ns = jnp.logspace(logN_min, logN_max, N_N).astype(int)
    Ms = 100*Ns
    isint = '_int' if exposure_quantities else ''
    for i, (N, M) in enumerate(zip(Ns, Ms), 1):
        
        print(f"  ({i}/{N_N}):  N = {N}, M = {M}")
        ## Data to condition on/predict from
        datafile = f'data/{N}{isint}.npz'
        if os.path.exists(datafile):
            data = jnp.load(datafile)['arr_0']
        else:
            data = get_data(true_kernel, N, yerr=yerr, exposure_quantities=exposure_quantities, save=True)
        if exposure_quantities:
            X_train, y_train, yerr_train = unpack_idata(data)
            t_train, texp_train, instid = X_train
        else:
            t_train, y_train, yerr_train = unpack_data(data)
            X_train = t_train

        # Test grid to predict at
        print("Generating data for benchmarking...")
        dt = 0.1 * (t_train.max() - t_train.min()) # how much to predict/retrodict
        t_test = jnp.linspace(t_train.min() - dt, t_train.max() + dt, M)

        ## Prepare GP objects
        if 'QSM' in kernels:
            gp_qs = tinygp.GaussianProcess(kernels['QSM'], X_train, diag=yerr_train**2)
        gp_gp = tinygp.GaussianProcess(kernels['GP'],  X_train, diag=yerr_train**2)
        gp_ss = smolgp.GaussianProcess(kernels['SSM'], X_train, diag=yerr_train**2)
        ## Pre-condition those that are compatible with it
        # _, condGPss = gp_ss.condition(y_train)
        ## Pack dict
        # gp = {'SSM': condGPss, 'QSM': gp_qs, 'GP': gp_gp}
        gp = {'SSM': gp_ss, 'GP': gp_gp}
        if 'QSM' in kernels:
            gp['QSM'] = gp_qs

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