import os
import pickle
import sys
import time
from abc import abstractmethod

import jax
import jax.numpy as jnp
import numpy as np
import tinygp
from funcs import unpack_data, unpack_idata

import smolgp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import multiprocessing as mp
import re
import subprocess
import threading

import psutil
import utils

mp.set_start_method("spawn", force=True)

key = jax.random.PRNGKey(0)


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


def get_data(true_kernel, N, yerr=0.3, exposure_quantities=None, save=True):
    # Generate data of length N
    if exposure_quantities:
        texp, readout = exposure_quantities
        t_train, y_train = utils.generate_integrated_data(
            N, true_kernel, texp=texp, readout=readout, yerr=yerr
        )
        texp_train = jnp.full_like(t_train, texp)
        yerr_train = jnp.full_like(t_train, yerr)
        instid = jnp.full_like(t_train, 0)
        data = jnp.array([t_train, y_train, yerr_train, texp_train, instid])
        savename = f"data/{N}_int.npz"
    else:
        t_train, y_train = utils.generate_data(N, true_kernel, yerr=yerr)
        yerr_train = jnp.full_like(t_train, yerr)
        data = jnp.array([t_train, y_train, yerr_train])
        savename = f"data/{N}.npz"
    if save:
        jnp.savez(savename, data)
    return data


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

    pid_re = re.compile(r"Process ID\s*:\s*(\d+)")
    mem_re = re.compile(r"Used GPU Memory\s*:\s*([\d\.]+)\s*MiB", re.IGNORECASE)
    name_re = re.compile(r"Name\s*:\s*(.*)")
    type_re = re.compile(r"Type\s*:\s*(\S)")

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
            results[pid]["used_memory"] = mem
            continue

        for reg in [name_re, type_re]:
            m = reg.search(line)
            if m:
                name = reg.pattern.split("\\")[0].lower()
                results[pid][name] = m.group(1)
                continue

    return results


class GPUMemorySampler(MemorySampler):
    def fetch_memory(self):
        smi = get_gpu_processes()
        tot_mem = 0
        for pid in smi:
            # if 'smolgp' in smi[pid]['name']:
            #     tot_mem += smi[pid]['used_memory']
            if smi[pid]["type"] == "C":
                tot_mem += smi[pid]["used_memory"]
        return tot_mem

    def record_baseline(self, interval=0.1):
        return 0


def tracer(fn_bytes, dat_bytes, obj_bytes, args_bytes, return_pipe, machine):
    """
    Time and trace memory of a JAX function inside an isolated subprocess.
    """
    # Unpickle function and arguments inside isolated subprocess
    fn = pickle.loads(fn_bytes)
    dat = pickle.loads(dat_bytes)  # this should be JAXArray
    obj = pickle.loads(obj_bytes)  # this can be anything (e.g. kernel/gp)
    args = pickle.loads(args_bytes)  # extra args for fn

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


def profile_jax_function(
    fn, data, obj, *args, n_repeat=5, machine="cpu", drop_outliers=False, **kwargs
):
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
            target=tracer,
            args=(fn_bytes, dat_bytes, obj_bytes, args_bytes, child_conn, machine),
        )
        p.start()
        result = parent_conn.recv()
        p.join()

        runtimes.append(result["runtime"])
        peaks.append(result["peak_mem"])
        output = pickle.loads(result["output"])
    runtimes = np.array(runtimes)
    peaks = np.array(peaks)
    if drop_outliers:
        runtimes = np.delete(runtimes, [runtimes.argmin(), runtimes.argmax()])
        peaks = np.delete(peaks, [peaks.argmin(), peaks.argmax()])

    return (
        (np.mean(runtimes), np.std(runtimes)),
        (np.mean(peaks), np.std(peaks)),
        output,
    )


def benchmark(
    funcs,
    data,
    objs,
    *args,
    n_repeat=5,
    cutoffs={},
    drop_outliers=False,
    use_gpu_profiler=False,
    max_seconds=None,
):
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

    runtime = {name: [] for name in funcs}
    memory = {name: [] for name in funcs}
    outputs = {name: [] for name in funcs}
    # Curves that have already blown the per-call time budget.
    _too_slow = {}
    Ns = []
    machine = "gpu" if use_gpu_profiler else "cpu"
    for n in range(len(data)):
        N = data[n].shape[-1]
        Ns.append(N)
        print(f"  ({n + 1}/{len(data)}):  N = {N}")
        for name in funcs:
            func = funcs[name]
            obj = objs[name]  # either kernel or gp
            cutoff = cutoffs.get(name, 3e4)

            # Runtime grows monotonically with size, so once a curve blows the
            # per-call budget every larger size will too: retire it rather than
            # re-measuring something we already know is too slow.
            if max_seconds and _too_slow.get(name):
                t, mem, val = (jnp.nan, jnp.nan), (jnp.nan, jnp.nan), jnp.nan
                print(f"    {name}: Skipped (exceeded {max_seconds:g}s budget at a smaller size)")
                runtime[name].append(t)
                memory[name].append(mem)
                outputs[name].append(val)
                continue

            if N <= cutoff:
                t, mem, val = profile_jax_function(
                    func,
                    data[n],
                    obj,
                    *args,
                    n_repeat=n_repeat,
                    machine=machine,
                    drop_outliers=drop_outliers,
                )
                basestr = f"    {name}: time = {t[0]:.4f} ± {t[1]:.4f} s"
                memstr = f", mem = {format_bytes(mem[0])} ± {format_bytes(mem[1])}"
                print(basestr + memstr)
                if max_seconds and t[0] > max_seconds:
                    _too_slow[name] = True
                    print(f"      (over the {max_seconds:g}s budget; retiring this curve)")
            else:
                t, mem, val = (jnp.nan, jnp.nan), (jnp.nan, jnp.nan), jnp.nan
                print(f"    {name}: Skipped (N={N} > cutoff={cutoff})")

            runtime[name].append(t)
            memory[name].append(mem)
            outputs[name].append(val)

            save_benchmark_data(
                f"results/individual/{func.__name__}_{N}_{machine}.pkl",
                [N],
                {name: [t]},
                {name: [mem]},
                {name: [val]},
            )

    return Ns, runtime, memory, outputs


def run_benchmark(
    true_kernel,
    funcs,
    kernels,
    yerr=0.3,
    N_N=10,
    logN_min=1,
    logN_max=7,
    n_repeat=5,
    cutoffs={},
    drop_outliers=False,
    use_gpu_profiler=False,
    exposure_quantities=None,
    max_seconds=None,
):
    """
    Generate data and benchmark the provided functions over a range of input sizes.
    """
    print("Generating data for benchmarking...")
    ## Generate all data ahead of time
    isint = "_int" if exposure_quantities else ""
    Ns = jnp.logspace(logN_min, logN_max, N_N).astype(int)
    data = []
    for N in Ns:
        datafile = f"data/{N}{isint}.npz"
        if os.path.exists(datafile):
            d = jnp.load(datafile)["arr_0"]
            print("  Loaded data from", datafile)
        else:
            print("  Generating data for N =", N)
            d = get_data(
                true_kernel,
                N,
                yerr=yerr,
                exposure_quantities=exposure_quantities,
                save=True,
            )
            print("  Generated and saved data to", datafile)
        data.append(d)

    print("Running benchmark...")
    Ns, runtime, memory, outputs = benchmark(
        funcs,
        data,
        kernels,
        n_repeat=n_repeat,
        cutoffs=cutoffs,
        drop_outliers=drop_outliers,
        use_gpu_profiler=use_gpu_profiler,
        max_seconds=max_seconds,
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
    n_repeat=5,
    cutoffs={},  # in M
    drop_outliers=False,
    use_gpu_profiler=False,
    exposure_quantities=None,
    max_seconds=None,
):
    runtime = {name: [] for name in funcs}
    memory = {name: [] for name in funcs}
    outputs = {name: [] for name in funcs}

    # Data (N) and test (M)
    Ns = jnp.logspace(logN_min, logN_max, N_N).astype(int)
    Ms = 100 * Ns
    isint = "_int" if exposure_quantities else ""
    for i, (N, M) in enumerate(zip(Ns, Ms), 1):
        skip = True
        for key in cutoffs:
            if M <= cutoffs[key]:
                skip = False
        if skip:
            print(f"  ({i}/{N_N}):  N = {N}, M = {M} -- Skipped (M > all cutoffs)")
            for name in funcs:
                runtime[name].append((jnp.nan, jnp.nan))
                memory[name].append((jnp.nan, jnp.nan))
                outputs[name].append(jnp.nan)
            continue

        print(f"  ({i}/{N_N}):  N = {N}, M = {M}")
        ## Data to condition on/predict from
        datafile = f"data/{N}{isint}.npz"
        if os.path.exists(datafile):
            data = jnp.load(datafile)["arr_0"]
        else:
            data = get_data(
                true_kernel,
                N,
                yerr=yerr,
                exposure_quantities=exposure_quantities,
                save=True,
            )
        if exposure_quantities:
            X_train, y_train, yerr_train = unpack_idata(data)
            t_train, texp_train, instid = X_train
        else:
            t_train, y_train, yerr_train = unpack_data(data)
            X_train = t_train

        # Test grid to predict at
        print("Generating data for benchmarking...")
        dt = 0.1 * (t_train.max() - t_train.min())  # how much to predict/retrodict
        t_test = jnp.linspace(t_train.min() - dt, t_train.max() + dt, M)

        ## Prepare GP objects
        if "QSM" in kernels:
            gp_qs = tinygp.GaussianProcess(kernels["QSM"], X_train, diag=yerr_train**2)
        gp_gp = tinygp.GaussianProcess(kernels["GP"], X_train, diag=yerr_train**2)
        gp_ss = smolgp.GaussianProcess(kernels["SSM"], X_train, noise=yerr_train**2)
        ## Pre-condition those that are compatible with it
        # _, condGPss = gp_ss.condition(y_train)
        ## Pack dict
        # gp = {'SSM': condGPss, 'QSM': gp_qs, 'GP': gp_gp}
        gp = {"SSM": gp_ss, "GP": gp_gp}
        if "QSM" in kernels:
            gp["QSM"] = gp_qs

        _, t, m, o = benchmark(
            funcs,
            [t_test],
            gp,
            y_train,
            n_repeat=n_repeat,
            cutoffs=cutoffs,
            drop_outliers=drop_outliers,
            use_gpu_profiler=use_gpu_profiler,
            max_seconds=max_seconds,
        )
        for name in funcs:
            if N <= maxN:
                runtime[name].append(t[name][0])
                memory[name].append(m[name][0])
                outputs[name].append(o[name][0])
            else:
                runtime[name].append((jnp.nan, jnp.nan))
                memory[name].append((jnp.nan, jnp.nan))
                outputs[name].append(jnp.nan)

    return Ns, runtime, memory, outputs


def run_prior_sample_benchmark(
    funcs,
    kernels,
    N_N=10,
    logM_min=1,
    logM_max=7,
    n_repeat=5,
    cutoffs={},  # in M
    drop_outliers=False,
    use_gpu_profiler=False,
    exposure_quantities=None,
    tmax=1e4,
    max_seconds=None,
):
    """Benchmark drawing from the *prior* as a function of M.

    A prior draw is conditioned on nothing, so there is no training set and no
    N: the only size parameter is M, the number of coordinates the realization
    is drawn at. That makes this the one benchmark whose x axis is M directly,
    rather than N with M following along.

    The sample coordinates are evenly spaced over a fixed window, so the
    process is progressively oversampled as M grows -- which is the realistic
    use (a dense draw of one realization), and keeps the kernel's correlation
    length fixed relative to the window rather than shrinking with M.
    """
    runtime = {name: [] for name in funcs}
    memory = {name: [] for name in funcs}
    outputs = {name: [] for name in funcs}

    Ms = jnp.logspace(logM_min, logM_max, N_N).astype(int)
    for i, M in enumerate(Ms, 1):
        skip = True
        for key in cutoffs:
            if M <= cutoffs[key]:
                skip = False
        if skip:
            print(f"  ({i}/{N_N}):  M = {M} -- Skipped (M > all cutoffs)")
            for name in funcs:
                runtime[name].append((jnp.nan, jnp.nan))
                memory[name].append((jnp.nan, jnp.nan))
                outputs[name].append(jnp.nan)
            continue

        print(f"  ({i}/{N_N}):  M = {M}")
        t_sample = jnp.linspace(0.0, tmax, M)
        if exposure_quantities:
            texp, _readout = exposure_quantities
            X_sample = (
                t_sample,
                jnp.full_like(t_sample, texp),
                jnp.zeros_like(t_sample).astype(int),
            )
        else:
            X_sample = t_sample

        _, t, m, o = benchmark(
            funcs,
            [X_sample],
            kernels,
            n_repeat=n_repeat,
            cutoffs=cutoffs,
            drop_outliers=drop_outliers,
            use_gpu_profiler=use_gpu_profiler,
            max_seconds=max_seconds,
        )
        for name in funcs:
            runtime[name].append(t[name][0])
            memory[name].append(m[name][0])
            outputs[name].append(o[name][0])

    return Ms, runtime, memory, outputs


# ---------------------------------------------------------------------------
# Size cutoffs from a memory budget
#
# The O(N) methods (SSM, QSM) are time-bound in practice, so they get a flat
# cap. The dense GP is memory-bound and is what actually decides how far a
# machine can go, so its cutoff is derived from the available RAM.
#
# The constants below are bytes consumed per unit of problem size, calibrated
# so that a 512 GB budget reproduces the cutoffs used for the currently
# deployed figures (GP: 6e4 in N for llh/cond, 1e6 in M for pred). They
# include workspace/copies, so they are several times the naive 8 bytes.
# ---------------------------------------------------------------------------
_BYTES_PER_N2 = 142  # dense GP, O(N^2) -- likelihood, conditioning, prior draw
_BYTES_PER_NM = 51  # dense GP, O(N*M) -- prediction, posterior draw

# Memory available per machine and device.
# workstation is Intel® Xeon® w53435X CPU + NVIDIA RTX 6000 Ada GPU
# macbook is Apple M3 Max 64 GB (Nov 2023)
# Pass --max-ram to override.
MACHINE_RAM_GB = {
    "workstation": {"cpu": 512, "gpu": 48},
    "macbook": {"cpu": 64, "gpu": 64},
}

# Fraction of MACHINE_RAM_GB the benchmark is allowed to target.
# An explicit --max-ram is taken literally.
RAM_HEADROOM = 0.85


def size_cutoffs(max_ram_gb, kind, max_N=1e7, max_M=1e6, m_per_n=100, gpu=False):
    """Per-curve size cutoffs for a given RAM budget.

    Args:
        max_ram_gb: memory available to the benchmark, in GB.
        kind: ``llh`` | ``cond`` | ``pred`` | ``sample-prior`` | ``sample-post``.
        max_N, max_M: flat caps for the O(N) methods (time-bound, not
            memory-bound).
        m_per_n: test/sample points per data point, for the M-scaled kinds.
        gpu: include the parallel-solver curves.

    Returns:
        dict of curve name -> maximum size, in whichever variable that kind's
        cutoffs are expressed (N for llh/cond/sample-prior, M for the rest).
    """
    ram = max_ram_gb * 1e9
    if kind in ("llh", "cond", "sample-prior"):
        cuts = {
            "GP": (ram / _BYTES_PER_N2) ** 0.5,
            "SSM": max_N,
            "QSM": max_N,
        }
    else:  # pred, sample-post -- cutoffs are in M, and the dense cost is N*M
        cuts = {
            "GP": (ram * m_per_n / _BYTES_PER_NM) ** 0.5,
            "SSM": max_M,
            "QSM": max_M,
        }
    # Parallel solvers only run on the GPU box, and never for sampling.
    if kind.startswith("sample"):
        cuts["pSSM"] = 0
        cuts["pQSM"] = 0
    else:
        cuts["pSSM"] = max_N if gpu else 0
        cuts["pQSM"] = max_N if gpu else 0
    return cuts
