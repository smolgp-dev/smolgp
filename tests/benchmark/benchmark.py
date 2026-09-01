import math
import os
import pickle
import signal
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

# Profiling runs each point in a spawned subprocess, and that child imports this 
# module, so need to enable here to ensure 64-bit precision in the benchmark
jax.config.update("jax_enable_x64", True)

# For reporting how a profiling subprocess died (see profile_jax_function).
SIGNAL_NAMES = {
    signal.SIGKILL: "SIGKILL, usually the OOM killer",
    signal.SIGSEGV: "SIGSEGV",
    signal.SIGABRT: "SIGABRT",
    signal.SIGBUS: "SIGBUS",
    signal.SIGTERM: "SIGTERM",
}

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
    """Peak-memory tracker for one timed call.

    The baseline is recorded *after* JIT warm-up, so the reported figure is
    the footprint of doing the operation once -- the matrices it has to hold
    and the workspace it needs -- and excludes compilation buffers. On Linux
    this is exact: XLA's CPU backend mmaps its large buffers and returns them
    on free, so the warm-up's allocation is not still resident when the
    baseline is taken. Verified against 2*N^2*8 for a dense GP likelihood to
    within 0.05% over N = 1778..23713.

    Note that the peak has to be *sampled during* the call. Comparing RSS
    before and after it returns reads ~0, because the buffers are already
    released by then.
    """

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
        for _ in range(max(1, int(interval / self.interval))):
            mem.append(self.fetch_memory())
            time.sleep(self.interval)
        self.baseline = np.mean(mem)

    def stop(self):
        self.running = False

    def measure(self, call):
        """Run ``call()`` and return ``(result, elapsed, peak_above_baseline)``."""
        self.record_baseline(0.1)
        self.start()
        start = time.perf_counter()
        out = call()
        elapsed = time.perf_counter() - start
        time.sleep(0.1)  # let the sampler see the tail of the allocation
        self.stop()
        return out, elapsed, self.peak - self.baseline


def vm_hwm():
    """Peak RSS the kernel has recorded for this process, in bytes.

    ``VmHWM`` in /proc/self/status. The kernel raises it on every page fault
    that pushes RSS higher, so unlike a sampling thread it cannot miss a peak.
    It is monotonic since process start, so it also carries the JIT warm-up's
    peak -- reset it with :func:`reset_vm_hwm` to scope it to one call.
    """
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def reset_vm_hwm():
    """Reset ``VmHWM`` to the current RSS. Returns True if it took.

    Writing "5" to /proc/self/clear_refs clears the high-water mark (Linux
    only, and only for the calling process). Lets the mark be scoped to the
    timed call rather than to the whole process lifetime.
    """
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


class CPUMemorySampler(MemorySampler):
    """Peak RSS from the kernel's own high-water mark, with a sampled fallback.

    ``VmHWM`` is authoritative. The kernel raises it on every page fault that
    pushes RSS higher, so it cannot miss a peak; the 1 ms sampling thread can,
    and above roughly 30 GB it does. Verified on the production path against
    dense GP likelihood, whose footprint is exactly 2*N^2*8 = 16 B/N^2 (the
    kernel matrix plus its Cholesky factor):

        N        reported   theory    ratio   B/N^2
        1778       0.05 GB   0.05 GB   1.00    15.9
        4216       0.28 GB   0.28 GB   1.00    16.0
        10000      1.61 GB   1.60 GB   1.01    16.1
        23713      9.02 GB   9.00 GB   1.00    16.0
        56234     50.57 GB  50.60 GB   1.00    16.0

    At N=56234 the sampled figure reads 33.6 GB against the same 50.60 GB, i.e.
    0.66x, while the two agree to within 1% at every smaller size. The
    consequence for the published figures was that the largest point of every
    dense curve read low -- 11 B/N^2 against a theoretical 16 -- which looked
    like an algorithmic change and was not: the runtimes across the same step
    continue clean N^3 with no kink.

    The sampled figure is still taken, as ``sampled``, purely as a cross-check;
    a large disagreement is reported by profile_jax_function. It also stands in
    as the reported value if ``VmHWM`` is unavailable, which is anywhere without
    a Linux /proc.
    """

    def fetch_memory(self):
        return self.proc.memory_info().rss

    def measure(self, call):
        self.record_baseline(0.1)
        # Scope the kernel mark to the timed call rather than to the whole
        # process, so it excludes the warm-up's compilation buffers -- the same
        # thing recording the baseline after warm-up does for the sampled path.
        reset_ok = reset_vm_hwm()
        base = self.fetch_memory()
        out, elapsed, sampled = super_measure(self, call)
        hwm = max(0, vm_hwm() - base) if reset_ok else 0
        self.sampled = sampled
        self.hwm = hwm
        # Absolute peak, no baseline subtracted: the process high-water mark is
        # what has to fit in RAM.
        self.absolute = vm_hwm()
        return out, elapsed, (hwm if hwm > 0 else sampled)


def super_measure(self, call):
    """MemorySampler.measure's body, minus the baseline it already recorded."""
    self.start()
    start = time.perf_counter()
    out = call()
    elapsed = time.perf_counter() - start
    time.sleep(0.1)  # let the sampler see the tail of the allocation
    self.stop()
    return out, elapsed, self.peak - self.baseline


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
    """Device memory from XLA's own allocator counter.

    Reports ``peak_bytes_in_use`` minus the ``bytes_in_use`` that was resident
    once the warm-up settled: the memory the operation needs on top of its
    already-loaded inputs. That is the same quantity the CPU sampler reports
    (peak RSS during the call, less settled RSS after warm-up), so the two
    devices' numbers mean the same thing.

    Getting here took two wrong turns, both worth recording:

    * Subtracting a *peak* baseline from the peak gives exactly zero. The mark
      is cumulative since process start and JAX offers no way to clear it --
      there is no ``reset_memory_stats`` on the device -- so the warm-up call
      has already driven it to the very figure being measured, and the timed
      call, repeating the same computation, never exceeds it.
    * Sampling ``bytes_in_use`` from a thread instead does not work either.
      XLA dispatches the whole executable asynchronously and allocates and
      frees its transients inside a single ``Execute``, so the live gauge sits
      flat at the input size for the entire call. Measured: a 4096x4096 f32
      matmul left ``bytes_in_use`` at 67 MB throughout while
      ``peak_bytes_in_use`` recorded 503 MB.

    So the peak counter is the only one that sees the transients, and the
    baseline has to come from the live gauge. No sampling thread is involved --
    the peak is exact, and each measurement runs in a fresh subprocess, so
    nothing earlier can have contaminated the mark.

    Against ``nvidia-smi``, which this replaced: exact byte counts rather than
    MiB rounding, no ~100 ms subprocess per sample, and -- on a shared
    workstation -- no contamination from other users' processes on the card.
    """

    #: Counter holding the high-water mark, including intra-execution transients.
    PEAK_KEY = "peak_bytes_in_use"
    #: Counter holding what is resident right now, used for the baseline.
    LIVE_KEY = "bytes_in_use"

    def __init__(self, interval=1e-3):
        super().__init__(interval=interval)
        self.device = jax.devices()[0]
        stats = self.device.memory_stats()
        if not stats or self.PEAK_KEY not in stats or self.LIVE_KEY not in stats:
            raise RuntimeError(
                f"{self.device} exposes no {self.PEAK_KEY}/{self.LIVE_KEY}; the GPU "
                "profiler needs a CUDA jaxlib (uv sync --group cuda). The CPU "
                f"backend returns None from memory_stats(). Got: {stats}"
            )

    def fetch_memory(self):
        return self.device.memory_stats()[self.LIVE_KEY]

    def measure(self, call):
        self.record_baseline(0.1)  # live bytes, settled after warm-up
        start = time.perf_counter()
        out = call()
        elapsed = time.perf_counter() - start
        peak = self.device.memory_stats()[self.PEAK_KEY]
        # Absolute device peak, baseline included: on a card, the CUDA context
        # and the resident inputs are memory you must genuinely have free, the
        # same sense in which the CPU's absolute figure includes the runtime.
        self.absolute = peak
        return out, elapsed, peak - self.baseline


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

    def call():
        out = fn_jit(dat)
        # Block on the whole result, whatever shape it is. The hasattr check
        # this replaces silently skipped blocking for anything that was not a
        # single array -- a (value, gradient) pair, for instance -- and then
        # timed JAX's asynchronous dispatch rather than the computation.
        out = jax.block_until_ready(out)
        return out

    # Warm up (JIT compilation). Everything it allocates is released before
    # the sampler takes its baseline, so compilation is excluded from the
    # reported footprint -- see MemorySampler.
    call()

    # 1 ms is as fine as time.sleep can usefully resolve; asking for 10 us
    # turns the sampler into a ~20 kHz spin doing a syscall per iteration,
    # which competes with the 32-thread BLAS being measured. The peak is a
    # plateau lasting the whole call, not a spike, so 1 ms is plenty.
    if machine == "gpu":
        sampler = GPUMemorySampler()
    elif machine == "cpu":
        sampler = CPUMemorySampler(interval=1e-3)
    else:
        raise ValueError(f"Unknown machine type: {machine}")

    # XLA's own buffer accounting: exact, static, and free of every runtime
    # overhead. temp + output + argument is the computation's working set --
    # scratch, result, and the input arrays. Available on both devices, and at
    # every size, so it has no measurement floor. The lower() re-traces but the
    # compile is served from JAX's cache, since fn_jit was just warmed up.
    try:
        stats = fn_jit.lower(dat).compile().memory_analysis()
        xla_mem = (
            stats.temp_size_in_bytes
            + stats.output_size_in_bytes
            + stats.argument_size_in_bytes
        )
    except Exception:  # noqa: BLE001 -- not worth failing a measurement over
        xla_mem = float("nan")

    out, runtime, peak_mem = sampler.measure(call)

    # Three numbers, deliberately:
    #   peak_mem  -- peak above the post-warm-up baseline. The historical
    #                quantity, kept so old and new points stay comparable.
    #   abs_mem   -- absolute peak. What a user must actually have free,
    #                including the interpreter and runtime, which is what the
    #                figures are meant to communicate.
    #   xla_mem   -- the computation alone, excluding all of that.
    return_pipe.send(
        {
            "output": pickle.dumps(out),
            "runtime": runtime,
            "peak_mem": peak_mem,
            "abs_mem": getattr(sampler, "absolute", float("nan")),
            "xla_mem": xla_mem,
            "hwm_mem": getattr(sampler, "hwm", None),
            "sampled_mem": getattr(sampler, "sampled", None),
        }
    )
    return_pipe.close()


#: How many repeats a point gets, chosen from how long its first call took:
#: ``(per-call seconds below which, repeats)``, ascending, last entry the
#: fallback. Repeat-to-repeat scatter is a function of runtime, not of N --
#: measured on the current results, std/mean is 5-7% below 10 ms but 0.9% in the
#: 1-10 s band and 0.4% above 10 s -- so a slow call needs far fewer samples to
#: pin its mean to well under a plot pixel. Adjust here if that changes.
NREPEAT_SCHEDULE = ((1.0, 7), (10.0, 5), (60.0, 3), (float("inf"), 1))

#: Print a per-repeat progress line once a point's calls are slower than this.
#: Below it the repeats are quick enough that the per-point summary is timely
#: and extra lines would only be noise; above it a single point can own the
#: sweep for an hour in complete silence, which is indistinguishable from a
#: wedged run. The remaining-time figure assumes each repeat costs 2t -- the
#: untimed warm-up plus the timed call -- the same model the runtime table uses
#: (see run/README.md).
PROGRESS_ABOVE_SECONDS = 60.0


def repeats_for(seconds):
    """Repeats for a point whose first call took ``seconds`` (see NREPEAT_SCHEDULE)."""
    for limit, n in NREPEAT_SCHEDULE:
        if seconds < limit:
            return n
    return NREPEAT_SCHEDULE[-1][1]


def _xla_child(fn_bytes, dat_bytes, obj_bytes, args_bytes, return_pipe):
    """Compile one point and report XLA's buffer accounting. No execution."""
    fn = pickle.loads(fn_bytes)
    dat = pickle.loads(dat_bytes)
    obj = pickle.loads(obj_bytes)
    args = pickle.loads(args_bytes)

    @jax.jit
    def fn_jit(x):
        return fn(x, obj, *args)

    try:
        st = fn_jit.lower(dat).compile().memory_analysis()
        total = (st.temp_size_in_bytes + st.output_size_in_bytes
                 + st.argument_size_in_bytes)
    except Exception as exc:  # noqa: BLE001
        return_pipe.send({"xla": float("nan"), "err": repr(exc)[:200]})
    else:
        return_pipe.send({"xla": total, "err": None})
    return_pipe.close()


def xla_footprint(fn, data, obj, *args, timeout=900.0):
    """Bytes the computation reserves, from XLA's buffer assignment.

    Static: it compiles but never executes, so it costs no timed run and has no
    measurement floor -- exact from a few kB upward, where sampled RSS bottoms
    out. Returns temp + output + argument, i.e. scratch, result and the input
    arrays: the computation's working set, excluding the interpreter, the JAX
    runtime, allocator slack and (on GPU) the CUDA context.

    Runs in a subprocess for the same reason profiling does -- compilation of a
    very large graph is not guaranteed to be well behaved, and a crash here
    should cost one point rather than the sweep.

    Returns NaN if compilation fails or the child dies.
    """
    parent_conn, child_conn = mp.Pipe()
    p = mp.Process(
        target=_xla_child,
        args=(pickle.dumps(fn), pickle.dumps(data), pickle.dumps(obj),
              pickle.dumps(args), child_conn),
    )
    p.start()
    result = None
    waited = 0.0
    while result is None:
        if parent_conn.poll(5.0):
            result = parent_conn.recv()
        elif not p.is_alive():
            break
        else:
            waited += 5.0
            if waited > timeout:
                p.terminate()
                break
    p.join()
    parent_conn.close()
    if result is None:
        return float("nan")
    if result["err"]:
        print(f"      (xla analysis failed: {result['err']})")
    return result["xla"]


def profile_jax_function(
    fn, data, obj, *args, n_repeat=None, machine="cpu", drop_outliers=False,
    max_seconds=None, **kwargs
):
    """
    JAX profiler for time benchmarking and memory tracing a function.

    ``n_repeat=None`` (the default) is adaptive: the first call is timed, and
    :func:`repeats_for` decides how many more to run. An explicit integer is
    taken literally at every size.

    ``max_seconds`` also stops the repeats early, which matters when a fixed
    ``n_repeat`` was requested -- adaptive already lands on 1 repeat for
    anything that slow.
    """
    fn_bytes = pickle.dumps(fn)
    dat_bytes = pickle.dumps(data)
    obj_bytes = pickle.dumps(obj)
    args_bytes = pickle.dumps(args)

    runtimes = []
    peaks = []
    absolutes = []   # absolute peak: what must be free to run this
    xlas = []        # XLA buffer accounting: the computation alone
    output = None

    adaptive = n_repeat is None
    max_repeats = max(n for _, n in NREPEAT_SCHEDULE) if adaptive else n_repeat
    target = max_repeats

    for _ in range(max_repeats):
        parent_conn, child_conn = mp.Pipe()
        p = mp.Process(
            target=tracer,
            args=(fn_bytes, dat_bytes, obj_bytes, args_bytes, child_conn, machine),
        )
        p.start()

        # Never block on a bare recv(). A child that dies without sending --
        # OOM-killed, or segfaulted inside XLA, which a large dense Cholesky
        # really does do -- leaves the parent waiting forever on a pipe that
        # will never be written, with the corpse unreaped because join() is
        # below. That silently wedges the whole sweep. Poll instead, and treat
        # a dead-and-silent child as a failed measurement.
        result = None
        while result is None:
            if parent_conn.poll(5.0):
                result = parent_conn.recv()
            elif not p.is_alive():
                break  # exited without sending: crashed
        p.join()
        parent_conn.close()

        if result is None:
            code = p.exitcode
            why = f"signal {-code} ({SIGNAL_NAMES.get(-code, '?')})" if code and code < 0 else f"exit code {code}"
            print(f"      (subprocess died with {why}; recording this point as failed)")
            return (np.nan, np.nan), (np.nan, np.nan, np.nan, np.nan), np.nan

        runtimes.append(result["runtime"])
        peaks.append(result["peak_mem"])
        absolutes.append(result.get("abs_mem", float("nan")))
        xlas.append(result.get("xla_mem", float("nan")))
        # The kernel's own high-water mark, for comparison. Sampling a peak from
        # a Python thread while 32 BLAS threads saturate the box can in
        # principle miss it, and the dense curves' largest points look like
        # exactly that -- 11 B/N^2 against a theoretical 16. Only reported when
        # the two disagree materially, so a healthy sweep stays quiet.
        hwm = result.get("hwm_mem")
        sampled = result.get("sampled_mem")
        if hwm and sampled and sampled > 0:
            ratio = hwm / sampled
            if ratio > 1.15 or ratio < 0.87:
                print(
                    f"      (VmHWM {format_bytes(hwm)} vs {format_bytes(sampled)}"
                    f" sampled, {ratio:.2f}x -- VmHWM is recorded; sampling"
                    " misses large peaks, see CPUMemorySampler)",
                    flush=True,
                )
        output = pickle.loads(result["output"])

        if adaptive and len(runtimes) == 1:
            target = repeats_for(result["runtime"])
            if target < max_repeats:
                print(
                    f"      (first call {result['runtime']:.3g}s -> {target}"
                    f" repeat{'s' if target > 1 else ''})"
                )
        if len(runtimes) >= target:
            break
        if runtimes[0] > PROGRESS_ABOVE_SECONDS:
            left = (target - len(runtimes)) * 2 * result["runtime"] / 60
            print(
                f"      (repeat {len(runtimes)}/{target}, {result['runtime']:.3g}s;"
                f" ~{left:.0f} min left on this point)",
                flush=True,
            )

        # Over budget: stop, rather than paying for the rest. Only reachable
        # with an explicit n_repeat; adaptive is already at 1 by here.
        if max_seconds is not None and result["runtime"] > max_seconds:
            print(
                f"      ({result['runtime']:.1f}s > {max_seconds:g}s budget;"
                f" skipping the remaining {target - len(runtimes)})"
            )
            break
    runtimes = np.array(runtimes)
    peaks = np.array(peaks)
    # Dropping the min and max costs two samples, so this only makes sense with
    # enough left over to still be a mean: below 5 repeats it is skipped
    # entirely. At 1 or 2 it would be outright broken -- argmin and argmax
    # cover the whole array and the mean of what remains is NaN -- and at 3 or 4
    # the one or two survivors are a mean in name only. Guarded here rather than
    # at the call site because --nrepeat makes low counts reachable from every
    # caller.
    if drop_outliers and len(runtimes) >= 5:
        runtimes = np.delete(runtimes, [runtimes.argmin(), runtimes.argmax()])
        peaks = np.delete(peaks, [peaks.argmin(), peaks.argmax()])

    # The memory entry is (mean, std, absolute, xla), widened from the original
    # (mean, std). Readers index [0] and [1], so the extra fields ride along
    # without touching the plotting code or invalidating older result files --
    # those simply have a 2-tuple, and anything wanting the new fields must
    # tolerate their absence.
    absolutes = np.asarray(absolutes, dtype=float)
    xlas = np.asarray(xlas, dtype=float)
    return (
        (np.mean(runtimes), np.std(runtimes)),
        (
            np.mean(peaks),
            np.std(peaks),
            float(np.nanmean(absolutes)) if absolutes.size else float("nan"),
            float(np.nanmean(xlas)) if xlas.size else float("nan"),
        ),
        output,
    )


def benchmark(
    funcs,
    data,
    objs,
    *args,
    n_repeat=5,
    cutoffs={},
    floors={},
    keep_over_budget=False,
    min_seconds=None,
    xla_only=False,
    no_checkpoint=False,
    drop_outliers=False,
    use_gpu_profiler=False,
    max_seconds=None,
    tag="",
    only_sizes=None,
    only_indices=None,
    sizes=None,
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
    _retired = {}
    Ns = []
    # (N, seconds) per curve, this sweep only, for projecting the next point's
    # cost before paying for it. Deliberately not seeded from earlier runs: a
    # solver change is exactly when the old numbers stop applying.
    history = {name: [] for name in funcs}
    machine = "gpu" if use_gpu_profiler else "cpu"
    for n in range(len(data)):
        if sizes is not None:
            N = int(sizes[n])
        else:
            # Integrated coordinates arrive as a tuple (t, texp, instid) rather
            # than a bare array, and every component has the same length, so
            # size off the first.
            coords = data[n]
            N = (coords[0] if isinstance(coords, tuple) else coords).shape[-1]
        Ns.append(N)

        # A None slot means the caller never built this dataset, because no
        # curve can run at this size. Record the row and move on -- the size
        # still belongs on the x axis, it just has no measurement.
        if data[n] is None:
            print(
                f"  ({n + 1}/{len(data)}):  N = {N} -- Skipped "
                "(above every cutoff, dataset not built)"
            )
            for name in funcs:
                runtime[name].append((jnp.nan, jnp.nan))
                memory[name].append((jnp.nan, jnp.nan, jnp.nan, jnp.nan))
                outputs[name].append(jnp.nan)
            continue

        print(f"  ({n + 1}/{len(data)}):  N = {N}")
        for name in funcs:
            func = funcs[name]
            obj = objs[name]  # either kernel or gp
            cutoff = cutoffs.get(name, 3e4)
            # A floor turns the usual "up to the cutoff" window into a band:
            # (floor, cutoff]. --long-runs-only sets the floor to the
            # production cutoff so a long run measures exactly the sizes a
            # production sweep declines, and re-measures none of the ones it
            # already has.
            floor = floors.get(name, 0.0)

            # Both cost and memory need grow monotonically with size, so a
            # curve that has already blown the per-call budget or failed to
            # allocate will do so at every larger size too. Retire it rather
            # than re-measuring what is already known. _retired holds the
            # reason, so the skip line says which of the two it was.
            if _retired.get(name):
                t, mem, val = (jnp.nan, jnp.nan), (jnp.nan, jnp.nan, jnp.nan, jnp.nan), jnp.nan
                print(f"    {name}: Skipped ({_retired[name]} at a smaller size)")
                runtime[name].append(t)
                memory[name].append(mem)
                outputs[name].append(val)
                continue

            measured = floor < N <= cutoff

            # Project this point's cost from the two most recent measurements of
            # the same curve, and skip it if the projection is over budget.
            # size_cutoffs already applies a coarse version up front, from
            # _COST's calibrated constants; this uses the sweep's own numbers,
            # so it tracks the machine and the current code. It replaces
            # measuring-then-discarding, which paid a point's full cost before
            # deciding to throw it away -- 949 s, once.
            #
            # Under-projection is the safe failure: the point simply gets
            # measured. Over-projection skips a feasible one, which is why the
            # slope comes from the last two points rather than a nominal
            # exponent -- a nominal N^3 read from inside a transition
            # over-predicts badly.
            if measured and max_seconds and len(history[name]) >= 2:
                (n0, t0), (n1, t1) = history[name][-2], history[name][-1]
                if n1 > n0 > 0 and t0 > 0 and t1 > 0:
                    power = math.log(t1 / t0) / math.log(n1 / n0)
                    projected = t1 * (N / n1) ** power
                    if min_seconds is not None and projected < min_seconds:
                        # Below the Tier 2 threshold: production's job, not
                        # this run's. See LONG_RUN_MIN_SECONDS.
                        print(
                            f"    {name}: Skipped (projected {projected:.0f}s"
                            f" < {min_seconds:g}s, belongs to the production"
                            " suite)"
                        )
                        runtime[name].append((jnp.nan, jnp.nan))
                        memory[name].append((jnp.nan, jnp.nan, jnp.nan, jnp.nan))
                        outputs[name].append(jnp.nan)
                        continue
                    if projected > max_seconds:
                        _retired[name] = (
                            f"projected over the {max_seconds:g}s budget"
                        )
                        print(
                            f"    {name}: Skipped (projected {projected:.0f}s"
                            f" via N^{power:.2f} from the last two points,"
                            f" over the {max_seconds:g}s budget)"
                        )
                        runtime[name].append((jnp.nan, jnp.nan))
                        memory[name].append((jnp.nan, jnp.nan, jnp.nan, jnp.nan))
                        outputs[name].append(jnp.nan)
                        continue

            if measured and xla_only:
                # Static pass: compile only. The other slots stay NaN so a merge
                # can tell "not measured this way" from "measured as zero", and
                # the checkpoint guard below -- which keys on a finite runtime --
                # will not overwrite a real per-point file.
                x = xla_footprint(func, data[n], obj, *args)
                t = (jnp.nan, jnp.nan)
                mem = (jnp.nan, jnp.nan, jnp.nan, x)
                val = jnp.nan
                print(f"    {name}: xla = {format_bytes(x)}")
            elif measured:
                t, mem, val = profile_jax_function(
                    func,
                    data[n],
                    obj,
                    *args,
                    n_repeat=n_repeat,
                    machine=machine,
                    drop_outliers=drop_outliers,
                    max_seconds=max_seconds,
                )
                if not np.isfinite(t[0]):
                    # profile_jax_function only returns NaN when the child died
                    # -- OOM-killed, or RESOURCE_EXHAUSTED inside XLA. Retire.
                    _retired[name] = "the subprocess died"
                basestr = f"    {name}: time = {t[0]:.4f} ± {t[1]:.4f} s"
                memstr = f", mem = {format_bytes(mem[0])} ± {format_bytes(mem[1])}"
                print(basestr + memstr)
                if max_seconds and t[0] > max_seconds:
                    # Flag first: the retirement decision needs the real time.
                    _retired[name] = f"exceeded the {max_seconds:g}s budget"
                    if keep_over_budget:
                        # A long run exists to buy exactly these points, so the
                        # one that finally goes over is the one worth keeping --
                        # it is the measurement, not an accident. Retire the
                        # curve so nothing larger is attempted, but bank it.
                        print(
                            f"      (over the {max_seconds:g}s budget; keeping"
                            " the point and retiring this curve)"
                        )
                    else:
                        print(
                            f"      (over the {max_seconds:g}s budget; retiring"
                            " this curve and discarding the point)"
                        )
                        # This is a size the sweep was told not to spend, so
                        # record it the way a cutoff-retired size is recorded.
                        # The measured value is still printed above.
                        t, mem, val = (jnp.nan, jnp.nan), (jnp.nan, jnp.nan, jnp.nan, jnp.nan), jnp.nan
            elif N <= floor:
                t, mem, val = (jnp.nan, jnp.nan), (jnp.nan, jnp.nan, jnp.nan, jnp.nan), jnp.nan
                print(f"    {name}: Skipped (N={N} <= floor={floor:.3g}, already"
                      " covered by the production sweep)")
            else:
                t, mem, val = (jnp.nan, jnp.nan), (jnp.nan, jnp.nan, jnp.nan, jnp.nan), jnp.nan
                print(f"    {name}: Skipped (N={N} > cutoff={cutoff:.3g})")

            if measured and np.isfinite(float(t[0])):
                history[name].append((float(N), float(t[0])))

            runtime[name].append(t)
            memory[name].append(mem)
            outputs[name].append(val)

            # Only checkpoint points we actually measured. These per-point
            # files are what a clobbered aggregate gets rebuilt from, so
            # writing a NaN placeholder over a real measurement -- which is
            # what a --quick pass would otherwise do at every skipped size,
            # or a crashed subprocess at a size that succeeded on an earlier
            # run -- destroys exactly the thing that makes recovery possible.
            # no_checkpoint: a merge-only pass (--absolute-only) measures with
            # a single repeat, so its numbers must not overwrite the averaged
            # per-point files that --rebuild reconstructs from.
            if measured and not no_checkpoint and not np.isnan(float(t[0])):
                save_benchmark_data(
                    f"results/individual/{func.__name__}_{N}_{machine}{tag}.pkl",
                    [N],
                    {name: [t]},
                    {name: [mem]},
                    {name: [val]},
                )

    return Ns, runtime, memory, outputs


def integrated_data_ceiling(texp, readout, buffer_frac=0.1):
    """OBSOLETE. Largest N the *old* integrated generator could build.

    Kept only to document a limit that no longer applies:
    ``generate_integrated_data`` now draws from the integrated kernel in O(N),
    so every grid size is buildable. Nothing calls this; use ``--max-n`` if you
    want to cap a grid for some other reason.

    Historically:

    ``utils.generate_integrated_data`` samples the truth on a 1 s grid spanning
    the whole observing baseline, so it materialises

        N * (texp + readout) * (1 + 2 * buffer_frac)

    points -- growing with the *baseline*, not with N. Past 2**31 of them the
    process dies: SIGSEGV on CPU, ``Invalid dimension size`` on GPU. Neither is
    catchable, so the ceiling has to be enforced before anything is attempted,
    both when building datasets and when choosing size cutoffs.

    At the suite's cadence (texp=140, readout=40) this lands at N ~ 9.9e6, just
    below the grid's largest point of 1e7 -- which is why one size took a whole
    sweep down with it.
    """
    cadence = texp + readout
    return int(2**31 / (cadence * (1 + 2 * buffer_frac)))


def select_sizes(sizes, only_sizes=None, only_indices=None):
    """Narrow a size grid to a chosen subset, for re-running individual points.

    Args:
        sizes: the full grid, as the runner built it.
        only_sizes: sizes to keep. Matched to the nearest grid point, since the
            grid comes from ``logspace(...).astype(int)`` and nobody wants to
            type 4216965 exactly.
        only_indices: 1-based positions to keep, matching the ``(i/17)`` the
            sweep prints, so a point can be named straight off the log.

    Returns:
        ``(kept_sizes, kept_positions)`` -- positions are indices into the
        original grid, so callers can report where in the sweep each point sat.
    """
    sizes = list(sizes)
    if not only_sizes and not only_indices:
        return sizes, list(range(len(sizes)))

    keep = set()
    for i in only_indices or []:
        if not 1 <= i <= len(sizes):
            raise ValueError(f"index {i} out of range 1..{len(sizes)}")
        keep.add(i - 1)
    for want in only_sizes or []:
        nearest = min(range(len(sizes)), key=lambda j: abs(int(sizes[j]) - int(want)))
        if int(sizes[nearest]) != int(want):
            print(
                f"  note: no grid point at {int(want)}; using the nearest, "
                f"{int(sizes[nearest])} (position {nearest + 1}/{len(sizes)})"
            )
        keep.add(nearest)

    order = sorted(keep)
    return [sizes[i] for i in order], order


def rebuild_from_points(kind, device, integrated=False, m_per_n=100, n_sizes=17,
                        logmin=1, logmax=7, curves=None, tag=""):
    """Reassemble an aggregate result file from the per-point checkpoints.

    Every measured point is written to ``results/individual/`` as it completes,
    so the aggregate is derivable from them. That makes it possible to re-run a
    single size in isolation and fold it back in, and it is also how a
    clobbered or interrupted sweep gets recovered -- which has been needed more
    than once.

    Missing points come back as NaN, exactly as a skipped size would.

    ``tag`` selects which family of checkpoints to read -- ``"_quick"`` for an
    abridged run, whose grid and per-point budget differ, so its points must not
    be mixed with production ones.
    """
    import glob

    prefixes = {
        "llh": {"SSM": "ss_llh", "QSM": "qs_llh", "GP": "gp_llh",
                "pSSM": "pss_llh", "pQSM": "pqs_llh"},
        "llh_value_and_grad": {
            "SSM": "ss_llh_vg", "QSM": "qs_llh_vg", "GP": "gp_llh_vg",
            "pSSM": "pss_llh_vg", "pQSM": "pqs_llh_vg",
        },
        "cond": {"SSM": "ss_cond", "QSM": "qs_cond", "GP": "gp_cond",
                 "pSSM": "pss_cond", "pQSM": "pqs_cond"},
        "pred": {"SSM": "ss_pred", "QSM": "qs_pred", "GP": "gp_pred"},
        "sample-prior": {"SSM": "ss_sample_prior", "QSM": "qs_sample_prior",
                         "GP": "gp_sample_prior"},
        "sample-post": {"SSM": "ss_sample_post", "QSM": "qs_sample_post",
                        "GP": "gp_sample_post"},
    }[kind]
    if integrated:
        prefixes = {k: "i" + v for k, v in prefixes.items() if k != "QSM"}

    if kind in _M_SCALED:
        logmax = logmax - round(math.log10(m_per_n))
    grid = [int(n) for n in np.logspace(logmin, logmax, n_sizes).astype(int)]

    # The M-scaled kinds checkpoint by M, not by N. benchmark() is handed the
    # test grid for those, so the size it sees -- and names the file after -- is
    # M = m_per_n * N, while the aggregate's x axis stays in N. Looking up by N
    # found only the three sizes that appear in both ladders (1000, 10000,
    # 100000) and returned NaN for the other fourteen, silently replacing a full
    # 17-point sweep with a 3-point file. Rebuild has never worked for pred or
    # sample-post; it went unnoticed because those kinds had not needed it.
    key_for = (lambda n: m_per_n * n) if kind in _M_SCALED else (lambda n: n)

    nan = (float("nan"), float("nan"))
    runtime, memory, outputs = {}, {}, {}
    found = 0
    for name, pre in prefixes.items():
        if curves and name not in curves:
            continue
        runtime[name], memory[name], outputs[name] = [], [], []
        for N in grid:
            f = f"results/individual/{pre}_{key_for(N)}_{device}{tag}.pkl"
            if os.path.exists(f):
                d = load_benchmark_data(f)
                runtime[name].append(d["runtime"][name][0])
                memory[name].append(d["memory"][name][0])
                outputs[name].append(d["outputs"][name][0])
                found += 1
            else:
                runtime[name].append(nan)
                memory[name].append(nan)
                outputs[name].append(float("nan"))
    # Preserve the absolute and xla slots. Checkpoints written before the entry was
    # widened to (mean, std, absolute, xla) carry only two fields, so rebuilding
    # from them silently discards a column that --xla-only had filled -- and
    # that column is static, so losing it means re-deriving it for no reason.
    # Any value the checkpoints do supply wins, since it is the newer one.
    prior = f"results/{device}_{kind}{'_int' if integrated else ''}{tag}_benchmark.pkl"
    if os.path.exists(prior):
        old_mem = load_benchmark_data(prior).get("memory", {})
        kept = 0
        for name, entries in memory.items():
            was = old_mem.get(name, [])
            for i, e in enumerate(entries):
                e = tuple(e) + (float("nan"),) * (4 - len(e))
                for slot in (2, 3):
                    if (e[slot] != e[slot] and i < len(was)
                            and len(was[i]) > slot and was[i][slot] == was[i][slot]):
                        e = e[:slot] + (was[i][slot],) + e[slot + 1:]
                        kept += 1
                entries[i] = e
        if kept:
            print(f"  kept {kept} xla values the checkpoints do not carry")

    print(f"  rebuilt {kind}{'_int' if integrated else ''} ({device}) from "
          f"{found} per-point files across {len(grid)} sizes")
    return grid, runtime, memory, outputs


def make_data_files(true_kernel, kind, yerr=0.3, exposure_quantities=None,
                    n_sizes=17, logmin=1, logmax=7, m_per_n=100,
                    only_sizes=None, only_indices=None, overwrite=False,
                    max_n=None):
    """Build the ``data/*.npz`` inputs for a kind's grid, without profiling.

    Split out from the sweeps so a single dataset can be repaired or rebuilt on
    its own -- regenerating one file should not mean sitting through a
    multi-hour benchmark, and a dataset that cannot be built should not be
    discovered halfway through one.

    Args:
        kind: which grid to build for. ``pred``/``sample-post`` index by N with
            M = m_per_n * N, so their N ladder stops m_per_n earlier.
        only_sizes, only_indices: restrict to particular sizes or 1-based grid
            positions (see :func:`select_sizes`).
        overwrite: rebuild files that already exist. Off by default, so a run
            fills gaps rather than redoing work.
        max_n: refuse sizes above this. Integrated data has a hard ceiling --
            generate_integrated_data samples the truth on a 1 s grid across the
            whole baseline, so it needs N * cadence * 1.2 points, which passes
            int32 at N ~ 9.9e6.

    Returns:
        ``(written, skipped, failed)`` lists of sizes.
    """
    if kind in _M_SCALED:
        logmax = logmax - round(math.log10(m_per_n))
    grid = [int(n) for n in np.logspace(logmin, logmax, n_sizes).astype(int)]
    grid, _ = select_sizes(grid, only_sizes, only_indices)

    isint = "_int" if exposure_quantities else ""
    written, skipped, failed = [], [], []
    for N in grid:
        path = f"data/{N}{isint}.npz"
        if max_n is not None and N > max_n:
            print(f"  {N:>9}  refused: above the max_n ceiling of {int(max_n)}")
            failed.append(N)
            continue
        if os.path.exists(path) and not overwrite:
            print(f"  {N:>9}  exists, skipping ({path})")
            skipped.append(N)
            continue
        verb = "rewriting" if os.path.exists(path) else "generating"
        print(f"  {N:>9}  {verb} {path} ...", flush=True)
        try:
            get_data(true_kernel, N, yerr=yerr,
                     exposure_quantities=exposure_quantities, save=True)
            size = os.path.getsize(path) if os.path.exists(path) else 0
            print(f"  {N:>9}  wrote {path} ({format_bytes(size)})", flush=True)
            written.append(N)
        except Exception as exc:  # noqa: BLE001 -- report and continue
            print(f"  {N:>9}  FAILED: {type(exc).__name__}: {str(exc)[:120]}",
                  flush=True)
            failed.append(N)
    print(f"\n  wrote {len(written)}, skipped {len(skipped)}, failed {len(failed)}")
    if failed:
        print(f"  failed sizes: {failed}")
    return written, skipped, failed


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
    floors={},
    keep_over_budget=False,
    min_seconds=None,
    xla_only=False,
    no_checkpoint=False,
    drop_outliers=False,
    use_gpu_profiler=False,
    exposure_quantities=None,
    max_seconds=None,
    tag="",
    only_sizes=None,
    only_indices=None,
):
    """
    Generate data and benchmark the provided functions over a range of input sizes.
    """
    print("Generating data for benchmarking...")
    isint = "_int" if exposure_quantities else ""
    Ns = jnp.logspace(logN_min, logN_max, N_N).astype(int)
    Ns, _ = select_sizes(Ns, only_sizes, only_indices)

    # Only build a dataset some curve will actually consume. This used to
    # generate every size unconditionally, so the largest was built even when
    # every cutoff had already retired it -- and when building it failed, the
    # sweep died before measuring anything at all. Integrated data at N = 1e7
    # does precisely that: generate_integrated_data samples the truth on a 1 s
    # grid across the whole baseline, needing N * cadence * 1.2 = 2.16e9 points,
    # past both int32 dimensions and any sane memory budget. Sixteen good sizes
    # were lost to the seventeenth.
    # Note this is deliberately driven by cutoffs only, never by floors: a long
    # run's targets are the sizes *above* the production cutoff, so folding
    # floors in here would skip building exactly the data it needs.
    reachable = max(cutoffs.values()) if cutoffs else float("inf")
    data = []
    for N in Ns:
        if int(N) > reachable:
            print(f"  Skipping data for N = {int(N)} (above every cutoff)")
            data.append(None)
            continue
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
        floors=floors,
        keep_over_budget=keep_over_budget,
        min_seconds=min_seconds,
        xla_only=xla_only,
        no_checkpoint=no_checkpoint,
        drop_outliers=drop_outliers,
        use_gpu_profiler=use_gpu_profiler,
        max_seconds=max_seconds,
        tag=tag,
        sizes=Ns,
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
    floors={},  # in M
    keep_over_budget=False,
    min_seconds=None,
    xla_only=False,
    no_checkpoint=False,
    drop_outliers=False,
    use_gpu_profiler=False,
    exposure_quantities=None,
    max_seconds=None,
    tag="",
    only_sizes=None,
    only_indices=None,
):
    runtime = {name: [] for name in funcs}
    memory = {name: [] for name in funcs}
    outputs = {name: [] for name in funcs}

    # Data (N) and test (M)
    Ns = jnp.logspace(logN_min, logN_max, N_N).astype(int)
    Ns, _ = select_sizes(Ns, only_sizes, only_indices)
    Ms = 100 * jnp.asarray(Ns)
    isint = "_int" if exposure_quantities else ""
    for i, (N, M) in enumerate(zip(Ns, Ms), 1):
        skip = True
        for key in cutoffs:
            if floors.get(key, 0.0) < M <= cutoffs[key]:
                skip = False
        if skip:
            why = "M > all cutoffs" if not floors else "M outside every (floor, cutoff]"
            print(f"  ({i}/{len(Ns)}):  N = {N}, M = {M} -- Skipped ({why})")
            for name in funcs:
                runtime[name].append((jnp.nan, jnp.nan))
                memory[name].append((jnp.nan, jnp.nan, jnp.nan, jnp.nan))
                outputs[name].append(jnp.nan)
            continue

        print(f"  ({i}/{len(Ns)}):  N = {N}, M = {M}")
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

        ## Prepare GP objects -- but only for curves this size will measure.
        #
        # A dense tinygp.GaussianProcess factorises in its *constructor*, and
        # un-jitted, so building one is not free bookkeeping: at N = 56234 it is
        # a 56234^2 kernel matrix plus its Cholesky, and the eager construction
        # peaks far above the 50 GB those two need. Doing that in the parent for
        # a curve the cutoffs immediately skip used to consume ~340 GB and take
        # the process out with a SIGSEGV -- which is how a --long-runs-only pred
        # sweep died at grid point 16 of 17, having measured only SSM.
        #
        # The band is the same one benchmark() gates on, including the same
        # default cutoff, so nothing that would be measured goes unbuilt. A
        # skipped curve gets None, which benchmark() never dereferences.
        def _in_band(curve):
            return floors.get(curve, 0.0) < M <= cutoffs.get(curve, 3e4)

        gp = {
            "SSM": (
                smolgp.GaussianProcess(kernels["SSM"], X_train, noise=yerr_train**2)
                if _in_band("SSM") else None
            ),
            "GP": (
                tinygp.GaussianProcess(kernels["GP"], X_train, diag=yerr_train**2)
                if _in_band("GP") else None
            ),
        }
        if "QSM" in kernels:
            gp["QSM"] = (
                tinygp.GaussianProcess(kernels["QSM"], X_train, diag=yerr_train**2)
                if _in_band("QSM") else None
            )

        _, t, m, o = benchmark(
            funcs,
            [t_test],
            gp,
            y_train,
            n_repeat=n_repeat,
            cutoffs=cutoffs,
            floors=floors,
            keep_over_budget=keep_over_budget,
            min_seconds=min_seconds,
            xla_only=xla_only,
            no_checkpoint=no_checkpoint,
            drop_outliers=drop_outliers,
            use_gpu_profiler=use_gpu_profiler,
            max_seconds=max_seconds,
            tag=tag,
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
    floors={},  # in M
    keep_over_budget=False,
    min_seconds=None,
    xla_only=False,
    no_checkpoint=False,
    drop_outliers=False,
    use_gpu_profiler=False,
    exposure_quantities=None,
    tmax=1e4,
    max_seconds=None,
    tag="",
    only_sizes=None,
    only_indices=None,
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
    Ms, _ = select_sizes(Ms, only_sizes, only_indices)
    for i, M in enumerate(Ms, 1):
        skip = True
        for key in cutoffs:
            if floors.get(key, 0.0) < M <= cutoffs[key]:
                skip = False
        if skip:
            why = "M > all cutoffs" if not floors else "M outside every (floor, cutoff]"
            print(f"  ({i}/{len(Ms)}):  M = {M} -- Skipped ({why})")
            for name in funcs:
                runtime[name].append((jnp.nan, jnp.nan))
                memory[name].append((jnp.nan, jnp.nan, jnp.nan, jnp.nan))
                outputs[name].append(jnp.nan)
            continue

        print(f"  ({i}/{len(Ms)}):  M = {M}")
        t_sample = jnp.linspace(0.0, tmax, M)
        if exposure_quantities:
            # Scale the exposure with the sample spacing rather than holding it
            # fixed. With a fixed texp on a window of fixed length, the spacing
            # shrinks as M grows until exposures overlap -- 140-fold by M=1e4 --
            # and the number of integral accumulators the draw needs grows with
            # M too (see assign_min_instids). The curve would then be measuring
            # M *and* that second, hidden dimension at once. Holding the duty
            # cycle fixed instead keeps exposures non-overlapping at every M, so
            # the scaling is in M alone.
            texp, readout = exposure_quantities
            duty = texp / (texp + readout)
            spacing = tmax / max(M - 1, 1)
            texp_M = duty * spacing
            X_sample = (
                t_sample,
                jnp.full_like(t_sample, texp_M),
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
            floors=floors,
            keep_over_budget=keep_over_budget,
            min_seconds=min_seconds,
            xla_only=xla_only,
            no_checkpoint=no_checkpoint,
            drop_outliers=drop_outliers,
            use_gpu_profiler=use_gpu_profiler,
            max_seconds=max_seconds,
            tag=tag,
        )
        for name in funcs:
            runtime[name].append(t[name][0])
            memory[name].append(m[name][0])
            outputs[name].append(o[name][0])

    return Ms, runtime, memory, outputs


# ---------------------------------------------------------------------------
# Size cutoffs from a memory budget and a wall-clock budget
#
# Every curve is bounded two ways, and which one binds depends on both the
# machine and the kind of benchmark:
#
#   memory -- mainly hardware-limited (to prevent OOM errors).
#   time   -- mainly patience-limited (not waiting years for a run to finish).
#
# size_cutoffs() takes the min of the two. Both come from the table below,
# whose entries are *measured*, not estimated: memory from one-subprocess-per-
# size runs of the real funcs, time from the tails of results/cpu_*.pkl.
#
# Two assumptions in the previous version of this code were wrong and are
# worth stating so they do not creep back:
#
#   1. "SSM and QSM are O(N), so they are time-bound and can take a flat cap."
#      True for SSM everywhere, and for QSM under llh/cond. NOT true for QSM
#      under pred or sample-post, where tinygp densifies: 56 B/(N*M) and
#      49 B/M^2 respectively. At the old flat 1e6 cap those want 560 GB and
#      49 TB.
#   2. "A posterior draw costs O(N*M), like a prediction."
#      No -- gp/qs_sample_post call condition(y, t_test) and then .sample(),
#      which Choleskys the full M x M posterior covariance. It is O(M^2).
# ---------------------------------------------------------------------------

# Memory coefficients recalibrated 2026-08-23 against the first complete suite
# measured with the corrected profiler (see CPUMemorySampler), via
# calibrate_costs.py. The dense GP entries were derived from first principles
# and needed no change -- all measured at 1.00x. Everything that moved is an
# O(N) state-space or quasisep curve whose constant was "the measured
# asymptote" and had gone stale:
#
#   llh  SSM       72 -> 160    cond SSM        184 -> 320
#   llh  QSM      153 -> 209    cond QSM        585 -> 633
#   pred QSM       48 ->  64    pred SSM        445 -> 489
#   llh  SSM int  352 -> 584    cond SSM int    904 -> 1344
#   pred SSM int 10860 -> 1631
#
# The SSM drifts are most likely the split-scan log_probability changing the
# footprint. pred QSM being 1.33x low is what admitted the M = 1e6 point that
# then tried to allocate 560 GB; pred SSM int was 6.7x *high*, an extrapolated
# guess that had been quietly costing that curve grid points. Re-run
# calibrate_costs.py after any full sweep rather than waiting for an OOM.
#
# exponents are (n_pow, m_pow): cost = coeff * N**n_pow * M**m_pow
_COST = {
    #                     memory (bytes)        time (seconds)
    #                     coeff   n,m           coeff     n,m
    ("llh", "GP"): ((16.0, (2, 0)), (2.7e-13, (3, 0))),
    ("llh", "SSM"): ((160.0, (1, 0)), (1.5e-06, (1, 0))),
    ("llh", "QSM"): ((209.0, (1, 0)), (5.4e-08, (1, 0))),
    ("cond", "GP"): ((24.0, (2, 0)), (9.9e-13, (3, 0))),
    ("cond", "SSM"): ((320.0, (1, 0)), (2.7e-06, (1, 0))),
    ("cond", "QSM"): ((633.0, (1, 0)), (7.8e-07, (1, 0))),
    ("pred", "GP"): ((24.0, (1, 1)), (1.04e-12, (2, 1))),
    # 48.0, not the 56 seen at N < 1e3: this constant only converges above
    # N ~ 1e3 (45.5 -> 50.8 -> 48.3 -> 48.00 across the production grid), and
    # the cutoff lives in the converged regime.
    ("pred", "QSM"): ((64.0, (1, 1)), (2.55e-08, (1, 1))),
    ("pred", "SSM"): ((489.0, (0, 1)), (1.74e-06, (0, 1))),
    # Time measured with smolgp.kernels.dense.SHOKernel, the kernel the
    # benchmark actually uses -- a generic tinygp.kernels.ExpSquared is ~7x
    # slower to build and gives a constant that truncates this curve at half
    # its real reach.
    ("sample-prior", "GP"): ((16.0, (0, 2)), (3.3e-13, (0, 3))),
    ("sample-prior", "SSM"): ((716.0, (0, 1)), (3.6e-06, (0, 1))),
    ("sample-prior", "QSM"): ((578.0, (0, 1)), (3.0e-07, (0, 1))),
    ("sample-post", "GP"): ((24.0, (0, 2)), (4.5e-13, (0, 3))),
    ("sample-post", "QSM"): ((49.0, (0, 2)), (1.5e-12, (0, 3))),
    ("sample-post", "SSM"): ((1050.0, (0, 1)), (4.9e-06, (0, 1))),
}

# Integrated-data (--int) variants. Same laws, different coefficients, and the
# differences are large enough to matter: cond GP is 72 B/N^2 against 24 for
# instantaneous, and pred SSM is 10860 B/M against 445, because the integrated
# kernels carry exposure/instrument state through every intermediate. Using the
# instantaneous numbers here would put cond GP a factor of 3 over budget.
#
# There is no QSM under --int (see the kernels dict in run_benchmark.py), so
# these cover SSM and GP only.
_COST_INT = {
    ("llh", "GP"): ((16.0, (2, 0)), (6.4e-13, (3, 0))),
    ("llh", "SSM"): ((584.0, (1, 0)), (1.9e-05, (1, 0))),
    ("cond", "GP"): ((72.0, (2, 0)), (2.2e-12, (3, 0))),
    ("cond", "SSM"): ((1344.0, (1, 0)), (2.6e-05, (1, 0))),
    ("pred", "GP"): ((24.0, (1, 1)), (4.42e-12, (2, 1))),
    ("pred", "SSM"): ((1631.0, (0, 1)), (1.44e-05, (0, 1))),
    ("sample-prior", "GP"): ((16.0, (0, 2)), (1.2e-12, (0, 3))),
    ("sample-post", "GP"): ((24.0, (0, 2)), (4.8e-13, (0, 3))),
    ("sample-prior", "SSM"): ((1212.0, (0, 1)), (1.1e-05, (0, 1))),
    # Measured, 2026-08-16: 1936-2108 B/M over M = 1.8e5..1e6, taking the
    # larger end. This entry used to be an extrapolated guess of 10900, which
    # was 5.6x too conservative -- it became measurable once sample() grew a
    # num_test_insts argument and the integrated posterior path could be
    # jitted (funcs.iss_sample_post passes 1, correct because those test
    # points are instantaneous).
    ("sample-post", "SSM"): ((2110.0, (0, 1)), (2.0e-05, (0, 1))),
}

# Kinds whose x axis (and therefore whose cutoffs) are expressed in M rather
# than N. For these, N = M / m_per_n is substituted before solving.
_M_SCALED = ("pred", "sample-post")

# Memory installed per machine and device, for targeting a box you are not
# sitting at via --machine. The CPU budget is normally auto-detected instead
# (see ram_budget); these are the fallbacks.
# workstation is Intel® Xeon® w5-3435X CPU + NVIDIA RTX 6000 Ada GPU
# macbook is Apple M3 Max 64 GB (Nov 2023)
MACHINE_RAM_GB = {
    "workstation": {"cpu": 503, "gpu": 48},
    "macbook": {"cpu": 64, "gpu": 64},
}

# Held back from the detected budget. `available` is a snapshot taken at
# startup, but a cond sweep runs for hours and can easily outlive whatever
# else gets launched during it. Capped as a fraction so a 64 GB laptop does
# not hand over a quarter of itself.
RESERVE_GB = 16.0
RESERVE_FRAC = 0.10

# Margin on the *measured* cost constants, not on the RAM. Applied inside the
# cost law so each curve's exponent dilutes it correctly -- 12% off a cutoff
# in N^2, 23% off one in N -- which a flat haircut on the budget cannot do.
_SAFETY = 1.3

# ---------------------------------------------------------------------------
# Reverse mode costs more memory than the forward pass it differentiates, so
# _COST/_COST_INT -- both calibrated on the forward pass -- are optimistic for
# --value-and-grad. Multipliers on the *memory* coefficient only, keyed by
# (kind, curve, integrated). The time coefficient is left alone: the gradient is
# slower too, but a bad time estimate only costs a wasted measurement, whereas a
# bad memory estimate is an OOM kill on a swapless box.
#
# Measured 2026-08-21 from the paired forward and value-and-grad result files,
# comparing B/N^exponent at the largest *reliable* point of each curve. The
# single largest point of every GP curve is deliberately excluded: it reports
# 11 B/N^2 against a theoretical exactly-16, i.e. the profiler under-reports
# there (see MemorySampler -- its verification stops at N=23713, which is where
# the discrepancy starts). Using it would build the factor out of a known-bad
# number.
#
#   kind curve int   fwd model   fwd measured   vg measured   factor
#   llh  SSM   no     72           159            553          3.5
#   llh  QSM   no    153           185            537          2.9
#   llh  GP    no     16            16             41          2.6
#   llh  SSM   yes   352           577           2643          4.6
#   llh  GP    yes    16            16          ~1570         98
#
# The integrated GP factor is the outlier that matters: the dense integrated
# kernel is built through an expression whose N x N intermediates the tape all
# retains, where the instantaneous kernel's is nearly free. At 16 B/N^2 the
# model put GP's cutoff at N = 9.8e4; the real limit is ~1e4, and the two grid
# points in between were spent on doomed 886 GB and 4.9 TB allocations.
#
# Note in passing that the forward SSM constants are themselves low by ~2x
# (72 against 159 measured, 352 against 577). That is a separate drift, most
# likely from the split-scan log_probability changing the footprint, and it is
# not corrected here -- _SAFETY absorbs part of it and the curves are cap-bound
# anyway, so nothing currently depends on it.
_GRAD_MEM_FACTOR = {
    ("llh", "SSM", False): 3.5,
    ("llh", "QSM", False): 2.9,
    ("llh", "GP", False): 2.6,
    ("llh", "SSM", True): 4.6,
    ("llh", "GP", True): 98.0,
}

#: Fallback for a (kind, curve) with no measurement yet. Deliberately on the
#: high side of the instantaneous factors: over-estimating costs a grid point,
#: under-estimating costs an OOM kill.
GRAD_MEM_FACTOR_DEFAULT = 5.0


def ram_budget(machine=None, device="cpu", max_ram_gb=None):
    """Memory budget for the size cutoffs, in bytes.

    An explicit ``max_ram_gb`` is taken literally: no reserve, no safety
    factor, so a deliberate edge-of-the-machine run is still possible.
    Otherwise the CPU budget is measured from this machine and the reserve is
    subtracted; the GPU budget comes from MACHINE_RAM_GB, since a device this
    process cannot see cannot be probed.
    """
    if max_ram_gb is not None:
        return max_ram_gb * 1e9
    if device == "gpu":
        total = MACHINE_RAM_GB[machine or "workstation"]["gpu"] * 1e9
        return total - min(RESERVE_GB * 1e9, RESERVE_FRAC * total)
    if machine is not None:
        total = avail = MACHINE_RAM_GB[machine]["cpu"] * 1e9
    else:
        vm = psutil.virtual_memory()
        total, avail = vm.total, vm.available
    return avail - min(RESERVE_GB * 1e9, RESERVE_FRAC * total)


def _solve(coeff, powers, budget, m_per_n):
    """Largest size satisfying ``coeff * N**n * M**m <= budget``.

    Returns the answer in whichever variable the caller's kind is indexed by;
    substituting N = M / m_per_n first means an (n, m) law in an M-scaled kind
    collapses to a single power of M.
    """
    n_pow, m_pow = powers
    # N = M / m_per_n, so N**n * M**m = M**(n+m) / m_per_n**n
    coeff = coeff / m_per_n**n_pow
    return (budget / coeff) ** (1.0 / (n_pow + m_pow))


def existing_floors(filename, curves, m_per_n=None):
    """Per-curve floor: the largest size that already has a real measurement.

    Used by ``--long-runs-only`` to define "the points a production sweep
    declined" without having to reconstruct which budget that sweep ran under.
    A curve with no finite point anywhere gets a floor of 0, so a long run also
    covers a curve that has never been measured at all.

    Args:
        filename: results/*.pkl aggregate. A missing file means no curve has a
            floor, which is the right answer -- everything is unmeasured.
        curves: curve names to report on, so the result lines up with the
            cutoffs dict whatever the file happens to contain.
        m_per_n: for the M-scaled kinds, whose cutoffs are in M while the stored
            ``Ns`` are in N. Pass the ratio to convert; ``None`` leaves the
            sizes as stored.

    Returns:
        dict of curve name -> floor, in the same variable as ``cutoffs``.
    """
    floors = {name: 0.0 for name in curves}
    if not os.path.exists(filename):
        return floors
    data = load_benchmark_data(filename)
    Ns = np.asarray(data.get("Ns", []), dtype=float)
    scale = 1.0 if m_per_n is None else float(m_per_n)
    for name in curves:
        points = data.get("runtime", {}).get(name)
        if not points or len(Ns) == 0:
            continue
        t = np.array([q[0] for q in points], dtype=float)
        n = min(len(t), len(Ns))
        good = np.isfinite(t[:n])
        if good.any():
            floors[name] = float(Ns[:n][good].max()) * scale
    return floors


def size_cutoffs(
    max_ram_bytes,
    kind,
    max_seconds=None,
    max_N=1e7,
    max_M=1e7,
    m_per_n=100,
    gpu=False,
    gpu_serial=False,
    integrated=False,
    safety=None,
    data_ceiling=None,
    value_and_grad=False,
    detail=False,
):
    """Per-curve size cutoffs from a memory budget and a time budget.

    Args:
        max_ram_bytes: memory available to the benchmark, in bytes (see
            ``ram_budget``).
        kind: ``llh`` | ``cond`` | ``pred`` | ``sample-prior`` | ``sample-post``.
        max_seconds: per-call wall-clock budget. ``None`` disables the time
            bound, leaving only memory (and the flat caps).
        max_N, max_M: flat caps, so a curve that is cheap in both time and
            memory still stops at the end of the intended grid.
        m_per_n: test/sample points per data point, for the M-scaled kinds.
        gpu: include the parallel-solver curves, and (by default) skip the
            serial ones -- see below.
        gpu_serial: also run SSM/QSM/GP on the GPU. Off by default because
            those curves come from the CPU sweep and a GPU result file's
            copies of them are never plotted. For one-off comparisons.
        integrated: use the --int coefficients (see _COST_INT). These differ
            by up to 24x, so this must match how the run is actually invoked.
        data_ceiling: largest size whose dataset can be built at all (see
            integrated_data_ceiling). Every curve is capped at it, because a
            size the data generator cannot reach is not measurable however much
            memory or time is on offer.
        safety: multiplier on the memory constants; defaults to _SAFETY. Pass
            1.0 alongside a hand-picked budget to take that budget literally
            -- the caller has then taken responsibility for the margin.
        value_and_grad: the run measures value *and* gradient, so scale the
            memory coefficients by _GRAD_MEM_FACTOR. Must match how the run is
            actually invoked: the factors range from 2.6x to 98x, and getting
            this wrong in the optimistic direction is an OOM kill.
        detail: also return, per curve, which bound was active.

    Returns:
        dict of curve name -> maximum size, in whichever variable that kind's
        cutoffs are expressed (N for llh/cond/sample-prior, M for the rest).
        With ``detail``, a second dict of curve name -> ``"memory"`` |
        ``"time"`` | ``"cap"``.

    Note:
        The time constants were calibrated on the workstation CPU. On another
        machine they are only indicative -- the ``_retired`` retirement in
        ``benchmark()`` is the backstop that catches the difference.
        pSSM/pQSM are GPU-only and have no calibration data at all, so they
        get the flat cap.
    """
    if kind in _M_SCALED:
        # Cutoffs are in M and the grid ties N to it, so an (n, m) law
        # collapses to a single power of M once N = M / m_per_n is substituted.
        flat, ratio = max_M, m_per_n
    elif kind == "sample-prior":
        # Also indexed by M, but a prior draw has no training set: the laws are
        # already pure powers of M, so there is nothing to substitute.
        flat, ratio = max_M, 1
    else:  # llh, cond -- indexed by N, laws are pure powers of N
        flat, ratio = max_N, 1

    table = _COST_INT if integrated else _COST
    safety = _SAFETY if safety is None else safety
    cuts, bounds = {}, {}
    for curve in ("SSM", "QSM", "GP"):
        cost = table.get((kind, curve))
        if cost is None:  # QSM has no integrated-data variant
            continue
        # The serial solvers are measured on the CPU; a GPU sweep only needs to
        # contribute the parallel curves. Running them on the card is not just
        # redundant, it is discarded -- make_benchmark_figure splices only
        # pSSM/pQSM out of a GPU result file (GPU_CURVES in plotting.py), so
        # nothing ever reads these. It is also slow: the sequential Kalman scan
        # does not parallelise, and SSM at N = 1e7 took 545 s per call on the
        # GPU against 15 s on the CPU, dominating the sweep for nothing.
        # Pass gpu_serial=True for a deliberate one-off comparison.
        if gpu and not gpu_serial:
            cuts[curve], bounds[curve] = 0, "skipped on GPU"
            continue
        (mem_coeff, mem_pow), (sec_coeff, sec_pow) = cost
        grad = (
            _GRAD_MEM_FACTOR.get(
                (kind, curve, integrated), GRAD_MEM_FACTOR_DEFAULT
            )
            if value_and_grad
            else 1.0
        )
        options = [
            (_solve(safety * grad * mem_coeff, mem_pow, max_ram_bytes, ratio), "memory"),
            (flat, "cap"),
        ]
        if data_ceiling is not None:
            options.append((data_ceiling, "data ceiling"))
        if max_seconds is not None and max_seconds != float("inf"):
            options.append((_solve(sec_coeff, sec_pow, max_seconds, ratio), "time"))
        cuts[curve], bounds[curve] = min(options)

    # Parallel solvers only run on the GPU box, and never for sampling.
    for curve in ("pSSM", "pQSM"):
        cuts[curve] = max_N if (gpu and not kind.startswith("sample")) else 0
        bounds[curve] = "cap"

    return (cuts, bounds) if detail else cuts
