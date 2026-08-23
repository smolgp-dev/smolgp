#!/usr/bin/env bash
# GPU sweeps. Everything applicable by default; pick individual ones with flags.
#
#   ./run/rungpu.sh                     all applicable: llh, llh-vg, cond, pred
#   ./run/rungpu.sh --llh               just the likelihood
#   ./run/rungpu.sh --cond --int        integrated conditioning
#   ./run/rungpu.sh --check             verify the profiler, run nothing
#
# Only the parallel solvers are measured here (pSSM/pQSM). The serial ones come
# from the CPU sweep, and a GPU file's copies of them are never plotted -- see
# --gpu-serial in run_benchmark.py if you want them anyway for a comparison.
#
# `--sample` is accepted but skipped: sampling is not implemented on the
# parallel solvers, so there is nothing for a GPU run to measure.
#
# This switches the venv to the CUDA jaxlib and leaves it there, so a CPU sweep
# afterwards needs `uv sync --dev` first (runcpu.sh does that itself).
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

parse_kinds "$@"

uv sync --group cuda || exit 1

# Pre-flight. This exercises the real GPUMemorySampler against a known
# allocation, rather than poking the counter by hand.
#
# The distinction matters: an earlier version of this check read
# peak_bytes_in_use in a *fresh* process, where the mark starts at zero, so it
# passed a sampler that reported 0 B for every point in the actual sweep. The
# profiler is never in that state -- it always measures after a warm-up call
# that has already driven the peak to its maximum. So the check now warms up
# first, exactly like tracer() does, which is the only way to catch a saturated
# or otherwise dead counter.
echo ">>> checking the GPU profiler"
uv run python - <<'PY' || { echo "GPU profiler check FAILED -- not starting the sweep"; exit 1; }
import sys
import jax, jax.numpy as jnp
sys.path.insert(0, ".")
from benchmark import GPUMemorySampler

dev = jax.devices()[0]
if dev.platform != "gpu":
    sys.exit(f"jax is on '{dev.platform}', not gpu: {jax.devices()}")

N = 4096
expect = N * N * 4          # one f32 4096x4096 buffer, 64 MiB

@jax.jit
def f(x):
    return (x @ x).sum() + x.sum()

x = jnp.ones((N, N), dtype=jnp.float32)
call = lambda: jax.block_until_ready(f(x))
call()                       # warm up, exactly as tracer() does

sampler = GPUMemorySampler(interval=1e-3)
_out, _t, mem = sampler.measure(call)
if mem <= 0:
    sys.exit(f"sampler reported {mem} B for a {expect/1e6:.0f} MB workload -- "
             "the counter is not tracking live allocations")
print(f"  OK: {dev}, sampler saw {mem/1e6:.1f} MB "
      f"(a single input buffer is {expect/1e6:.0f} MB)")
PY

for a in ${EXTRA[@]+"${EXTRA[@]}"}; do
    if [ "$a" = "--check" ]; then
        echo "check only; stopping here"
        exit 0
    fi
done

for kind in $(expand_kinds "$KINDS"); do
    case "$kind" in
        sample-*)
            echo
            echo ">>> skipping $kind on GPU: sampling is not implemented on the"
            echo "    parallel solvers, so no curve would be measured"
            continue
            ;;
    esac
    run "$kind$TAG (GPU)" $(kind_args "$kind") --gpu ${EXTRA[@]+"${EXTRA[@]}"} \
        $GPU_BUDGET --plot
done

summary
