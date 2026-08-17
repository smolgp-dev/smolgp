#!/usr/bin/env bash
# CPU sweeps. Everything by default; pick individual ones with flags.
#
#   ./run/runcpu.sh                     all: llh, cond, pred, sample
#   ./run/runcpu.sh --llh               just the likelihood
#   ./run/runcpu.sh --cond --pred       two of them
#   ./run/runcpu.sh --sample            prior and posterior draws
#   ./run/runcpu.sh --llh --int         integrated data
#
# `--sample` is shorthand for the two sampling sweeps, sample-prior and
# sample-post. Anything that is not a kind selector is passed straight through
# to run_benchmark.py, so --int, --quick, --max-seconds, --sizes and friends all
# work here too.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

parse_kinds "$@"

uv sync --dev || exit 1

# `uv sync --dev` does not necessarily remove an already-installed CUDA plugin,
# and if jax finds one it will try to initialise CUDA in every one of the
# hundreds of profiling subprocesses -- printing driver errors, wasting seconds
# of startup each, and reserving a few hundred MB on the card for a run that is
# not using it. Pin the platform instead of relying on what happens to be
# installed.
export JAX_PLATFORMS=cpu

for kind in $(expand_kinds "$KINDS"); do
    run "$kind$TAG (CPU)" "$kind" ${EXTRA[@]+"${EXTRA[@]}"} $CPU_BUDGET --plot
done

summary
