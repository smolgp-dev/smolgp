#!/usr/bin/env bash
# Tier 2: fill in the points a production sweep declines.
#
#   ./run/runlong.sh            CPU then GPU
#   ./run/runlong.sh --cpu      CPU bands only
#   ./run/runlong.sh --gpu      GPU bands only
#
# Each sweep runs with --long-runs-only, which measures the band above what is
# already recorded -- see "Filling in the long points" in README.md. nrepeat
# defaults to 1 and the per-call budget to 1800 s. A sweep with nothing to add
# exits early saying so, which is not a failure.
#
# ~2.3 h projected; `uv run estimate_runtime.py --long-runs` prints the
# breakdown. Expect the dense curves to stop on memory rather than on the
# budget: that is the point, since it establishes where the wall actually is
# instead of where the model guesses it is.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

DO_CPU=1; DO_GPU=1
for a in "$@"; do
    case "$a" in
        --cpu) DO_GPU=0 ;;
        --gpu) DO_CPU=0 ;;
    esac
done

LONG="--long-runs-only"

if [ "$DO_CPU" = 1 ]; then
    uv sync --dev || exit 1
    export JAX_PLATFORMS=cpu

    for kind in llh cond pred sample-prior sample-post; do
        run "long $kind"           $kind $LONG
        run "long $kind int"       $kind --int $LONG
    done
    run "long llh-vg"              llh --value-and-grad $LONG
    run "long llh-vg int"          llh --int --value-and-grad $LONG

    for kind in llh cond pred sample-prior sample-post; do
        run "rebuild $kind"        $kind --rebuild --plot
        run "rebuild $kind int"    $kind --int --rebuild --plot
    done
    run "rebuild llh-vg"           llh --value-and-grad --rebuild --plot
    run "rebuild llh-vg int"       llh --int --value-and-grad --rebuild --plot
fi

if [ "$DO_GPU" = 1 ]; then
    # Rebuilt while the CUDA jaxlib is still installed: --gpu names a device,
    # so a gpu rebuild after switching back to the CPU wheel would be naming a
    # device that is no longer there.
    unset JAX_PLATFORMS
    uv sync --group cuda || exit 1

    for kind in llh cond pred; do
        run "long $kind gpu"       $kind --gpu $LONG
        run "long $kind int gpu"   $kind --int --gpu $LONG
    done
    run "long llh-vg gpu"          llh --value-and-grad --gpu $LONG
    run "long llh-vg int gpu"      llh --int --value-and-grad --gpu $LONG

    for kind in llh cond pred; do
        run "rebuild $kind gpu"    $kind --gpu --rebuild
        run "rebuild $kind int gpu" $kind --int --gpu --rebuild
    done
    run "rebuild llh-vg gpu"       llh --value-and-grad --gpu --rebuild
    run "rebuild llh-vg int gpu"   llh --int --value-and-grad --gpu --rebuild

    # Back on the CPU wheel for the final figures: plotting splices the
    # gpu_*.pkl files in from disk and needs no device of its own.
    uv sync --dev || exit 1
    export JAX_PLATFORMS=cpu
    for kind in llh cond pred; do
        run "plot $kind"           $kind --plot-only
        run "plot $kind int"       $kind --int --plot-only
    done
    run "plot llh-vg"              llh --value-and-grad --plot-only
    run "plot llh-vg int"          llh --int --value-and-grad --plot-only
fi

summary
