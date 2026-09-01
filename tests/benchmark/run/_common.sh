# Shared setup for the production sweep scripts. Sourced, not executed.
#
# Deliberately NOT `set -e`. These run unattended for hours and the sweeps are
# independent: a failure in one must not throw away the rest. Each records its
# exit status and `summary` prints them all at the end.
set -uo pipefail

# Scripts live in run/, but run_benchmark.py resolves results/, data/ and
# figures relative to the working directory, so everything happens one level up.
BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BENCH_DIR" || exit 1
mkdir -p logs

# LaTeX labels: matplotlib's usetex shells out to latex/dvipng rather than
# linking against them, so they only need to be on PATH -- no separate venv is
# involved. Pass --no-tex to fall back to mathtext.
module load texlive/20240312 2>/dev/null || \
    echo "note: texlive module not loaded; add --no-tex if LaTeX rendering fails"

# 485 GB is deliberately close to this box's ~488 GB of available RAM: it is
# what the largest QSM prediction point (M = 1e6) needs, 480 GB, leaving ~8 GB
# of slack with swap off. Run on an otherwise idle machine. Taken literally --
# no reserve and no safety factor on the cost constants (see size_cutoffs).
CPU_BUDGET="--max-ram 485 --max-seconds 600"
# Never pass --max-ram to a GPU run: the card has 48 GB, and 485 would size
# every cutoff for ten times the memory that exists. The default budget is
# derived from the device instead.
GPU_BUDGET="--max-seconds 600"

declare -a _NAMES=()
declare -a _CODES=()

#: Everything a sweep can measure. "sample" is a shorthand -- there is no
#: single `sample` benchmark, it expands to the prior and posterior draws.
#: "llh-vg" is the likelihood measured the way a hyperparameter fit calls it,
#: value AND gradient. It is a separate kind rather than a variant of "llh"
#: because reverse mode changes the cost profile enough to need its own
#: results, figure and memory budget -- and because it is easy to forget, which
#: is precisely why it is in the default list.
ALL_KINDS="llh llh-vg cond pred sample"

parse_kinds() {
    # Splits arguments into kind selectors and everything else. Selectors are
    # --llh/--llh-vg/--cond/--pred/--sample; anything else (--int, --no-tex, --quick,
    # --max-seconds N ...) is passed straight through to run_benchmark.py.
    #
    # Precedence: explicit flags > KINDS env var > all of them.
    local kinds=() rest=()
    for a in "$@"; do
        case "$a" in
            --llh|--llh-vg|--cond|--pred|--sample) kinds+=("${a#--}") ;;
            *) rest+=("$a") ;;
        esac
    done
    if [ ${#kinds[@]} -gt 0 ]; then
        KINDS="${kinds[*]}"
    else
        KINDS="${KINDS:-$ALL_KINDS}"
    fi
    EXTRA=()
    [ ${#rest[@]} -gt 0 ] && EXTRA=("${rest[@]}")
    TAG=""
    for a in ${EXTRA[@]+"${EXTRA[@]}"}; do [ "$a" = "--int" ] && TAG=" --int"; done
}

expand_kinds() {
    # `sample` is two sweeps, prior and posterior.
    local out=()
    for k in $1; do
        if [ "$k" = "sample" ]; then
            out+=("sample-prior" "sample-post")
        else
            out+=("$k")
        fi
    done
    echo "${out[*]}"
}

kind_args() {
    # The run_benchmark.py positional (and any kind-specific flag) for a kind.
    # Deliberately unquoted at the call site so `llh-vg` expands to two words.
    case "$1" in
        llh-vg) echo "llh --value-and-grad" ;;
        *)      echo "$1" ;;
    esac
}

kind_cpu_budget() {
    # Per-kind CPU budget. Every kind now uses CPU_BUDGET, including the
    # gradient sweep.
    #
    # This used to override llh-vg to --max-ram 170. _COST is calibrated on the
    # forward pass while reverse mode keeps the forward pass's intermediates, so
    # at 485 GB the model allowed far larger sizes than really fit and the small
    # budget was standing in for the missing factor. size_cutoffs now takes a
    # value_and_grad argument and scales the memory coefficients by
    # _GRAD_MEM_FACTOR -- measured per (kind, curve, integrated), 2.6x to 98x --
    # so the model derives the right cutoff itself. Verified at --max-ram 485:
    # GP stops at grid point 5.6e4 instantaneous and 1.0e4 integrated, which are
    # exactly the largest sizes that measured successfully, and SSM/QSM stay
    # cap-bound at 1e7.
    #
    # llh-vg keeps a raised *time* budget. Its dense GP point at N=56234 takes
    # 949 s, and it is a wanted point -- it is the last GP marker in the
    # deployed figure. Under the 600 s default the sweep now projects it as
    # over-budget and skips it, which is cheaper than the old measure-then-
    # discard but loses it just the same. 1200 s admits it while still bounding
    # the sweep; the next grid point up is refused on memory regardless.
    case "$1" in
        llh-vg) echo "--max-ram 485 --max-seconds 1200" ;;
        *)      echo "$CPU_BUDGET" ;;
    esac
}

# XLA logs this at ERROR level on every process start, and it is pure noise:
# it parses the NVIDIA kernel-mode driver version expecting X.Y.Z, but this
# machine reports two components (580.142), so the parse fails. Nothing is
# broken -- jax initialises fine either way -- but with hundreds of profiling
# subprocesses per sweep it drowns the log: 1096 of 1253 lines, 87%, in one
# GPU run. Dropped by exact match, so genuine errors still come through.
XLA_NOISE='kernel mode driver version'

run() {
    local label="$1"; shift
    echo
    echo "=============================================================="
    echo ">>> $label   [started $(date '+%F %T')]"
    echo "=============================================================="
    # Combined stream through one line-buffered filter: keeps stdout/stderr in
    # order, and the line buffering also stops the log going stale for hours
    # behind a block buffer, which made a healthy run look hung.
    uv run run_benchmark.py "$@" 2>&1 | grep --line-buffered -vE "$XLA_NOISE"
    local code=${PIPESTATUS[0]}
    _NAMES+=("$label"); _CODES+=("$code")
    echo ">>> $label finished with exit code $code at $(date '+%F %T')"
}

summary() {
    echo
    echo "=============================== SUMMARY ==============================="
    local bad=0
    for i in "${!_NAMES[@]}"; do
        printf '  %-26s exit %s\n' "${_NAMES[$i]}" "${_CODES[$i]}"
        [ "${_CODES[$i]}" -ne 0 ] && bad=1
    done
    echo "ALLDONE $(date '+%F %T')"
    return $bad
}
