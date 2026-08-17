# Production benchmark sweeps

Scripts that generate the figures on the docs' benchmarks page. They wrap
`run_benchmark.py`, which does the actual profiling; everything here is about
running it with the right budgets, on the right device, without stepping on a
sweep already in flight.

| script | what it runs | device |
|---|---|---|
| `runcpu.sh` | all four sweeps by default; select with `--llh --cond --pred --sample` | CPU |
| `rungpu.sh` | the same, minus `--sample` (nothing to measure there) | GPU |

There is no separate sampling script: `--sample` is shorthand for the two
sampling sweeps, `sample-prior` and `sample-post`.

Anything that is not a kind selector passes straight through to
`run_benchmark.py`, so `--int`, `--no-tex`, `--max-seconds 60`, `--quick`,
`--sizes` and friends work on either script.

## Quick start

```bash
cd tests/benchmark

./run/rungpu.sh --check          # verify the GPU profiler, run nothing
./run/runcpu.sh                  # everything on CPU: llh, cond, pred, sample
./run/runcpu.sh --int            # the same, integrated data
./run/runcpu.sh --sample         # just the two sampling sweeps
./run/runcpu.sh --llh --cond     # just likelihood and conditioning
./run/rungpu.sh                  # llh, cond, pred on GPU (sample skipped)
```

A full sweep takes hours. Detach it so it survives losing the terminal:

```bash
setsid nohup ./run/runcpu.sh > logs/cpu_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
```

`setsid` reparents to PID 1, so an SSH or VS Code disconnect will not take it
down. Follow with `tail -f logs/<file>`, and check the `SUMMARY` block at the
end for per-sweep exit codes — the scripts deliberately do **not** `set -e`, so
one failed sweep does not discard the others.

## Flags

Everything below is a `run_benchmark.py` flag and therefore usable on either
script, except the kind selectors and `KINDS`, which the scripts read
themselves rather than passing through.

| flag | default | what it does |
|---|---|---|
| *(positional)* | — | `llh`, `cond`, `pred`, `sample-prior`, `sample-post` |
| `--llh` `--cond` `--pred` `--sample` | all four | **script selectors**, handled by `runcpu.sh`/`rungpu.sh` rather than passed through. Combine freely; `--sample` expands to `sample-prior sample-post`, and is skipped by `rungpu.sh` since sampling has no parallel-solver implementation |
| `KINDS="..."` | all four | env-var equivalent of the selectors, for scripting: `KINDS="llh sample" ./run/runcpu.sh`. Explicit flags win over it |
| `--int` | off | exposure-integrated kernels. No QSM curve exists for these |
| `--gpu` | off | run on the GPU. `rungpu.sh` sets this for you |
| `--gpu-serial` | off | **also** run SSM/QSM/GP on the GPU. Off because those curves come from the CPU sweep and a GPU file's copies are never plotted — see below |
| `--max-ram GB` | detected | memory budget. Taken *literally*: no 16 GB reserve, no safety factor on the cost constants |
| `--machine {workstation,macbook}` | detected | use a preset budget instead of measuring this box, e.g. to preview another machine's cutoffs |
| `--max-seconds S` | **600** (5 under `--quick`) | per-call wall-clock budget. Sets the size cutoffs up front *and* retires a curve that blows it. `inf` disables both |
| `--sizes N[,N…]` | all | re-run only these grid sizes, snapped to the nearest grid point. **Partial run — does not write the aggregate**, see below |
| `--indices i[,i…]` | all | same, but by 1-based grid position, so a point can be named straight off the `(11/17)` in the log |
| `--rebuild` | off | skip profiling; reassemble the aggregate from the per-point checkpoints in `results/individual/`. Folds in `--sizes`/`--indices` runs, and recovers a lost aggregate |
| `--quick` | off | 9 sizes instead of 17, smaller maximum, 5 s budget. Writes to `*_quick_benchmark.pkl`, so it cannot clobber production data |
| `--plot` | off | write the figure after running |
| `--plot-only` | off | skip profiling entirely, just redraw from existing `results/*.pkl` |
| `--no-tex` | off | mathtext instead of LaTeX, for machines without a TeX install |

### One-off recipes

Filling a single missing point, or extending one curve, does not need a whole
sweep. Because every measured point is checkpointed to
`results/individual/`, a targeted run can be merged back later.

**Re-running a single point.** `--sizes` / `--indices` narrow the grid, and
`--rebuild` merges the result back:

```bash
# The point the sweep logs as (11/17) -- equivalently N = 56234
uv run run_benchmark.py cond --indices 11 --max-seconds 2000 --max-ram 485
uv run run_benchmark.py cond --rebuild --plot     # fold it in, redraw

# Or name the size directly; it snaps to the nearest grid point
uv run run_benchmark.py llh --sizes 56234,133352 --max-seconds 2000
```

A partial run **deliberately leaves the aggregate alone** — writing two points
over a full sweep is exactly the accident this avoids. The measurements go to
`results/individual/` as usual, and `--rebuild` reassembles the whole file from
there. That is also how to recover an aggregate that was lost or clobbered.

```bash
# One kind only, on either device
./run/runcpu.sh --cond
./run/rungpu.sh --pred

# Push one curve further out: raise the time budget so the cutoff moves.
# cond GP stops at N=6.7e4 with the 600 s default; 2000 s reaches the next
# grid point. Check what you are asking for first -- --plot-only prints the
# cutoffs and which bound is active without running anything.
uv run run_benchmark.py cond --max-seconds 2000 --max-ram 485 --plot

# Recover a point the default budget excludes. pred QSM at M=1e6 needs
# 480 GB, just over the auto-detected budget, so it needs an explicit one:
uv run run_benchmark.py pred --max-ram 490 --max-seconds 600 --plot

# Compare serial-vs-parallel on the GPU (normally skipped, see the flag table)
uv run run_benchmark.py llh --gpu --gpu-serial --max-seconds 600

# See the cutoffs for another machine without running anything
uv run run_benchmark.py llh --machine macbook --plot-only
```

### Figure options

These are keyword arguments to `make_benchmark_figure` in `plotting.py`, not
CLI flags, so they need a couple of lines in a REPL or notebook rather than a
sweep:

| argument | default | effect |
|---|---|---|
| `substitute_theory` | `True` | replace memory points below the measurement floor with the theoretical footprint, drawn as **hollow** markers |
| `show_theory` | `False` | additionally overlay each derived theoretical curve as a thin line across the whole range |
| `annotate` | `True` | draw the auto-placed scaling labels (`N³`, `N+M`, …) |
| `gpu_data` | `None` | second result dict whose pSSM/pQSM curves get spliced in |

## Before you run

- **The box must be idle.** `CPU_BUDGET` is `--max-ram 485`, roughly 99% of the
  ~488 GB this machine has available, because the largest QSM prediction point
  (M = 1e6) genuinely needs 480 GB. Swap is off, so overshooting is a hard kill,
  not a slowdown. Check with `free -g` and `pgrep -af run_benchmark` first.
- **Do not edit `benchmark.py`, `plotting.py`, `funcs.py`, `gp.py` or
  `helpers.py` while a sweep is running.** Profiling uses `mp` spawn, so every
  measured point re-imports those modules in a fresh subprocess; editing one
  mid-run has already killed a sweep with an `ImportError`.
- **Do not `uv sync` while a sweep is running** — same reason, it swaps the
  jaxlib underneath the subprocesses. Note `rungpu.sh` leaves the venv on CUDA;
  `runcpu.sh` calls `uv sync --dev` itself, so running it afterwards is fine.
- **GPU: always `./run/rungpu.sh --check` first.** It confirms jax is actually
  on the GPU (not silently fallen back to CPU) and then drives the real
  `GPUMemorySampler` against a known 64 MB workload, *after a warm-up call*.
  The warm-up matters: an earlier version of this check read the counter in a
  fresh process, where it starts at zero, and so passed a sampler that reported
  0 B for every point in the actual sweep.

### Wiping before a clean regeneration

Figures mix versions if some curves come from an old run and some from a new
one. To regenerate `llh`/`cond`/`pred` from scratch:

```bash
rm -f results/{cpu,gpu}_{llh,cond,pred}{,_int}_benchmark.pkl
rm -f results/individual/{gp,ss,qs,pss,pqs,igp,iss,ipss}_{llh,cond,pred}_*.pkl
```

Leave `data/*.npz` alone — those are generated inputs, and regenerating
N = 1e7 is slow. Wiping the `gpu_*` aggregates commits you to running the GPU
half too, or those figures lose their pSSM/pQSM curves.

## Recovery

`results/individual/<func>_<N>_<device>.pkl` holds one measured point each, and
the aggregate can be rebuilt from them if it is lost or clobbered — this has
been needed twice. Only successful measurements are checkpointed, so a crashed
or skipped point never overwrites a good one, and `--quick` writes to
`*_quick_benchmark.pkl` so it cannot clobber production data.
