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

## Known issues / TODO

- [ ] **Investigate the GP `sample-post` segfault at M = 1e5.** Reproducible:
  it happened in both the instantaneous and integrated sweeps, same size, exit
  139 (SIGSEGV) inside XLA. It is *not* memory (110–240 GB needed against
  ~450 GB free) and not an obvious 32-bit index overflow, since M = 56234
  already exceeds 2^31 matrix elements and works fine. The profiler now records
  the point as NaN and carries on, so the sweep completes — but that curve
  loses its largest point until someone bisects this.
- [ ] **Skip serial SSM on GPU by default.** Only the parallel solver is worth
  measuring there: the sequential Kalman scan does not parallelise, so running
  it on the card is strictly slower than the CPU and it dominates the sweep's
  wall clock for nothing. Measured on `llh` at N = 4.2e6: **96.9 s on GPU
  against 6.15 s on CPU**, a 16x slowdown, and SSM at N = 1e7 was the long pole
  of the whole GPU run. Give SSM a cutoff of 0 when `gpu=True` in
  `size_cutoffs`, mirroring how pSSM/pQSM are already zeroed on CPU, with a
  flag to opt back in if the comparison is ever wanted for a figure.
- [x] ~~**Measure the provisional `_COST_INT` entries.**~~ Done:
  `("sample-post", "SSM")` measures **1936–2108 B/M** against the extrapolated
  10900, i.e. the guess was 5.6x too conservative. Written into `benchmark.py`;
  no PROVISIONAL entries remain.
- [ ] **Regenerate the remaining GPU memory curves.** `llh` is done and now
  reads correctly (GP lands on 16.0 B/N^2, matching CPU and theory exactly).
  `cond` and `pred` still hold pre-fix data, where the old sampler never
  subtracted a baseline and summed every process on the card — which is why
  those curves are flat ~5 GB lines independent of N.
- [x] ~~**Settle the SSM memory law.**~~ Done: it is **linear** in the state
  dimension, `8*(d+9)` B/N, validated at d = 2/4/6/8 (slope 0.999 doubles per
  unit d). The `2d^2`-style guesses were wrong and only looked right because
  they coincide at d=2. See README.md.
- [ ] **Account for the 9 d-independent arrays** in that law. Three are
  presumably t/y/yerr; the rest would need a read through the state-space
  solver to name. Would turn the empirical law into a full derivation.
- [ ] **Fix a stale help string.** `--max-seconds` advertises `default: 300`
  in `run_benchmark.py`, but `DEFAULT_MAX_SECONDS` is 600. The behaviour is
  correct; only the help text is wrong. (Could not be fixed at the time it was
  spotted: a sweep was running, and every profiling subprocess re-imports that
  module.)
- [ ] **Calibrate pSSM/pQSM.** They are GPU-only and have no entries in
  `_COST`, so they fall back to the flat cap rather than a real memory/time
  bound.
- [ ] **Time constants are workstation-calibrated.** On another machine the
  runtime cutoffs are indicative only; the `_too_slow` retirement inside
  `benchmark()` is the backstop that catches the difference.
- [ ] **Result files are dominated by `outputs`.** `cpu_cond_benchmark.pkl` is
  529 MB because it stores every function's full return array; the timings and
  memory numbers are kilobytes. Dropping or subsampling `outputs` would cut the
  directory by ~99%.
- [ ] **`assign_instids` relabelled the probe groups.** Any valid colouring is
  statistically equivalent, but a given PRNG key now yields a different draw, so
  `docs/tutorials/sample_anim/` will regenerate differently.

### Deliberate, not bugs

- Memory points below the measurement floor are replaced by the theoretical
  footprint and drawn with **hollow** markers. They are computed, not observed,
  and the marker style is what says so. Pass `substitute_theory=False` to see
  the raw measurements instead, or `show_theory=True` to overlay the full
  theoretical curve.
- A legend entry with no curve (e.g. pQSM) means that figure is *expected* to
  have that curve and it has not been run, or it errored. The gap is the signal.
