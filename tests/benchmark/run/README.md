# Production benchmark sweeps

Scripts that benchmark runtime and memory usage for each use-case of `smolgp`. See the benchmarks page on the docs for the resulting plots. They wrap `run_benchmark.py`, which does the profiling.

| script | runs | device |
|---|---|---|
| `runcpu.sh` | All by default, or selectable with flags (e.g. `--llh --llh-vg --cond --pred --sample`) | CPU |
| `rungpu.sh` | Same, for those defined on GPU | GPU |
| `runlong.sh` | Extend curves to longer $N$ | both |

Note these run the instantaneous kernel tests. Pass `--int` to run the exposure-integrated variants.

## Quick start

```bash
cd tests/benchmark

./run/rungpu.sh --check       # verify the GPU profiler, run nothing
./run/runcpu.sh               # full non-integrated CPU sweeps
./run/runcpu.sh --int         # same for integrated kernels
./run/runcpu.sh --llh --cond  # just the likelihood and conditioning
```

**For a full re-run**: four commands:

```bash
./run/runcpu.sh && ./run/runcpu.sh --int
./run/rungpu.sh && ./run/rungpu.sh --int
```

A full suite takes hours, so detach it:

```bash
setsid nohup ./run/runcpu.sh > logs/cpu_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
```

Follow with `tail -f`; the `SUMMARY` at the end gives per-sweep exit codes. The
scripts do not `set -e`, so one failure does not discard the rest.

To stop a detached run, kill its **session** — `pkill -f` misses
`multiprocessing` spawn children and orphans them:

```bash
pkill -s $(ps -o sid= -p $(pgrep -f runcpu.sh | head -1) | tr -d ' ')
```

## Flags

| flag | default | effect |
|---|---|---|
| *(positional)* | — | `llh`, `cond`, `pred`, `sample-prior`, `sample-post` |
| `--llh` `--llh-vg` `--cond` `--pred` `--sample` | all | script selectors; `--sample` expands to both sampling kinds |
| `KINDS="..."` | all | env-var form of the selectors |
| `--int` | off | exposure-integrated kernels (no QSM curve exists) |
| `--gpu` | off | run on the GPU |
| `--gpu-serial` | off | *also* run SSM/QSM/GP on the GPU (normally skipped) |
| `--value-and-grad` | off | value **and** gradient, as a hyperparameter fit pays. `llh` only |
| `--max-ram GB` | detected | memory budget, taken literally (no reserve, no safety factor) |
| `--machine {workstation,macbook}` | detected | preset budget instead of measuring this box |
| `--max-seconds S` | 600 | per-call budget: sets the cutoffs and retires a curve that exceeds them |
| `--nrepeat n` | adaptive | repeats per point; see `NREPEAT_SCHEDULE` |
| `--sizes N[,N…]` / `--indices i[,i…]` | all | restrict to grid sizes / 1-based positions. **Partial run** |
| `--curves C[,C…]` | all | restrict to `SSM`, `QSM`, `GP`, `pSSM`, `pQSM`. **Partial run** |
| `--long-runs-only` | off | only points above what each curve already has. **Partial run** |
| `--absolute-only` | off | re-measure only the absolute peak; merges into the aggregate |
| `--xla-only` | off | compile only, record XLA's buffer accounting; merges. No timed run |
| `--rebuild` | off | reassemble the aggregate from `results/individual/` |
| `--quick` | off | 9 sizes, smaller maximum, 5 s budget; writes `*_quick_*.pkl` |
| `--max-n N` / `--make-data` | — | cap the grid / build datasets and stop |
| `--plot` / `--plot-only` | off | draw the figure after running / instead of running |
| `--no-tex` | off | mathtext instead of LaTeX |

**Partial run** = the aggregate is not written. Points go to
`results/individual/`; `--rebuild` folds them in. This is what makes it safe to
re-measure a few points without overwriting a full sweep.

## What gets recorded

Each memory entry is `(mean, std, absolute, xla)`:

| field | meaning |
|---|---|
| `mean`, `std` | peak above the post-warm-up baseline |
| `absolute` | whole-process peak — what must be free to run it. Plotted solid |
| `xla` | XLA buffer accounting: the computation alone. Plotted faded |

CPU reads `VmHWM`; GPU reads `peak_bytes_in_use`. Both are exact — sampled RSS
misses peaks above ~30 GB and is no longer used.

## Settings

`_common.sh`: `CPU_BUDGET` (`--max-ram 485 --max-seconds 600`), `GPU_BUDGET`
(`--max-seconds 600`), and a raised 1200 s budget for `llh-vg`.

`benchmark.py`:

| name | value | controls |
|---|---|---|
| `NREPEAT_SCHEDULE` | 7 / 5 / 3 / 1 | repeats, by per-call time (<1 s, <10 s, <60 s, else) |
| `PROGRESS_ABOVE_SECONDS` | 60 | per-repeat progress lines above this |
| `_SAFETY` | 1.3 | margin on the memory constants |
| `RESERVE_GB` / `RESERVE_FRAC` | 16 GB / 10% | held back from the detected budget |
| `_COST` / `_COST_INT` | per (kind, curve) | memory and time laws behind the cutoffs |
| `_GRAD_MEM_FACTOR` | 2.5–98× | reverse-mode memory multipliers |

`run_benchmark.py`: `DEFAULT_MAX_SECONDS` 600, `LONG_RUN_MAX_SECONDS` 1800.
`plotting.py`: `MEM_FLOOR_BYTES` 2e6 — below this the solid curve stops, since
the measurement is not meaningful there; the faded curve continues.

Re-derive all the constants after a full sweep with
`uv run calibrate_costs.py`; it prints source-vs-measured and flags drift.

## Before you run

- **The box must be idle.** `CPU_BUDGET` is ~99% of available RAM and swap is
  off. Check `free -g` and `uptime`.
- **Do not edit `benchmark.py`, `plotting.py`, `funcs.py`, `gp.py` or
  `helpers.py` mid-sweep** — every point re-imports them in a fresh subprocess.
- **Do not `uv sync` mid-sweep.** `rungpu.sh` leaves the venv on CUDA.
- **GPU: `./run/rungpu.sh --check` first.**
- **LaTeX** comes from the `texlive` module, loaded by `_common.sh`. A plot step
  outside `run()` will not have it; use `--no-tex` or load it yourself.

## Recipes

```bash
# One point, then fold it in
uv run run_benchmark.py cond --indices 11 --max-seconds 2000 --max-ram 485
uv run run_benchmark.py cond --rebuild --plot

# One implementation, leaving GP/QSM (tinygp) alone
uv run run_benchmark.py llh --curves SSM --max-seconds 600 --max-ram 485
uv run run_benchmark.py llh --rebuild --plot

# Preview cutoffs without running anything
uv run run_benchmark.py cond --plot-only
uv run run_benchmark.py llh --machine macbook --plot-only

# A point the default budget excludes
uv run run_benchmark.py pred --max-ram 490 --max-seconds 600 --plot
```

### Wiping before a clean regeneration

```bash
rm -f results/{cpu,gpu}_{llh,cond,pred}{,_int}_benchmark.pkl
rm -f results/individual/{gp,ss,qs,pss,pqs,igp,iss,ipss}_{llh,cond,pred}_*.pkl
```

Leave `data/*.npz` alone. Wiping a `gpu_*` aggregate commits you to re-running
the GPU half.

## Recovery

`results/individual/<func>_<N>_<device>.pkl` holds one point each; `--rebuild`
reassembles the aggregate from them. Only successful measurements are
checkpointed, so a crash never overwrites a good point.

## Runtimes

Regenerate with `uv run estimate_runtime.py --write`.

A point costs `n x (2t + c)`: each repeat is a fresh subprocess that makes an
untimed warm-up call before the timed one, so the compute is paid twice. `c` is
fork + `import jax` + compile — 2.4 s CPU, 13.3 s GPU, 6.2 / 22.6 s with
`--int` (a fresh CUDA context, and larger arrays to unpickle). The low–high
range is ±25% on `c` alone, so a wide range means the estimate is mostly model.

<!-- RUNTIME TABLE START -->
| sweep | device | pts | runtime | basis |
|---|---|---|---|---|
| cond | 🟪 GPU | 34 | 🛑 75 -- 98 m | measured |
| llh | 🟪 GPU | 34 | ⚠️ 40 -- 66 m | measured |
| llh-vg | 🟪 GPU | 33 | ⚠️ 40 -- 64 m | measured |
| llh-vg | 🟦 CPU | 45 | ⚠️ 51 -- 56 m | measured |
| pred `--int` | 🟦 CPU | 30 | ⚠️ 46 -- 54 m | measured |
| cond `--int` | 🟪 GPU | 16 | ⚠️ 33 -- 52 m | measured |
| llh `--int` | 🟪 GPU | 16 | ⚠️ 32 -- 52 m | measured |
| llh-vg `--int` | 🟦 CPU | 26 | ⚠️ 42 -- 50 m | measured |
| sample-post `--int` | 🟦 CPU | 25 | ⚠️ 42 -- 48 m | measured |
| llh-vg `--int` | 🟪 GPU | 15 | ⚠️ 29 -- 48 m | measured |
| cond `--int` | 🟦 CPU | 28 | ⚠️ 38 -- 46 m | measured |
| sample-post | 🟦 CPU | 33 | ⚠️ 36 -- 40 m | measured |
| sample-prior `--int` | 🟦 CPU | 28 | ⚠️ 28 -- 36 m | measured |
| pred | 🟦 CPU | 42 | ⚠️ 28 -- 33 m | measured |
| llh `--int` | 🟦 CPU | 28 | ⚠️ 24 -- 33 m | measured |
| cond | 🟦 CPU | 45 | ✅ 23 -- 29 m | measured |
| sample-prior | 🟦 CPU | 45 | ✅ 22 -- 28 m | measured |
| llh | 🟦 CPU | 45 | ✅ 15 -- 22 m | measured |
| pred | 🟪 GPU | — | ❌ — | never run |
| pred `--int` | 🟪 GPU | — | ❌ — | never run |
| **TOTAL (18 of 20 sweeps)** | | | **644 -- 857 m ≈ 10.7 -- 14.3 h** | |
<!-- RUNTIME TABLE END -->

The two GPU `pred` rows can never run: there is no parallel-solver prediction
implementation, so those sweeps have no curves.

### Tier 2 (`--long-runs-only`)

Regenerate with `uv run estimate_runtime.py --long-runs --write`. Projections,
not measurements — each row takes the larger of the curve's local slope and the
cost table's law.

<!-- LONG RUN TABLE START -->
| sweep | device | curve | N to add | local slope | slowest call | est. runtime |
|---|---|---|---|---|---|---|
| llh `--int` | 🟦 CPU | GP | 133,352 | N^1.95 | 25 min | ⚠️ 51 min |
| sample-prior | 🟦 CPU | GP | 133,352 | N^2.75 | 13 min | ✅ 26 min |
| llh | 🟦 CPU | GP | 133,352 | N^2.73 | 11 min | ✅ 21 min |
| **TOTAL (3 curve-bands)** | | | | | | **1.6 h** |
<!-- LONG RUN TABLE END -->

Tier 2 covers only points whose *individual call* takes 10 min or more
(`LONG_RUN_MIN_SECONDS`). Anything faster belongs to the production suite,
which already runs a 600 s per-call budget — a cheap point missing from it is
missing because of its memory cutoff or because it crashed, and neither is
fixed by running it here.

These remaining bands are dense GP, which hits a SIGSEGV ceiling at ~1e10
matrix elements, so expect them to fail until that is understood. A failed
point is stored as NaN, indistinguishable from one never attempted, so it will
be re-proposed each time; check against the logs before spending a night on
it.
