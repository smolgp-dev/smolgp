# Benchmark suite

Runtime and peak-memory profiling for three GP implementations
1. `smolgp`'s state-space solver (**SSM**)
2. `tinygp`'s quasiseparable solver (**QSM**), 
3. and a dense `tinygp` reference (**GP**) — across problem size.

See [`run/`](run/README.md) for running the benchmarking scripts.

This file documents the memory profiling, including meta-benchmarking test results and measured/derived scaling constants. See `_COST` / `_COST_INT` in `benchmark.py`, which set the the size cutoffs that keep a sweep from exhausting the machine.

## What is measured

Each point is profiled in a fresh subprocess, so nothing carries between sizes. Three memory figures are recorded per point:

| field | how | meaning |
|---|---|---|
| `absolute` | `VmHWM` (CPU) / `peak_bytes_in_use` (GPU) | whole-process peak — what must be free to run it |
| `xla` | `memory_analysis()` | the computation alone: scratch + output + inputs |

The figures plot `absolute` solid and `xla` faded. The former includes a (fixed) overhead, about ~330 MB (interpreter + JAX) on CPU and about ~500 MB (CUDA context) on GPU.


## Scalings

Everything is float64. `d` is the state-space dimension and `J` the
quasiseparable rank; both are 2 for the SHO kernel used throughout. Measured values are the converged asymptote, which several curves do not reach until N ≳ 10³.

| func | | scaling | what is held | derived | measured |
|---|---|---|---|---|---|
| **`llh`** (x=N) | GP | N² | kernel matrix + Cholesky | 2·8 = **16** | 16 |
| | QSM | N | `SymmQSM` + `LowerTriQSM` + data + workspace | — | 209 |
| | SSM | N | one d-vector per point + fixed N-arrays | — | 160 |
| **`cond`** (x=N) | GP | N² | + posterior covariance | 3·8 = **24** | 24 |
| | QSM | N | | — | 633 |
| | SSM | N | | — | 320 |
| **`pred`** (x=M=100N) | GP | N·M | | — | 24 |
| | QSM | N·M | | — | 64 |
| | SSM | M | | — | 489 |
| **`sample-prior`** (x=M) | GP | M² | M×M kernel + Cholesky | 2·8 = **16** | 16 |
| | QSM | M | | — | 585 |
| | SSM | M | | — | 732 |
| **`sample-post`** (x=M=100N) | GP | **M²** | M×M posterior covariance + Cholesky + draw | 3·8 = **24** | 24 |
| | QSM | **M²** | | — | 49 |
| | SSM | M | | — | 1045 |

Integrated variants (`--int`) use different kernels, so their constants live in `_COST_INT`. **There is no QSM curve.** `tinygp` has no integrated quasiseparable kernel, so the dense O(N³) path is the only comparison.

| func | | derived | measured |
|---|---|---|---|
| `llh` | GP / SSM | 16 B/N² | 16 / 584 |
| `cond` | GP / SSM | 72 B/N² | 72 / 1344 |
| `pred` | GP / SSM | 24 B/N·M | 24 / 1631 |
| `sample-prior` | GP / SSM | 16 B/M² | 16 / 1229 |
| `sample-post` | GP / SSM | 24 B/M² | 24 / 1988 |

Reverse mode (`--value-and-grad`) needs more, so `size_cutoffs` scales the memory coefficients by `_GRAD_MEM_FACTOR`: 2.5× (GP), 2.6× (QSM), 3.5× (SSM), 4.6× (SSM `--int`), and **98×** for integrated GP, whose dense kernel is built through an expression the tape retains every N×N intermediate of.

## Maintaining the constants

`uv run calibrate_costs.py` re-derives all three families from the current result files and flags drift. **Run it after any full sweep.** The derived GP entries never move; the measured O(N) ones do — a solver change alters the footprint, and a stale constant is not cosmetic:

- too low admits a size that then OOMs (`pred` QSM was 48 against a real 64,
  which let through a point that tried to allocate 560 GB);
- too high silently costs grid points (`pred --int` SSM was 10860 against a
  real 1631).

## Caveats

- **`MEM_FLOOR_BYTES = 2e6`** in `plotting.py`: below this the solid curve
  stops, because a sub-MB reading taken against a ~330 MB process is noise. The
  faded `xla` curve continues down to a few kB.
- **Dense factorisations fail above ~10¹⁰ matrix elements** with a SIGSEGV,
  while a matmul of the same size succeeds. `_COST`'s memory law does not
  predict this — memory is not what fails — so the dense curves can crash at a
  size the cutoffs allow.
- **A failed point is stored as NaN**, indistinguishable from one never
  attempted, so `--long-runs-only` re-proposes known failures.
- **Constants are per-machine.** The time laws were calibrated on this
  workstation's CPU; elsewhere they are indicative and the `_retired`
  backstop in `benchmark()` catches the difference.
