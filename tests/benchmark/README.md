# Benchmark suite: memory scalings

Runtime and peak-memory profiling for three GP implementations — `smolgp`'s
state-space solver (**SSM**), `tinygp`'s quasiseparable solver (**QSM**), and a
dense `tinygp` reference (**GP**) — across problem size.

Scripts for running the sweeps are in [`run/`](run/README.md). This file covers
the *memory model*: what each implementation must hold in memory, where those
numbers come from, and how far to trust them. The constants live in `_COST` /
`_COST_INT` in `benchmark.py`, where they set the size cutoffs that keep a sweep
from exhausting the machine.

## Derived scalings

Everything below is float64, so one stored value is 8 bytes. `d` is the
state-space dimension and `J` the quasiseparable rank; for the SHO kernel used
throughout, both are **2** (`SHO.design_matrix()` is 2×2, and `tinygp`'s
quasisep SHO has J=2).

A dash in *predicted* means the constant is **measured only** — a fitted
asymptote, with no derivation behind it yet. The size each measurement is
quoted at matters, and is given: see the caveats on non-constancy below.

| func | model | scaling | what is held | predicted | measured |
|---|---|---|---|---|---|
| **`llh`** <br>(x = N) | GP | N² | kernel matrix + Cholesky factor | 2·8 = **16 B/N²** | 16.0 &nbsp;<sub>N = 1778–10⁴</sub> |
| | QSM | N | `SymmQSM` + `LowerTriQSM` + data + workspace | 23·8 = **184 B/N** | 185.0 &nbsp;<sub>N = 10⁷</sub> |
| | SSM | N | one d-vector per point + 9 fixed N-arrays | 8·(d+9) = **88 B/N** | 88.3 &nbsp;<sub>N = 10⁷</sub> |
| **`cond`** <br>(x = N) | GP | N² | + posterior covariance | 3·8 = **24 B/N²** | 24.0 &nbsp;<sub>N = 56234</sub> |
| | QSM | N | — | — | 585.1 &nbsp;<sub>N = 10⁷</sub> |
| | SSM | N | — | — | 184.1 &nbsp;<sub>N = 10⁷</sub> |
| **`pred`** <br>(x = M = 100N) | GP | N·M | — | — | 24.0 &nbsp;<sub>M = 10⁶</sub> |
| | QSM | N·M | — | — | 48.0 &nbsp;<sub>M = 10⁶</sub> |
| | SSM | M | — | — | 444 B/M &nbsp;<sub>M = 5.6×10⁶</sub> |
| **`sample-prior`** <br>(x = M) | GP | M² | M×M kernel + Cholesky | 2·8 = **16 B/M²** | 16.0 &nbsp;<sub>M = 23713</sub> |
| | QSM | M | — | — | 584 B/M &nbsp;<sub>M = 7.5×10⁵</sub> |
| | SSM | M | — | — | 812 B/M &nbsp;<sub>M = 7.5×10⁵</sub> |
| **`sample-post`** <br>(x = M = 100N) | GP | **M²** | M×M posterior covariance + Cholesky + draw | 3·8 = **24 B/M²** | 24.7 &nbsp;<sub>M = 10⁴</sub> |
| | QSM | **M²** | — | — | 48.9 &nbsp;<sub>M = 10⁴</sub> |
| | SSM | M | — | — | 1066 B/M &nbsp;<sub>M = 10⁶</sub> |

## Integrated variants (`--int`)

Exposure-integrated data uses different kernels, and their constants live
separately in `_COST_INT`. There is **no QSM curve here** — `tinygp` has no
integrated quasiseparable kernel — so these sweeps run GP, SSM and pSSM only.

| func | model | scaling | predicted | measured | vs instantaneous |
|---|---|---|---|---|---|
| **`llh`** <br>(x = N) | GP | N² | 2·8 = **16 B/N²** | 15.8 &nbsp;<sub>N = 23713</sub> | 1.00× |
| | SSM | N | — | 349.9 B/N &nbsp;<sub>N = 10⁷</sub> | **3.96×** |
| **`cond`** <br>(x = N) | GP | N² | — | 56.3 B/N² &nbsp;<sub>N = 23713</sub> | **2.41×** |
| | SSM | N | — | 906.1 B/N &nbsp;<sub>N = 10⁷</sub> | **3.36×** |
| **`pred`** <br>(x = M = 100N) | GP | N·M | — | 22.3 B/N·M &nbsp;<sub>M = 3.2×10⁵</sub> | 1.01× |
| | SSM | M | — | 1680 B/M &nbsp;<sub>M = 10⁶</sub> | **3.64×** |
| **`sample-prior`** <br>(x = M) | GP | M² | 2·8 = **16 B/M²** | 16.0 &nbsp;<sub>M = 23713</sub> | 1.00× |
| | SSM | M | — | 1228 B/M &nbsp;<sub>M = 7.5×10⁵</sub> | 1.51× |
| **`sample-post`** <br>(x = M = 100N) | GP | M² | 3·8 = **24 B/M²** | 22.1 B/M² &nbsp;<sub>M = 3.2×10⁴</sub> | 1.00× |
| | SSM | M | — | 1936 B/M &nbsp;<sub>M = 10⁶</sub> | **1.82×** |

Every ratio is computed at the **same size** for both variants, at the largest
grid point where neither has yet crossed the in-place-factorisation transition.
That matters: several of these constants drift with N, so quoting the two
variants at different sizes would invent differences that are not there.

Two patterns are worth reading off that last column.

**The dense curve barely notices.** GP is *identical* integrated or not for
`llh`, `pred`, `sample-prior` and `sample-post` — 1.00× or 1.01× in every case.
The kernel matrix is N×N whichever kernel fills it, so the same 2 or 3 matrices
are held either way and the derivations carry over unchanged. The single
exception is `cond`, at 2.4×, where integrated conditioning keeps materially
more.

**The state-space curve pays for the augmented state**, from 1.5× to 4.0×,
because the integrated model carries an exposure-integral accumulator through
every intermediate. The ratios are *not* a single clean factor — 3.96× for
`llh` against 3.36× for `cond` and 1.51× for `sample-prior` — so no formula is
claimed for them; each is a measurement.

These integrated numbers were re-measured on 2026-08-17 against a rewritten
data generator (see below), so they supersede the constants still sitting in
`_COST_INT`. The two agree closely where both exist — `llh` SSM measures
349.9 B/N against a table value of 352.1 — but `cond` GP is now 56.3 B/N² at
N = 23713 where the table says 72, because the older figure was quoted at a
smaller size before the constant had finished drifting.

**The integrated datasets are drawn differently now.** `generate_integrated_data`
used to sample a dense "true" signal on a 1 s grid spanning the whole baseline
and average it over each exposure window: O(N × cadence), which at N = 10⁷ meant
a 2.16×10⁹-element array — past 2³¹, dying with a SIGSEGV on CPU and an int32
dimension error on GPU, and taking every smaller size in the sweep with it. It
now draws straight from the integrated kernel at the exposure windows, so the
model does the integration: O(N), 58 s and 2.95 GB at N = 10⁷. The notion of
"truth" changed with it, which is immaterial for timing, and the measurements
bear that out — SSM at N = 10⁷ agrees with the historical value to 0.6%.

### Dense GP

The only fully first-principles case. A dense likelihood forms the N×N kernel
matrix and its Cholesky factor and holds both, so 2·N²·8 = 16 B/N². Measured
16.00 to within 0.05% across N = 1778…23713. Conditioning keeps a third matrix
(24 B/N²), and a posterior draw Choleskys the full M×M posterior covariance —
which is why `sample-post` is **O(M²) and not O(N·M)** despite otherwise
mirroring `pred`.

### State-space solver (SSM)

Memory is **linear in the state dimension `d`, not quadratic**, at

```
    (d + 9) doubles per point  =  8·(d + 9) bytes per N
```

which is worth stating plainly because the obvious guess is wrong. A Kalman
filter carries `d×d` covariances, so the natural expectation is that per-point
storage goes as `d²` — and at the SHO's d=2 the measured 11.04 doubles/point is
fit perfectly well by `2d²+3` or `2d²+d+1`, both of which happen to equal 11
there. Neither survives contact with a larger state:

| components | d | measured B/N | doubles/pt | `2d²+3` predicts |
|---|---|---|---|---|
| 1 (SHO) | 2 | 88.55 | 11.07 | 11 |
| 2 | 4 | 104.49 | 13.06 | 35 |
| 3 | 6 | 120.47 | 15.06 | 75 |
| 4 | 8 | 136.49 | 17.06 | 131 |

Measured at N = 5×10⁶ with sums of SHO components. The slope is **0.999
doubles per unit d** where any `2d²` term would need ~20, so the covariances
are evidently not retained across the scan: for the likelihood only a running
state is carried, and the per-point cost is one `d`-vector on top of nine
N-length arrays that do not depend on `d` at all.

Those nine are not yet individually accounted for — three are presumably the
data (t, y, yerr), and identifying the rest would need a read through the
solver. So this is a validated empirical law in `d` rather than a
first-principles derivation, but unlike the d=2-only constant it can be
extrapolated to other kernels with confidence.

### Quasiseparable solver (QSM)

`tinygp`'s `QuasisepSolver` declares exactly two fields, `matrix: SymmQSM` and
`factor: LowerTriQSM`. Both decompose into a `DiagQSM` (`d`, one value per
point) and a `StrictLowerTriQSM` (`p` and `q` of shape (N, J), `a` of shape
(N, J, J)), giving `1 + 2J + J²` doubles each:

```
  SymmQSM        1 + 2J + J²   = 9      (covariance)
  LowerTriQSM    1 + 2J + J²   = 9      (its factor)
  data                       3          (t, y, diag)
  solve workspace            J = 2
  ------------------------------------
  total       2(1 + 2J + J²) + J + 3 = 23 doubles = 184 B/N
```

Measured 185.0 B/N — 0.5% agreement, from the solver's declared structure
rather than a fit.

## How these were determined

Each point is profiled in a **fresh subprocess** (`tracer` in `benchmark.py`),
which is what makes the numbers comparable: no state carries between sizes.

**On CPU**, a sampler thread polls RSS at 1 ms while the timed call runs, and
subtracts a baseline taken *after* JIT warm-up. Both halves matter. The
baseline must follow warm-up so compilation buffers are excluded, and the peak
must be sampled *during* the call — comparing RSS before and after the call
returns reads ≈0, because XLA has already released the buffers by then. That
second point caused a 285× under-report in an earlier diagnostic.

**On GPU**, RSS is meaningless, so the sampler reads XLA's own allocator
counters: `peak_bytes_in_use` after the call, minus the `bytes_in_use` resident
once warm-up settled. Two other combinations were tried and both fail:

- *peak minus peak* reads exactly **0** for every size. The peak is a
  high-water mark since process start with no way to reset it (there is no
  `reset_memory_stats` on the device object), so warm-up has already driven it
  to the figure being measured.
- *sampling `bytes_in_use` from a thread* also reads **0**. XLA dispatches the
  whole executable asynchronously and allocates and frees its transients inside
  one `Execute`, so the live gauge never moves. Measured directly: a 4096²
  matmul held `bytes_in_use` flat at 67 MB throughout while
  `peak_bytes_in_use` recorded 503 MB.

Constants are then read off the **converged regime**, which matters because
several do not settle until N ≳ 10³. `pred` QSM, for instance, runs
45.5 → 50.8 → 48.3 → 48.00 B/N·M across the grid; taking the small-N value
would overstate it by 17%.

## Caveats

**There is a measurement floor, and it differs by device.** Below roughly
N ≈ 10⁴ the SSM/QSM footprint is a few hundred kB, measured as the difference
between two ~300 MB RSS readings — so CPU reports either exactly 0 B or a
spurious sub-MB value. In the figures these points are replaced by the
theoretical footprint and drawn as hollow markers (`substitute_theory` in
`plotting.py`), so the curve stays readable without presenting computed values
as measured ones. The tell is non-monotonicity: SSM reads 1.001 MB at
N=4216 but 0.931 MB at N=10000. GPU has the same problem for a different
reason: XLA hands out allocator chunks, so small sizes quantise to exact
multiples of 16 MB (±0 B across all repeats). `nvidia-smi` is *worse* than
either — MiB granularity cannot resolve a 200 kB allocation at all.

**The dense constants are not scale-invariant.** Every dense curve drops by
roughly half at the largest size measured:

| curve | …23713 / 31600 | 56234 / 56200 |
|---|---|---|
| `llh` GP | 15.7 B/N² | **10.8** |
| `sample-prior` GP | 16.0 B/M² | **11.1** |
| `sample-post` GP | 22.1 B/M² | **11.8** |
| `sample-post` QSM | 34.5 B/M² | **12.0** |

All four converge on ~11–12 B/unit², about 1.4 matrices rather than 2–3, above
~3×10⁹ matrix elements (≈25 GB for one f64 matrix) — consistent with XLA
switching to an in-place factorisation that overwrites its input. The `_COST`
constants keep the smaller-size values, so they are **conservative by ~2× at
exactly the sizes where the cutoffs bite**. Safe, but it means the sweeps stop
earlier than the hardware requires.

**The constants are version-dependent.** Re-running `llh` against current
`smolgp`/`tinygp` moved both O(N) constants by ~22% from the values fitted to
the previously published data:

| | was | now |
|---|---|---|
| `llh` SSM | 72 B/N | **88.3** |
| `llh` QSM | 153 B/N | **185.0** |

`cond` and `pred` still carry constants fitted to the older data and should be
expected to shift similarly when they are re-run. Treat `_COST` as calibration
for a specific commit, not a property of the algorithms.
