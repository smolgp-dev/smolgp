<p align="center">
<img width="100" height="100" alt="smolgp-logo" src="https://github.com/user-attachments/assets/66c691c9-c4d3-4253-9587-82f50adda047"/><br>
<strong>smolgp</strong><br>
<i>State Space Models for O(Linear/Log) Gaussian Processes</i>
</p>

[![Tests](https://github.com/smolgp-dev/smolgp/actions/workflows/tests.yml/badge.svg)](https://github.com/smolgp-dev/smolgp/actions/workflows/tests.yml)
[![codecov](https://codecov.io/github/smolgp-dev/smolgp/branch/main/graph/badge.svg?token=KQLRPBCV9X)](https://codecov.io/github/smolgp-dev/smolgp)

[`smolgp`](https://github.com/smolgp-dev/smolgp) is a Python/JAX extension of the [`tinygp`](https://github.com/dfm/tinygp) package that implements
1. A Kalman filter and RTS smoother as a `StateSpaceSolver` compatible with `tinygp`-like GP kernels (see `smolgp.kernels`)
2. An `IntegratedStateSpaceSolver` that can handle integrated (and possibly overlapping) measurements from multiple instruments (see Rubenzahl and Hattori et al. submitted)
3. Parallelized versions of 1 (`ParallelStateSpaceSolver`, see [Särkkä and García-Fernández 2020](https://ieeexplore.ieee.org/document/9013038)) and 2 (`ParallelIntegratedStateSpaceSolver`, see Rubenzahl and Hattori et al. submitted) using `jax.lax.associative_scan`
    - see also [Yaghoobi and Särkkä 2024](https://ieeexplore.ieee.org/abstract/document/10804629) and its [implementation](https://github.com/Fatemeh-Yaghoobi/Parallel-integrated-method?tab=readme-ov-file)
4. Approximations of popular GP kernels that lack quasiseparability (e.g., ExpSineSquared, Quasiperiodic) but can utilize the O(N) states space solvers.
