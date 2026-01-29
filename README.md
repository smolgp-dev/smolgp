<p align="center">
<img width="100" height="100" alt="smolgp-logo" src="https://github.com/user-attachments/assets/66c691c9-c4d3-4253-9587-82f50adda047"/><br>
<strong>smolgp</strong><br>
<i>State Space Models for O(Linear/Log) Gaussian Processes</i>
</p>

[![docs](https://readthedocs.org/projects/smolgp/badge/?version=latest)](https://smolgp.readthedocs.io/en/latest/index.html)
[![Tests](https://github.com/smolgp-dev/smolgp/actions/workflows/tests.yml/badge.svg)](https://github.com/smolgp-dev/smolgp/actions/workflows/tests.yml)
[![codecov](https://codecov.io/github/smolgp-dev/smolgp/branch/main/graph/badge.svg?token=KQLRPBCV9X)](https://codecov.io/github/smolgp-dev/smolgp)
[![arXiv](https://img.shields.io/badge/arXiv-2601.02527-b31b1b.svg)](https://arxiv.org/abs/2601.02527)
[![DOI](https://zenodo.org/badge/1065470871.svg)](https://doi.org/10.5281/zenodo.18418837)

[`smolgp`](https://github.com/smolgp-dev/smolgp) is a Python/JAX standalone extension of the [`tinygp`](https://github.com/dfm/tinygp) package that implements
1. A Kalman filter and RTS smoother as a `StateSpaceSolver` compatible with `tinygp`-like GP kernels (see `smolgp.kernels`)
2. An `IntegratedStateSpaceSolver` that can handle integrated (and possibly overlapping) measurements from multiple instruments (see [Rubenzahl and Hattori et al. submitted](https://arxiv.org/abs/2601.02527))
3. Parallelized versions of 1 (`ParallelStateSpaceSolver`, see [Särkkä and García-Fernández 2020](https://ieeexplore.ieee.org/document/9013038)) and 2 (`ParallelIntegratedStateSpaceSolver`, see [Rubenzahl and Hattori et al. submitted](https://arxiv.org/abs/2601.02527)) using `jax.lax.associative_scan`
    - see also [Yaghoobi and Särkkä 2024](https://ieeexplore.ieee.org/abstract/document/10804629) and its [implementation](https://github.com/Fatemeh-Yaghoobi/Parallel-integrated-method?tab=readme-ov-file)
4. Approximations of popular GP kernels that lack quasiseparability (e.g., ExpSineSquared, Quasiperiodic) but can utilize the O(N) states space solvers.

This package (and its documentation) is still under heavy active development, with tutorials coming soon. Please raise issues here and/or reach out to [Ryan Rubenzahl](https://github.com/rrubenza) and/or [So Hattori](https://github.com/soichiro-hattori). 

## Installation

For the most up-to-date version of the code, clone this repository and install locally.

There is also a version on PyPI (TODO: will be auto-updated with most recent version of this repo):

```uv add smolgp```

Note that `tinygp` dependencies require the latest version of the [tinygp GitHub repository](https://github.com/dfm/tinygp), rather than the version on PyPI. [uv](https://docs.astral.sh/uv/) should handle this automatically.
