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

[`smolgp`](https://github.com/smolgp-dev/smolgp) is a standalone extension of the [`tinygp`](https://github.com/dfm/tinygp) package that implements scalable & GPU-parallelizable Gaussian Processes in JAX using the state space representation. It is particularly suited for integrated measurements (such as long exposures in astronomy), jointly modeling data from multiple instruments, and for scalable implementations of popular kernels that traditionally lack quasiseparable structure (e.g. the quasiperiodic kernel).

The `smolgp` API is designed to be as similar to `tinygp` as possible. In almost all cases, you can simply find-and-replace "smol" with "tiny" in your existing code.

## Main features
1. A Kalman filter and RTS smoother compatible with `tinygp`-like GP kernels.
3. Scalable (O(N)) solving with integrated (and possibly overlapping) measurements from multiple instruments (see [Rubenzahl and Hattori et al. 2026](https://arxiv.org/abs/2601.02527)).
4. Parallelized versions of 1 (see [Särkkä and García-Fernández 2020](https://ieeexplore.ieee.org/document/9013038)) and 2 (see [Rubenzahl and Hattori et al. 2026](https://arxiv.org/abs/2601.02527)).
    - see also [Yaghoobi and Särkkä 2024](https://ieeexplore.ieee.org/abstract/document/10804629) and its [implementation](https://github.com/Fatemeh-Yaghoobi/Parallel-integrated-method?tab=readme-ov-file).
5. Approximations of popular GP kernels that lack quasiseparability (e.g., ExpSineSquared, Quasiperiodic) that can utilize the O(N) state space solvers.
6. A convenient and optimally-efficient model-building framework to assemble multicomponent GPs and compute per-component distributions.

Check out the docs for more information, including tutorials: https://smolgp.readthedocs.io/

Please raise issues here and/or reach out to [Ryan Rubenzahl](https://github.com/rrubenza) and/or [So Hattori](https://github.com/soichiro-hattori). 

## Installation

You can install the most recent release from [PyPI](https://pypi.org/project/smolgp/), e.g. with [uv](https://docs.astral.sh/uv/):

```
uv add smolgp
```

Or, you can simply clone this repository and install locally:

```
git clone https://github.com/smolgp-dev/smolgp.git
cd smolgp
uv pip install -e .
```

Note that `tinygp` dependencies require the latest version of the [tinygp GitHub repository](https://github.com/dfm/tinygp), rather than the version on PyPI. [uv](https://docs.astral.sh/uv/) should handle this automatically.

## Citation
[![DOI](https://zenodo.org/badge/1065470871.svg)](https://doi.org/10.5281/zenodo.18418837)
[![arXiv](https://img.shields.io/badge/arXiv-2601.02527-b31b1b.svg)](https://arxiv.org/abs/2601.02527)

If you use `smolgp` in your research, please cite the relevant [software release](https://zenodo.org/records/18418838) and [paper](https://ui.adsabs.harvard.edu/abs/2026arXiv260102527R/abstract). The [`cffconvert` tool](https://github.com/citation-file-format/cffconvert) can be used to generate a bibtex entry from the included [CITATION.cff](https://github.com/smolgp-dev/smolgp/blob/main/CITATION.cff) (or just use the "cite this repository" button on the GitHub sidebar).

## Author & Contact 
[![GitHub followers](https://img.shields.io/github/followers/rrubenza?label=Follow&style=social)](https://github.com/rrubenza)
[![GitHub followers](https://img.shields.io/github/followers/soichiro-hattori?label=Follow&style=social)](https://github.com/soichiro-hattori)

This repo is maintained by [Ryan Rubenzahl](https://github.com/rrubenza) and [So Hattori](https://github.com/soichiro-hattori). 
