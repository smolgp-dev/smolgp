# smolgp

_**S**tate Space **M**odels for **O**(**L**inear/Log) **G**aussian **P**rocesses_

[![docs](https://readthedocs.org/projects/smolgp/badge/?version=latest)](https://smolgp.readthedocs.io/en/latest/index.html)
[![Tests](https://github.com/smolgp-dev/smolgp/actions/workflows/tests.yml/badge.svg)](https://github.com/smolgp-dev/smolgp/actions/workflows/tests.yml)
[![codecov](https://codecov.io/github/smolgp-dev/smolgp/branch/main/graph/badge.svg?token=KQLRPBCV9X)](https://codecov.io/github/smolgp-dev/smolgp)
[![arXiv](https://img.shields.io/badge/arXiv-2601.02527-b31b1b.svg)](https://arxiv.org/abs/2601.02527)


[`smolgp`](https://github.com/smolgp-dev/smolgp) is a Python/JAX standalone extension of the [`tinygp`](https://github.com/dfm/tinygp) package that uses the state space representation of Gaussian Process to achieve substantial performance boosts. Like `tinygp` it is built on top of [`jax`](https://github.com/google/jax) and so can utilize just-in-time compliation, automatic differentiation, and GPU-accelerated linear algebra. It can even be [parallelized](tutorials/parallel) for a further performance boost.

To get started, check out the {ref}`guide` and then the {ref}`quickstart` to hit the ground running. There are also many useful {ref}`tutorials` with example usage, including {ref}`introssm` for those interested in the framework that powers `smolgp`. For all the nitty-gritty details, see the [full API documentation](api-ref). 

If you use `smolgp` in your research, please see {ref}`citing`.

```{admonition} When should I use smolgp instead of tinygp?
:class: tip

<div style="display: flex"><div style="font-size: 40px;margin-right:16px">⚡️</div><div>  If you want scalable (O(N) or better) performance for GP kernels which do not have quasiseparable representations but can be approximated by a state space model, such as the quasiperiodic kernel (see {ref}`kernels`).</div></div>
<br>
<div style="display: flex"><div style="font-size: 40px;margin-right:16px">∬</div><div> If your measurements are integrated over finite time intervals that are appreciable compared to the variability timescale of the GP. `smolgp` can correctly account for the integrated covariance while maintaining scalable performance. See {ref}`integrated` for more details.</div></div>
<br>
<div style="display: flex"><div style="font-size: 40px;margin-right:16px">📊</div><div> If you are jointly modeling data from multiple instruments and those measurements overlap with one another. `smolgp` naturally accounts for the covariances during the overlap by construction.</div></div>
```

If you find any bugs, please raise them on the [GitHub issues
page](https://github.com/smolgp-dev/smolgp/issues).


## Table of contents

```{toctree}
:maxdepth: 2

guide
tutorials
api/index
GitHub Repository <https://github.com/smolgp-dev/smolgp>
```

## Authors & license

Copyright 2025, 2026 Simons Foundation, Inc.

`smolgp` is built and maintained by [Ryan Rubenzahl](https://github.com/rrubenza) and [Soichiro Hattori](https://github.com/soichiro-hattori), but contributions from all via the [Issue
Tracker](https://github.com/smolgp-dev/smolgp/issues) are welcome. Licensed under the MIT license (see `LICENSE`).