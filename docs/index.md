# smolgp

_**S**tate Space **M**odels for **O**(**L**inear/Log) **G**aussian **P**rocesses_

[`smolgp`](https://github.com/smolgp-dev/smolgp) is a Python/JAX standalone extension of the [`tinygp`](https://github.com/dfm/tinygp) package that uses the state space representation of Gaussian Process to achieve substantial performance boosts. Like `tinygp` it is built on top of [`jax`](https://github.com/google/jax) and so can utilize just-in-time compliation, automatic differentiation, and GPU-accelerated linear algebra. It can even be [parallelized](tutorials/parallel) for a further performance boost.

To get started, check out the {ref}`guide` and then the {ref}`quickstart` to hit the ground running. There are also many useful {ref}`tutorials` with example usage, including {ref}`introssm` for those interested in the framework that powers `smolgp`. For all the nitty-gritty details, see the [full API documentation](api-ref). 

If you use `smolgp` in your research, please see {ref}`citing`.

```{admonition} When should I use smolgp instead of tinygp?
:class: tip

⚡️ If you want scalable (O(N) or better) performance for GP kernels which do not have quasiseparable representations but can be approximated by a state space model, such as the quasiperiodic kernel (see {ref}`quasiperiodic`).

∬ If your measurements are integrated over finite time intervals that are appreciable compared to the variability timescale of the GP. `smolgp` can correctly account for the integrated covariance while maintaining scalable performance. See {ref}`integrated` for more details.

📊 If you are jointly modeling data from multiple instruments and those measurements overlap with one another. `smolgp` naturally accounts for the covariances during the overlap by construction.
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