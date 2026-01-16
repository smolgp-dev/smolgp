(benchmarks)=

# Benchmarks

This page summarizes the key plots from Section 4 in [Rubenzahl et al. 2026](https://ui.adsabs.harvard.edu/abs/2026arXiv260102527R/abstract), which benchmark the performance of `smolgp` on CPU (blue curves) and GPU (purple curve) as compared to the same computations implemented in `tinygp` (green curve). Each set of plots benchmark the runtime and memory usage for computing the likelihood, conditioned mean and variance at the data points, and predicted mean and variance on a grid of points 100x as dense as the data. 

## Instantaneous measurements
For instantaneous measurements, quasiseparable matrix (QSM, orange curves) optimizations can be leveraged with `tinygp` to achieve very similar performance as the state space method. However, prediction is much faster and significantly less memory intensive with `smolgp`.

::::{grid} 3
:gutter: 1
:align: bottom

:::{grid-item}
:::{image} _static/llh_benchmark.png
:::

:::{grid-item}
:::{image} _static/cond_benchmark.png
:::

:::{grid-item}
:::{image} _static/pred_benchmark.png
:::

::::

## Integrated measurements
When the measurements individually span a finite time interval with variable length and/or overlap with other measurements, we cannot take advantage of any quasiseparable optimizations in `tinygp` and so are forced to use the $\mathcal{O}(N^3)$ solution there. This is the scenario in which `smolgp` has the most impactful advantage over previous methods.

::::{grid} 3
:gutter: 1
:align:bottom

:::{grid-item}
:::{image} _static/llh_int_benchmark.png
:::

:::{grid-item}
:::{image} _static/cond_int_benchmark.png
:::

:::{grid-item}
:::{image} _static/pred_int_benchmark.png
:::

::::