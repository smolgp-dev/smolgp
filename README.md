<p align="center">
<img width="100" height="100" alt="smolgp-logo" src="https://github.com/user-attachments/assets/66c691c9-c4d3-4253-9587-82f50adda047"/><br>
<strong>smolgp</strong><br>
<i>State Space Models for O(Linear/Log) Gaussian Processes</i>
</p>

`smolgp` is a Python/JAX extension of the [`tinygp`](https://github.com/dfm/tinygp) package that implements
1. A Kalman filter and RTS smoother as a `StateSpaceSolver` compatible with `tinygp`-like GP kernels.
2. An `IntegratedStateSpaceSolver` that can handle integrated (and possibly overlapping) measurements from mutliple instruments (see Rubenzahl and Hattori et al. in prep)
3. TODO: Parallelized versions of 1 (see [Särkkä and García-Fernández 2020](https://ieeexplore.ieee.org/document/9013038)) and 2 (see [Yaghoobi and Särkkä 2024](https://ieeexplore.ieee.org/abstract/document/10804629) and its [implementation](https://github.com/Fatemeh-Yaghoobi/Parallel-integrated-method?tab=readme-ov-file)) togglable with the `num_parallel_workers` flag

TODO:
- benchmark plots from paper/showing full GP vs. QSM GP vs. SSM vs. parallel SSM
- doc/example useage
- tests

Possible additions
- define other kernels not in tinygp
