r"""
This subpackage implements the Kalman filter and RTS smoother for an augmented
state space model in its parallel form, which includes an integral state as a
:class:`ParallelIntegratedStateSpaceSolver`. This is intended to be used with
:class:`smolgp.kernels.integrated` state space models to properly account
for integrated (e.g. exposure-averaged) measurements.

The parallel method is described in Section 3.2.4 of
`Rubenzahl and Hattori et al. (2026) <https://iopscience.iop.org/article/10.3847/1538-3881/ae4d0b>`_,

See :doc:`/tutorials/parallel` for a tutorial on using the parallel solvers.
"""
