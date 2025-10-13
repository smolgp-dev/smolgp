"""
In ``ssmolgp``, "solvers" provide a swappable low-level interface for
implementing the filtering and smoothing algorithms required to condition
Gaussian Process models. New solvers can be implemented as external packages
or as pull requests to the main ``ssmolgp`` GitHub project.

The built in solvers are:

1. :class:`StateSpaceSolver`: A solver that uses the standard Kalman filter
   and RTS smoothing algorithms to derive the optimal predictive means and
   covariances for a linear Gaussian state space model. This is the default
   solver, and is intended to be used with the instantaneous kernels
   implemented in `ssmolgp.kernels.base`. [TODO: compare benchmarking
   to `tinygp.solvers.QuasisepSolver`].

2. :class:`IntegratedStateSpaceSolver`: TODO....

3. :class:`ParallelStateSpaceSolver`: TODO....

4. :class:`IntegratedParallelStateSpaceSolver`: TODO....

Up to numerical precision, these are all *exact* solvers.

Users generally won't instantiate these solvers directly; ``ssmolgp`` should
automatically use the appropriate one given the form of the kernel.

TODO: how will the solver handle a mixed kernel where some are integrated
and some are instantaneous? It should default to the IntegratedStateSpaceSolver,
simply skipping updates for the instantaneous parts at the exposure starts,
but this needs to be tested.

The details for the included solvers are given below, but this is a pretty
low-level feature and the details are definitely subject to change!
"""

__all__ = ["StateSpaceModel"]

from ssmolgp.solvers.solver import StateSpaceSolver
from ssmolgp.solvers.parallel.solver import ParallelStateSpaceSolver
