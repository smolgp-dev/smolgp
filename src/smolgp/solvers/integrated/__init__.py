r"""
This subpackage implements the Kalman filter and RTS smoother for an augmented
state space model which includes an integral state as a
:class:`IntegratedStateSpaceSolver`. This is intended to be used with
:class:`smolgp.kernels.integrated` state space models to properly account
for integrated (e.g. exposure-averaged) measurements.

See :ref:`integrated` for a tutorial on using the integrated solvers.
"""

from smolgp.solvers.integrated.solver import (
    IntegratedStateSpaceSolver as IntegratedStateSpaceSolver,
)
from smolgp.solvers.integrated.parallel.solver import (
    ParallelIntegratedStateSpaceSolver as ParallelIntegratedStateSpaceSolver,
)
