"""
In ``smolgp``, "solvers" provide a swappable low-level interface for the
Bayesian filtering and smoothing algorithms required for GP conditioning.
New solvers can be contributed as external packages or pull requests to the
`smolgp GitHub repository <https://github.com/smolgp-dev/smolgp>`_.

The four built-in solvers are:

1. :class:`StateSpaceSolver`: Standard Kalman filter and RTS smoother for
   instantaneous kernels (see :mod:`smolgp.kernels.base`). This is the
   default solver.

2. :class:`IntegratedStateSpaceSolver`: Kalman filter and RTS smoother for
   integrated (time-averaged) measurement kernels
   (see :mod:`smolgp.kernels.integrated`).

3. :class:`ParallelStateSpaceSolver`: GPU-parallelised version of
   :class:`StateSpaceSolver` with :math:`O(\log N)` complexity on compatible
   hardware.

4. :class:`ParallelIntegratedStateSpaceSolver`: GPU-parallelised version of
   :class:`IntegratedStateSpaceSolver`.

All four inherit from :class:`Solver`, which fixes the interface (``Kalman``,
``RTS``, ``condition``, ``predict``) and supplies the shared state-order
bookkeeping and a default marginal likelihood built from any filter's
innovations. The parallel solvers subclass their sequential counterparts, 
implement associative scans for their Kalman and RTS methods, and fix
``log_probability`` to the default behavior rather than inherit a sequential scan.

All solvers are exact up to numerical precision.

Users generally do not need to instantiate solvers directly; :class:`~smolgp.GaussianProcess`
selects the appropriate solver automatically based on the kernel type.
"""

from smolgp.solvers.base import Solver as Solver
from smolgp.solvers.solver import StateSpaceSolver as StateSpaceSolver
from smolgp.solvers.parallel.solver import ParallelStateSpaceSolver as ParallelStateSpaceSolver
from smolgp.solvers.integrated.solver import (
    IntegratedStateSpaceSolver as IntegratedStateSpaceSolver,
)
from smolgp.solvers.integrated.parallel.solver import (
    ParallelIntegratedStateSpaceSolver as ParallelIntegratedStateSpaceSolver,
)
