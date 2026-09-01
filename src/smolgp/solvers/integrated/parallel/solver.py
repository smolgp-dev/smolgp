from __future__ import annotations

from typing import Any

from tinygp.helpers import JAXArray

from smolgp.kernels.base import StateSpaceModel
from smolgp.solvers.integrated.parallel.kalman import ParallelIntegratedKalmanFilter
from smolgp.solvers.integrated.parallel.rts import ParallelIntegratedRTSSmoother
from smolgp.solvers.integrated.solver import IntegratedStateSpaceSolver


class ParallelIntegratedStateSpaceSolver(IntegratedStateSpaceSolver):
    """
    A solver that uses ``jax.lax.associative_scan`` to implement
    parallel Kalman filtering and RTS smoothing for integrated measurements
    """

    _instid_per_state: JAXArray


    def __init__(
        self,
        kernel: StateSpaceModel,
        X: JAXArray,
        noise: JAXArray,
    ):
        """Build a :class:`ParallelIntegratedStateSpaceSolver` for a given kernel and coordinates

        Args:
            kernel: The kernel function.
            X: The input coordinates. The coordinates for an integrated model should be a tuple of
                    X = (t, delta, instid),
                where `t` is the usual coordinate (e.g. time) at the measurements (midpoints),
                `delta` is the integration range (e.g. exposure time) for each measurement,
                and `instid` is an index encoding which instrument the measurement corresponds to.
            noise: Observation noise covariance array of shape ``(D, D, N)``.
        """
        # Sets self.kernel, self.X, self.noise, and self.state_coords (whose
        # instid is indexed per-observation, length N).
        super().__init__(kernel, X, noise)

        # Unlike the sequential integrated Kalman/RTS functions, the parallel
        # ones index a per-*state* instid array (length K) directly rather
        # than doing the instid[obsid[k]] gather themselves, so hoist that
        # gather here (once) instead of into every per-state vmapped call.
        self._instid_per_state = self.state_coords.instid_per_state()

    def log_probability(self, y) -> JAXArray:
        """The marginal log likelihood, reduced from this solver's own filter.

        Overrides :meth:`IntegratedStateSpaceSolver.log_probability` deliberately, 
        as that is an optimized *sequential* scan. The generic path to reuse the
        Kalman filtered ``v`` and ``S`` is better here, as those are determined
        via associative scan, hence the likelihood stays log-depth.
        """
        return self._log_probability_from_filter(y)

    def Kalman(self, y, return_v_S=True) -> Any:
        """Wrapper for Kalman filter used with this solver"""
        sc = self.state_coords
        # noise (D, D, N) → R (N, D, D); y (..., N) → (N, D)
        y_nd = y[:, None] if y.ndim == 1 else y
        return ParallelIntegratedKalmanFilter(
            self.kernel,
            self.X,
            y_nd,
            sc.t_states,
            sc.obsid,
            self._instid_per_state,
            sc.stateid,
            self.noise,
            return_v_S=return_v_S,
        )

    def RTS(self, kalman_results) -> Any:
        """Wrapper for RTS smoother used with this solver"""
        sc = self.state_coords
        return ParallelIntegratedRTSSmoother(
            self.kernel,
            sc.t_states,
            sc.stateid,
            self._instid_per_state,
            kalman_results,
        )
