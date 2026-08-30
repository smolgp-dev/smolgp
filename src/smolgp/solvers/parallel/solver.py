from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from tinygp.helpers import JAXArray
from tinygp.solvers.quasisep.solver import QuasisepSolver

from smolgp.kernels.base import StateSpaceModel
from smolgp.solvers.solver import StateSpaceSolver
from smolgp.solvers.parallel.kalman import ParallelKalmanFilter
from smolgp.solvers.parallel.rts import ParallelRTSSmoother
from smolgp.solvers.rts import rts_gains
from smolgp.solvers.state_coords import StateCoords


class ParallelStateSpaceSolver(StateSpaceSolver):
    """
    A solver that uses ``jax.lax.associative_scan`` to implement
    parallel Kalman filtering and RTS smoothing.

    Inherits from :class:`StateSpaceSolver` and overrides the Kalman and RTS methods
    to use the parallel implementations. Methods which do not benefit from associative
    scans are inherited from :class:`StateSpaceSolver`.
    """

    def Kalman(self, y, return_v_S=True) -> Any:
        """Wrapper for Kalman filter used with this solver"""
        # noise (N, D, D) → R (N, D, D); y (..., N) → (N, D)
        y_nd = y[:, None] if y.ndim == 1 else y
        X_sorted, y_sorted, noise_sorted = self._to_state_order(
            self.X, y_nd, self.noise
        )
        return ParallelKalmanFilter(
            self.kernel, X_sorted, y_sorted, noise_sorted, return_v_S=return_v_S
        )

    def log_probability(self, y) -> JAXArray:
        """The marginal log likelihood, reduced from this solver's own filter.

        Overrides :meth:`StateSpaceSolver.log_probability` deliberately, as
        that is an optimized *sequential* scan. The generic path to reuse the
        Kalman filtered ``v`` and ``S`` is better here, as those are determined
        via associative scan, hence the likelihood stays log-depth.
        """
        return self._log_probability_from_filter(y)

    def RTS(self, kalman_results) -> Any:
        """Wrapper for RTS smoother used with this solver"""
        (X_sorted,) = self._to_state_order(self.X)
        return ParallelRTSSmoother(self.kernel, X_sorted, kalman_results)

    def condition(self, y, return_v_S=False) -> JAXArray:
        """
        Compute the Kalman predicted, filtered, and RTS smoothed
        means and covariances at each of the input coordinates
        """

        # Kalman filtering
        kalman_results = self.Kalman(y, return_v_S=return_v_S)
        if return_v_S:
            m_filtered, P_filtered, m_predicted, P_predicted, v, S = kalman_results
            v_S = (v, S)
        else:
            m_filtered, P_filtered, m_predicted, P_predicted = kalman_results
            v_S = None

        # RTS smoothing
        rts_results = self.RTS((m_filtered, P_filtered))
        _, m_smoothed, P_smoothed = rts_results

        # Pack-up results and return
        conditioned_states = (
            (m_predicted, P_predicted),
            (m_filtered, P_filtered),
            (m_smoothed, P_smoothed),
        )
        return self.state_coords, conditioned_states, v_S
