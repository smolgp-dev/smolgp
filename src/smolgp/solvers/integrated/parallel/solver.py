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

    _state_coords: JAXArray

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
        # Sets self.kernel, self.X, self.noise, and self.state_coords
        # (= (t_states, instid, obsid, stateid), with instid indexed
        # per-observation -- see class docstring below for why the parallel
        # Kalman/RTS functions additionally need a per-*state* instid array).
        super().__init__(kernel, X, noise)

        t_states, instid, obsid, stateid = self.state_coords
        # instid[obsid[k]] gathered once here (rather than inside every
        # per-state vmapped call in parallel/kalman.py and parallel/rts.py,
        # which index a per-state array directly instead of doing this
        # gather themselves).
        self._state_coords = (t_states, instid[obsid], obsid, stateid)

    def Kalman(self, y, return_v_S=True) -> Any:
        """Wrapper for Kalman filter used with this solver"""
        t_states, instid, obsid, stateid = self._state_coords
        # noise (D, D, N) → R (N, D, D); y (..., N) → (N, D)
        y_nd = y[:, None] if y.ndim == 1 else y
        return ParallelIntegratedKalmanFilter(
            self.kernel,
            self.X,
            y_nd,
            t_states,
            obsid,
            instid,
            stateid,
            self.noise,
            return_v_S=return_v_S,
        )

    def RTS(self, kalman_results) -> Any:
        """Wrapper for RTS smoother used with this solver"""
        t_states, instid, _obsid, stateid = self._state_coords
        return ParallelIntegratedRTSSmoother(
            self.kernel,
            t_states,
            stateid,
            instid,
            kalman_results,
        )
