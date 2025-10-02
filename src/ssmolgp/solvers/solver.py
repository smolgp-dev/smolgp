from __future__ import annotations

__all__ = ["StateSpaceSolver"]

from typing import Any

import equinox as eqx

from tinygp.helpers import JAXArray
from ssmolgp.kernels.stationary import StateSpaceModel
from tinygp.noise import Noise

from tinygp.solvers.quasisep.solver import QuasisepSolver

from ssmolgp.solvers.kalman import KalmanFilter
from ssmolgp.solvers.rts import RTSSmoother

class StateSpaceSolver(eqx.Module):
    """
    A solver that uses ``jax.lax.scan`` to implement Kalman filtering and RTS smoothing
    """

    X: JAXArray
    y: JAXArray | None = None
    means: JAXArray
    covariances: JAXArray

    def __init__(
        self,
        kernel: StateSpaceModel,
        X: JAXArray,
        y: JAXArray | None,
        noise: Noise,
    ):
        """Build a :class:`DirectSolver` for a given kernel and coordinates

        Args:
            kernel: The kernel function.
            X: The input coordinates.
            y: The measurement values.
            noise: The noise model for the process.
        """
        self.kernel = kernel
        self.X = X
        self.y = y
        self.noise = noise
        # self.variance_value = kernel(X) + noise.diagonal()
        # if covariance is None:
        #     covariance = kernel(X, X) + noise
        # self.covariance_value = covariance

    def variance(self) -> JAXArray:
        return self.variance_value

    def covariance(self) -> JAXArray:
        return self.covariance_value

    def normalization(self) -> JAXArray:
        # TODO: do we want/can we implement this in state space? for now, fall back to quasisep
        return QuasisepSolver(self.kernel, self.X, self.noise).normalization()

    def condition(self) -> Any:
        """
        Compute the Kalman predicted, filtered, and RTS smoothed 
        means and covariances at each of the input coordinates
        """

        # Kalman filtering
        kalman_results = KalmanFilter(self.kernel, self.X, self.y, self.noise)
        m_filtered, P_filtered, m_predicted, P_predicted = kalman_results

        # RTS smoothing
        rts_results = RTSSmoother(self.kernel, self.X, kalman_results)
        m_smoothed, P_smoothed = rts_results
    
        # # Project predictive mean/var to observation space
        ## TODO; move/rewrite this into GaussianProcess.condition
        # y_kal = (H @ m_filtered.T).squeeze()
        # yvar_kal = (H @ P_filtered @ H.T).squeeze()
        # yerr_kal = jnp.sqrt(yvar_kal)

        # y_rts = (H @ m_smooth.T).squeeze()
        # yvar_rts  = (H @ P_smooth @ H.T).squeeze()
        # yerr_rts = jnp.sqrt(yvar_rts)

        return (m_predicted, P_predicted), (m_filtered, P_filtered), (m_smoothed, P_smoothed)