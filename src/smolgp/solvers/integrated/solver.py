from __future__ import annotations

__all__ = ["IntegratedStateSpaceSolver"]

from typing import Any

import jax
import jax.numpy as jnp
import equinox as eqx

from tinygp.helpers import JAXArray
from tinygp.noise import Noise
from tinygp.solvers.quasisep.solver import QuasisepSolver
from smolgp.kernels.base import StateSpaceModel
from smolgp.solvers.kalman import KalmanFilter
from smolgp.solvers.rts import RTSSmoother

class IntegratedStateSpaceSolver(eqx.Module):
    """
    A solver that uses ``jax.lax.scan`` to implement Kalman filtering 
    and RTS smoothing for integrated measurements
    """

    X: JAXArray
    y: JAXArray | None
    kernel : StateSpaceModel
    noise  : Noise

    def __init__(
        self,
        kernel: StateSpaceModel,
        X: JAXArray,
        y: JAXArray | None,
        noise: Noise,
    ):
        """Build a :class:`IntegratedStateSpaceSolver` for a given kernel and coordinates

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
