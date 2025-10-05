from __future__ import annotations

__all__ = ["StateSpaceSolver"]

from typing import Any

import jax
import jax.numpy as jnp
import equinox as eqx

from tinygp.helpers import JAXArray
from tinygp.noise import Noise
from tinygp.solvers.quasisep.solver import QuasisepSolver
from ssmolgp.kernels.base import StateSpaceModel
from ssmolgp.solvers.kalman import KalmanFilter
from ssmolgp.solvers.rts import RTSSmoother

class StateSpaceSolver(eqx.Module):
    """
    A solver that uses ``jax.lax.scan`` to implement Kalman filtering and RTS smoothing
    """

    X: JAXArray
    y: JAXArray | None = None

    def __init__(
        self,
        kernel: StateSpaceModel,
        X: JAXArray,
        y: JAXArray | None,
        noise: Noise,
        conditioned_states: JAXArray | None,
    ):
        """Build a :class:`StateSpaceSolver` for a given kernel and coordinates

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
        self.conditioned_states = None

    def normalization(self) -> JAXArray:
        # TODO: do we want/can we implement this in state space? for now, fall back to quasisep
        return QuasisepSolver(self.kernel, self.X, self.noise).normalization()

    def condition(self, return_v_S=False) -> JAXArray:
        """
        Compute the Kalman predicted, filtered, and RTS smoothed 
        means and covariances at each of the input coordinates
        """

        # Kalman filtering
        kalman_results = KalmanFilter(self.kernel, self.X, self.y, self.noise, return_v_S=return_v_S)

        # RTS smoothing
        rts_results = RTSSmoother(self.kernel, self.X, kalman_results)

        # Unpack and return
        m_smoothed, P_smoothed = rts_results
        if return_v_S:
            m_filtered, P_filtered, m_predicted, P_predicted, v, S = kalman_results
            v_S = (v, S)
        else:
            m_filtered, P_filtered, m_predicted, P_predicted = kalman_results
            v_S = None

        # Save conditioned states for self.predict
        conditioned_states = (m_predicted, P_predicted), (m_filtered, P_filtered), (m_smoothed, P_smoothed)
        self.conditioned_states = conditioned_states
        return conditioned_states, v_S

    @jax.jit
    def predict(self, X_test, observation_model=None)  -> JAXArray:
        """
        TODO: write this docstring 
        TODO: add option to parallelize over X_test

        observation_model : H for the test points
                            should be a function just like 
                            self.kernel.observation_model
        """

        N = len(self.X) # number of data points
        M = len(X_test) # number of test points

        # If solver hasn't been conditioned, do so now
        if self.conditioned_states is None:
            self.condition()
            # raise RuntimeError("Must call .condition() before .predict()")
        
        # Unpack conditioned results
        (m_predicted, P_predicted),\
            (m_filtered, P_filtered), \
                (m_smooth, P_smooth) = self.conditioned_states
    
        Pinf = self.kernel.stationary_covariance()

        # Observation model to call at each of the X_test
        H = self.kernel.observation_model \
            if observation_model is None else observation_model

        # Nearest (future) datapoint
        ## TODO: we assume self.X is sorted here; should we enforce that?
        k_nexts = jnp.searchsorted(self.X, X_test, side='right')

        # Which method to use for each test point:
        past   = (k_nexts<=0)    # Retrodiction
        future = (k_nexts>=N)    # Forecast
        during = ~past & ~future # Interpolate
        cases = (past.astype(int)*0 + during.astype(int)*1 + future.astype(int)*2)

        # Shorthand for matrices
        A = self.kernel.transition_matrix
        Q = self.kernel.process_noise

        def predict(k, ktest):
            '''
            Kalman prediction from most recent 
            filtered (not smoothed) state
            '''
            dt = X_test[ktest] - self.X[k]
            m_k = m_filtered[k]
            P_k = P_filtered[k]
            A_star = A(0,dt) # transition matrix from t_k to t_star
            Q_star = Q(0,dt) # process noise from t_k to t_star
            m_star_pred = A_star @ m_k
            P_star_pred = A_star @ P_k @ A_star.T + Q_star
            # No Kalman update since we have no data at t_star, so we're done
            return m_star_pred, P_star_pred

        def smooth(k_next, ktest, m_star_pred, P_star_pred):
            '''
            RTS smooth the prediction (ktest) using 
            the nearest future (smoothed) state (k_next)

            m_star_pred and P_star_pred are the output of predict(k, k_star)
            '''
            # Next (future) data point predicted & smoothed state
            m_pred_next = m_predicted[k_next] # prediction (no kalman update) at next data point
            P_pred_next = P_predicted[k_next] # prediction (no kalman update) at next data point
            m_hat_next = m_smooth[k_next]     # RTS smoothed state at next data point
            P_hat_next = P_smooth[k_next]     # RTS smoothed covariance at next data point

            # Time-lag between states
            dt = self.X[k_next] - X_test[ktest]

            # Transition matrix for this step
            A_k = A(0,dt) 

            # Compute smoothing gain
            # P_pred_next_inv = jnp.linalg.inv(P_pred_next)
            # G_k = P_star_pred @ A_k.T @ P_pred_next_inv # smoothing gain
            G_k = jnp.linalg.solve(P_pred_next.T, (P_star_pred @ A_k.T).T).T # more stable
            
            # Update state and covariance
            m_star_hat = m_star_pred + G_k @ (m_hat_next - m_pred_next)
            P_star_hat = P_star_pred + G_k @ (P_hat_next - P_pred_next) @ G_k.T
            
            return m_star_hat, P_star_hat

        def project(ktest, m_star, P_star):
            ''' Project the state vector to the observation space '''
            Htest = H(X_test[ktest])
            pred_mean = (Htest @ m_star.T).squeeze()
            pred_var  = (Htest @ P_star @ Htest.T).squeeze()
            return pred_mean, pred_var

        def retrodict(ktest):
            ''' Reverse-extrapolate from first datapoint t_star '''
            m_star, P_star = smooth(0, ktest, 0, Pinf)
            return project(ktest, m_star, P_star)

        def interpolate(ktest):
            ''' Interpolate between nearest data points '''
            
            # Get nearest data point before and after the test point
            k_next = k_nexts[ktest]
            k_prev = k_next - 1

            # 1. Kalman predict from most recent data point (in past)
            m_star_pred, P_star_pred = predict(k_prev, ktest)

            # 2. RTS smooth from next nearest data point (in future)
            m_star_hat, P_star_hat = smooth(k_next, ktest, m_star_pred, P_star_pred)

            return project(ktest, m_star_hat, P_star_hat)

        def extrapolate(ktest):
            ''' Kalman predict from from last datapoint t_star '''
            m_star, P_star = predict(-1, ktest)
            return project(ktest, m_star, P_star)

        # Calculate predictions
        ktests = jnp.arange(0, M, 1)
        branches = (retrodict, interpolate, extrapolate)
        (pred_mean, pred_var) = jax.vmap(lambda ktest: jax.lax.switch(cases[ktest], branches, (ktest)))(ktests)

        return pred_mean, pred_var