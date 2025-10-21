from __future__ import annotations

__all__ = ["StateSpaceSolver"]

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

class StateSpaceSolver(eqx.Module):
    """
    A solver that uses ``jax.lax.scan`` to implement Kalman filtering and RTS smoothing
    """

    X: JAXArray
    kernel : StateSpaceModel
    noise  : Noise

    def __init__(
        self,
        kernel: StateSpaceModel,
        X: JAXArray,
        noise: Noise,
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
        self.noise = noise

    def normalization(self) -> JAXArray:
        # TODO: do we want/can we implement this in state space? for now, fall back to quasisep
        return QuasisepSolver(self.kernel, self.X, self.noise).normalization()

    def Kalman(self, X, y, noise, return_v_S=False) -> Any:
        """Wrapper for Kalman filter used with this solver"""
        X_states = X # states are at the data points here
        return X_states, KalmanFilter(self.kernel, X_states, y, noise, return_v_S=return_v_S)

    def RTS(self, X, kalman_results) -> Any:
        """Wrapper for RTS smoother used with this solver"""
        return RTSSmoother(self.kernel, X, kalman_results)

    def condition(self, y, return_v_S=False) -> JAXArray:
        """
        Compute the Kalman predicted, filtered, and RTS smoothed 
        means and covariances at each of the input coordinates
        """

        # Kalman filtering
        X_states, kalman_results = self.Kalman(self.X, y, self.noise, return_v_S=return_v_S)
        if return_v_S:
            m_filtered, P_filtered, m_predicted, P_predicted, v, S = kalman_results
            v_S = (v, S)
        else:
            m_filtered, P_filtered, m_predicted, P_predicted = kalman_results
            v_S = None
        
        # RTS smoothing
        rts_results = self.RTS(self.X, (m_filtered, P_filtered, m_predicted, P_predicted))
        m_smoothed, P_smoothed = rts_results

        # Pack-up results and return
        conditioned_states = (m_predicted, P_predicted), (m_filtered, P_filtered), (m_smoothed, P_smoothed)
        return X_states, conditioned_states, v_S
    
    def predict(self, X_test, conditioned_results, observation_model=None) -> JAXArray:
        """
        Wrapper fot jitted StateSpaceSolver._predict. 
        
        TODO: add option to parallelize over X_test
        Args:
            X_test              : The test coordinates.
            conditioned_results : The output of self.condition()
            observation_model   : (optional) H for the test points
                                  should be a function just like 
                                  self.kernel.observation_model
        """
        # Observation model to call at each of the X_test
        H = self.kernel.observation_model \
            if observation_model is None else observation_model
        return self._predict(X_test, conditioned_results, H)
    
    @jax.jit
    def _predict(self, X_test, conditioned_results, H) -> JAXArray:
        """
        Algorithm for making predictions at arbitrary coordinates X_test

        There are three cases:
            1. Retrodiction  : smoothing from the first data point
                               using the prior as the prediction
            2. Interpolation : filtering from most recent data point
                               and smoothing from next future point
            3. Extrapolation : predicting from final filtered point
        """

        # Unpack conditioned results
        X_states, conditioned_states, _ = conditioned_results
        (m_predicted, P_predicted), \
        (m_filtered, P_filtered), \
        (m_smoothed, P_smoothed) = conditioned_states

        N = len(self.X)   # number of data points
        K = len(X_states) # number of states
        M = len(X_test)   # number of test points
                
        Pinf = self.kernel.stationary_covariance()
        if not isinstance(Pinf, JAXArray):
            Pinf = Pinf.to_dense() # need dense version for jnp.linalg.solve in retrodict

        # Nearest (future) datapoint
        ## TODO: we assume X_states is sorted here; should we enforce that?
        k_nexts = jnp.searchsorted(X_states, X_test, side='right')

        # Which method to use for each test point:
        past   = (k_nexts<=0)    # Retrodiction
        future = (k_nexts>=K)    # Forecast
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
            m_hat_next = m_smoothed[k_next]     # RTS smoothed state at next data point
            P_hat_next = P_smoothed[k_next]     # RTS smoothed covariance at next data point

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