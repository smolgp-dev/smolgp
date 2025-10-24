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
from smolgp.solvers.integrated.kalman import IntegratedKalmanFilter
from smolgp.solvers.integrated.rts import IntegratedRTSSmoother

class IntegratedStateSpaceSolver(eqx.Module):
    """
    A solver that uses ``jax.lax.scan`` to implement Kalman filtering 
    and RTS smoothing for integrated measurements
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
        """Build a :class:`IntegratedStateSpaceSolver` for a given kernel and coordinates

        Args:
            kernel: The kernel function.
            X: The input coordinates. The coordinates for an integrated model should be a tuple of
                    X = (t, delta, instid), 
                where `t` is the usual coordinate (e.g. time) at the measurements (midpoints),
                `delta` is the integration range (e.g. exposure time) for each measurement,
                and `instid` is an index encoding which instrument the measurement corresponds to.

            y: The measurement values.
            noise: The noise model for the process.
        """
        self.kernel = kernel
        self.X = X
        self.noise = noise

    def normalization(self) -> JAXArray:
            # TODO: do we want/can we implement this in state space? for now, fall back to quasisep
            return QuasisepSolver(self.kernel, self.X, self.noise).normalization()
    
    def Kalman(self, X, y, X_states, return_v_S=False) -> Any:
            """Wrapper for Kalman filter used with this solver"""
            t_states, instid, obsid, stateid = X_states # this gets encoded in 'condition' below
            return IntegratedKalmanFilter(self.kernel, X, y, t_states, obsid, instid, stateid, self.noise, return_v_S=return_v_S)

    def RTS(self, X_states, kalman_results) -> Any:
        """Wrapper for RTS smoother used with this solver"""
        t_states, instid, obsid, stateid = X_states # this gets encoded in 'condition' below
        return IntegratedRTSSmoother(self.kernel, t_states, obsid, instid, stateid, kalman_results)

    def condition(self, y, return_v_S=False) -> JAXArray:
        """
        Compute the Kalman predicted, filtered, and RTS smoothed 
        means and covariances at each of the input coordinates
        """

        tmid, delta, instid = self.X  # unpack coordinates

        ## Bookkeeping/prepwork to assign labels to each observation/state
        # obsid   -- array len(K): which observation (0,...,N-1) is being made at each state k
        # instids -- array len(N): which instrument (0,...,Ninst-1) recorded observation n
        # stateid -- array len(K): 0 for exposure-start, 1 for exposure-end
       
        ## Construct interleaved time array of chronological exposure start/stop times
        ts = tmid - delta/2  # Exposure start times
        te = tmid + delta/2  # Exposure end times
        obsid = jnp.arange(len(tmid)).repeat(2)  # which observation does each time belong to

        # Interleave start and end times into one array (fastest)
        # https://stackoverflow.com/questions/5347065/interleaving-two-numpy-arrays-efficiently
        t_states = jnp.empty((ts.size + te.size,), dtype=tmid.dtype)
        t_states = t_states.at[0::2].set(ts)  # evens are start times
        t_states = t_states.at[1::2].set(te)  # odds are end times
        # Have to re-sort because exposures can overlap
        sortidx  = jnp.argsort(t_states)
        t_states = t_states[sortidx]
        obsid    = obsid[sortidx] 
        stateid = jnp.tile(jnp.array([0,1]), len(tmid))[sortidx] # 0 for t_s, 1 for t_e

        # Pack-up X_states for Kalman and RTS functions
        X_states = (t_states, instid, obsid, stateid)
        
        # Kalman filtering
        kalman_results = self.Kalman(self.X, y, X_states, return_v_S=return_v_S)
        if return_v_S:
            m_filtered, P_filtered, m_predicted, P_predicted, v, S = kalman_results
            v_S = (v, S)
        else:
            m_filtered, P_filtered, m_predicted, P_predicted = kalman_results
            v_S = None
        
        # RTS smoothing
        rts_results = self.RTS(X_states, (m_filtered, P_filtered, m_predicted, P_predicted))
        m_smoothed, P_smoothed = rts_results

        # Pack-up results and return
        conditioned_states = (m_predicted, P_predicted), (m_filtered, P_filtered), (m_smoothed, P_smoothed)
        return t_states, conditioned_states, v_S
    
    def predict(self, X_test, conditioned_results, observation_model=None) -> JAXArray:
        """
        Wrapper fot jitted StateSpaceSolver._predict. 
        
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
        t_states, conditioned_states, _ = conditioned_results
        (m_predicted, P_predicted), \
        (m_filtered, P_filtered), \
        (m_smooth, P_smooth) = conditioned_states

        # Unpack test coordinates
        t_test = self.kernel.coord_to_sortable(X_test)

        # Array shapes
        N = len(self.X)   # number of data points
        K = len(t_states) # number of states
        M = len(t_test)   # number of test points

        # Prior covariance for retrodiction
        Pinf = self.kernel.stationary_covariance()
        if not isinstance(Pinf, JAXArray): # if multicomponent model
            Pinf = Pinf.to_dense()  # needs to be array form here

        # Prior mean for retrodiction
        mean = jnp.zeros(self.kernel.d) # TODO: mean function of base kernel
        m0 = jnp.block([mean] + self.kernel.num_insts*[jnp.zeros(self.kernel.d)])

        # Nearest (future) datapoint
        k_nexts = jnp.searchsorted(t_states, t_test, side='right')

        # Method to use for test point
        past   = (k_nexts<=0)    # Retrodict
        future = (k_nexts>=N)    # Extrapolate
        during = ~past & ~future # Interpolate
        cases = (past.astype(int)*0 + during.astype(int)*1 + future.astype(int)*2)

        # Shorthand for matrices
        A_aug = lambda dt: self.kernel.transition_matrix(0, dt)
        Q_aug = lambda dt: self.kernel.process_noise(0, dt)

        @jax.jit
        def kalman(k, ktest):
            '''
            Kalman prediction from most recent 
            filtered (but not RTS smoothed) state
            '''
            dt = t_test[ktest] - t_states[k]
            m_k = m_filtered[k]
            P_k = P_filtered[k]
            A_star = A_aug(dt) # transition matrix from t_k to t_star
            Q_star = Q_aug(dt) # process noise from t_k to t_star
            m_star_pred = A_star @ m_k
            P_star_pred = A_star @ P_k @ A_star.T + Q_star
            return m_star_pred, P_star_pred
        
        @jax.jit
        def smooth(k_next, ktest, m_star_pred, P_star_pred):
            '''
            RTS smooth the prediction (ktest) using 
            the nearest future data point (k_next)

            m_star_pred and P_star_pred are the output of kalman(k, k_star)
            '''
            # Next (future) data point predicted & smoothed state
            m_pred_next = m_predicted[k_next] # prediction (no kalman update) at next data point
            P_pred_next = P_predicted[k_next] # prediction (no kalman update) at next data point
            m_hat_next = m_smooth[k_next]     # RTS smoothed state at next data point
            P_hat_next = P_smooth[k_next]     # RTS smoothed covariance at next data point
            
            # Transition matrix
            dt = t_states[k_next] - t_test[ktest]
            A_k = A_aug(dt)

            # RTS update
            G_k = jnp.linalg.solve(P_pred_next.T, (P_star_pred @ A_k.T).T).T
            m_star_hat = m_star_pred + G_k @ (m_hat_next - m_pred_next)
            P_star_hat = P_star_pred + G_k @ (P_hat_next - P_pred_next) @ G_k.T
            
            return m_star_hat, P_star_hat

        @jax.jit
        def project(ktest, m_star, P_star):
            ''' Project the state vector to the observation space '''
            # TODO: if user specifies exposure time here, need to:
            ## 1. predict to start state & set z to zero
            ## 2. predict to the end state, then project using H and texp_test
            Hk = H(X_test[ktest])
            pred_mean = (Hk @ m_star.T).squeeze()
            pred_var  = (Hk @ P_star @ Hk.T).squeeze()
            return pred_mean, pred_var

        def retrodict(ktest):
            ''' Reverse-extrapolate from first datapoint t_star '''
            m_star, P_star = smooth(0, ktest, m0, Pinf, retro=True)
            return project(ktest, m_star, P_star)

        @jax.jit
        def interpolate(ktest):
            ''' Interpolate between nearest data points '''
            
            # Get nearest data point before and after the test point
            k_next = k_nexts[ktest]
            k_prev = k_next - 1

            # 1. Kalman predict from most recent data point (in past)
            m_star_pred, P_star_pred = kalman(k_prev, ktest)

            # 2. RTS smooth from next nearest data point (in future)
            m_star_hat, P_star_hat = smooth(k_next, ktest, m_star_pred, P_star_pred)

            return project(ktest, m_star_hat, P_star_hat)

        @jax.jit
        def extrapolate(ktest):
            ''' Kalman predict from from last datapoint t_star '''
            m_star, P_star = kalman(-1, ktest)
            return project(ktest, m_star, P_star)
        
        # Calculate predictions
        ktests = jnp.arange(0, M, 1)
        branches = (retrodict, interpolate, extrapolate)
        (pred_mean, pred_var) = jax.vmap(lambda ktest: jax.lax.switch(cases[ktest], branches, (ktest)))(ktests)

        return pred_mean, pred_var