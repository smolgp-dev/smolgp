from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = ["KalmanFilter", "kalman_filter"]


def KalmanFilter(kernel, X, y, noise, return_v_S=False):
    '''
    Wrapper for Kalman filter

    Parameters:
        kernel: StateSpaceModel kernel
        X: input coordinates
        y: observations
        noise: Noise model
        
    Returns:
        m_filtered: filtered means
        P_filtered: filtered covariances
        m_predicted: predicted means
        P_predicted: predicted covariances
    '''
    H = kernel.observation_model
    A = kernel.transition_matrix
    Q = kernel.process_noise
    R = noise.diagonal() if noise is not None else jnp.zeros_like(y)
    m0 = jnp.zeros(kernel.dimension)
    P0 = kernel.stationary_covariance()

    output = kalman_filter(A, Q, H, R, X, y, m0, P0)
    if return_v_S:
        return output
    else:
        m_filtered, P_filtered, m_predicted, P_predicted, v, S = output
        return m_filtered, P_filtered, m_predicted, P_predicted

@jax.jit
def kalman_filter(A, Q, H, R, t, y, m0, P0, return_v_S=False):
    '''
    Jax implementation of the Kalman filter algorithm

    See Theorem 4.2 (pdf page 77) in "Bayesian Filtering and Smoothing" 
    by Simo S{\"a}rkk{\"a} for detailed description of the algorithm and notation.

    e.g. _prev is _{k-1} in Sarkka notation
         _pred is _k^{-} in Sarkka notation

    Total runtime complexity is O(N*d^3) where N is the number 
    of time steps and d is the dimension of the state vector.
    '''
    N = len(t) # number of data points

    def step(carry, k):
        '''
        Routine for a single step of the Kalman filter

        Parameters:
            carry: (x_prev, P_prev) - previous state and covariance
            k: index of the current time step
        
        Returns:
        - Conditioned state (m_k) and covariance (P_k) to carry to next iteration
        - Full output for completed scan (m_k, P_k, m_pred, P_pred)
        '''

        # Unpack previous state and covariance
        m_prev, P_prev = carry

        # Logic to check if first time step:
        # If k==0 we use the prior x0, P0
        # and zero time-lag (Delta=0)
        Delta = jax.lax.cond(k > 0,
                             lambda i: t[i] - t[i-1],
                             lambda _: 0.0,
                             k)
        
        # Get transition matrix
        A_prev = A(0,Delta)
        Q_prev = Q(0,Delta)
        
        # Predict (Eq. 4.20)
        m_pred = A_prev @ m_prev
        P_pred = A_prev @ P_prev @ A_prev.T + Q_prev 
        
        # Update (Eq. 4.21)
        H_k = H(t[k])                     # observation model for this time step
        y_pred = H_k @ m_pred             # predicted observation
        v_k = y[k] - y_pred               # "innovation" or "surprise" term
        S_k = H_k @ P_pred @ H_k.T + R[k] # uncertainy in predicted observation
        # S_k_inv = jnp.linalg.inv(S_k)
        # K_k = P_pred @ H_k.T @ S_k_inv    # Kalman gain
        K_k = jnp.linalg.solve(S_k.T, (P_pred @ H_k.T).T).T # more stable
        m_k = m_pred + K_k @ v_k          # conditioned state estimate
        P_k = P_pred - K_k @ S_k @ K_k.T  # conditioned covariance estimate
        
        return (m_k, P_k), (m_k, P_k, m_pred, P_pred, v_k, S_k)
    
    # Initialize carry with prior state and covariance
    init_carry = (m0, P0)

    # Run the filter over all time steps, unpack, and return results
    _, outputs = jax.lax.scan(step, init_carry, jnp.arange(N))
    return outputs
