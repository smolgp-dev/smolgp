from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = ["RTSSmoother", "rts_smoother"]


def RTSSmoother(kernel, X, kalman_results):
    '''
    Wrapper for RTS smoother

    Parameters:
        kernel: StateSpaceModel kernel
        X: input coordinates
        kalman_results: output from Kalman filter (m_filtered, P_filtered, m_predicted, P_predicted)

    Returns:
        m_smooth: smoothed means
        P_smooth: smoothed covariances
    '''
    A = kernel.transition_matrix
    return rts_smoother(A, X, *kalman_results)

@jax.jit
def rts_smoother(A, t, m_filtered, P_filtered, m_predicted, P_predicted):
    '''
    Jax implementation of the Rauch-Tung-Striebel (RTS) smoothing algorithm

    See Theorem 8.2 (pdf page 156) in "Bayesian Filtering and Smoothing" 
    by Simo S{\"a}rkk{\"a} for detailed description of the algorithm and notation.
    '''
    N = len(t) # number of data points

    def step(carry, k):
        '''
        Routine for a single step of the RTS smoother

        Parameters:
            carry: (m_next, P_next) - next state and covariance
            k: index of the current time step

            Recall we are iterating backwards, so _next is k+1

        Returns:
        - Smoothed state (m_k_hat) and covariance (P_k_hat) to carry to next iteration
        - Full output for completed scan (m_k_hat, P_k_hat)
        '''

        # Outputs from Kalman filter, unpacked for notational consistency
        m_k = m_filtered[k]
        P_k = P_filtered[k]
        m_pred_next = m_predicted[k+1] # has superscript minus
        P_pred_next = P_predicted[k+1] # has superscript minus

        # Unpack state and covariance from last iteration
        m_hat_next, P_hat_next = carry

        # Time-lag between states
        Delta_k = t[k+1] - t[k]

        # Transition matrix
        A_k = A(0,Delta_k)

        # Compute smoothing gain
        # P_pred_next_inv = jnp.linalg.inv(P_pred_next)
        # G_k = P_k @ A_k.T @ P_pred_next_inv # smoothing gain
        G_k = jnp.linalg.solve(P_pred_next.T, (P_k @ A_k.T).T).T # more stable
        
        # Update state and covariance
        m_hat_k = m_k + G_k @ (m_hat_next - m_pred_next)
        P_hat_k = P_k + G_k @ (P_hat_next - P_pred_next) @ G_k.T
        
        return (m_hat_k, P_hat_k), (m_hat_k, P_hat_k)

    # Start smoothing from final filtered state
    init_carry = (m_filtered[-1], P_filtered[-1])

    # Run backward from N-2 down to 0
    _, outputs = jax.lax.scan(step, init_carry, jnp.arange(N-2, -1, -1))
    m_smooth_reversed, P_smooth_reversed = outputs

    # Reverse outputs to match time order
    m_smooth = jnp.vstack([m_smooth_reversed[::-1], m_filtered[-1][None, :]])
    P_smooth = jnp.vstack([P_smooth_reversed[::-1], P_filtered[-1][None, :, :]])
    return m_smooth, P_smooth