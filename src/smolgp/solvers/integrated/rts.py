from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = ["IntegratedRTSSmoother", "integrated_rts_smoother"]

def IntegratedRTSSmoother(kernel, t_states, obsid, instid, stateid, kalman_results):
    '''
    Wrapper for jitted integrated_rts_smoother function

    Parameters:
        kernel  : IntegratedStateSpaceModel kernel
        t_states: Array of size K, sorted time coordinate of all states (exposure starts and ends)
        obsid   : Array of size N, which observation (0,...,N-1) is being made at each state k
        instid  : Array of size N, which instrument (0,...,Ninst-1) recorded observation n
        stateid : Array of size K, 0 for exposure-start, 1 for exposure-end
        kalman_results: output from Kalman filter (m_filtered, P_filtered, m_predicted, P_predicted)

    Returns:
        m_filtered: filtered means
        P_filtered: filtered covariances
        m_predicted: predicted means
        P_predicted: predicted covariances
    '''

    # Model components
    A_aug = kernel.transition_matrix
    RESET = kernel.reset_matrix
    
    return integrated_rts_smoother(A_aug, RESET, t_states, 
                                     obsid, instid, stateid,
                                     *kalman_results)

@jax.jit
def integrated_rts_smoother(A_aug, RESET, t_states,
                             obsid, instid, stateid,
                             m_filtered, P_filtered,
                             m_predicted, P_predicted):

    def step(carry, k):
        # Outputs from Kalman filter, unpacked for notational consistency
        m_k = m_filtered[k]
        P_k = P_filtered[k]
        m_pred_next = m_predicted[k+1] # has superscript minus
        P_pred_next = P_predicted[k+1] # has superscript minus

        # Unpack state and covariance from last iteration
        m_hat_next, P_hat_next = carry

        # Time-lag between states
        Delta = t_states[k+1] - t_states[k]
        
        # Compute smoothing gain
        A_k = A_aug(Delta)

        # If transition is from te_k to ts_k (i.e., over the exposure)
        def smooth_start():
            ''' Back-propagate state during an exposure '''

            ## What we're working with:
            # pre-reset  at k+1: m_pred_next, P_pred_next (predicted)
            # post-reset at k+1: m_hat_next, P_hat_next (smoothed)
            # post-reset at k  : m_k, P_k (filtered)

            ## 1. t_e to post-reset t_s
            ##    aka t_k+1 to t_k+2/3
            ##    it is the RTS equations over the exposure interval
            G_k_post = jnp.linalg.solve(P_pred_next.T, (P_k @ A_k.T).T).T
            m_hat_k_post = m_k + G_k_post @ (m_hat_next - m_pred_next)
            P_hat_k_post = P_k + G_k_post @ (P_hat_next - P_pred_next) @ G_k_post.T

            ## 2. post-reset t_s to pre-reset t_s
            ##    aka t_k+2/3 to t_k+1/3
            ##    it is RTS but with 'Reset' as our 'transition matrix'
            m_k_pre = m_predicted[k]  # pre-reset start state
            P_k_pre = P_predicted[k]  # pre-reset start covariance
            # After undoing the reset, add a nonzero value to the diagonal at the zeroed-out z
            # This let's us calculate the inverse, but does not affect the end result
            # since we immedietely multiply by Reset.T which deletes those rows/cols again
            Reset = RESET(instid[obsid[k]])
            # P_pred_post = Reset @ P_k_pre @ Reset.T + (jnp.eye(len(Reset))-Reset)  #### This changed
            P_pred_post = P_k + (jnp.eye(len(Reset))-Reset)  #### This changed
            G_k_pre = jnp.linalg.solve(P_pred_post.T, (P_k_pre @ Reset.T).T).T 

            ## Final smoothed state at k
            m_hat_k = m_k_pre + G_k_pre @ (m_hat_k_post - m_k)
            P_hat_k = P_k_pre + G_k_pre @ (P_hat_k_post - P_k) @ G_k_pre.T

            return m_hat_k, P_hat_k

        # If transition is from ts_k+1 to te_k (i.e., over the gap)
        def smooth_end():
            ''' Back-propagate state between exposures '''

            ## 3. pre-reset t_s to previous t_e
            ##    aka t_k+1/3 to t_k
            ##    this is simply the normal RTS update equations
            G_k = jnp.linalg.solve(P_pred_next.T, (P_k @ A_k.T).T).T
            m_hat_k = m_k + G_k @ (m_hat_next - m_pred_next)
            P_hat_k = P_k + G_k @ (P_hat_next - P_pred_next) @ G_k.T

            return m_hat_k, P_hat_k

        m_hat_k, P_hat_k = jax.lax.cond(
                    stateid[k]==0,
                    lambda _: smooth_start(), 
                    lambda _: smooth_end(),   
                    operand=None
                )
        
        return (m_hat_k, P_hat_k), (m_hat_k, P_hat_k)

    # Start smoothing from final filtered state
    init_carry = (m_filtered[-1], P_filtered[-1])

    # Run backward from N-2 down to 0
    K = len(t_states) # number of iterations
    _, outputs = jax.lax.scan(step, init_carry, jnp.arange(K-2, -1, -1))
    m_smooth_reversed, P_smooth_reversed = outputs

    # Reverse outputs to match time order
    m_smooth = jnp.vstack([m_smooth_reversed[::-1], m_filtered[-1][None, :]])
    P_smooth = jnp.vstack([P_smooth_reversed[::-1], P_filtered[-1][None, :, :]])
    return m_smooth, P_smooth