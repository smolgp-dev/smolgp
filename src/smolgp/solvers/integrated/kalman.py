from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = ["IntegratedKalmanFilter", "integrated_kalman_filter"]


def IntegratedKalmanFilter(kernel, X, y, t_states, obsid, instid, stateid, noise=None, return_v_S=False):
    '''
    Wrapper for integrated_kalman_filter function

    Parameters:
        kernel  : IntegratedStateSpaceModel kernel
        X       : Array of size N, data coordinates (e.g. (time, texp, instid))
        y       : Array of size N, measurements at the data coordinates
        t_states: Array of size K, sorted time coordinate of all states (exposure starts and ends)
        obsid   : Array of size N, which observation (0,...,N-1) is being made at each state k
        instid  : Array of size N, which instrument (0,...,Ninst-1) recorded observation n
        stateid : Array of size K, 0 for exposure-start, 1 for exposure-end
        noise   : Noise model
        return_v_S : Whether to return innovation and its covariance (for likelihood computation)

    Returns:
        m_filtered : filtered means
        P_filtered : filtered covariances
        m_predicted: predicted means
        P_predicted: predicted covariances
    '''

    # Model components
    H_aug = kernel.observation_model
    A_aug = kernel.transition_matrix
    Q_aug = kernel.process_noise
    RESET = kernel.reset_matrix
    R = noise.diagonal() if noise is not None else jnp.zeros_like(y)
    
    # Initial state and covariance
    mean = jnp.zeros(kernel.d) # TODO: mean function of base kernel
    m0 = jnp.block([mean] + kernel.num_insts*[jnp.zeros(kernel.d)])
    P0 = kernel.stationary_covariance()

    output = integrated_kalman_filter(A_aug, Q_aug, H_aug, R, RESET,
                                      X, y, t_states, obsid, instid, stateid,
                                      m0, P0)
    if return_v_S:
        return output
    else:
        m_filtered, P_filtered, m_predicted, P_predicted, v, S = output
        return m_filtered, P_filtered, m_predicted, P_predicted


def integrated_kalman_filter(A_aug, Q_aug, H_aug, R, RESET,
                             X, y, t_states, obsid, instid, stateid,
                             m0, P0):
    """
    Jax implementation of the integrated Kalman filter algorithm

    See Section 3.2.1 in Rubenzahl & Hattori et al. (in prep) 
    for detailed description of the algorithm and notation.
    """

    @jax.jit
    def step(carry, k):
        # Unpack previous state and covariance
        m_prev, P_prev = carry

        # If k==0 we use the prior m0, Pinf and zero time-lag (dt=0)
        Delta = jax.lax.cond(k > 0,
                          lambda i: t_states[i] - t_states[i-1],
                          lambda _: 0.0,
                          k)
        n = obsid[k]
        
        # Get transition matrix
        A_prev = A_aug(0, Delta)
        Q_prev = Q_aug(0, Delta)

        # Predict step is same for exposure or gap
        m_pred = A_prev @ m_prev
        P_pred = A_prev @ P_prev @ A_prev.T + Q_prev 

        def get_ith_elems(i, X):
            return jax.tree.map(lambda x: x[i], X)

        # Update the end of the exposure
        def update_end():
            Hk = H_aug(get_ith_elems(n, X))
            y_pred = Hk @ m_pred            # predicted observation
            v_k = y[n] - y_pred             # "innovation" or "surprise" term
            S_k = Hk @ P_pred @ Hk.T + R[n] # uncertainy in predicted observation
            K_k = jnp.linalg.solve(S_k.T, (P_pred @ Hk.T).T).T # Kalman gain
            m_k = m_pred + K_k @ v_k          # conditioned state estimate
            P_k = P_pred - K_k @ S_k @ K_k.T  # conditioned covariance estimate
            return m_k, P_k, m_pred, P_pred, v_k, S_k

        # Update the start of the exposure, aka reset its z to zero
        def update_start():
            Reset = RESET(instid[n])
            m_k = Reset @ m_pred 
            P_k = Reset @ P_pred @ Reset.T
            Hk = H_aug(get_ith_elems(n, X))    # TODO: Hacky way to 
            v_k =  0 * (Hk @ m_pred)           # get v_k and S_k to 
            S_k = Hk @ P_pred @ Hk.T + jnp.inf # have correct shape
            return m_k, P_k, m_pred, P_pred, v_k, S_k
        
        m_k, P_k, m_pred, P_pred, v, S = jax.lax.cond(
                        stateid[k]==0,
                        lambda _: update_start(), # k=2,4,... is a t_e->t_s aka gap
                        lambda _: update_end(),   # k=1,3,... is a t_s->t_e aka exposure
                        operand=None              # note k=0 is -inf->t_s is also a 'gap' update
                    )  
        
        return (m_k, P_k), (m_k, P_k, m_pred, P_pred, v, S)
    
    # Initialize carry with prior state and covariance
    init_carry = (m0, P0)

    # Run the filter over all time steps, unpack, and return results
    _, outputs = jax.lax.scan(step, init_carry, jnp.arange(len(t_states)))
    m_filtered, P_filtered, m_predicted, P_predicted, v, S = outputs

    # only return v,S at exposure ends (where there is data)
    ends = jnp.argwhere(stateid == 1)

    return m_filtered, P_filtered, m_predicted, P_predicted, v[ends], S[ends]