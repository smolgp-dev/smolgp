from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = ["IntegratedKalmanFilter", "integrated_kalman_filter"]


def IntegratedKalmanFilter(kernel, X, y, noise, obsid, stateid, return_v_S=False):
    '''
    Wrapper for jitted integrated_kalman_filter function

    Parameters:
        kernel: IntegratedStateSpaceModel kernel
        X: Data coordinates with the following structure: (t, texp, instid), where
            t      : time coordinate
            texp   : exposure time (same unit as t)
            instid : instrument id (integer)
        y: Measurements at the data coordinates
        noise: Noise model
        obsid: Array of size N, which observation (0,...,N-1) is being made at each state k
        instids: Array of size N, which instrument (0,...,Ninst-1) recorded observation n
        stateid: Array of size K, 0 for exposure-start, 1 for exposure-end

    Returns:
        m_filtered: filtered means
        P_filtered: filtered covariances
        m_predicted: predicted means
        P_predicted: predicted covariances
    '''

    # Unpack data coordinates
    t, texp, instid = X

    # Model components
    H_aug = kernel.observation_model
    A_aug = kernel.transition_matrix
    Q_aug = kernel.process_noise
    RESET = kernel.reset_matrix
    R = noise.diagonal() if noise is not None else jnp.zeros_like(y)
    
    # Initial state and covariance
    mean = jnp.zeros(kernel.d) # TODO: mean function of base kernel
    m0 = jnp.block([mean] + kernel.num_inst*[jnp.zeros(kernel.d)])
    P0 = kernel.stationary_covariance()

    output = integrated_kalman_filter(A_aug, Q_aug, H_aug,
                                      R, t, y, texp, obsid, instid, stateid,
                                      m0, P0)
    if return_v_S:
        return output
    else:
        m_filtered, P_filtered, m_predicted, P_predicted, v, S = output
        return m_filtered, P_filtered, m_predicted, P_predicted

@jax.jit
def integrated_kalman_filter(A_aug, Q_aug, H_aug, RESET,
                             R, t, y, texp, obsid, instid, stateid, 
                             m0, P0):
    """
    Jax implementation of the integrated Kalman filter algorithm

    See Section 3.2.1 in Rubenzahl & Hattori et al. (in prep) 
    for detailed description of the algorithm and notation.
    """
    
    K = len(t)    # number of iterations (2N)
    N = int(K/2)

    if len(H) != N:
        H = jnp.full((N, *H.shape), H)  # case of constant observation model

    def step(carry, k):
        '''
        '''

        # Unpack previous state and covariance
        m_prev, P_prev = carry

        # If k==0 we use the prior m0, Pinf and zero time-lag (dt=0)
        Delta = jax.lax.cond(k > 0,
                          lambda i: t[i] - t[i-1],
                          lambda _: 0.0,
                          k)
        n = obsid[k]        # which observation are we working on
        instid = instid[n] # which instrument this observation is from
        
        # Get transition matrix
        A_prev = A_aug(Delta)
        Q_prev = Q_aug(Delta)

        # Predict step is same for exposure or gap
        m_pred = A_prev @ m_prev
        P_pred = A_prev @ P_prev @ A_prev.T + Q_prev 

        # Update the end of the exposure
        def update_end():

            y_pred = H_aug @ m_pred             # predicted observation
            v_k = y[n] - y_pred               # "innovation" or "surprise" term
            S_k = H_aug @ P_pred @ H_aug.T + R[n] # uncertainy in predicted observation
            K_k = jnp.linalg.solve(S_k.T, (P_pred @ H_aug.T).T).T # Kalman gain
            m_k = m_pred + K_k @ v_k          # conditioned state estimate
            P_k = P_pred - K_k @ S_k @ K_k.T  # conditioned covariance estimate
            return m_k, P_k, m_pred, P_pred

        # Update the start of the exposure, aka reset its z to zero
        def update_start():
            Reset = RESET(instid)
            m_k = Reset @ m_pred 
            P_k = Reset @ P_pred @ Reset.T
            return m_k, P_k, m_pred, P_pred
        
        m_k, P_k, m_pred, P_pred = jax.lax.cond(
                        stateid[k]==0,
                        lambda _: update_start(), # k=2,4,... is a t_e->t_s aka gap
                        lambda _: update_end(),   # k=1,3,... is a t_s->t_e aka exposure
                        operand=None              # note k=0 is -inf->t_s is also a 'gap' update
                    )  
        
        return (m_k, P_k), (m_k, P_k, m_pred, P_pred)
    
    # Initialize carry with prior state and covariance
    init_carry = (m0, P0)

    # Run the filter over all time steps, unpack, and return results
    _, outputs = jax.lax.scan(step, init_carry, jnp.arange(K))
    m_filtered, P_filtered, m_predicted, P_predicted = outputs
    return m_filtered, P_filtered, m_predicted, P_predicted