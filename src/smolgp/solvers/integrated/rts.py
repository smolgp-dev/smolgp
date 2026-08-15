from __future__ import annotations

import jax
import jax.numpy as jnp

from smolgp.helpers import get_smoothing_gain


def IntegratedRTSSmoother(kernel, t_states, obsid, instid, stateid, kalman_results):
    """
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
    """

    # Model components
    A_aug = kernel.transition_matrix
    RESET = kernel.reset_matrix

    return integrated_rts_smoother(A_aug, RESET, t_states, obsid, instid, stateid, *kalman_results)


@jax.jit
def integrated_rts_smoother(
    A_aug,
    RESET,
    t_states,
    obsid,
    instid,
    stateid,
    m_filtered,
    P_filtered,
    m_predicted,
    P_predicted,
):
    """
    Jax implementation of the integrated RTS smoothing algorithm

    See Section 3.2.2 in Rubenzahl & Hattori et al. (in prep)
    for detailed description of the algorithm and notation.
    """

    def step(carry, k):
        # Outputs from Kalman filter, unpacked for notational consistency
        m_k = m_filtered[k]
        P_k = P_filtered[k]
        m_pred_next = m_predicted[k + 1]  # has superscript minus
        P_pred_next = P_predicted[k + 1]  # has superscript minus

        # Unpack state and covariance from last iteration
        m_hat_next, P_hat_next = carry

        # Compute smoothing gain
        Delta = t_states[k + 1] - t_states[k]
        A_k = A_aug(0, Delta)

        def smooth_start():
            """RTS smooth an exposure-start state"""

            m_k_pre = m_predicted[k]  # pre-reset start state
            P_k_pre = P_predicted[k]  # pre-reset start covariance

            Reset = RESET(instid[obsid[k]])
            AR = A_k @ Reset
            G_k = get_smoothing_gain(P_pred_next, P_k_pre @ AR.T)
            m_hat_k = m_k_pre + G_k @ (m_hat_next - m_pred_next)
            P_hat_k = P_k_pre + G_k @ (P_hat_next - P_pred_next) @ G_k.T
            return m_hat_k, P_hat_k

        def smooth_end():
            """RTS smooth an exposure-end state"""
            G_k = get_smoothing_gain(P_pred_next, P_k @ A_k.T)
            m_hat_k = m_k + G_k @ (m_hat_next - m_pred_next)
            P_hat_k = P_k + G_k @ (P_hat_next - P_pred_next) @ G_k.T
            return m_hat_k, P_hat_k

        m_hat_k, P_hat_k = jax.lax.cond(
            stateid[k] == 0,
            lambda _: smooth_start(),
            lambda _: smooth_end(),
            operand=None,
        )

        return (m_hat_k, P_hat_k), (m_hat_k, P_hat_k)

    # Start smoothing from final filtered state
    init_carry = (m_filtered[-1], P_filtered[-1])

    # Run backward from N-2 down to 0
    K = len(t_states)  # number of iterations
    _, outputs = jax.lax.scan(step, init_carry, jnp.arange(K - 2, -1, -1))
    m_smooth_reversed, P_smooth_reversed = outputs

    # Reverse outputs (with final filtered=smoothed state) to match time order
    m_smooth = jnp.vstack([m_smooth_reversed[::-1], m_filtered[-1][None, :]])
    P_smooth = jnp.vstack([P_smooth_reversed[::-1], P_filtered[-1][None, :, :]])
    return m_smooth, P_smooth


@jax.jit
def integrated_rts_gains(A_aug, RESET, t_states, obsid, instid, stateid, P_filtered, P_predicted):
    """
    The `y`-independent smoothing gains for the integrated smoother,
    mirroring smooth_start/smooth_end's gain formulas exactly (both use
    get_smoothing_gain, matching integrated_rts_smoother). Pointwise in k
    (jax.vmap, no scan needed).

    Returns:
        G_all: shape (K-1, dim, dim)
    """
    K = len(t_states)

    def gain_at_k(k):
        P_pred_next = P_predicted[k + 1]
        Delta = t_states[k + 1] - t_states[k]
        A_k = A_aug(0, Delta)

        def start_gain():
            P_k_pre = P_predicted[k]
            Reset = RESET(instid[obsid[k]])
            AR = A_k @ Reset
            return get_smoothing_gain(P_pred_next, P_k_pre @ AR.T)

        def end_gain():
            P_k = P_filtered[k]
            return get_smoothing_gain(P_pred_next, P_k @ A_k.T)

        return jax.lax.cond(
            stateid[k] == 0, lambda _: start_gain(), lambda _: end_gain(), operand=None
        )

    return jax.vmap(gain_at_k)(jnp.arange(K - 1))


@jax.jit
def integrated_rts_smoother_batched_mean(G_all, m_filtered_batch, m_predicted_batch, stateid):
    """
    Batched-mean-path replay of integrated_rts_smoother, given precomputed
    G_all (integrated_rts_gains) and the BATCHED filtered/predicted means
    (integrated_kalman_filter_batched_mean).

    `stateid` selects, per state, whether the mean carried forward is the
    pre-reset prediction (m_predicted_T[k], matching smooth_start's
    m_k_pre) or the filtered mean (m_filtered_T[k], matching smooth_end's
    m_k) -- Reset itself only enters via G_k (already baked into
    integrated_rts_gains), not applied again here.

    Parameters:
        G_all: (K-1, dim, dim)
        m_filtered_batch, m_predicted_batch: (M, K, dim)
        stateid: (K,)

    Returns:
        m_smoothed_batch: (M, K, dim)
    """
    K = m_filtered_batch.shape[1]
    m_filtered_T = jnp.moveaxis(m_filtered_batch, 0, 1)
    m_predicted_T = jnp.moveaxis(m_predicted_batch, 0, 1)

    def step(carry, k):
        m_hat_next_batch = carry
        m_pred_next_batch = m_predicted_T[k + 1]
        G_k = G_all[k]

        m_k_batch = jax.lax.cond(
            stateid[k] == 0,
            lambda _: m_predicted_T[k],
            lambda _: m_filtered_T[k],
            operand=None,
        )
        m_hat_k_batch = m_k_batch + jnp.einsum(
            "ij,mj->mi", G_k, m_hat_next_batch - m_pred_next_batch
        )
        return m_hat_k_batch, m_hat_k_batch

    init_carry = m_filtered_T[-1]
    _, m_smooth_reversed_T = jax.lax.scan(step, init_carry, jnp.arange(K - 2, -1, -1))
    m_smooth_T = jnp.concatenate(
        [m_smooth_reversed_T[::-1], m_filtered_T[-1][None, :, :]], axis=0
    )
    return jnp.moveaxis(m_smooth_T, 0, 1)
