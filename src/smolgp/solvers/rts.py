from __future__ import annotations

import jax
import jax.numpy as jnp

from smolgp.helpers import get_smoothing_gain


def RTSSmoother(kernel, X, kalman_results):
    """
    Wrapper for RTS smoother

    Parameters:
        kernel: StateSpaceModel kernel
        X: data coordinates, e.g. time or (time, texp, instid)
        kalman_results: output from Kalman filter (m_filtered, P_filtered, m_predicted, P_predicted)

    Returns:
        m_smooth: smoothed means
        P_smooth: smoothed covariances
    """
    A = kernel.transition_matrix
    t = kernel.coord_to_sortable(X)
    return rts_smoother(A, t, *kalman_results)


@jax.jit
def rts_smoother(A, t, m_filtered, P_filtered, m_predicted, P_predicted):
    """
    Jax implementation of the Rauch-Tung-Striebel (RTS) smoothing algorithm

    See Theorem 8.2 (pdf page 156) in "Bayesian Filtering and Smoothing"
    by Simo Särkkä for detailed description of the algorithm and notation.
    """
    N = len(t)  # number of data points

    def step(carry, k):
        """
        Routine for a single step of the RTS smoother

        Parameters:
            carry: (m_next, P_next) - next state and covariance
            k: index of the current time step

            Recall we are iterating backwards, so _next is k+1

        Returns:
        - Smoothed state (m_k_hat) and covariance (P_k_hat) to carry to next iteration
        - Full output for completed scan (m_k_hat, P_k_hat)
        """

        # Outputs from Kalman filter, unpacked for notational consistency
        m_k = m_filtered[k]
        P_k = P_filtered[k]
        m_pred_next = m_predicted[k + 1]  # has superscript minus
        P_pred_next = P_predicted[k + 1]  # has superscript minus

        # Unpack state and covariance from last iteration
        m_hat_next, P_hat_next = carry

        # Time-lag between states
        Delta_k = t[k + 1] - t[k]

        # Transition matrix
        A_k = A(0, Delta_k)

        # Compute smoothing gain
        # P_pred_next_inv = jnp.linalg.inv(P_pred_next)
        # G_k = P_k @ A_k.T @ P_pred_next_inv # smoothing gain
        G_k = jnp.linalg.solve(P_pred_next.T, (P_k @ A_k.T).T).T  # more stable

        # Update state and covariance
        m_hat_k = m_k + G_k @ (m_hat_next - m_pred_next)
        P_hat_k = P_k + G_k @ (P_hat_next - P_pred_next) @ G_k.T

        return (m_hat_k, P_hat_k), (m_hat_k, P_hat_k)

    # Start smoothing from final filtered state
    init_carry = (m_filtered[-1], P_filtered[-1])

    # Run backward from N-2 down to 0
    _, outputs = jax.lax.scan(step, init_carry, jnp.arange(N - 2, -1, -1))
    m_smooth_reversed, P_smooth_reversed = outputs

    # Reverse outputs to match time order
    m_smooth = jnp.vstack([m_smooth_reversed[::-1], m_filtered[-1][None, :]])
    P_smooth = jnp.vstack([P_smooth_reversed[::-1], P_filtered[-1][None, :, :]])
    return m_smooth, P_smooth


@jax.jit
def rts_gains(A, t, P_filtered, P_predicted):
    """
    The `y`-independent smoothing gains G_k, k=0..N-2, derived from
    P_filtered/P_predicted (from ONE prior kalman_filter call). Pointwise in
    k (jax.vmap, no scan needed).

    Uses helpers.get_smoothing_gain (rather than the raw jnp.linalg.solve
    rts_smoother inlines) so a degenerate/near-singular P_predicted[k+1] is
    handled the same robust way the integrated smoother already relies on --
    a strict superset of rts_smoother's own solve, since get_smoothing_gain
    falls back to its generic (plain solve) branch whenever P_predicted[k+1]
    is well-conditioned.

    Returns:
        G_all: shape (N-1, dim, dim)
    """
    N = len(t)

    def gain_at_k(k):
        P_k = P_filtered[k]
        P_pred_next = P_predicted[k + 1]
        Delta_k = t[k + 1] - t[k]
        A_k = A(0, Delta_k)
        return get_smoothing_gain(P_pred_next, P_k @ A_k.T)

    return jax.vmap(gain_at_k)(jnp.arange(N - 1))


@jax.jit
def rts_smoother_batched_mean(G_all, m_filtered_batch, m_predicted_batch):
    """
    Batched-mean-path replay of rts_smoother, given precomputed G_all
    (rts_gains) and the BATCHED filtered/predicted means
    (kalman_filter_batched_mean).

    Parameters:
        G_all: (N-1, dim, dim)
        m_filtered_batch, m_predicted_batch: (M, N, dim)

    Returns:
        m_smoothed_batch: (M, N, dim)
    """
    N = m_filtered_batch.shape[1]
    m_filtered_T = jnp.moveaxis(m_filtered_batch, 0, 1)  # (N, M, dim)
    m_predicted_T = jnp.moveaxis(m_predicted_batch, 0, 1)  # (N, M, dim)

    def step(carry, k):
        m_hat_next_batch = carry
        m_k_batch = m_filtered_T[k]
        m_pred_next_batch = m_predicted_T[k + 1]
        G_k = G_all[k]
        m_hat_k_batch = m_k_batch + jnp.einsum(
            "ij,mj->mi", G_k, m_hat_next_batch - m_pred_next_batch
        )
        return m_hat_k_batch, m_hat_k_batch

    init_carry = m_filtered_T[-1]  # (M, dim)
    _, m_smooth_reversed_T = jax.lax.scan(step, init_carry, jnp.arange(N - 2, -1, -1))
    m_smooth_T = jnp.concatenate(
        [m_smooth_reversed_T[::-1], m_filtered_T[-1][None, :, :]], axis=0
    )
    return jnp.moveaxis(m_smooth_T, 0, 1)  # (M, N, dim)
