from __future__ import annotations

import jax
import jax.numpy as jnp
from tinygp.helpers import JAXArray


def KalmanFilter(kernel, X, y, R, return_v_S=False):
    """
    Wrapper for jitted kalman_filter function

    Parameters:
        kernel: StateSpaceModel kernel
        X: data coordinates, e.g. time or (time, texp, instid)
        y: observations, shape (N, D)
        R: observation noise covariance, shape (N, D, D)

    Returns:
        m_filtered: filtered means
        P_filtered: filtered covariances
        m_predicted: predicted means
        P_predicted: predicted covariances
    """
    H = kernel.observation_model
    A = kernel.transition_matrix
    Q = kernel.process_noise
    m0 = jnp.zeros(kernel.dimension)
    P0 = kernel.stationary_covariance()
    if not isinstance(P0, JAXArray):
        P0 = P0.to_dense()  # needed for carry in jax.lax.scan

    t = kernel.coord_to_sortable(X)
    # Evaluate the observation model over the FULL X (which may be a tuple
    # like (t, texp, instid)), not the sortable-scalar timeline -- a kernel's
    # H is allowed to depend on any part of its coordinate, e.g. a per-output
    # amplitude selected by an id channel. Mirrors the integrated filter,
    # which has always done this. Passing t[k] here instead would silently
    # strip every non-sortable channel.
    H_all = jax.vmap(H)(X)
    output = kalman_filter(A, Q, H_all, R, t, y, m0, P0)
    if return_v_S:
        return output
    else:
        m_filtered, P_filtered, m_predicted, P_predicted, v, S = output
        return m_filtered, P_filtered, m_predicted, P_predicted


@jax.jit
def kalman_filter(A, Q, H_all, R, t, y, m0, P0):
    """
    Jax implementation of the Kalman filter algorithm

    See Theorem 4.2 (pdf page 77) in "Bayesian Filtering and Smoothing"
    by Simo S{\"a}rkk{\"a} for detailed description of the algorithm and notation.

    e.g. _prev is _{k-1}
         _pred is _k^{-}

    Total runtime complexity is O(N*d^3) where N is the number
    of time steps and d is the dimension of the state vector.
    """
    N = len(t)  # number of data points

    def step(carry, k):
        """
        Routine for a single step of the Kalman filter

        Parameters:
            carry: (x_prev, P_prev) - previous state and covariance
            k: index of the current time step

        Returns:
        - Conditioned state (m_k) and covariance (P_k) to carry to next iteration
        - Full output for completed scan (m_k, P_k, m_pred, P_pred)
        """

        # Unpack previous state and covariance
        m_prev, P_prev = carry

        # Logic to check if first time step:
        # If k==0 we use the prior x0, P0
        # and zero time-lag (Delta=0)
        Delta = jax.lax.cond(
            k > 0,
            lambda i: t[i] - t[i - 1],
            lambda _: 0.0,
            k,
        )

        # Get transition matrix
        A_prev = A(0, Delta)
        Q_prev = Q(0, Delta)

        # Predict (Eq. 4.20)
        m_pred = A_prev @ m_prev
        P_pred = A_prev @ P_prev @ A_prev.T + Q_prev

        # Update (Eq. 4.21)
        H_k = H_all[k]  # observation model for this time step
        y_pred = H_k @ m_pred  # predicted observation
        v_k = y[k] - y_pred  # "innovation" or "surprise" term
        S_k = H_k @ P_pred @ H_k.T + R[k]  # uncertainy in predicted observation
        # S_k_inv = jnp.linalg.inv(S_k)
        # K_k = P_pred @ H_k.T @ S_k_inv    # Kalman gain
        K_k = jnp.linalg.solve(S_k.T, (P_pred @ H_k.T).T).T  # more stable
        m_k = m_pred + K_k @ v_k  # conditioned state estimate
        P_k = P_pred - K_k @ S_k @ K_k.T  # conditioned covariance estimate

        return (m_k, P_k), (m_k, P_k, m_pred, P_pred, v_k, S_k)

    # Initialize carry with prior state and covariance
    init_carry = (m0, P0)

    # Run the filter over all time steps, unpack, and return results
    _, outputs = jax.lax.scan(step, init_carry, jnp.arange(N))
    return outputs


@jax.jit
def kalman_gains(A, H_all, R, t, P_predicted):
    """
    The `y`-independent per-step quantities needed to replay kalman_filter's
    mean-path recursion for many different observation vectors, given
    P_predicted from ONE prior call to kalman_filter (any `y` -- P_predicted
    never depends on it).

    Pointwise in k (jax.vmap, no scan needed): A_k, H_k, and the Kalman gain
    K_k (given P_predicted[k]) don't depend on any other step.

    Args:
        H_all: shape (N, D, dim) -- the observation model already evaluated
            over the full coordinates (``jax.vmap(kernel.observation_model)(X)``),
            NOT the bare function; see KalmanFilter's own note on why.

    Returns:
        A_all: shape (N, dim, dim) -- A(0, Delta_k)
        H_all: shape (N, D, dim)   -- passed through unchanged
        K_all: shape (N, dim, D)   -- Kalman gain at step k
    """
    N = len(t)

    def gains_at_k(k, P_pred_k):
        Delta = jax.lax.cond(k > 0, lambda i: t[i] - t[i - 1], lambda _: 0.0, k)
        A_k = A(0, Delta)
        H_k = H_all[k]
        S_k = H_k @ P_pred_k @ H_k.T + R[k]
        K_k = jnp.linalg.solve(S_k.T, (P_pred_k @ H_k.T).T).T
        return A_k, H_k, K_k

    return jax.vmap(gains_at_k)(jnp.arange(N), P_predicted)


@jax.jit
def kalman_filter_batched_mean(A_all, H_all, K_all, y_batch, m0):
    """
    Batched-mean-path replay of kalman_filter's forward recursion, given
    PRECOMPUTED, `y`-independent gains (kalman_gains). Processes M
    observation/residual batches in a single jax.lax.scan.

    Runtime: O(N * M * dim * D), vs. O(M * N * dim^3) for M independent
    calls to kalman_filter.

    Parameters:
        A_all: (N, dim, dim), H_all: (N, D, dim), K_all: (N, dim, D) -- from kalman_gains
        y_batch: (M, N, D) -- batch of M observation/residual arrays
        m0: (dim,) -- prior mean, shared/broadcast across the batch

    Returns:
        m_filtered_batch: (M, N, dim)
        m_predicted_batch: (M, N, dim)
    """
    N = A_all.shape[0]
    M = y_batch.shape[0]
    dim = m0.shape[0]

    def step(carry, k):
        m_prev_batch = carry  # (M, dim)
        A_k, H_k, K_k = A_all[k], H_all[k], K_all[k]

        m_pred_batch = jnp.einsum("ij,mj->mi", A_k, m_prev_batch)
        y_pred_batch = jnp.einsum("di,mi->md", H_k, m_pred_batch)
        v_batch = y_batch[:, k, :] - y_pred_batch
        m_k_batch = m_pred_batch + jnp.einsum("id,md->mi", K_k, v_batch)

        return m_k_batch, (m_k_batch, m_pred_batch)

    init_carry = jnp.broadcast_to(m0, (M, dim))
    _, (m_filtered_T, m_predicted_T) = jax.lax.scan(step, init_carry, jnp.arange(N))
    return jnp.moveaxis(m_filtered_T, 0, 1), jnp.moveaxis(m_predicted_T, 0, 1)
