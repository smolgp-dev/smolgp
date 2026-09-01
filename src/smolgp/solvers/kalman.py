from __future__ import annotations

import jax
import jax.numpy as jnp
from tinygp.helpers import JAXArray

from smolgp.helpers import kalman_gain, transition_sequence
from smolgp.solvers.base import log_prob_from_v_S


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
    A_all, Q_all = transition_sequence(A, Q, t)

    def step(carry, data):
        """
        Routine for a single step of the Kalman filter

        Parameters:
            carry: (x_prev, P_prev) - previous state and covariance
            data: (A_prev, Q_prev, H_k, R_k, y_k) for the current time step

        Returns:
        - Conditioned state (m_k) and covariance (P_k) to carry to next iteration
        - Full output for completed scan (m_k, P_k, m_pred, P_pred)
        """

        # Unpack previous state and covariance
        m_prev, P_prev = carry
        A_prev, Q_prev, H_k, R_k, y_k = data

        # Predict (Eq. 4.20)
        m_pred = A_prev @ m_prev
        P_pred = A_prev @ P_prev @ A_prev.T + Q_prev

        # Update (Eq. 4.21)
        y_pred = H_k @ m_pred  # predicted observation
        v_k = y_k - y_pred  # "innovation" or "surprise" term
        S_k = H_k @ P_pred @ H_k.T + R_k  # uncertainy in predicted observation
        K_k = kalman_gain(S_k, P_pred @ H_k.T)  # Kalman gain
        m_k = m_pred + K_k @ v_k  # conditioned state estimate
        P_k = P_pred - K_k @ S_k @ K_k.T  # conditioned covariance estimate

        return (m_k, P_k), (m_k, P_k, m_pred, P_pred, v_k, S_k)

    # Initialize carry with prior state and covariance
    init_carry = (m0, P0)

    # Run the filter over all time steps, unpack, and return results
    _, outputs = jax.lax.scan(step, init_carry, (A_all, Q_all, H_all, R, y))
    return outputs


def KalmanLoglike(kernel, X, y, R):
    """
    Wrapper for the jitted kalman_loglike function
    
    Same arguments as :func:`KalmanFilter`, but returns 
    just the marginal log likelihood.
    """
    A = kernel.transition_matrix
    Q = kernel.process_noise
    m0 = jnp.zeros(kernel.dimension)
    P0 = kernel.stationary_covariance()
    if not isinstance(P0, JAXArray):
        P0 = P0.to_dense()  # needed for carry in jax.lax.scan

    t = kernel.coord_to_sortable(X)
    H_all = jax.vmap(kernel.observation_model)(X)  # see KalmanFilter's note
    return kalman_loglike(A, Q, H_all, R, t, y, m0, P0)


@jax.jit
def kalman_loglike(A, Q, H_all, R, t, y, m0, P0):
    r"""
    Optimized function to return just the marginal log likelihood.
    The performance boost here comes from three optimisations:

    1. Replace `solve` with a simple division in 
        :func:`~smolgp.helpers.kalman_gain` when `D==1`.
    2. Prebuilds A(Δ) and Q(Δ) with vmap instead of computing them 
        on-the-fly inside the scan.
    3. Splitting: separate scans for the y-independant covariance and the 
        y-dependent mean, so each carries one array instead of (m, P). 

    #3 is only relevant for `d == 2` kernels (with nontrivial Q). At that size,
    the covariance half of the scan fits just under a code-generation threshold, 
    so splitting it from the mean scan (which is always small) is a ~10x speedup
    as both scans compile into short instructions whereas carrying both is over
    the threshold and compiles to ~7x more instructions per step.  d == 1 fits 
    as a single scan, so splitting is actually marginally slower (10%), and 
    d > 2 is over the threshold regardless, so splitting is irrelevant.
    """
    D = R.shape[-1]
    A_all, Q_all = transition_sequence(A, Q, t)

    def factor(carry, data):
        """Pass 1: the y-independent covariance recursion,
        accumulating the log-determinant."""
        P_prev, logdet = carry
        A_prev, Q_prev, H_k, R_k = data

        # Predict (Eq. 4.20)
        P_pred = A_prev @ P_prev @ A_prev.T + Q_prev

        # Update (Eq. 4.21), covariance path only
        PHt = P_pred @ H_k.T
        S_k = H_k @ PHt + R_k
        K_k = kalman_gain(S_k, PHt)
        # K S K^T = K (P H^T)^T
        P_k = P_pred - K_k @ PHt.T

        # For D == 1 the log-determinant is one log, so accumulate it here
        # rather than materializing it. For D > 1 it comes out of the batched
        # Cholesky in log_prob_from_v_S instead.
        if D == 1:
            logdet = logdet + jnp.log(S_k[0, 0])
        return (P_k, logdet), (K_k, S_k)

    (_, logdet), (K_all, S_all) = jax.lax.scan(
        factor, (P0, jnp.zeros((), dtype=jnp.result_type(P0))), (A_all, Q_all, H_all, R)
    )

    if D == 1:
        def solve(carry, data):
            """Pass 2: the y-dependent mean recursion, 
            accumulating the quadratic form."""
            m_prev, quad = carry
            A_prev, H_k, K_k, S_k, y_k = data
            m_pred = A_prev @ m_prev
            v_k = y_k - H_k @ m_pred
            m_k = m_pred + K_k @ v_k
            return (m_k, quad + v_k[0] * v_k[0] / S_k[0, 0]), None

        (_, quad), _ = jax.lax.scan(
            solve,
            (m0, jnp.zeros((), dtype=jnp.result_type(P0))),
            (A_all, H_all, K_all, S_all, y),
        )
        # The D*log(2*pi) term is per-step but constant, so it is summed once here.
        loglike = -0.5 * (quad + logdet + len(t) * jnp.log(2.0 * jnp.pi))
        return jnp.where(jnp.isfinite(loglike), loglike, -jnp.inf)

    def solve(carry, data):
        """Pass 2 for D > 1: emit the innovations for one batched reduction."""
        m_prev = carry
        A_prev, H_k, K_k, y_k = data
        m_pred = A_prev @ m_prev
        v_k = y_k - H_k @ m_pred
        return m_pred + K_k @ v_k, v_k

    _, v_all = jax.lax.scan(solve, m0, (A_all, H_all, K_all, y))
    return log_prob_from_v_S(v_all, S_all)


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
        K_k = kalman_gain(S_k, P_pred_k @ H_k.T)
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
