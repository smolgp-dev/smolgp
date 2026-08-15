from __future__ import annotations

import jax
import jax.numpy as jnp


def IntegratedKalmanFilter(
    kernel, X, y, t_states, obsid, instid, stateid, R, return_v_S=False
):
    """
    Wrapper for integrated_kalman_filter function

    Parameters:
        kernel  : IntegratedStateSpaceModel kernel
        X       : Array of size N, data coordinates (e.g. (time, texp, instid))
        y       : Array of size (N, D), measurements at the data coordinates
        t_states: Array of size K, sorted time coordinate of all states (exposure starts and ends)
        obsid   : Array of size N, which observation (0,...,N-1) is being made at each state k
        instid  : Array of size N, which instrument (0,...,Ninst-1) recorded observation n
        stateid : Array of size K, 0 for exposure-start, 1 for exposure-end
        R       : Observation noise covariance, shape (N, D, D)
        return_v_S : Whether to return innovation and its covariance (for likelihood computation)

    Returns:
        m_filtered : filtered means
        P_filtered : filtered covariances
        m_predicted: predicted means
        P_predicted: predicted covariances
    """

    # Model components
    H_aug = kernel.observation_model
    A_aug = kernel.transition_matrix
    Q_aug = kernel.process_noise
    RESET = kernel.reset_matrix

    # Initial state and covariance
    # mean = jnp.zeros(kernel.d) # TODO: mean function of base kernel
    # m0 = jnp.block([mean] + kernel.num_insts*[jnp.zeros(kernel.d)])
    m0 = jnp.zeros(kernel.dimension)
    P0 = kernel.stationary_covariance()

    output = integrated_kalman_filter(
        A_aug, Q_aug, H_aug, R, RESET, X, y, t_states, obsid, instid, stateid, m0, P0
    )
    if return_v_S:
        return output
    else:
        m_filtered, P_filtered, m_predicted, P_predicted, v, S = output
        return m_filtered, P_filtered, m_predicted, P_predicted


@jax.jit
def integrated_kalman_filter(
    A_aug, Q_aug, H_aug, R, RESET, X, y, t_states, obsid, instid, stateid, m0, P0
):
    """
    Jax implementation of the integrated Kalman filter algorithm

    See Section 3.2.1 in Rubenzahl & Hattori et al. (in prep)
    for detailed description of the algorithm and notation.
    """

    H = jax.vmap(H_aug)(X)

    @jax.jit
    def step(carry, k):
        # Unpack previous state and covariance
        m_prev, P_prev = carry

        # If k==0 we use the prior m0, Pinf and zero time-lag (dt=0)
        Delta = jax.lax.cond(
            k > 0, lambda i: t_states[i] - t_states[i - 1], lambda _: 0.0, k
        )
        n = obsid[k]

        # Get transition matrix
        A_prev = A_aug(0, Delta)
        Q_prev = Q_aug(0, Delta)

        # Predict step is same
        m_pred = A_prev @ m_prev
        P_pred = A_prev @ P_prev @ A_prev.T + Q_prev

        # Update the end of the exposure
        def update_end():
            Hk = H[n]
            y_pred = Hk @ m_pred  # predicted observation
            v_k = y[n] - y_pred  # "innovation" or "surprise" term
            S_k = Hk @ P_pred @ Hk.T + R[n]  # uncertainy in predicted observation
            K_k = jnp.linalg.solve(S_k.T, (P_pred @ Hk.T).T).T  # Kalman gain
            m_k = m_pred + K_k @ v_k  # conditioned state estimate
            P_k = P_pred - K_k @ S_k @ K_k.T  # conditioned covariance estimate
            return m_k, P_k, m_pred, P_pred, v_k, S_k

        # Update the start of the exposure, aka reset its z to zero
        def update_start():
            Reset = RESET(instid[n])
            m_k = Reset @ m_pred
            P_k = Reset @ P_pred @ Reset.T
            Hk = H[n]  # TODO: change this and next two lines to use shapes?
            v_k = jnp.zeros_like(Hk @ m_pred)  # maybe e.g. jax broadcast_shapes?
            S_k = jnp.zeros_like(Hk @ P_pred @ Hk.T)
            return m_k, P_k, m_pred, P_pred, v_k, S_k

        m_k, P_k, m_pred, P_pred, v_k, S_k = jax.lax.cond(
            stateid[k] == 0,
            lambda _: update_start(),
            lambda _: update_end(),
            operand=None,
        )

        return (m_k, P_k), (m_k, P_k, m_pred, P_pred, v_k, S_k)

    # Initialize carry with prior state and covariance
    init_carry = (m0, P0)

    # Run the filter over all time steps, unpack, and return results
    _, outputs = jax.lax.scan(step, init_carry, jnp.arange(len(t_states)))
    m_filtered, P_filtered, m_predicted, P_predicted, v, S = outputs

    # only return v,S at exposure ends (where there is data)
    ends_idx = jnp.nonzero(stateid == 1, size=y.shape[0])[0]
    v_sel = jnp.take(v, ends_idx, axis=0)
    S_sel = jnp.take(S, ends_idx, axis=0)

    return m_filtered, P_filtered, m_predicted, P_predicted, v_sel, S_sel


@jax.jit
def integrated_kalman_gains(A_aug, H_aug, RESET, R, X, t_states, obsid, instid, stateid, P_predicted):
    """
    The `y`-independent per-state quantities needed to replay
    integrated_kalman_filter's mean-path recursion for many different
    observation/residual arrays, given P_predicted from ONE prior call to
    integrated_kalman_filter (any y -- P_predicted never depends on it).

    Computes BOTH the Kalman-gain ingredients (meaningful at exposure-end
    states) and the Reset matrix (meaningful at exposure-start states) at
    every state k unconditionally (cheap; no scan needed) -- the
    batched-mean recursion selects the right one per state via the same
    stateid==0 jax.lax.cond dispatch integrated_kalman_filter uses.

    Returns:
        A_all: shape (K, dim, dim)     -- A_aug(0, Delta_k)
        H_all: shape (N, D, dim)       -- H_aug(X), i.e. jax.vmap(H_aug)(X)
        K_all: shape (K, dim, D)       -- Kalman gain (meaningful at end states)
        Reset_all: shape (K, dim, dim) -- RESET(instid[obsid[k]]) (meaningful at start states)
    """
    K = len(t_states)
    H_all = jax.vmap(H_aug)(X)

    def gains_at_k(k):
        Delta = jax.lax.cond(
            k > 0, lambda i: t_states[i] - t_states[i - 1], lambda _: 0.0, k
        )
        A_k = A_aug(0, Delta)
        n = obsid[k]
        Hk = H_all[n]
        P_pred_k = P_predicted[k]
        S_k = Hk @ P_pred_k @ Hk.T + R[n]
        K_k = jnp.linalg.solve(S_k.T, (P_pred_k @ Hk.T).T).T
        Reset_k = RESET(instid[n])
        return A_k, K_k, Reset_k

    A_all, K_all, Reset_all = jax.vmap(gains_at_k)(jnp.arange(K))
    return A_all, H_all, K_all, Reset_all


@jax.jit
def integrated_kalman_filter_batched_mean(
    A_all, H_all, K_all, Reset_all, obsid, stateid, y_batch, m0
):
    """
    Batched-mean-path replay of integrated_kalman_filter, using PRECOMPUTED
    (integrated_kalman_gains) A_all/H_all/K_all/Reset_all.

    The exposure-start "reset" update is a pure per-sample-batched einsum
    with no y/data dependence at all (the innovation is always zero at a
    start state) -- `jnp.einsum("ij,mj->mi", Reset_k, m_pred_batch)`. The
    stateid==0 dispatch itself (which state gets which update) is
    data-independent (known from t_states/obsid alone), so the same
    jax.lax.cond structure as integrated_kalman_filter applies unchanged,
    just wrapping batched einsums instead of unbatched matrix-vector `@`.

    Parameters:
        A_all: (K, dim, dim), H_all: (N, D, dim), K_all: (K, dim, D),
        Reset_all: (K, dim, dim) -- all from integrated_kalman_gains
        obsid, stateid: (K,) bookkeeping arrays (same as integrated_kalman_filter)
        y_batch: (M, N, D) -- batch of M residual/observation arrays
        m0: (dim,)

    Returns:
        m_filtered_batch, m_predicted_batch: (M, K, dim)
    """
    K = A_all.shape[0]
    M = y_batch.shape[0]
    dim = m0.shape[0]

    def step(carry, k):
        m_prev_batch = carry
        A_k = A_all[k]
        m_pred_batch = jnp.einsum("ij,mj->mi", A_k, m_prev_batch)
        n = obsid[k]

        def update_end_batch():
            Hk, K_k = H_all[n], K_all[k]
            y_pred_batch = jnp.einsum("di,mi->md", Hk, m_pred_batch)
            v_batch = y_batch[:, n, :] - y_pred_batch
            return m_pred_batch + jnp.einsum("id,md->mi", K_k, v_batch)

        def update_start_batch():
            Reset_k = Reset_all[k]
            return jnp.einsum("ij,mj->mi", Reset_k, m_pred_batch)

        m_k_batch = jax.lax.cond(
            stateid[k] == 0,
            lambda _: update_start_batch(),
            lambda _: update_end_batch(),
            operand=None,
        )
        return m_k_batch, (m_k_batch, m_pred_batch)

    init_carry = jnp.broadcast_to(m0, (M, dim))
    _, (m_filtered_T, m_predicted_T) = jax.lax.scan(step, init_carry, jnp.arange(K))
    return jnp.moveaxis(m_filtered_T, 0, 1), jnp.moveaxis(m_predicted_T, 0, 1)
