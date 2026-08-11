from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from smolgp.helpers import get_smoothing_gain


def predict_exposure(
    kernel,
    X,
    y,
    R,
    state_coords,
    conditioned_states,
    t_star: float,
    delta_star: float,
    instid_star: int,
):
    r"""
    Predict the exposure-integrated posterior for a single out-of-sample test point
    :math:`(t_*, \delta_*, \mathrm{instid}_*)` with :math:`\delta_* > 0`.

    Returns the raw, unprojected augmented state (mean of shape ``(n,)`` and
    covariance of shape ``(n, n)``, where ``n = kernel.dimension``), matching
    the signature of :meth:`IntegratedStateSpaceSolver.predict` for
    instantaneous queries. That is, the returned result has the test point's
    exposure-integrated mean/variance staged at state index ``kernel.d + instid_star``.
    The ``kernel.observation_model`` is applied afterward in GaussianProcess.predict().

    The algorithm mirrors the instantaneous predict algorithm (Algorithm 1 in Rubenzahl
    & Hattori et al. 2026) but includes replaying the Kalman steps for any data points
    that overlap with the test exposure. A virtual extra instrument index is used to
    hold the test exposure's state, which is reset at the start of the exposure.

    1. Treat the test exposure as a new, *unobserved* measurement on a
       virtual extra instrument index ``num_insts`` (one past the real
       ones), by building ``kernel_ext`` with ``num_insts + 1``. Let the
       test exposure span the interval :math:`[a, b) = [t_* - \delta_*/2, t_* + \delta_*/2)`.
    2. **Phase A**: Transition from the filtered data point (or the prior,
       if retrodictive) immedietely before the test exposure start to the
       test state :math:`a`, then apply ``kernel_ext.reset_matrix`` to zero
       the virtual instrument there.
    3. **Phase B**: scan over every real state strictly inside :math:`[a, b)`
       and replay the Kalman filter predict/reset/update steps. This correctly
       updates the etst prediction with overlapping real observations.
    4. **Phase C**: one final predict-only transition from wherever Phase B
       left off to :math:`b`.
    5. **Phase D**: RTS-smooth the result against the nearest future real
       state on or after :math:`b`. This is skipped if the test point ends
       after all observed data.

    Because the test point is computed on a fully private index throughout,
    ``instid_star`` colliding with a real training instrument's id is harmless.
    ``instid_star`` is only used to choose where in the returned ``(n,)``/``(n, n)``
    arrays to stage the final probe mean/variance, so that the GP applies the
    observation model for the correct instrument.
    """
    t_states, instid, obsid, stateid = state_coords
    (m_predicted, P_predicted), (m_filtered, P_filtered), (m_smoothed, P_smoothed) = (
        conditioned_states
    )
    K = t_states.shape[0]
    n = kernel.dimension
    kernel_ext = dataclasses.replace(kernel, num_insts=kernel.num_insts + 1)
    n_ext = n + 1
    probe_idx = n

    Pinf = kernel.stationary_covariance()
    m0 = jnp.zeros(n)

    a = t_star - delta_star / 2
    b = t_star + delta_star / 2

    k_a = jnp.searchsorted(t_states, a, side="right")
    k_b = jnp.searchsorted(t_states, b, side="right")

    # ---- Phase A: filtered state immediately before "a", extended + reset ----
    idx_anchor = jnp.clip(k_a - 1, 0, K - 1)
    use_prior = k_a <= 0
    m_anchor = jnp.where(use_prior, m0, m_filtered[idx_anchor])
    P_anchor = jnp.where(use_prior, Pinf, P_filtered[idx_anchor])
    # When using the prior, set t_anchor=a so the hop below is a Delta=0 no-op
    # (the prior m0/Pinf is the stationary distribution, valid at any time).
    t_anchor = jnp.where(use_prior, a, t_states[idx_anchor])

    m_anchor_ext = jnp.concatenate([m_anchor, jnp.zeros(1)])
    P_anchor_ext = jnp.zeros((n_ext, n_ext)).at[:n, :n].set(P_anchor)

    dt_a = a - t_anchor
    A1 = kernel_ext.transition_matrix(0, dt_a)
    Q1 = kernel_ext.process_noise(0, dt_a)
    m_at_a_pred = A1 @ m_anchor_ext
    P_at_a_pred = A1 @ P_anchor_ext @ A1.T + Q1

    Reset0 = kernel_ext.reset_matrix(probe_idx)
    m_init = Reset0 @ m_at_a_pred
    P_init = Reset0 @ P_at_a_pred @ Reset0.T

    # ---- Phase B: masked walk through any real states inside [a, b) ----
    # H, padded with one zero column for the probe (which never appears in
    # any real observation).
    H_all_ext = jnp.pad(jax.vmap(kernel.observation_model)(X), ((0, 0), (0, 0), (0, 1)))

    def step(carry, j):
        m_carry, P_carry, t_ref = carry
        is_active = (j >= k_a) & (j < k_b)
        is_first = j == k_a
        t_from = jnp.where(is_first, a, t_ref)
        dt_j = jnp.where(is_active, t_states[j] - t_from, 0.0)

        Aj = kernel_ext.transition_matrix(0, dt_j)
        Qj = kernel_ext.process_noise(0, dt_j)
        m_p = Aj @ m_carry
        P_p = Aj @ P_carry @ Aj.T + Qj

        n_obs = obsid[j]

        def do_start(_):
            Reset_j = kernel_ext.reset_matrix(instid[n_obs])
            return Reset_j @ m_p, Reset_j @ P_p @ Reset_j.T

        def do_end(_):
            Hk = H_all_ext[n_obs]
            v_k = y[n_obs] - Hk @ m_p
            S_k = Hk @ P_p @ Hk.T + R[n_obs]
            K_k = jnp.linalg.solve(S_k.T, (P_p @ Hk.T).T).T
            return m_p + K_k @ v_k, P_p - K_k @ S_k @ K_k.T

        m_k, P_k = jax.lax.cond(stateid[j] == 0, do_start, do_end, operand=None)

        # Snap the real (non-probe) block to the already-validated arrays --
        # a no-op in exact arithmetic, and a guard against any drift.
        m_k = m_k.at[:n].set(m_filtered[j])
        P_k = P_k.at[:n, :n].set(P_filtered[j])

        new_m = jnp.where(is_active, m_k, m_carry)
        new_P = jnp.where(is_active, P_k, P_carry)
        new_t_ref = jnp.where(is_active, t_states[j], t_ref)
        return (new_m, new_P, new_t_ref), None

    (m_walk, P_walk, t_ref), _ = jax.lax.scan(step, (m_init, P_init, a), jnp.arange(K))

    # ---- Phase C: close the window, hop from t_ref to b (predict-only) ----
    dt_b = b - t_ref
    Ac = kernel_ext.transition_matrix(0, dt_b)
    Qc = kernel_ext.process_noise(0, dt_b)
    m_star_pred = Ac @ m_walk
    P_star_pred = Ac @ P_walk @ Ac.T + Qc

    # ---- Phase D: RTS-smooth against the nearest future real state ----
    idx_next = jnp.clip(k_b, 0, K - 1)
    dt_next = t_states[idx_next] - b
    A_real = kernel.transition_matrix(0, dt_next)
    A_rect = jnp.zeros((n, n_ext)).at[:, :n].set(A_real)
    numerator = P_star_pred @ A_rect.T
    G_k = get_smoothing_gain(P_predicted[idx_next], numerator)
    m_smooth_res = m_star_pred + G_k @ (m_smoothed[idx_next] - m_predicted[idx_next])
    P_smooth_res = (
        P_star_pred + G_k @ (P_smoothed[idx_next] - P_predicted[idx_next]) @ G_k.T
    )

    is_extrapolate = k_b >= K
    m_final = jnp.where(is_extrapolate, m_star_pred, m_smooth_res)
    P_final = jnp.where(is_extrapolate, P_star_pred, P_smooth_res)

    # ---- Readout: stage the probe result into the (n,)-dim base state ----
    z_mean = m_final[probe_idx]
    z_var = P_final[probe_idx, probe_idx]
    slot = kernel.d + instid_star
    m_out = jnp.zeros(n).at[slot].set(z_mean)
    P_out = jnp.zeros((n, n)).at[slot, slot].set(z_var)
    return m_out, P_out
