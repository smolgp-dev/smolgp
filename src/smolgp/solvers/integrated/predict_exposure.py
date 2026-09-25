from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from smolgp.helpers import smoothing_gain
from smolgp.kernels.base import Sum, Wrapper, extract_all_components
from smolgp.kernels.integrated import IntegratedStateSpaceModel


def _extend_kernel(kernel):
    """Add one virtual instrument slot to every integrated leaf of ``kernel``.

    Returns ``(kernel_ext, E, zslots)``:

    - ``kernel_ext``: the kernel with ``num_insts + 1`` on every integrated leaf.
    - ``E``: the ``(n_ext, n)`` 0/1 embedding of the base state into the
      extended state, so ``m_ext = E @ m``. Its columns are orthonormal.
    - ``zslots``: for each integrated leaf, ``(z0, virtual)``: the base-state
      index of that leaf's first integral state, and the extended-state index
      of its virtual (test exposure) integral state.
    """
    if isinstance(kernel, Sum):
        k1, E1, z1 = _extend_kernel(kernel.kernel1)
        k2, E2, z2 = _extend_kernel(kernel.kernel2)
        n1, n1_ext = E1.shape[1], E1.shape[0]
        E = np.zeros((E1.shape[0] + E2.shape[0], E1.shape[1] + E2.shape[1]))
        E[:n1_ext, :n1] = E1
        E[n1_ext:, n1:] = E2
        z2 = [(z0 + n1, v + n1_ext) for z0, v in z2]
        return Sum(k1, k2), E, z1 + z2
    if isinstance(kernel, Wrapper):
        inner, E, zslots = _extend_kernel(kernel.kernel)
        if not zslots:
            return kernel, E, zslots
        updated = dataclasses.replace(kernel, kernel=inner)
        if updated.kernel is not inner:
            raise NotImplementedError(
                f"Cannot add an exposure slot inside a {type(kernel).__name__} wrapper"
            )
        return updated, E, zslots
    if isinstance(kernel, IntegratedStateSpaceModel):
        n = kernel.dimension
        E = np.eye(n + 1, n)  # virtual slot is the new last state
        kernel_ext = dataclasses.replace(kernel, num_insts=kernel.num_insts + 1)
        return kernel_ext, E, [(kernel.d, n)]
    # Instantaneous leaves (and Products, which cannot contain integrated kernels)
    return kernel, np.eye(kernel.dimension), []


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
    instantaneous queries. Every real state is the smoothed state at the end
    of the test exposure, except that each integrated component's integral
    state for ``instid_star`` holds that component's integral over the test
    exposure. The ``kernel.observation_model`` is applied afterward in
    GaussianProcess.predict(), which gives the total and per-component
    exposure averages (instantaneous components of a mixed Sum are read at
    the end of the exposure, as in training; see the Sum docstring).

    Works for any kernel tree (e.g. a Sum of integrated and instantaneous
    kernels): each integrated leaf gets its own virtual integral state, mapped
    into place by an embedding matrix.

    The algorithm mirrors the instantaneous predict algorithm (Algorithm 1 in Rubenzahl
    & Hattori et al. 2026) but includes replaying the Kalman steps for any data points
    that overlap with the test exposure. A virtual extra instrument index is used to
    hold the test exposure's state, which is reset at the start of the exposure.

    1. Treat the test exposure as a new, *unobserved* measurement on a
       virtual extra instrument index ``num_insts`` (one past the real
       ones), by building ``kernel_ext`` with ``num_insts + 1`` on every
       integrated component. Let the
       test exposure span the interval :math:`[a, b) = [t_* - \delta_*/2, t_* + \delta_*/2)`.
    2. **Phase A**: Transition from the filtered data point (or the prior,
       if retrodictive) immedietely before the test exposure start to the
       test state :math:`a`, then apply ``kernel_ext.reset_matrix`` to zero
       the virtual instrument there.
    3. **Phase B**: loop over every real state inside :math:`[a, b)`
       and replay the Kalman filter predict/reset/update steps. This correctly
       updates the etst prediction with overlapping real observations.
    4. **Phase C**: one final predict-only transition from wherever Phase B
       left off to :math:`b`.
    5. **Phase D**: RTS-smooth the result against the nearest future real
       state on or after :math:`b`. This is skipped if the test point ends
       after all observed data.

    Because the test point is computed on a fully private index throughout,
    ``instid_star`` colliding with a real training instrument's id is harmless.
    ``instid_star`` is only used to choose which integral state in the returned
    ``(n,)``/``(n, n)`` arrays holds the test exposure, so that the GP applies
    the observation model for the correct instrument.
    """
    t_states, instid, obsid, stateid = (
        state_coords.t_states,
        state_coords.instid,
        state_coords.obsid,
        state_coords.stateid,
    )
    (m_predicted, P_predicted), (m_filtered, P_filtered), (m_smoothed, P_smoothed) = (
        conditioned_states
    )
    K = t_states.shape[0]
    n = kernel.dimension
    kernel_ext, E, zslots = _extend_kernel(kernel)
    E = jnp.asarray(E)
    Pr = E @ E.T  # projector onto the real (non-virtual) states
    # The virtual instrument index, one past the real ones
    num_insts = next(
        k.num_insts
        for k in extract_all_components(kernel)
        if isinstance(k, IntegratedStateSpaceModel)
    )

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

    m_anchor_ext = E @ m_anchor
    P_anchor_ext = E @ P_anchor @ E.T

    dt_a = a - t_anchor
    A1 = kernel_ext.transition_matrix(0, dt_a)
    Q1 = kernel_ext.process_noise(0, dt_a)
    m_at_a_pred = A1 @ m_anchor_ext
    P_at_a_pred = A1 @ P_anchor_ext @ A1.T + Q1

    Reset0 = kernel_ext.reset_matrix(num_insts)
    m_init = Reset0 @ m_at_a_pred
    P_init = Reset0 @ P_at_a_pred @ Reset0.T

    # ---- Phase B: walk through the real states inside [a, b) ----
    # H for the extended kernel; real observations never touch the virtual
    # integral states, since their instids are all < num_insts.
    H_all_ext = jax.vmap(kernel_ext.observation_model)(X)

    # A while_loop over j in [k_a, k_b) only, rather than a masked scan over all
    # K states: under vmap it runs as many iterations as the fullest window in
    # the batch (typically a handful), not K per test point.
    def step(carry):
        j, m_carry, P_carry, t_ref = carry
        t_from = jnp.where(j == k_a, a, t_ref)
        dt_j = t_states[j] - t_from

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

        # Snap the real (non-probe) block to the already-validated arrays
        # A no-op in exact arithmetic, and a guard against any drift.
        m_k = m_k - Pr @ m_k + E @ m_filtered[j]
        P_k = P_k - Pr @ P_k @ Pr + E @ P_filtered[j] @ E.T

        return j + 1, m_k, P_k, t_states[j]

    _, m_walk, P_walk, t_ref = jax.lax.while_loop(
        lambda carry: carry[0] < k_b, step, (k_a, m_init, P_init, a)
    )

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
    A_rect = A_real @ E.T
    numerator = P_star_pred @ A_rect.T
    G_k = smoothing_gain(P_predicted[idx_next], numerator)
    m_smooth_res = m_star_pred + G_k @ (m_smoothed[idx_next] - m_predicted[idx_next])
    P_smooth_res = (
        P_star_pred + G_k @ (P_smoothed[idx_next] - P_predicted[idx_next]) @ G_k.T
    )

    is_extrapolate = k_b >= K
    m_final = jnp.where(is_extrapolate, m_star_pred, m_smooth_res)
    P_final = jnp.where(is_extrapolate, P_star_pred, P_smooth_res)

    # ---- Readout: map back to the (n,)-dim base state, with each integrated
    # component's instid_star integral state replaced by its virtual one ----
    S = E.T
    for z0, virtual in zslots:
        row = z0 + instid_star
        S = S.at[row].set(0.0).at[row, virtual].set(1.0)
    m_out = S @ m_final
    P_out = S @ P_final @ S.T
    return m_out, P_out
