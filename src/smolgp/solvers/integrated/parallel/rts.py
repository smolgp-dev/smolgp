from __future__ import annotations

import jax
import jax.numpy as jnp


def ParallelIntegratedRTSSmoother(
    kernel,
    t_states,
    stateid,
    instid,
    kalman_results,
):
    """
    Wrapper for Parallel RTS smoother

    Parameters:
        kernel: StateSpaceModel kernel
        X: input coordinates
        kalman_results: output from Kalman filter:
            m_pred, P_pred, m_filter, P_filter

    Returns:
        E:
        g: smoothed means
        L: smoothed covariances
    """
    m_pred, P_pred, m_filter, P_filter = kalman_results
    Phi_aug = kernel.transition_matrix
    Q_aug = kernel.process_noise
    RESET = kernel.reset_matrix

    asso_params = make_associative_params(
        Phi_aug,
        Q_aug,
        RESET,
        t_states,
        stateid,
        instid,
        m_pred,
        P_pred,
        m_filter,
        P_filter,
    )
    E, g, L = parallel_integrated_rts_smoother(asso_params)
    return (E, g, L)


@jax.jit
def make_associative_params(
    Phi_aug,
    Q_aug,
    RESET,
    t_states,
    stateid,
    instid,
    m_pred,
    P_pred,
    m_filter,
    P_filter,
):
    """Generate the associative parameters needed for parallel RTS

    See eqns in Section 4B of Sarkka & Garcia-Fernandez (2020)
    """

    def make_last_params(mf_last, Pf_last):
        return (jnp.zeros_like(Pf_last), mf_last, Pf_last)

    def make_generic_params(
        Phi_dt,
        Q_dt,
        Reset,
        mp,
        Pp,
        mf,
        Pf,
        sid_current,
    ):
        def end_state():
            Phik = Phi_dt
            Qk = Q_dt
            mk = mf
            Pk = Pf
            return Phik, Qk, mk, Pk

        def start_state():
            Phik = Phi_dt @ Reset
            Qk = Q_dt
            mk = mp
            Pk = Pp
            return Phik, Qk, mk, Pk

        Phik, Qk, mk, Pk = jax.lax.cond(
            sid_current == 0,
            start_state,
            end_state,
        )

        A = Phik @ Pk @ Phik.T + Qk
        b = Pk @ Phik.T

        E = jax.scipy.linalg.solve(A.T, b.T, assume_a="pos").T
        g = mk - E @ (Phik @ mk)
        L = Pk - E @ Phik @ Pk

        return (E, g, L)

    t_delta = jnp.diff(t_states)
    Phis = jax.vmap(Phi_aug, in_axes=(None, 0))(0, t_delta)
    Qs = jax.vmap(Q_aug, in_axes=(None, 0))(0, t_delta)
    Resets = jax.vmap(RESET)(instid[:-1])

    E, g, L = jax.vmap(
        make_generic_params,
    )(
        Phis,
        Qs,
        Resets,
        m_pred[:-1],
        P_pred[:-1],
        m_filter[:-1],
        P_filter[:-1],
        stateid[:-1],
    )

    EN, gN, LN = make_last_params(m_filter[-1], P_filter[-1])

    E_all = jnp.concatenate([E, EN[jnp.newaxis, ...]], axis=0)
    g_all = jnp.concatenate([g, gN[jnp.newaxis, ...]], axis=0)
    L_all = jnp.concatenate([L, LN[jnp.newaxis, ...]], axis=0)

    return (E_all, g_all, L_all)


def _combine_per_pair(left, right):
    Ei, gi, Li = left
    Ej, gj, Lj = right

    # The indices need to be swapped for some reason...
    Eij = Ej @ Ei
    gij = Ej @ gi + gj
    Lij = Ej @ Li @ Ej.T + Lj

    return (Eij, gij, Lij)


@jax.jit
def parallel_integrated_rts_smoother(asso_params):
    """
    Jax implementation of the parallel RTS smoother algorithm
    for integrated measurements.

    See Section 4B of Sarkka & Garcia-Fernandez (2020) for
    a detailed description of the algorithm and notation,
    and Section 3.2.4 of Rubenzahl & Hattori et al. (2025)
    for the integrated case.

    Total runtime (span) complexity is O(N/T + logT) where N is the
    number of time steps and T is the number of parallel threads.
    """
    return jax.lax.associative_scan(
        jax.vmap(_combine_per_pair),
        asso_params,
        reverse=True,
    )
