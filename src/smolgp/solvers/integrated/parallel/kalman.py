from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = ["IntegratedKalmanFilter", "integrated_kalman_filter"]


def IntegratedKalmanFilter(
    kernel,
    X,
    y,
    t_states,
    obsid,
    instid,
    stateid,
    noise=None,
    return_v_S=False,
):
    """
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
    """

    H_aug = kernel.observation_model
    Phi_aug = kernel.transition_matrix
    Q_aug = kernel.process_noise
    RESET = kernel.reset_matrix
    R = noise.diagonal() if noise is not None else jnp.zeros_like(y)

    m0 = jnp.zeros(kernel.dimension)
    P0 = kernel.stationary_covariance()

    asso_params = make_associative_params(
        Phi_aug,
        H_aug,
        Q_aug,
        RESET,
        R,
        X,
        y,
        t_states,
        obsid,
        instid,
        stateid,
        m0,
        P0,
    )
    A, b, C, eta, J = integrated_kalman_filter(asso_params)
    m_pred, P_pred, v, S = postprocess(
        Phi_aug,
        Q_aug,
        H_aug,
        R,
        X,
        y,
        t_states,
        obsid,
        stateid,
        b,
        C,
        m0,
        P0,
    )
    return (
        (A, b, C, eta, J),
        (m_pred, P_pred, v, S),
    )


@jax.jit
def make_associative_params(
    Phi_aug,
    H_aug,
    Q_aug,
    RESET,
    R,
    X,
    y,
    t_states,
    obsid,
    instid,
    stateid,
    m0,
    P0,
):
    """Generate the associative parameters needed for parallel Kalman

    See Eqns. 10, 11, 12 from Sarkka & Garcia-Fernandez (2020)

    """

    # precompute H at data coordinates
    H_array = jax.vmap(H_aug)(X)  # index with obsid

    state_dim = H_array[0].shape[-1]

    def make_first_params(
        Phi_aug,
        m0,
        P0,
        Reset,
    ):
        Phi0 = Phi_aug(0, 0)
        jax.debug.print("Phi0: {phi0}", phi0=Phi0)

        transition = Reset @ Phi0

        m = transition @ m0
        P = transition @ P0 @ transition.T  # Q(0,0) = 0

        jax.debug.print("m {m}", m=m)
        jax.debug.print("P {p}", p=P)

        A = Reset
        b = jnp.squeeze(m)
        C = P

        eta = jnp.zeros(state_dim)
        J = jnp.zeros_like(Phi0)

        return (A, b, C, eta, J)

    def to_start_params(ops):
        (
            Phi_dt,
            Q_dt,
            Reset,
            obsid,
        ) = ops

        A = Reset @ Phi_dt
        b = jnp.zeros(Phi_dt.shape[-1])
        C = Reset @ Q_dt @ Reset.T
        eta = jnp.zeros(state_dim)
        J = jnp.zeros_like(Phi_dt)
        return (A, b, C, eta, J)

    def to_end_params(ops):
        (
            Phi_dt,
            Q_dt,
            Reset,
            obsid,
        ) = ops

        I_nx = jnp.eye(Phi_dt.shape[-1])

        Hk = H_array[obsid]
        yk = y[obsid]
        rk = R[obsid]

        Sk = Hk @ Q_dt @ Hk.T + rk
        Kk = jnp.linalg.solve(Sk.T, (Q_dt @ Hk.T).T).T
        factor = I_nx - Kk @ Hk
        A = factor @ Phi_dt
        b = jnp.squeeze(Kk @ jnp.atleast_1d(yk))
        C = factor @ Q_dt
        C = 0.5 * (C + C.T)

        _a = Phi_dt.T @ Hk.T
        _b = jnp.linalg.solve(Sk, jnp.atleast_1d(yk))
        eta = _a @ _b
        _c = jnp.linalg.solve(Sk, Hk @ Phi_dt)
        J = _a @ _c
        return (A, b, C, eta, J)

    A0, b0, C0, eta0, J0 = make_first_params(
        Phi_aug,
        m0,
        P0,
        RESET(instid[0]),
    )
    t_delta = jnp.diff(t_states)

    Phis = jax.vmap(
        Phi_aug,
        in_axes=(
            None,
            0,
        ),
    )(0, t_delta)

    Qs = jax.vmap(
        Q_aug,
        in_axes=(
            None,
            0,
        ),
    )(0, t_delta)

    Resets = jax.vmap(RESET)(instid[1:])
    ops = (
        Phis,
        Qs,
        Resets,
        obsid[1:],
    )
    A, b, C, eta, J = jax.vmap(
        lambda sid, op: jax.lax.cond(
            sid == 0,
            to_start_params,
            to_end_params,
            op,
        ),
    )(stateid[1:], ops)
    A_all = jnp.concatenate([A0[jnp.newaxis, ...], A], axis=0)
    b_all = jnp.concatenate([b0[jnp.newaxis, ...], b], axis=0)
    C_all = jnp.concatenate([C0[jnp.newaxis, ...], C], axis=0)
    eta_all = jnp.concatenate([eta0[jnp.newaxis, ...], eta], axis=0)
    J_all = jnp.concatenate([J0[jnp.newaxis, ...], J], axis=0)

    return (A_all, b_all, C_all, eta_all, J_all)


def _combine_per_pair(left, right):
    """See Eqn. 13 & 14 of Sarkka & Garcia-Fernandez (2020) for
    a the algorithm and notation.

    """
    Ai, bi, Ci, etai, Ji = left
    Aj, bj, Cj, etaj, Jj = right

    dim = Ai.shape[-1]
    I = jnp.eye(dim)

    D = I + Ci @ Jj
    E = I + Jj @ Ci

    Aij = Aj @ jnp.linalg.solve(D, Ai)
    bij = Aj @ jnp.linalg.solve(D, bi + Ci @ etaj) + bj
    Cij = Aj @ jnp.linalg.solve(D, Ci) @ Aj.T + Cj
    etaij = Ai.T @ jnp.linalg.solve(E, etaj - Jj @ bi) + etai
    Jij = Ai.T @ jnp.linalg.solve(E, Jj) @ Ai + Ji

    Cij = 0.5 * (Cij + Cij.T)
    Jij = 0.5 * (Jij + Jij.T)

    return (Aij, bij, Cij, etaij, Jij)


@jax.jit
def integrated_kalman_filter(asso_params):
    """
    Jax implementation of the parallel Kalman filter algorithm

    See Section 4A of Sarkka & Garcia-Fernandez (2020) for
    a detailed description of the algorithm and notation.

    Total runtime (span) complexity is ~O(logN) where N is the number
    of time steps.
    """

    A, b, C, eta, J = jax.lax.associative_scan(
        jax.vmap(_combine_per_pair),
        asso_params,
    )

    return (A, b, C, eta, J)


@jax.jit
def _calc_kf_predictions(
    Phi_aug,
    Q_aug,
    H_aug,
    t_states,
    b,
    C,
    m0,
    P0,
):
    t_delta = jnp.diff(t_states)
    Phi0 = Phi_aug(0, 0)
    Phik = jax.vmap(Phi_aug, in_axes=(None, 0))(0, t_delta)
    Phis = jnp.concatenate([Phi0[jnp.newaxis, ...], Phik], axis=0)

    Q0 = Q_aug(0, 0)
    Qk = jax.vmap(Q_aug, in_axes=(None, 0))(0, t_delta)
    Qs = jnp.concatenate([Q0[jnp.newaxis, ...], Qk], axis=0)

    m_prev = jnp.concatenate(
        [m0[jnp.newaxis, ...], b[:-1]],
        axis=0,
    )
    P_prev = jnp.concatenate(
        [P0[jnp.newaxis, ...], C[:-1]],
        axis=0,
    )

    m_pred = jax.vmap(lambda _Phi, _m: _Phi @ _m)(Phis, m_prev)
    P_pred = jax.vmap(lambda _Phi, _P_prev, _Q: _Phi @ _P_prev @ _Phi.T + _Q)(
        Phis,
        P_prev,
        Qs,
    )

    return (m_pred, P_pred)


@jax.jit
def _calc_vS(
    H_aug,
    R,
    m_pred,
    P_pred,
    X,
    y,
    stateid,
    obsid,
):
    ends_idx = jnp.nonzero(
        stateid == 1,
        size=y.shape[0],
    )[0]  # end states where there is a measurement
    obsidx_in_ends_order = jnp.take(obsid, ends_idx)

    mm = jnp.take(m_pred, ends_idx, axis=0)
    Pm = jnp.take(P_pred, ends_idx, axis=0)
    Xm = jax.tree.map(lambda x: jnp.take(x, obsidx_in_ends_order), X)
    Hm = jax.vmap(H_aug)(Xm)
    ym = jnp.take(y, obsidx_in_ends_order)
    Rm = jnp.take(R, obsidx_in_ends_order)

    y_pred = jax.vmap(
        lambda H, m: H @ m,
        in_axes=(0, 0),
    )(Hm, mm)

    v = ym[..., jnp.newaxis] - y_pred
    S = jax.vmap(
        lambda H, P, R: H @ P @ H.T + R,
        in_axes=(0, 0, 0),
    )(
        Hm,
        Pm,
        Rm,
    )

    return (v, S)


@jax.jit
def postprocess(
    Phi_aug,
    Q_aug,
    H_aug,
    R,
    X,
    y,
    t_states,
    obsid,
    stateid,
    b,
    C,
    m0,
    P0,
):
    m_pred, P_pred = _calc_kf_predictions(
        Phi_aug,
        Q_aug,
        H_aug,
        t_states,
        b,
        C,
        m0,
        P0,
    )

    v, S = _calc_vS(
        H_aug,
        R,
        m_pred,
        P_pred,
        X,
        y,
        stateid,
        obsid,
    )

    return (m_pred, P_pred, v, S)
