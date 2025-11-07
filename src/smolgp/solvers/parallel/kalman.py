from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = ["KalmanFilter", "kalman_filter"]


def KalmanFilter(
    kernel,
    X,
    y,
    noise,
    return_v_S=False,
):
    """
    Wrapper for the parallel Kalman filter.

    Parameters:
        kernel: StateSpaceModel kernel
        X: input coordinates
        y: observations
        noise: Noise model

    Returns:
        A:
        b: filtered means
        C: filtered covariances
        eta:
        J:
    """
    H = kernel.observation_model
    Phi = kernel.transition_matrix
    Q = kernel.process_noise
    R = noise.diagonal() if noise is not None else jnp.zeros_like(y)
    m0 = jnp.zeros(kernel.dimension)
    P0 = kernel.stationary_covariance()

    asso_params = make_associative_params(Phi, H, Q, R, X, y, m0, P0)
    A, b, C, eta, J = kalman_filter(asso_params)
    m_pred, P_pred, v, S = postprocess(
        Phi,
        H,
        Q,
        R,
        X,
        y,
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
    Phi,
    H,
    Q,
    R,
    X,
    y,
    m0,
    P0,
):
    """Generate the associative parameters needed for parallel Kalman

    See Eqns. 10, 11, 12 from Sarkka & Garcia-Fernandez (2020)

    """

    def make_first_params(
        Phi,
        H,
        m0,
        P0,
        y0,
        r0,
    ):
        Phi0 = Phi(0, 0)
        H0 = H(0)  # this is sort of unnecessary but we'll keep it for now

        m = Phi0 @ m0
        P = Phi0 @ P0 @ Phi0.T  # Q(0,0) = 0
        S = H0 @ P @ H0.T + r0
        S_inv = S**-1  # We might have to change this later
        K = P @ H0.T @ S_inv

        A = jnp.zeros_like(Phi0)
        b = jnp.squeeze(m + K @ (y0 - H0 @ m))
        C = P - K @ S @ K.T

        eta = jnp.squeeze(
            Phi0.T @ H0.T @ (S_inv @ jnp.atleast_1d(y0))
        )  # this might change
        J = Phi0.T @ H0.T @ S_inv @ H0 @ Phi0

        return (A, b, C, eta, J)

    def make_generic_params(
        Phi,
        H,
        Q,
        t_delta,
        y,
        r,
    ):
        Phi_dt = Phi(0, t_delta)
        I = jnp.eye(Phi_dt.shape[-1])

        Hk = H(t_delta)  # TODO: this is wrong, pass data coordinate here
        Q_dt = Q(0, t_delta)

        S = Hk @ Q_dt @ Hk.T + r
        S_inv = S**-1
        K = Q_dt @ Hk.T @ S_inv

        A = (I - K @ Hk) @ Phi_dt
        b = jnp.squeeze(K @ jnp.atleast_1d(y))  # remove atleast_1d?
        C = (I - K @ Hk) @ Q_dt

        eta = jnp.squeeze(Phi_dt.T @ Hk.T @ (S_inv * jnp.atleast_1d(y)))
        J = Phi_dt.T @ Hk.T @ S_inv @ Hk @ Phi_dt

        return (A, b, C, eta, J)

    A0, b0, C0, eta0, J0 = make_first_params(Phi, H, m0, P0, y[0], R[0])
    t_delta = jnp.diff(X)
    A, b, C, eta, J = jax.vmap(
        make_generic_params,
        in_axes=(None, None, None, 0, 0, 0),
    )(Phi, H, Q, t_delta, y[1:], R[1:])

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
def kalman_filter(asso_params):
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
def postprocess(
    Phi,
    H,
    Q,
    R,
    X,
    y,
    b,
    C,
    m0,
    P0,
):

    t_delta = jnp.diff(X)
    dim = b.shape[-1]
    I = jnp.eye(dim)

    Phis = jax.vmap(Phi, in_axes=(None, 0))(0, t_delta)
    Qs = jax.vmap(Q, in_axes=(None, 0))(0, t_delta)

    Phi_all = jnp.concatenate(
        [I[jnp.newaxis, ...], Phis],
        axis=0,
    )
    Q_all = jnp.concatenate(
        [jnp.zeros_like(I)[jnp.newaxis, ...], Qs],
        axis=0,
    )
    H_all = jax.vmap(H, in_axes=(0,))(X)
    R_all = R

    m_prev = jnp.concatenate(
        [m0[jnp.newaxis, ...], b[:-1]],
        axis=0,
    )
    P_prev = jnp.concatenate(
        [P0[jnp.newaxis, ...], C[:-1]],
        axis=0,
    )

    m_pred = jax.vmap(lambda _Phi, _m: _Phi @ _m)(
        Phi_all,
        m_prev,
    )
    P_pred = jax.vmap(lambda _Phi, _P_prev, _Q: _Phi @ _P_prev @ _Phi.T + _Q)(
        Phi_all,
        P_prev,
        Q_all,
    )

    y_pred = jax.vmap(lambda _H, _m: _H @ _m, in_axes=(0, 0))(
        H_all,
        m_pred,
    )

    v = y[..., jnp.newaxis] - y_pred

    S = jax.vmap(lambda _H, _P, _R: _H @ _P @ _H.T + _R)(
        H_all,
        P_pred,
        R_all,
    )

    return (m_pred, P_pred, v, S)
