from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = ["RTSSmoother", "rts_smoother"]


def RTSSmoother(kernel, X, kalman_results):
    """
    Wrapper for Parallel RTS smoother

    Parameters:
        kernel: StateSpaceModel kernel
        X: input coordinates
        kalman_results: output from Kalman filter
            these are the filtered state means (b) and covariances (C)

    Returns:
        E:
        g: smoothed means
        L: smoothed covariances
    """
    mu, P = kalman_results

    Phi = kernel.transition_matrix
    Q = kernel.process_noise

    asso_params = make_associative_params(Phi, Q, X, mu, P)
    E, g, L = rts_smoother(asso_params)
    return (E, g, L)


@jax.jit
def make_associative_params(Phi, Q, X, mu, P):
    """Generate the associative parameters needed for parallel RTS

    See eqns in Section 4B of Sarkka & Garcia-Fernandez (2020)
    """

    def make_last_params(mu_last, P_last):
        return (jnp.zeros_like(P_last), mu_last, P_last)

    def make_generic_params(Phi, Q, t_delta, mu, P):
        Phi_dt = Phi(0, t_delta)
        Q_dt = Q(0, t_delta)

        # Placeholder variables (not A, b from parallel KF)
        A = Phi_dt @ P @ Phi_dt.T + Q_dt
        b = P @ Phi_dt.T

        E = jax.scipy.linalg.solve(A.T, b.T, assume_a="pos").T
        g = mu - E @ (Phi_dt @ mu)
        L = P - E @ Phi_dt @ P

        return (E, g, L)

    t_delta = jnp.diff(X)
    E, g, L = jax.vmap(
        make_generic_params,
        in_axes=(None, None, 0, 0, 0),
    )(
        Phi,
        Q,
        t_delta,
        mu[:-1],
        P[:-1],
    )

    EN, gN, LN = make_last_params(mu[-1], P[-1])

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
def rts_smoother(asso_params):
    """
    Jax implementation of the parallel RTS smoother algorithm

    See Section 4B of Sarkka & Garcia-Fernandez (2020) for
    a detailed description of the algorithm and notation.

    Total runtime (span) complexity is ~O(logN) where N is the number
    of time steps.
    """
    return jax.lax.associative_scan(
        jax.vmap(_combine_per_pair),
        asso_params,
        reverse=True,
    )
