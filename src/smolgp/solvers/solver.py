from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from tinygp.helpers import JAXArray
from tinygp.solvers.quasisep.solver import QuasisepSolver

from smolgp.kernels.base import StateSpaceModel
from smolgp.solvers.base import Solver
from smolgp.solvers.kalman import (
    KalmanFilter,
    KalmanLoglike,
    kalman_filter_batched_mean,
    kalman_gains,
)
from smolgp.solvers.rts import RTSSmoother, rts_gains, rts_smoother_batched_mean
from smolgp.solvers.state_coords import StateCoords


class StateSpaceSolver(Solver):
    r"""A solver that implements Kalman filtering and RTS smoothing for state space GPs.

    Given a :class:`~smolgp.kernels.StateSpaceModel` kernel and a set of observed
    coordinates, this solver computes the Kalman filtered and
    Rauch-Tung-Striebel (RTS) smoothed posterior means and covariances using
    ``jax.lax.scan`` for efficient JIT-compiled sequential computation.

    Predictions at arbitrary test coordinates are handled by :meth:`predict`,
    which dispatches among retrodiction, interpolation, and extrapolation
    depending on whether each test point falls before, between, or after
    the observed data.

    :param kernel: The kernel function; must be a
        :class:`~smolgp.kernels.StateSpaceModel` instance.
    :type kernel: StateSpaceModel
    :param X: The input coordinates with leading dimension of size ``N``.
    :type X: JAXArray
    :param noise: Observation noise covariance array of shape ``(N, D, D)``,
        where ``D`` is the observation dimension.
    :type noise: JAXArray

    Attributes:
        kernel (StateSpaceModel): The kernel defining the state space model.
        X (JAXArray): The observed input coordinates.
        noise (JAXArray): Per-observation noise covariance matrices, shape ``(N, D, D)``.
        state_coords (StateCoords): State-level bookkeeping. For an
            instantaneous kernel this is the degenerate one-state-per-observation
            case (see :meth:`~smolgp.solvers.state_coords.StateCoords.instantaneous`).
    """

    def __init__(
        self,
        kernel: StateSpaceModel,
        X: JAXArray,
        noise: JAXArray,
    ):
        """Build a :class:`StateSpaceSolver` for a given kernel and coordinates

        Args:
            kernel: The kernel function.
            X: The input coordinates.
            noise: Observation noise covariance array of shape ``(N, D, D)``.
        """
        self.kernel = kernel
        self.X = X
        self.noise = noise
        self.state_coords = StateCoords.instantaneous(kernel.coord_to_sortable(X))

    def Kalman(self, y, return_v_S=False) -> Any:
        """Wrapper for Kalman filter used with this solver"""
        y_nd = y[:, None] if y.ndim == 1 else y
        X_sorted, y_sorted, noise_sorted = self._to_state_order(
            self.X, y_nd, self.noise
        )
        return KalmanFilter(
            self.kernel, X_sorted, y_sorted, noise_sorted, return_v_S=return_v_S
        )

    def log_probability(self, y) -> JAXArray:
        """The marginal log likelihood, without running the full filter.

        Uses :func:`~smolgp.solvers.kalman.kalman_loglike`, whose scan only 
        computes the parts of the Kalman filter needed for the likelihood
        """
        y_nd = y[:, None] if y.ndim == 1 else y
        X_sorted, y_sorted, noise_sorted = self._to_state_order(
            self.X, y_nd, self.noise
        )
        return KalmanLoglike(self.kernel, X_sorted, y_sorted, noise_sorted)

    def RTS(self, kalman_results) -> Any:
        """Wrapper for RTS smoother used with this solver"""
        return RTSSmoother(self.kernel, self.t_states, kalman_results)

    def smoothing_gains(self, P_filtered, P_predicted) -> JAXArray:
        """The RTS smoothing gains G_k for this solver's state timeline.

        These are ``y``-independent, so they can be rebuilt on demand from the
        covariances a previous :meth:`condition` already produced
        """
        return rts_gains(
            self.kernel.transition_matrix, self.t_states, P_filtered, P_predicted
        )

    def condition_batched_mean(self, y_batch: JAXArray) -> JAXArray:
        """
        Batched-mean-path variant of condition(): computes the RTS-smoothed
        posterior mean for M residual/observation vectors that all share
        this solver's (kernel, X, noise), at O(M) cost in the cheap
        mean-path recursion only -- the O(N*dim^3) covariance recursion
        runs exactly ONCE, via the unmodified Kalman filter (called with a
        dummy y=zeros; exact, since P_filtered/P_predicted never depend on
        y), rather than once per sample.

        Args:
            y_batch: shape (M, N) or (M, N, D)

        Returns:
            m_smoothed_batch: shape (M, N, dim)
        """
        y_batch_nd = y_batch[:, :, None] if y_batch.ndim == 2 else y_batch
        _M, N, D = y_batch_nd.shape

        # 1. Covariance path, ONCE (dummy y; only P_filtered/P_predicted are used).
        y_dummy = jnp.zeros((N, D))
        _m_f, P_filtered, _m_p, P_predicted = self.Kalman(y_dummy)

        # 2. y-independent gains, ONCE. Everything here is indexed by state,
        #    so the per-observation arrays are gathered into state order too.
        X_sorted, noise_sorted = self._to_state_order(self.X, self.noise)
        A_all, H_all, K_all = kalman_gains(
            self.kernel.transition_matrix,
            jax.vmap(self.kernel.observation_model)(X_sorted),
            noise_sorted,
            self.t_states,
            P_predicted,
        )
        G_all = rts_gains(
            self.kernel.transition_matrix, self.t_states, P_filtered, P_predicted
        )

        # 3. Batched mean-path recursion, O(M) cheap.
        m0 = jnp.zeros(self.kernel.dimension)
        (y_batch_sorted,) = self._to_state_order(
            jnp.swapaxes(y_batch_nd, 0, 1)
        )  # (M, N, D) -> gather on N -> back to (M, N, D)
        m_filtered_batch, m_predicted_batch = kalman_filter_batched_mean(
            A_all, H_all, K_all, jnp.swapaxes(y_batch_sorted, 0, 1), m0
        )
        return rts_smoother_batched_mean(G_all, m_filtered_batch, m_predicted_batch)

    @jax.jit
    def predict(self, X_test, conditioned_results) -> JAXArray:
        """Algorithm for making predictions at arbitrary coordinates ``X_test``.

        Args:
            X_test (JAXArray): The test coordinates; same shape as ``self.X``.
            conditioned_results (tuple): The output of :meth:`condition`.

        Returns:
            tuple: A pair ``(pred_mean, pred_var)`` of arrays with leading
            dimension ``M = len(X_test)``, giving the predicted state means
            and covariances at each test coordinate.

        Each test point is handled by one of three cases depending on its
        position relative to the observed data:

        1. **Retrodiction** — test point precedes all observations: smoothed
           backward from the first data point using the stationary prior.
        2. **Interpolation** — test point falls between two observations:
           Kalman-predicted forward from the nearest past point, then
           RTS-smoothed backward from the nearest future point.
        3. **Extrapolation** — test point follows all observations:
           Kalman-predicted forward from the final filtered state.
        """

        # Unpack conditioned results
        state_coords, conditioned_states, _ = conditioned_results
        (
            (m_predicted, P_predicted),
            (m_filtered, P_filtered),
            (m_smoothed, P_smoothed),
        ) = conditioned_states
        t_states = state_coords.t_states

        # Unpack test coordinates
        t_test = self.kernel.coord_to_sortable(X_test)

        # N = len(self.X)   # number of data points (unused)
        K = len(t_states)  # number of states
        M = len(t_test)  # number of test points

        Pinf = self.kernel.stationary_covariance()
        if not isinstance(Pinf, JAXArray):  # if multicomponent model
            # need dense version for jnp.linalg.solve in retrodict
            Pinf = Pinf.to_dense()

        # Prior mean for retrodiction
        # mean = jnp.zeros(self.kernel.d)  # TODO: mean function of base kernel
        # m0 = jnp.block([mean] + self.kernel.num_insts * [jnp.zeros(self.kernel.d)])
        m0 = jnp.zeros(self.kernel.dimension)

        # Nearest (future) datapoint
        ## TODO: we assume X_states is sorted here; should we enforce that?
        k_nexts = jnp.searchsorted(t_states, t_test, side="right")

        # Which method to use for each test point:
        past = k_nexts <= 0  # Retrodiction
        future = k_nexts >= K  # Forecast
        during = ~past & ~future  # Interpolate
        cases = past.astype(int) * 0 + during.astype(int) * 1 + future.astype(int) * 2

        # Shorthand for matrices
        A = self.kernel.transition_matrix
        Q = self.kernel.process_noise

        def kalman(k_prev, ktest):
            """
            Kalman prediction from most recent
            filtered (not smoothed) state
            """
            dt = t_test[ktest] - t_states[k_prev]
            m_k = m_filtered[k_prev]
            P_k = P_filtered[k_prev]
            A_star = A(0, dt)  # transition matrix from t_k to t_star
            Q_star = Q(0, dt)  # process noise from t_k to t_star
            m_star_pred = A_star @ m_k
            P_star_pred = A_star @ P_k @ A_star.T + Q_star
            # No Kalman update since we have no data at t_star, so we're done
            return m_star_pred, P_star_pred

        def smooth(k_next, ktest, m_star_pred, P_star_pred):
            """
            RTS smooth the prediction (ktest) using
            the nearest future (smoothed) state (k_next)

            m_star_pred and P_star_pred are the output of kalman(k, k_star)
            """
            # Next (future) data point predicted & smoothed state
            m_pred_next = m_predicted[k_next]
            P_pred_next = P_predicted[k_next]
            m_hat_next = m_smoothed[k_next]
            P_hat_next = P_smoothed[k_next]

            # Time-lag between states
            dt = t_states[k_next] - t_test[ktest]

            # Transition matrix for this step
            A_k = A(0, dt)

            # Compute smoothing gain
            # P_pred_next_inv = jnp.linalg.inv(P_pred_next)
            # G_k = P_star_pred @ A_k.T @ P_pred_next_inv # smoothing gain
            G_k = jnp.linalg.solve(
                P_pred_next.T, (P_star_pred @ A_k.T).T
            ).T  # more stable

            # Update state and covariance
            m_star_hat = m_star_pred + G_k @ (m_hat_next - m_pred_next)
            P_star_hat = P_star_pred + G_k @ (P_hat_next - P_pred_next) @ G_k.T

            return m_star_hat, P_star_hat

        def retrodict(ktest):
            """Reverse-extrapolate from first datapoint t_star"""
            m_star, P_star = smooth(0, ktest, m0, Pinf)
            return m_star, P_star

        def interpolate(ktest):
            """Interpolate between nearest data points"""

            # Get nearest data point before and after the test point
            k_next = k_nexts[ktest]
            k_prev = k_next - 1

            # 1. Kalman predict from most recent data point (in past)
            m_star_pred, P_star_pred = kalman(k_prev, ktest)

            # 2. RTS smooth from next nearest data point (in future)
            m_star_hat, P_star_hat = smooth(k_next, ktest, m_star_pred, P_star_pred)

            return m_star_hat, P_star_hat

        def extrapolate(ktest):
            """Kalman predict from from last datapoint t_star"""
            m_star, P_star = kalman(-1, ktest)
            return m_star, P_star

        def predict_point(ktest):
            """
            Switch between retrodiction, interpolation, and extrapolation
            for a single test point ktest
            """
            return jax.lax.switch(
                cases[ktest], (retrodict, interpolate, extrapolate), (ktest)
            )

        # Calculate predictions
        ktests = jnp.arange(0, M, 1)
        (pred_mean, pred_var) = jax.vmap(predict_point)(ktests)

        return pred_mean, pred_var
