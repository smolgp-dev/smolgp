from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from tinygp.helpers import JAXArray
from tinygp.solvers.quasisep.solver import QuasisepSolver

from smolgp.helpers import get_smoothing_gain
from smolgp.kernels.base import StateSpaceModel
from smolgp.solvers.integrated.kalman import (
    IntegratedKalmanFilter,
    integrated_kalman_filter_batched_mean,
    integrated_kalman_gains,
)
from smolgp.solvers.integrated.predict_exposure import predict_exposure
from smolgp.solvers.integrated.rts import (
    IntegratedRTSSmoother,
    integrated_rts_gains,
    integrated_rts_smoother_batched_mean,
)
from smolgp.solvers.state_coords import StateCoords


class IntegratedStateSpaceSolver(eqx.Module):
    """
    A solver that uses ``jax.lax.scan`` to implement Kalman filtering
    and RTS smoothing for integrated measurements
    """

    X: JAXArray
    kernel: StateSpaceModel
    noise: JAXArray  # shape (N, D, D): observation noise covariance per time step
    state_coords: StateCoords

    def __init__(
        self,
        kernel: StateSpaceModel,
        X: JAXArray,
        noise: JAXArray,
    ):
        """Build a :class:`IntegratedStateSpaceSolver` for a given kernel and coordinates

        Args:
            kernel: The kernel function.
            X: The input coordinates. The coordinates for an integrated model should be a tuple of
                    X = (t, delta, instid),
                where `t` is the usual coordinate (e.g. time) at the measurements (midpoints),
                `delta` is the integration range (e.g. exposure time) for each measurement,
                and `instid` is an index encoding which instrument the measurement corresponds to.
            noise: Observation noise covariance array of shape ``(N, D, D)``.
            state_coords: Bookkeeping indices for the discretized states used in Kalman/RTS
        """
        self.kernel = kernel
        self.X = X
        self.noise = noise

        ## Preprocess state coordinates (exposure start/stops)
        ## and assign labels to each observation/state for bookkeeping:
        ## obsid   -- array len(K): which observation (0,...,N-1) is being made at each state k
        ## instids -- array len(N): which instrument (0,...,Ninst-1) recorded observation n
        ## stateid -- array len(K): 0 for exposure-start, 1 for exposure-end
        tmid, delta, instid = self.X  # unpack coordinates

        ## Construct interleaved time array of chronological exposure start/stop times
        ts = tmid - delta / 2  # Exposure start times
        te = tmid + delta / 2  # Exposure end times
        obsid = jnp.arange(len(tmid)).repeat(2)

        # Interleave start and end times into one array (fastest)
        # https://stackoverflow.com/questions/5347065/interleaving-two-numpy-arrays-efficiently
        t_states = jnp.empty((ts.size + te.size,), dtype=tmid.dtype)
        t_states = t_states.at[0::2].set(ts)  # evens are start times
        t_states = t_states.at[1::2].set(te)  # odds are end times
        stateid = jnp.tile(jnp.array([0, 1]), len(tmid))  # 0 for start, 1 for end
        # Have to re-sort because exposures can overlap
        # enforce end times before start times at same t
        sortidx = jnp.lexsort((-stateid, t_states))
        t_states = t_states[sortidx]
        obsid = obsid[sortidx]
        stateid = stateid[sortidx]  # 0 for t_s, 1 for t_e

        # Pack-up state_coords for Kalman and RTS functions. Note instid stays
        # per-*observation* (length N) while the others are per-*state*
        # (length K=2N) -- see StateCoords' own docstring.
        self.state_coords = StateCoords(
            t_states=t_states, instid=instid, obsid=obsid, stateid=stateid
        )

    def normalization(self) -> JAXArray:
        # TODO: do we want/can we implement this in state space? for now, fall back to quasisep
        class _NoiseAdapter:
            def __init__(self, n):
                self._n = n

            def diagonal(self):
                return self._n[0, 0, :]

        return QuasisepSolver(
            self.kernel, self.X, _NoiseAdapter(self.noise)
        ).normalization()

    def Kalman(self, y, return_v_S=False) -> Any:
        """Wrapper for Kalman filter used with this solver"""
        sc = self.state_coords
        # noise (N, D, D) → R (N, D, D); y (..., N) → (N, D)
        y_nd = y[:, None] if y.ndim == 1 else y
        return IntegratedKalmanFilter(
            self.kernel,
            self.X,
            y_nd,
            sc.t_states,
            sc.obsid,
            sc.instid,
            sc.stateid,
            self.noise,
            return_v_S=return_v_S,
        )

    def RTS(self, kalman_results) -> Any:
        """Wrapper for RTS smoother used with this solver"""
        sc = self.state_coords
        return IntegratedRTSSmoother(
            self.kernel, sc.t_states, sc.obsid, sc.instid, sc.stateid, kalman_results
        )

    def condition(self, y, return_v_S=False) -> JAXArray:
        """
        Compute the Kalman predicted, filtered, and RTS smoothed
        means and covariances at each of the input coordinates
        """

        # Kalman filtering
        kalman_results = self.Kalman(y, return_v_S=return_v_S)
        if return_v_S:
            m_filtered, P_filtered, m_predicted, P_predicted, v, S = kalman_results
            v_S = (v, S)
        else:
            m_filtered, P_filtered, m_predicted, P_predicted = kalman_results
            v_S = None

        # RTS smoothing
        rts_results = self.RTS((m_filtered, P_filtered, m_predicted, P_predicted))
        m_smoothed, P_smoothed = rts_results

        conditioned_states = (
            (m_predicted, P_predicted),
            (m_filtered, P_filtered),
            (m_smoothed, P_smoothed),
        )

        return self.state_coords, conditioned_states, v_S

    def condition_batched_mean(self, y_batch: JAXArray) -> JAXArray:
        """
        Batched-mean-path variant of condition() for M residual/observation
        arrays sharing this solver's (kernel, X, noise). See
        StateSpaceSolver.condition_batched_mean for the general idea; this
        additionally threads the reset-matrix bookkeeping through
        integrated_kalman_gains/integrated_kalman_filter_batched_mean.

        Args:
            y_batch: shape (M, N) or (M, N, D)

        Returns:
            m_smoothed_batch: shape (M, K, dim)
        """
        y_batch_nd = y_batch[:, :, None] if y_batch.ndim == 2 else y_batch
        _M, N, D = y_batch_nd.shape
        sc = self.state_coords
        t_states, instid, obsid, stateid = (
            sc.t_states,
            sc.instid,
            sc.obsid,
            sc.stateid,
        )

        # 1. Covariance path, ONCE (dummy y; only P_filtered/P_predicted are used).
        y_dummy = jnp.zeros((N, D))
        _m_f, P_filtered, _m_p, P_predicted = self.Kalman(y_dummy)

        # 2. y-independent gains, ONCE.
        A_aug = self.kernel.transition_matrix
        H_aug = self.kernel.observation_model
        RESET = self.kernel.reset_matrix
        A_all, H_all, K_all, Reset_all = integrated_kalman_gains(
            A_aug,
            H_aug,
            RESET,
            self.noise,
            self.X,
            t_states,
            obsid,
            instid,
            stateid,
            P_predicted,
        )
        G_all = integrated_rts_gains(
            A_aug, RESET, t_states, obsid, instid, stateid, P_filtered, P_predicted
        )

        # 3. Batched mean-path recursion, O(M) cheap.
        m0 = jnp.zeros(self.kernel.dimension)
        m_filtered_batch, m_predicted_batch = integrated_kalman_filter_batched_mean(
            A_all, H_all, K_all, Reset_all, obsid, stateid, y_batch_nd, m0
        )
        return integrated_rts_smoother_batched_mean(
            G_all, m_filtered_batch, m_predicted_batch, stateid
        )

    @jax.jit
    def predict(self, X_test, conditioned_results, y=None) -> JAXArray:
        """
        Algorithm for making predictions at arbitrary coordinates X_test

        Args:
            X_test              : The test coordinates. If a tuple
                                   ``(t, delta, instid)`` with any ``delta > 0``
                                   *and* ``y`` is also given, those test points
                                   are predicted as exposure-integrated
                                   averages (see
                                   :func:`~smolgp.solvers.integrated.predict_exposure.predict_exposure`)
                                   rather than instantaneous values.
            conditioned_results : The output of self.condition()
            y                   : The training data used to condition this
                                   solver. Only needed for ``delta>0`` test points.

        Returns:
            pred_mean : Predicted means of the states at X_test
            pred_var  : Predicted variances of the states at X_test

        There are three cases for each test point:
            1. Retrodiction  : smoothing from the first data point
                               using the prior as the prediction
            2. Interpolation : filtering from most recent data point
                               and smoothing from next future point
            3. Extrapolation : predicting from final filtered point
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
        if isinstance(X_test, tuple):
            _, delta_test, instid_test = X_test
        else:
            delta_test, instid_test = None, None

        # Array shapes
        # N = len(self.X)  # number of data points
        K = len(t_states)  # number of states
        M = len(t_test)  # number of test points

        # Prior covariance for retrodiction
        Pinf = self.kernel.stationary_covariance()
        if not isinstance(Pinf, JAXArray):  # if multicomponent model
            Pinf = Pinf.to_dense()  # needs to be array form here

        # Prior mean for retrodiction
        # mean = jnp.zeros(self.kernel.d)  # TODO: mean function of base kernel
        # m0 = jnp.block([mean] + self.kernel.num_insts * [jnp.zeros(self.kernel.d)])
        m0 = jnp.zeros(self.kernel.dimension)

        # Nearest/next past/future state for each datapoint
        k_nexts = jnp.searchsorted(t_states, t_test, side="right")

        # Method to use for test point
        past = k_nexts <= 0  # Retrodict
        future = k_nexts >= K  # Extrapolate
        during = ~past & ~future  # Interpolate
        cases = past.astype(int) * 0 + during.astype(int) * 1 + future.astype(int) * 2

        # Shorthand for matrices
        A_aug = lambda dt: self.kernel.transition_matrix(0, dt)
        Q_aug = lambda dt: self.kernel.process_noise(0, dt)

        def kalman(k_prev, ktest):
            """
            Kalman prediction from most recent
            filtered (but not RTS smoothed) state
            """
            dt = t_test[ktest] - t_states[k_prev]
            m_k = m_filtered[k_prev]
            P_k = P_filtered[k_prev]
            A_star = A_aug(dt)
            Q_star = Q_aug(dt)
            m_star_pred = A_star @ m_k
            P_star_pred = A_star @ P_k @ A_star.T + Q_star
            return m_star_pred, P_star_pred

        def smooth(k_next, ktest, m_star_pred, P_star_pred):
            """
            RTS smooth the prediction (ktest) using
            the nearest future data point (k_next)

            m_star_pred and P_star_pred are the output of kalman(k, k_star)
            """
            # Next (future) predicted & smoothed state
            m_pred_next = m_predicted[k_next]
            P_pred_next = P_predicted[k_next]
            m_hat_next = m_smoothed[k_next]
            P_hat_next = P_smoothed[k_next]

            # Transition matrix
            dt = t_states[k_next] - t_test[ktest]
            A_k = A_aug(dt)

            # RTS update
            G_k = get_smoothing_gain(P_pred_next, P_star_pred @ A_k.T)
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

        def predict_instantaneous(ktest):
            """
            Switch between retrodiction, interpolation, and extrapolation
            for a single instantaneous test point ktest
            """
            return jax.lax.switch(
                cases[ktest], (retrodict, interpolate, extrapolate), (ktest)
            )

        if delta_test is not None and y is not None:
            # Exposure-integrated prediction requires a non-zero exposure time (delta),
            # an instrument id (instid), and the training data (y).
            # If any of these are missing, fall back to instantaneous prediction.
            def predict_point(ktest):
                return jax.lax.cond(
                    delta_test[ktest] > 0,
                    lambda kt: predict_exposure(
                        self.kernel,
                        self.X,
                        y,
                        self.noise,
                        state_coords,
                        conditioned_states,
                        t_test[kt],
                        delta_test[kt],
                        instid_test[kt],
                    ),
                    predict_instantaneous,
                    ktest,
                )
        else:
            predict_point = predict_instantaneous

        # Calculate predictions
        ktests = jnp.arange(0, M, 1)
        (pred_mean, pred_var) = jax.vmap(predict_point)(ktests)

        return pred_mean, pred_var
