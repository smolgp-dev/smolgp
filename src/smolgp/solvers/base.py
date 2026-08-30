""":class:`Solver` helps organize the common elements of all solvers, including
    1. the required methods (:meth:`Kalman`, :meth:`RTS`, :meth:`smoothing_gains`, :meth:`predict`), 
    2. the shared state-order bookkeeping definitions,
    3. and a default implementation of the marginal likelihood in terms of the
       filter's innovations, which every Kalman filter produces. A subclass may 
       override :meth:`log_probability` with a more efficient scan that only 
       accumulates the likelihood contributions (rather than the full filter outputs).

Subclasses of :class:`Solver` only inherit when the parent's method would be still 
be correct (though perhaps slower). As such, the inheritance tree looks like:

    Solver
    |-- StateSpaceSolver                        instantaneous, sequential Kalman/RTS
    |   `-- ParallelStateSpaceSolver            associative-scan Kalman/RTS
    `-- IntegratedStateSpaceSolver              exposure-aware, sequential Kalman/RTS (over K = 2N states)
        `-- ParallelIntegratedStateSpaceSolver  associative-scan exposure-aware Kalman/RTS (over K = 2N states)
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from tinygp.helpers import JAXArray

from smolgp.kernels.base import StateSpaceModel
from smolgp.solvers.state_coords import StateCoords


def log_prob_from_v_S(v: JAXArray, S: JAXArray) -> JAXArray:
    r"""The Gaussian log probability from a Kalman filter's innovations.

    Every Kalman filter produces the innovation :math:`v_k = y_k - H_k m_k^-`
    and its covariance :math:`S_k`, from which the marginal log likelihood is

    .. math::

        \log p(y) = -\tfrac{1}{2} \sum_k \left(
            v_k^T S_k^{-1} v_k + \log\det S_k + D \log 2\pi \right),

    Args:
        v: shape ``(N, D)``, the innovations.
        S: shape ``(N, D, D)``, the innovation covariances.
    """

    L = jax.vmap(jnp.linalg.cholesky)(S)  # (N, D, D)
    w = jax.scipy.linalg.solve_triangular(L, v[..., None], lower=True)
    w = jnp.squeeze(w, axis=-1)
    quad = jnp.sum(w**2, axis=1)
    logdetS = 2.0 * jnp.sum(jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1)), axis=1)
    D = v.shape[1]
    loglike = -0.5 * jnp.sum(quad + logdetS + D * jnp.log(2.0 * jnp.pi))

    return jnp.where(jnp.isfinite(loglike), loglike, -jnp.inf)


class Solver(eqx.Module):
    r"""Base class for a smolgp solver.

    Subclasses must implement :meth:`Kalman`, :meth:`RTS`, :meth:`smoothing_gains` 
    and :meth:`predict`. :meth:`condition` is provided here too, being only a
    sequencing of :meth:`Kalman` and :meth:`RTS`, but a solver whose smoother 
    takes different arguments overrides it.

    The likelihood is implemented here in terms of the filter's innovations,
    which every Kalman filter produces, but a subclass may override 
    :meth:`log_probability` with a more efficient scan that only accumulates 
    the likelihood contributions (rather than the full filter outputs).

    Attributes:
        kernel (StateSpaceModel): The kernel defining the state space model.
        X (JAXArray): The observed input coordinates.
        noise (JAXArray): Per-observation noise covariance, shape ``(N, D, D)``.
        state_coords (StateCoords): State-level bookkeeping. One state per
            observation for an instantaneous kernel, two (exposure start and
            end) for an integrated kernel.
    """

    X: JAXArray
    kernel: StateSpaceModel
    noise: JAXArray
    state_coords: StateCoords

    @property
    def t_states(self) -> JAXArray:
        """The chronologically sorted time coordinate of every state."""
        return self.state_coords.t_states

    def _to_state_order(self, *arrays: JAXArray) -> tuple[JAXArray, ...]:
        """Gather per-observation arrays into the solver's (chronologically 
        sorted) state order.

        ``self.X`` and everything derived from it (``y``, ``noise``) are kept
        in the caller's input order; the filter/smoother step chronologically,
        so they need the sorted order instead. ``state_coords.obsid`` is exactly
        that permutation. Results come back in state order and are mapped
        back by the usual sort-by-``obsid`` machinery.
        """
        obsid = self.state_coords.obsid
        return tuple(jax.tree_util.tree_map(lambda a: a[obsid], arr) for arr in arrays)

    def Kalman(self, y, return_v_S: bool = False) -> Any:
        """Run this solver's Kalman filter.

        Returns ``(m_filtered, P_filtered, m_predicted, P_predicted)``, 
            plus ``(v, S)`` when ``return_v_S`` is True.
        """
        raise NotImplementedError

    def RTS(self, kalman_results) -> Any:
        """Run this solver's RTS smoother over :meth:`Kalman`'s output."""
        raise NotImplementedError

    def smoothing_gains(self, P_filtered, P_predicted) -> JAXArray:
        """The ``y``-independent RTS smoothing gains for this state timeline."""
        raise NotImplementedError

    def condition(self, y, return_v_S: bool = False) -> Any:
        """Filter then smooth, giving the posterior at the data.

        Implemented here rather than per solver: it is only a sequencing of
        :meth:`Kalman` and :meth:`RTS` plus packaging, so every solver whose
        ``RTS`` takes the filter's four outputs shares it verbatim. A solver
        whose smoother has a different signature overrides it -- see
        :class:`~smolgp.solvers.ParallelStateSpaceSolver`, whose parallel
        smoother consumes only the filtered pair.
        """
        kalman_results = self.Kalman(y, return_v_S=return_v_S)
        if return_v_S:
            m_filtered, P_filtered, m_predicted, P_predicted, v, S = kalman_results
            v_S = (v, S)
        else:
            m_filtered, P_filtered, m_predicted, P_predicted = kalman_results
            v_S = None
        rts_results = self.RTS((m_filtered, P_filtered, m_predicted, P_predicted))
        m_smoothed, P_smoothed = rts_results
        conditioned_states = (
            (m_predicted, P_predicted),
            (m_filtered, P_filtered),
            (m_smoothed, P_smoothed),
        )
        return self.state_coords, conditioned_states, v_S

    def predict(self, X_test, conditioned_results) -> Any:
        """The posterior at arbitrary test coordinates."""
        raise NotImplementedError

    def _log_probability_from_filter(self, y) -> JAXArray:
        """The likelihood via the Kalman filter outputs."""
        *_, v, S = self.Kalman(y, return_v_S=True)
        return log_prob_from_v_S(v, S)

    def log_probability(self, y) -> JAXArray:
        """The marginal log likelihood of the data, ``y``.

        By default, runs the Kalman filter and reduces its innovations.
        However, the Kalman filter computes more than is necessary if one
        only wants the likelihood. Hence, a `Solver` can override this
        function with an optimized method, e.g.
        :meth:`~smolgp.solvers.solver.StateSpaceSolver.log_probability`
        """
        return self._log_probability_from_filter(y)
