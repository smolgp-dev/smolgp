from __future__ import annotations

__all__ = ["GaussianProcess"]

from collections.abc import Sequence
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    NamedTuple,
)

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from tinygp import kernels, means
from tinygp.helpers import JAXArray
from tinygp.noise import Diagonal, Noise

from smolgp.kernels import StateSpaceModel
from smolgp.kernels.integrated import IntegratedStateSpaceModel  # TODO: define this

from smolgp.solvers import StateSpaceSolver
from smolgp.solvers import ParallelStateSpaceSolver
from smolgp.solvers.integrated import IntegratedStateSpaceSolver

if TYPE_CHECKING:
    from tinygp.numpyro_support import TinyDistribution


class ConditionedStates(eqx.Module):
    """
    An object to hold the conditioned means and variances

    times : timestamp corresponding to each state
    predicted_mean/var : Kalman predicted state
    filtered_mean/var  : Kalman filtered state
    smoothed_mean/var  : RTS smoothed state
    """

    times: JAXArray
    predicted_mean: JAXArray
    filtered_mean: JAXArray
    smoothed_mean: JAXArray
    predicted_var: JAXArray
    filtered_var: JAXArray
    smoothed_var: JAXArray

    def __init__(
        self,
        times,
        m_pred: JAXArray,
        P_pred: JAXArray,
        m_filt: JAXArray,
        P_filt: JAXArray,
        m_smooth: JAXArray,
        P_smooth: JAXArray,
    ):
        self.times = times
        self.predicted_mean = m_pred
        self.predicted_var = P_pred
        self.filtered_mean = m_filt
        self.filtered_var = P_filt
        self.smoothed_mean = m_smooth
        self.smoothed_var = P_smooth

    def __call__(self):
        packaged_results = (
            (self.predicted_mean, self.predicted_var),
            (self.filtered_mean, self.filtered_var),
            (self.smoothed_mean, self.smoothed_var),
        )
        return self.times, packaged_results, None


class GaussianProcess(eqx.Module):
    """An interface for designing a Gaussian Process regression model

    Args:
        kernel (Kernel): The kernel function
        X (JAXArray): The input coordinates. This can be any PyTree that is
            compatible with ``kernel`` where the zeroth dimension is ``N_data``,
            the size of the data set.
        diag (JAXArray, optional): The value to add to the diagonal of the
            covariance matrix, often used to capture measurement uncertainty.
            This should be a scalar or have the shape ``(N_data,)``. If not
            provided, this will default to the square root of machine epsilon
            for the data type being used. This can sometimes be sufficient to
            avoid numerical issues, but if you're getting NaNs, try increasing
            this value.
        noise (Noise, optional): Used to implement more expressive observation
            noise models than those supported by just ``diag``. This can be any
            object that implements the :class:`tinygp.noise.Noise` protocol. If
            this is provided, the ``diag`` parameter will be ignored.
        mean (Callable, optional): A callable or constant mean function that
            will be evaluated with the ``X`` as input: ``mean(X)``
        solver: The solver type to be used to execute the required linear
            algebra.
    """

    num_data: int = eqx.field(static=True)
    dtype: np.dtype = eqx.field(static=True)
    kernel: kernels.Kernel
    X: JAXArray
    mean_function: means.MeanBase
    mean: JAXArray
    var: JAXArray | None
    noise: Noise
    solver: StateSpaceSolver
    states: ConditionedStates

    def __init__(
        self,
        kernel: kernels.Kernel,
        X: JAXArray,
        *,
        diag: JAXArray | None = None,
        noise: Noise | None = None,
        mean: means.MeanBase | Callable[[JAXArray], JAXArray] | JAXArray | None = None,
        solver: Any | None = None,
        mean_value: JAXArray | None = None,
        variance_value: JAXArray | None = None,
        covariance_value: Any | None = None,
        states: JAXArray | None = None,
        **solver_kwargs: Any,
    ):
        self.kernel = kernel
        self.X = X

        if isinstance(mean, means.MeanBase):
            self.mean_function = mean
        elif mean is None:
            self.mean_function = means.Mean(jnp.zeros(()))
        else:
            self.mean_function = means.Mean(mean)
        if mean_value is None:
            mean_value = jax.vmap(self.mean_function)(self.X)
        self.num_data = mean_value.shape[0]
        self.dtype = mean_value.dtype
        self.mean = mean_value
        self.var = variance_value
        self.states = states
        if self.mean.ndim != 1:
            raise ValueError(
                "Invalid mean shape: " f"expected ndim = 1, got ndim={self.mean.ndim}"
            )

        if noise is None:
            diag = _default_diag(self.mean) if diag is None else diag
            noise = Diagonal(diag=jnp.broadcast_to(diag, self.mean.shape))
        self.noise = noise

        if solver is None:
            if isinstance(kernel, IntegratedStateSpaceModel):
                # TODO: if any leaf kernels are integrated, use this
                # will have to define a Sum between integrated/non
                # (and Product) that creates a new integrated model
                solver = IntegratedStateSpaceSolver
            elif isinstance(self.kernel, StateSpaceModel):
                solver = StateSpaceSolver
            else:
                raise ValueError(
                    "Must provide a solver if the kernel is not "
                    "a StateSpaceModel or IntegratedStateSpaceModel"
                )

            self.solver = solver(
                kernel,
                self.X,
                self.noise,
            )
        # If solver type is passed
        elif solver is ParallelStateSpaceSolver:
            self.solver = solver(
                kernel,
                self.X,
                self.noise,
            )
        # If an instantiated solver is passed like condGP
        else:
            self.solver = solver

    @property
    def loc(self) -> JAXArray:
        # TODO: Return H @ m_smooth
        return self.mean

    @property
    def variance(self) -> JAXArray:
        # TODO: Return H @ P_smooth @ H.T
        return self.var

    @property
    def covariance(self) -> JAXArray:
        # TODO: Eq. 12.55 in Sarkka & Solin 2019
        return self.covariance_value

    def log_probability(self, y: JAXArray) -> JAXArray:
        """Compute the log probability of this multivariate normal

        Args:
            y (JAXArray): The observed data. This should have the shape
                ``(N_data,)``, where ``N_data`` was the zeroth axis of the ``X``
                data provided when instantiating this object.

        Returns:
            The marginal log probability of this multivariate normal model,
            evaluated at ``y``.
        """
        if isinstance(self.solver, StateSpaceSolver):
            _, _, _, _, v, S = self.solver.Kalman(
                self.X, y, self.noise, return_v_S=True
            )
        elif isinstance(self.solver, ParallelStateSpaceSolver):
            _, _, outputs = self.solver.Kalman(
                self.X,
                y,
                self.noise,
                return_v_S=True,
            )
            _, _, v, S = outputs

        return self._compute_log_prob(v, S)

    def condition(
        self,
        y: JAXArray,
        X_test: JAXArray | None = None,
        *,
        diag: JAXArray | None = None,  # TODO: is this needed?
        noise: Noise | None = None,  # TODO: is this needed?
        include_mean: bool = True,
        kernel: kernels.Kernel | None = None,  # TODO: select a component kernel
    ) -> ConditionResult:
        """Condition the model on observed data

        Args:
            y (JAXArray): The observed data. This should have the shape
                ``(N_data,)``, where ``N_data`` was the zeroth axis of the ``X``
                data provided when instantiating this object.
            X_test (JAXArray, optional): The coordinates where the prediction
                should be evaluated. This should have a data type compatible
                with the ``X`` data provided when instantiating this object. If
                it is not provided, ``X`` will be used by default, so the
                predictions will be made.
            diag (JAXArray, optional): Will be passed as the diagonal to the
                conditioned ``GaussianProcess`` object, so this can be used to
                introduce, for example, observational noise to predicted data.
            include_mean (bool, optional): If ``True`` (default), the predicted
                values will include the mean function evaluated at ``X_test``.
            kernel (Kernel, optional): A kernel to optionally specify the
                covariance between the observed data and predicted data. See
                :ref:`mixture` for an example.

        Returns:
            A named tuple where the first element ``log_probability`` is the log
            marginal probability of the model, and the second element ``gp`` is
            the :class:`GaussianProcess` object describing the conditional
            distribution evaluated at ``X_test``.
        """

        # If X_test is provided, we need to check that the tree structure
        # matches that of the input data, and that the shapes are all compatible
        # (i.e. the dimension of the inputs must match). This is slightly
        # convoluted since we need to support arbitrary pytrees.
        if X_test is not None:
            matches = jax.tree_util.tree_map(
                lambda a, b: jnp.ndim(a) == jnp.ndim(b)
                and jnp.shape(a)[1:] == jnp.shape(b)[1:],
                self.X,
                X_test,
            )
            if not jax.tree_util.tree_reduce(lambda a, b: a and b, matches):
                raise ValueError(
                    "`X_test` must have the same tree structure as the input `X`, "
                    "and all but the leading dimension must have matching sizes"
                )

        ## Condition on the data and return likelihood ingredients
        conditioned_results = self.solver.condition(y, return_v_S=True)

        ## unpack into prediction at the states
        X_states, conditioned_states, (v, S) = conditioned_results
        (
            (m_predicted, P_predicted),
            (m_filtered, P_filtered),
            (m_smoothed, P_smoothed),
        ) = conditioned_states

        ## Grab likelihood
        log_prob = self._compute_log_prob(v, S)

        # If no component kernel passed, use the full model
        if kernel is None:
            observation_model = self.kernel.observation_model
        else:
            kernel = self.kernel
            # Pick out just the component of interest
            observation_model = self.kernel.observation_model
            # TODO: zero out all the other components

        if X_test is not None:
            # If X_test was given, also predit at those points
            mu, var = self.solver.predict(X_test, conditioned_results)
        else:
            # Otherwise, project the conditioned states
            # (at the data points) to observation space
            X_test = X_states
            H = jax.vmap(observation_model)(X_test)
            ks = jnp.arange(len(X_test))
            # TODO: when we have integral states, we should have the solver
            # return only the exposure-end states along with the data timestamps
            # that way here we don't have to worry about H at exposure starts technically being "zero"
            # if we implement it that way, we can remove X_test from everywhere
            mu = jax.vmap(lambda k: H[k] @ m_smoothed[k])(ks).squeeze()
            var = jax.vmap(lambda k: H[k] @ P_smoothed[k] @ H[k].T)(ks).squeeze()

        # Save the conditioned state values to a new GP object
        # so we can use them to make quick predictions at test
        # points with subsequent calls to self.predict
        states = ConditionedStates(
            X_states,
            m_predicted,
            P_predicted,
            m_filtered,
            P_filtered,
            m_smoothed,
            P_smoothed,
        )

        condGP = GaussianProcess(
            kernel=self.kernel,
            X=X_test,
            noise=self.noise,
            mean=self.mean,
            solver=self.solver,
            mean_value=mu,
            variance_value=var,
            states=states,
        )

        # Return the likelihood and conditioned GP
        return log_prob, condGP

    def predict(
        self,
        X_test: JAXArray | None = None,
        y: JAXArray | None = None,
        *,
        # kernel: kernels.Kernel | None = None, # TODO: here
        # include_mean: bool = True,
        return_var: bool = False,
        # return_cov: bool = False,
        observation_model: Any | None = None,
    ) -> JAXArray | tuple[JAXArray, JAXArray]:
        """Predict the GP model at new test points conditioned on observed data

        Args:
            X_test (JAXArray, optional): The coordinates where the prediction
                should be evaluated. This should have a data type compatible
                with the ``X`` data provided when instantiating this object. If
                it is not provided, ``X`` will be used by default, so the
                predictions will be made at the data coordinates.
            y (JAXArray): The observed data. Only needs to be given if the GP
                has not yet been conditioned. This should have the shape
                ``(N_data,)``, where ``N_data`` was the zeroth axis of the ``X``
                data provided when instantiating this object.
            include_mean (bool, optional): If ``True`` (default), the predicted
                values will include the mean function evaluated at ``X_test``.
            return_var (bool, optional): If ``True`` (default), the variance of the
                predicted values at ``X_test`` will be returned.
            return_cov (bool, optional): If ``True``, the covariance of the
                predicted values at ``X_test`` will be returned. If
                ``return_var`` is ``True``, this flag will be ignored.
            observation_model (Any, optional): optionally provide a function of
                                X_test to define the output observation model.
                                Default will use that of the kernel.
            TODO: add an option to predict just at a given component kernel? or bake that into observation_model above


        Returns:
            The mean of the predictive model evaluated at ``X_test``, with shape
            ``(N_test,)`` where ``N_test`` is the zeroth dimension of
            ``X_test``. If either ``return_var`` or ``return_cov`` is ``True``,
            the variance or covariance of the predicted process will also be
            returned with shape ``(N_test,)`` or ``(N_test, N_test)``
            respectively.
        """

        if self.states is None:
            # Need to condition the GP first
            assert (
                y is not None
            ), "The GP has not been conditioned yet, and no data array `y` was given."
            llh, condGP = self.condition(
                y, X_test
            )  # condition on data, also predicts at X_test
            mu, var = condGP.loc, condGP.var  # which we can read off like so
        else:
            if X_test is None:
                mu = self.loc  # get pre-computed mean
                var = self.var  # & var at the data points
            else:
                # TODO: if component kernel is passed, fold that into H_test here
                H_test = (
                    self.kernel.observation_model
                    if observation_model is None
                    else observation_model
                )
                mu, var = self.solver.predict(X_test, self.states(), H_test)
        if return_var:
            return mu, var
        # if return_cov:
        #     return mu, var
        return mu

    ## TODO: how to define the sample function?
    def sample(
        self,
        key: jax.random.KeyArray,
        shape: Sequence[int] | None = None,
    ) -> JAXArray:
        """Generate samples from the prior process

        Args:
            key: A ``jax`` random number key array. shape (tuple, optional): The
            number and shape of samples to
                generate.

        Returns:
            The sampled realizations from the process with shape ``(N_data,) +
            shape`` where ``N_data`` is the zeroth dimension of the ``X``
            coordinates provided when instantiating this process.
        """
        return self._sample(key, shape)

    def numpyro_dist(self, **kwargs: Any) -> TinyDistribution:
        """Get the numpyro MultivariateNormal distribution for this process"""
        from tinygp.numpyro_support import TinyDistribution

        return TinyDistribution(self, **kwargs)

    @partial(jax.jit, static_argnums=(2,))
    def _sample(
        self,
        key: jax.random.KeyArray,
        shape: Sequence[int] | None,
    ) -> JAXArray:
        raise NotImplementedError
        ## TODO: implement sampling for state space model
        ##        or alternatively call the tinygp version? copied below:
        # if shape is None:
        #     shape = (self.num_data,)
        # else:
        #     shape = (self.num_data,) + tuple(shape)
        # normal_samples = jax.random.normal(key, shape=shape, dtype=self.dtype)
        # return self.mean + jnp.moveaxis(
        #     self.solver.dot_triangular(normal_samples), 0, -1
        # )

    @jax.jit
    def _compute_log_prob(self, v: JAXArray, S: JAXArray) -> JAXArray:
        """
        Compute the log-likelihood given v and S from the Kalman filter
        """
        # def llh(k):
        #     v_k, S_k = v[k], S[k]
        #     L_k = jnp.linalg.cholesky(S_k)
        #     w = jax.scipy.linalg.solve_triangular(L_k, v_k, lower=True)
        #     quad = jnp.dot(w, w)
        #     logdetS_k = 2.0 * jnp.sum(jnp.log(jnp.diag(L_k)))
        #     d = v_k.shape[0]
        #     return quad + logdetS_k + d*jnp.log(2*jnp.pi)

        # loglike = -0.5 * jnp.sum(jax.vmap(llh)(jnp.arange(len(v))))
        L = jax.vmap(jnp.linalg.cholesky)(S)  # [T, D, D]
        w = jax.scipy.linalg.solve_triangular(L, v[..., None], lower=True)
        w = jnp.squeeze(w, axis=-1)
        quad = jnp.sum(w**2, axis=1)
        logdetS = 2.0 * jnp.sum(jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1)), axis=1)
        d = v.shape[1]
        log_probs = quad + logdetS + d * jnp.log(2.0 * jnp.pi)
        loglike = -0.5 * jnp.sum(log_probs)

        return jnp.where(jnp.isfinite(loglike), loglike, -jnp.inf)

    # @partial(jax.jit, static_argnums=(3,))
    # def _condition(
    #     self,
    #     y: JAXArray,
    #     X_test: JAXArray | None,
    #     include_mean: bool,
    #     kernel: kernels.Kernel | None = None,
    # ) -> tuple[JAXArray, JAXArray, JAXArray]:
    #     alpha = self._get_alpha(y)
    #     log_prob = self._compute_log_prob(alpha)

    #     return alpha, log_prob, mean_value


class ConditionResult(NamedTuple):
    """The result of conditioning a :class:`GaussianProcess` on data

    This has two entries, ``log_probability`` and ``gp``, that are described
    below.
    """

    log_probability: JAXArray
    """The log probability of the conditioned model

    In other words, this is the marginal likelihood for the kernel parameters,
    given the observed data, or the multivariate normal log probability
    evaluated at the given data.
    """

    gp: GaussianProcess
    """A :class:`GaussianProcess` describing the conditional distribution

    This will have a mean and covariance conditioned on the observed data, but
    it is otherwise a fully functional GP that can sample from or condition
    further (although that's probably not going to be very efficient).
    """


def _default_diag(reference: JAXArray) -> JAXArray:
    """Default to adding some amount of jitter to the diagonal, just in case,
    we use sqrt(eps) for the dtype of the mean function because that seems to
    give sensible results in general.
    """
    return jnp.sqrt(jnp.finfo(reference).eps)
