from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    NamedTuple,
)

import equinox as eqx
import jax
import jax.numpy as jnp
from tinygp import kernels, means
from tinygp.helpers import JAXArray

from smolgp.helpers import assign_instids, count_min_instids, robust_sqrt
from smolgp.kernels import Product, StateSpaceModel, Sum, Wrapper
from smolgp.kernels.base import extract_all_components, extract_leaf_kernels
from smolgp.kernels.integrated import IntegratedStateSpaceModel
from smolgp.solvers import ParallelStateSpaceSolver, StateSpaceSolver
from smolgp.solvers.integrated import (
    IntegratedStateSpaceSolver,
    ParallelIntegratedStateSpaceSolver,
)
from smolgp.solvers.sample import (
    data_order_indices,
    merge_exposure_test_coords,
    merge_test_coords,
    project_exposure_test_points,
    project_trajectory_at_data,
    project_trajectory_at_positions,
    sample_prior_trajectory,
)
from smolgp.solvers.integrated.rts import integrated_rts_gains
from smolgp.solvers.rts import rts_gains
from smolgp.solvers.state_coords import StateCoords

if TYPE_CHECKING:
    from tinygp.numpyro_support import TinyDistribution

import dataclasses


def assign_unique_kernel_names(kernel: StateSpaceModel) -> StateSpaceModel:
    """Return a new kernel where duplicated leaf kernel names are made unique by appending _1, _2, etc.

    For example, if the original kernel has three components
    named "SHO", "Matern", and "Matern", they will be renamed
    to "SHO", "Matern_1", and "Matern_2". This is useful for
    ensuring that the component kernels can be uniquely identified
    when making predictions at test points or when extracting
    component contributions.
    """
    leaves = extract_leaf_kernels(kernel)
    names = [k.name for k in leaves]
    # Early exit if all names are unique (no duplicates)
    if len(set(names)) == len(names):
        return kernel

    # Otherwise, count occurrences
    counts = {}
    for k in leaves:
        counts[k.name] = counts.get(k.name, 0) + 1
    # counter for how many times we've used each duplicated name
    used = {name: 1 for name, c in counts.items() if c > 1}

    def _rename(k: StateSpaceModel) -> StateSpaceModel:
        if isinstance(k, Sum):
            k1 = _rename(k.kernel1)
            k2 = _rename(k.kernel2)
            return Sum(k1, k2)
        if isinstance(k, Product):
            k1 = _rename(k.kernel1)
            k2 = _rename(k.kernel2)
            return Product(k1, k2)
        # Leaf
        if counts[k.name] > 1:
            idx = used[k.name]
            used[k.name] += 1
            newname = f"{k.name}_{idx}"
            return dataclasses.replace(k, name=newname)
        else:
            # Single occurrence: leave unchanged
            return k

    return _rename(kernel)


def assign_num_insts(kernel: StateSpaceModel, num_insts: int) -> StateSpaceModel:
    """Return a new kernel where every IntegratedStateSpaceModel component has
    num_insts set to the given value, reinitializing any that don't already match.

    Traverses Sum, Product, and Wrapper nodes to find IntegratedStateSpaceModel
    leaves anywhere in the kernel tree.
    """
    if isinstance(kernel, (Sum, Product)):
        k1 = assign_num_insts(kernel.kernel1, num_insts)
        k2 = assign_num_insts(kernel.kernel2, num_insts)
        if k1 is kernel.kernel1 and k2 is kernel.kernel2:
            return kernel
        return type(kernel)(k1, k2)
    if isinstance(kernel, IntegratedStateSpaceModel):
        if kernel.num_insts == num_insts:
            return kernel
        warnings.warn(
            f"Kernel '{kernel.name}' has num_insts={kernel.num_insts}, but the data's "
            f"instid array implies {num_insts} instrument(s). Reinitializing this "
            f"kernel component with num_insts={num_insts}."
        )
        return dataclasses.replace(kernel, num_insts=num_insts)
    if isinstance(kernel, Wrapper):
        inner = assign_num_insts(kernel.kernel, num_insts)
        if inner is kernel.kernel:
            return kernel
        updated = dataclasses.replace(kernel, kernel=inner)
        if updated.kernel is not inner:
            raise NotImplementedError(
                f"Cannot automatically update num_insts inside a {type(kernel).__name__} "
                "wrapper; construct its integrated component with the correct num_insts "
                "directly."
            )
        return updated
    return kernel


class ConditionedStates(eqx.Module):
    """
    An object to hold the conditioned means and variances

    X: len(N) data coordinates
    y: len(N) observed data this was conditioned on
    t_states: len(K) time coordinates of all states
    instid  : len(N) instrument ID for each measurement
    obsid   : len(K) observation IDs corresponding to the measurement at each state
    stateid : len(K) state IDs corresponding to each state (0 for exposure-start, 1 for exposure-end)
    predicted_mean/var : len(K) Kalman predicted state
    filtered_mean/var  : len(K) Kalman filtered state
    smoothed_mean/var  : len(K) RTS smoothed state
    """

    X: JAXArray
    y: JAXArray
    t_states: JAXArray
    instid: JAXArray
    obsid: JAXArray
    stateid: JAXArray
    predicted_mean: JAXArray
    filtered_mean: JAXArray
    smoothed_mean: JAXArray
    predicted_cov: JAXArray
    filtered_cov: JAXArray
    smoothed_cov: JAXArray

    def __init__(
        self,
        X,
        y: JAXArray,
        t_states: JAXArray,
        instid: JAXArray,
        obsid: JAXArray,
        stateid: JAXArray,
        m_pred: JAXArray,
        P_pred: JAXArray,
        m_filt: JAXArray,
        P_filt: JAXArray,
        m_smooth: JAXArray,
        P_smooth: JAXArray,
    ):
        self.X = X
        self.y = y
        self.t_states = t_states
        self.instid = instid
        self.obsid = obsid
        self.stateid = stateid
        self.predicted_mean = m_pred
        self.predicted_cov = P_pred
        self.filtered_mean = m_filt
        self.filtered_cov = P_filt
        self.smoothed_mean = m_smooth
        self.smoothed_cov = P_smooth

    def __call__(self):
        state_coords = StateCoords(
            t_states=self.t_states,
            instid=self.instid,
            obsid=self.obsid,
            stateid=self.stateid,
        )
        packaged_results = (
            (self.predicted_mean, self.predicted_cov),
            (self.filtered_mean, self.filtered_cov),
            (self.smoothed_mean, self.smoothed_cov),
        )
        # This should match the output of solver.condition
        return state_coords, packaged_results, None

    def project_at_data(self, observation_model):
        """
        Project the states with measurements (e.g. exposure-ends)
        and sort back into original order as the data
        """

        @jax.jit
        def project(X, m, P):
            H = observation_model(X)
            mu = H @ m
            var = H @ P @ H.T
            return mu, var

        N = jnp.array(self.X).shape[-1]
        ends_idx = jnp.nonzero(self.stateid == 1, size=N)[0]
        sort = jnp.argsort(self.obsid[ends_idx])
        idx = ends_idx[sort]
        m_sel = jnp.take(self.smoothed_mean, idx, axis=0)
        P_sel = jnp.take(self.smoothed_cov, idx, axis=0)
        mu, var = jax.vmap(project)(self.X, m_sel, P_sel)
        return mu.squeeze(), var.squeeze()


class PredictedStates(eqx.Module):
    """
    An object to hold the full predictive states

    t_states: time coordinates at each state
    mean : predictive mean vector for each state
    cov  : predictive covariance for each state
    """

    t_states: JAXArray
    mean: JAXArray
    cov: JAXArray
    kernel: StateSpaceModel

    def __init__(
        self,
        t_states: JAXArray,
        m: JAXArray,
        P: JAXArray,
        kernel: StateSpaceModel,
    ):
        self.t_states = t_states
        self.mean = m
        self.cov = P
        self.kernel = kernel

    def project_mean(self, observation_model) -> JAXArray:
        """
        The projected mean at self.t_states given an observation model.
        """

        def _project(H, m):
            mu = H @ m
            return mu

        H = jax.vmap(observation_model)(self.t_states)
        mu = jax.vmap(_project)(H, self.mean)
        return mu.squeeze()

    def project_variance(self, observation_model) -> JAXArray:
        """
        The projected variance at self.t_states given an observation model.
        """

        def _project(H, P):
            var = H @ P @ H.T
            return var

        H = jax.vmap(observation_model)(self.t_states)
        var = jax.vmap(_project)(H, self.cov)
        return var.squeeze()

    @property
    def loc(self) -> JAXArray:
        """
        The overall mean at the predicted states
        """
        return self.project_mean(self.kernel.observation_matrix)

    @property
    def variance(self) -> JAXArray:
        """
        The overall variance at the predicted states
        """
        return self.project_variance(self.kernel.observation_matrix)

    def get_component(
        self,
        component: str | list[str],
        return_var: bool = False,
    ) -> PredictedStates:
        """
        Extract the predicted states corresponding to a component kernel
        """

        if isinstance(component, str):
            component = [component]

        ## Get effective observation model for the desired component(s)
        def H_comp(X):
            H = self.kernel.observation_model(X, component=component[0])
            for k, name in enumerate(component[1:]):
                H += self.kernel.observation_model(X, component=name)
            return H

        ## Project at test coordinates with component observation model
        component_mean = self.project_mean(H_comp)
        component_var = self.project_variance(H_comp)

        if return_var:
            return component_mean, component_var
        else:
            return component_mean

    def get_all_components(self, return_var: bool = False) -> dict[str, Any]:
        """
        Extract the predicted mean/variance corresponding to each component kernel
        """
        components = extract_leaf_kernels(self.kernel)
        results = {}
        for kernel in components:
            name = kernel.name
            if return_var:
                mu_m, var_m = self.get_component(name, return_var=True)
                results[name] = (mu_m, var_m)
            else:
                mu_m = self.get_component(name, return_var=False)
                results[name] = mu_m
        return results


class GaussianProcess(eqx.Module):
    r"""An interface for designing a Gaussian Process regression model.

    :param kernel: The kernel function.
    :type kernel: Kernel
    :param X: The input coordinates — any PyTree compatible with ``kernel``
        whose leading dimension has size ``N_data``.
        For integrated kernels, pass ``(t, texp)`` where ``t`` is the array of
        exposure midpoints and ``texp`` is the array of exposure durations.
    :type X: JAXArray
    :param noise: Observation noise covariance matrices with shape
        ``(N, D, D)``, where ``N`` is the number of data points and ``D`` is
        the observation dimension (usually 1). Each slice ``noise[k]`` is the
        :math:`D \times D` noise covariance for the ``k``-th observation.
        Two shorthands are accepted and broadcast up to ``(N, 1, 1)``: a 1-D
        array of shape ``(N,)`` is interpreted as scalar per-observation
        variances, and a scalar as a single homoscedastic variance shared by
        every observation (i.e. ``noise=0.01`` is equivalent to
        ``noise=jnp.full(N, 0.01)``). Defaults to
        :math:`\sqrt{\varepsilon_{\mathrm{machine}}} \cdot I` for all
        observations.
    :type noise: JAXArray | float, optional
    :param mean: A callable or constant mean function evaluated as
        ``mean(X)``.
    :type mean: Callable, optional
    :param solver: Solver class for filtering and smoothing. If ``None``
        (default), selected automatically based on the kernel type.
    """

    num_data: int = eqx.field(static=True)
    dtype: jnp.dtype = eqx.field(static=True)
    kernel: kernels.Kernel
    X: JAXArray
    mean_function: means.MeanBase
    mean: JAXArray
    var: JAXArray | None
    noise: JAXArray
    solver: StateSpaceSolver
    states: ConditionedStates

    def __init__(
        self,
        kernel: kernels.Kernel,
        X: JAXArray,
        *,
        noise: JAXArray | float | None = None,
        mean: means.MeanBase | Callable[[JAXArray], JAXArray] | JAXArray | None = None,
        solver: Any | None = None,
        mean_value: JAXArray | None = None,
        variance_value: JAXArray | None = None,
        covariance_value: Any | None = None,
        states: JAXArray | None = None,
        use_unique_names: bool = True,
        **solver_kwargs: Any,
    ):
        # First, assign unique kernel names if needed
        if use_unique_names:
            self.kernel = assign_unique_kernel_names(kernel)
        else:
            self.kernel = kernel

        # Check if the kernel contains any integrated components
        # (fully unwraps Sum/Product/Wrapper so e.g. 2.0 * IntegratedExp(...) is detected)
        kernels = extract_all_components(self.kernel)
        is_integrated = any(isinstance(k, IntegratedStateSpaceModel) for k in kernels)
        is_instantaneous = all(isinstance(k, StateSpaceModel) for k in kernels)

        # If using an integrated solver, ensure X has both coords and bin sizes
        if is_integrated:
            assert isinstance(X, tuple) and len(X) > 1, (
                "IntegratedStateSpaceSolver requires both the data coordinates (e.g. times)"
                " and bin sizes (e.g. exposure times). These should be passed as X=(t, texp)"
                " where t is the midpoint of each measurement and texp is the exposure time"
                " (i.e. each measurement is over the interval [t - texp/2, t + texp/2])."
            )

            # If instid is provided (X = (t, texp, instid)), validate its format
            # and reconcile num_insts across any integrated kernel components.
            if len(X) > 2:
                t_coord, _texp, instid = X
                instid = jnp.asarray(instid)
                if not jnp.issubdtype(instid.dtype, jnp.integer):
                    raise ValueError(
                        f"instid must be an integer array, got dtype {instid.dtype}"
                    )
                if instid.shape[0] != jnp.shape(t_coord)[0]:
                    raise ValueError(
                        "instid must have the same length as the data coordinates "
                        f"(got {instid.shape[0]} vs {jnp.shape(t_coord)[0]})"
                    )
                try:
                    # cannot be jitted, so if this fails
                    unique_insts = jnp.unique(instid)
                except jax.errors.ConcretizationTypeError:
                    # just retain the kernel's configured num_insts
                    unique_insts = None
                if unique_insts is not None:
                    num_insts = int(unique_insts.shape[0])
                    if not jnp.array_equal(unique_insts, jnp.arange(num_insts)):
                        raise ValueError(
                            "instid must contain consecutive integer instrument ids "
                            f"0, 1, ..., num_insts-1; got unique values {unique_insts}"
                        )

                    # Only auto-resize for fresh solvers; preserve pre-built 
                    # solver kernels for when test-time instid is a partial subset.
                    building_fresh_solver = solver is None or solver in [
                        StateSpaceSolver,
                        IntegratedStateSpaceSolver,
                        ParallelStateSpaceSolver,
                        ParallelIntegratedStateSpaceSolver,
                    ]
                    if building_fresh_solver:
                        self.kernel = assign_num_insts(self.kernel, num_insts)

        # Data coordinates (or tuple of coordinates)
        self.X = X

        # Mean function
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
        if self.mean.ndim > 2:
            raise ValueError(
                f"Invalid mean shape: expected ndim = 1 or 2, got ndim={self.mean.ndim}"
            )

        # Observation noise: shape (N, D, D)
        # A 0-D scalar is treated as one homoscedastic variance shared by every
        # observation, and a 1-D array of shape (N,) as per-observation scalar
        # variances; both are broadcast up to (N, 1, 1).
        if noise is None:
            jitter = _default_jitter(self.mean)
            noise = jnp.full((self.num_data, 1, 1), jitter, dtype=self.dtype)
        elif jnp.ndim(noise) == 0:
            noise = jnp.full((self.num_data, 1, 1), noise, dtype=self.dtype)
        elif jnp.ndim(noise) == 1:
            noise = jnp.asarray(noise)[:, None, None]
        self.noise = noise

        # Set up the solver
        # TODO: add parallel flag and if so use ParallelIntegratedStateSpaceSolver?
        if solver is None:
            if is_integrated:
                solver = IntegratedStateSpaceSolver
            elif is_instantaneous:
                solver = StateSpaceSolver
            else:
                raise ValueError(
                    "Must provide a solver if the kernel is not "
                    "a StateSpaceModel or IntegratedStateSpaceModel"
                )

            self.solver = solver(
                self.kernel,
                self.X,
                self.noise,
                **solver_kwargs,
            )
        # If solver type (uninstantiated) is passed
        elif solver in [
            StateSpaceSolver,
            IntegratedStateSpaceSolver,
            ParallelStateSpaceSolver,
            ParallelIntegratedStateSpaceSolver,
        ]:
            self.solver = solver(
                self.kernel,
                self.X,
                self.noise,
                **solver_kwargs,
            )
        # If a pre-instantiated solver is passed (e.g. like condGP)
        else:
            self.solver = solver

    @property
    def loc(self) -> JAXArray:
        """
        If conditioned, this will be the mean at the data points
        Otherwise, it is just the prior mean.
        """
        return self.mean

    @property
    def variance(self) -> JAXArray:
        r"""The marginal variance at each coordinate, i.e.
        :math:`\mathrm{diag}(\texttt{covariance})`.

        If conditioned, this is the posterior variance at this GP's coordinates.
        Otherwise it is the prior variance plus the observation noise.

        Computed directly rather than constructing the full covariance
        matrix and taking its diagonal (more expensive).
        """
        if self.var is not None:
            return self.var
        prior_var = jax.vmap(self.kernel.evaluate)(self.X, self.X)
        return prior_var + self._noise_diagonal()

    def _noise_diagonal(self) -> JAXArray:
        """Per-observation noise variance, as a length-``N`` vector."""
        return jnp.diagonal(self.noise, axis1=-2, axis2=-1).squeeze(-1)

    @property
    def covariance(self) -> JAXArray:
        r"""The full covariance matrix at this GP's coordinates.
        For just the diagonal, use :attr:`variance`.

        .. warning::
            This materializes an :math:`N \times N` matrix, which is exactly
            the cost ``smolgp`` exists to avoid: :math:`O(N^2)` memory and,
            for the conditioned case, :math:`O(N^2 d^3)` time. It is provided
            for small problems, for validation against dense references, and
            for cases genuinely needing the joint distribution. Prefer
            :attr:`variance` (:math:`O(N)`) when the marginals suffice.

        The unconditioned case is just the prior covariance plus measurement
        noise :math:`k(X, X) + \Sigma_n`

        The conditioned case returns the posterior covariance at the data using
        the smoother cross-covariance identity (Eq. 12.55 of Särkkä & Solin 2019).
        The diagonal is the observed smoothed variances,

        .. math::
            \Sigma_{k,k} = H_k P_k^s H_k^T,

        and the lower triangle of the symmetric covariance matrix is

        .. math::
            \Sigma_{i,j}
                = H_i \left(\prod_{m=i}^{j-1} G_m\right) P^s_j H_j^T, \quad i < j,

        with :math:`G_k` the RTS smoothing gains. The gains are recomputed on
        demand from the cached filtered/predicted covariances in :math:`O(N d^3)`,
        negligible compared to forming the matrix itself.

        Raises:
            NotImplementedError: for a GP returned by
                ``condition(y, X_test=...)``, whose coordinates are the test
                points rather than the training states. The cross-covariance
                *between* two arbitrary test points is not produced by the
                current predict machinery.
        """
        if self.states is None:
            return self.kernel(self.X, self.X) + jnp.diag(self._noise_diagonal())

        n_train = self.states.y.shape[0]
        if self.num_data != n_train:
            raise NotImplementedError(
                "covariance is only available at the training coordinates; this "
                "GP was built by condition(y, X_test=...), so its coordinates are "
                f"{self.num_data} test points rather than the {n_train} training "
                "points. Condition without X_test to get the joint posterior "
                "covariance at the data."
            )
        return self._posterior_covariance()

    def _posterior_covariance(self) -> JAXArray:
        r"""The joint posterior covariance at the data points, via the RTS
        smoother cross-covariance recursion (see :attr:`covariance`).

        See Eq. 12.55 of Särkkä & Solin 2019. We expand this definition to
        include integrated SSMs by first building the ``(K, K, d, d)``
        state-space cross-covariance, then selecting and projecting the
        ``N`` data-carrying states into observation space in data order.
        """
        sc = self.state_coords
        t_states, stateid = sc.t_states, sc.stateid
        K = t_states.shape[0]
        N = self.states.y.shape[0]

        P_filt = self.states.filtered_cov
        P_pred = self.states.predicted_cov
        P_smooth = self.states.smoothed_cov

        # recompute smoothing gains from the cached covariances.
        if isinstance(
            self.solver,
            (IntegratedStateSpaceSolver, ParallelIntegratedStateSpaceSolver),
        ):
            G = integrated_rts_gains(
                self.kernel.transition_matrix,
                self.kernel.reset_matrix,
                t_states,
                sc.obsid,
                sc.instid,
                stateid,
                P_filt,
                P_pred,
            )
        else:
            G = rts_gains(self.kernel.transition_matrix, t_states, P_filt, P_pred)

        # Row i of the block matrix: running product G_i G_{i+1} ... G_{j-1},
        # right-multiplied by P^s_j. Scanning forward from each i keeps this at
        # one (d, d) matmul per block rather than re-deriving the product.
        eye = jnp.eye(self.kernel.dimension, dtype=P_smooth.dtype)

        def row(i):
            def step(prod, j):
                # prod holds G_i...G_{j-1}; advance it with G_{j-1} for j > i.
                prod_next = jnp.where(j > i, prod @ G[jnp.clip(j - 1, 0, K - 2)], prod)
                block = jnp.where(j >= i, prod_next @ P_smooth[j], jnp.zeros_like(eye))
                return prod_next, block

            _, blocks = jax.lax.scan(step, eye, jnp.arange(K))
            return blocks  # (K, d, d), valid for j >= i

        upper = jax.lax.map(row, jnp.arange(K))  # (K, K, d, d), upper triangle

        # Mirror the upper triangle into the lower one: Cov(j, i) = Cov(i, j)^T.
        lower = jnp.swapaxes(jnp.transpose(upper, (1, 0, 2, 3)), -1, -2)
        iu = jnp.arange(K)[:, None] <= jnp.arange(K)[None, :]
        C = jnp.where(iu[..., None, None], upper, lower)

        # Select the data-carrying states, in data order, and project.
        idx = data_order_indices(sc, N)
        C_data = jnp.take(jnp.take(C, idx, axis=0), idx, axis=1)
        H = jax.vmap(self.kernel.observation_model)(self.X)  # (N, D, dim)

        def project(H_a, row_blocks):
            return jax.vmap(lambda H_b, blk: (H_a @ blk @ H_b.T)[0, 0])(H, row_blocks)

        return jax.vmap(project)(H, C_data)

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
        _, _, _, _, v, S = self.solver.Kalman(y, return_v_S=True)
        return self._compute_log_prob(v, S)

    @property
    def state_coords(self) -> StateCoords:
        """The :class:`~smolgp.solvers.state_coords.StateCoords` for this GP's
        states, shared by :meth:`condition` and :meth:`sample`.

        If already conditioned, rebuilds it from the cached ``self.states``
        fields. Otherwise reuses ``self.solver.state_coords``, which every
        solver builds once in its own ``__init__`` (the instantaneous solvers
        via :meth:`StateCoords.instantaneous`).
        """
        if self.states is not None:
            return StateCoords(
                t_states=self.states.t_states,
                instid=self.states.instid,
                obsid=self.states.obsid,
                stateid=self.states.stateid,
            )
        return self.solver.state_coords

    def condition(
        self,
        y: JAXArray,
        X_test: JAXArray | None = None,
        *,
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
            include_mean (bool, optional): If ``True`` (default), the predicted
                values will include the mean function evaluated at ``X_test``.
            kernel (Kernel, optional): A kernel to optionally specify the component
                kernel to be used for predicting after conditioning. See
                :ref:`multicomponent` for an example.

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
                lambda a, b: (
                    jnp.ndim(a) == jnp.ndim(b) and jnp.shape(a)[1:] == jnp.shape(b)[1:]
                ),
                self.X,
                X_test,
            )
            if not jax.tree_util.tree_reduce(lambda a, b: a and b, matches):
                raise ValueError(
                    "`X_test` must have the same tree structure as the input `X`, "
                    "and all but the leading dimension must have matching sizes"
                )

        # Condition on the data and return likelihood ingredients
        conditioned_results = self.solver.condition(y, return_v_S=True)

        # unpack conditioned_results, but discard the solver's own
        # state_coords in favor of self.state_coords, which is the
        # same StateCoords but also handles the already-conditioned case)
        _, conditioned_states, (v, S) = conditioned_results
        (
            (m_predicted, P_predicted),
            (m_filtered, P_filtered),
            (m_smoothed, P_smoothed),
        ) = conditioned_states

        sc = self.state_coords

        # Save the conditioned state values to a new GP object
        # so we can use them to make quick predictions at test
        # points with subsequent calls to self.predict
        states = ConditionedStates(
            self.X,
            y,
            sc.t_states,
            sc.instid,
            sc.obsid,
            sc.stateid,
            m_predicted,
            P_predicted,
            m_filtered,
            P_filtered,
            m_smoothed,
            P_smoothed,
        )

        ## Grab likelihood (v and S will already be
        ## filtered down to the "at the data" states)
        log_prob = self._compute_log_prob(v, S)

        ## Make predictions at X_test if given
        if kernel is None:
            # If no component kernel passed, use the full model
            observation_model = self.kernel.observation_model
        else:
            # Otherwise use the observation model of the passed
            # kernel, where we zero out all the other components
            observation_model = lambda X: self.kernel.observation_model(
                X, component=kernel.name
            )

        if X_test is not None:
            # If X_test was given, also predict at those points.
            if isinstance(
                self.solver,
                (IntegratedStateSpaceSolver, ParallelIntegratedStateSpaceSolver),
            ):
                mean, variance = self.solver.predict(X_test, states(), y=y)
            else:
                mean, variance = self.solver.predict(X_test, states())
            H = jax.vmap(observation_model)(X_test)
            mu = jax.vmap(lambda H_i, m: H_i @ m)(H, mean).squeeze()
            var = jax.vmap(lambda H_i, P: H_i @ P @ H_i.T)(H, variance).squeeze()
        else:
            # Otherwise, project the conditioned states
            # (at the data points) to observation space
            X_test = self.X
            mu, var = states.project_at_data(observation_model)

        ## Create the conditioned GP
        condGP = GaussianProcess(
            kernel=self.kernel,
            X=X_test,
            noise=self.noise,
            # mean=self.mean,
            solver=self.solver,
            mean_value=mu,
            variance_value=var,
            states=states,
        )

        # Return the likelihood and conditioned GP
        return ConditionResult(log_probability=log_prob, gp=condGP)

    def predict(
        self,
        X_test: JAXArray | None = None,
        y: JAXArray | None = None,
        *,
        return_full_state: bool = False,
        kernel: int | None = None,
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
            y (JAXArray, optional): The observed data. Only needs to be given
                if the GP has not yet been conditioned. Once conditioned,
                the data, if needed, is recalled automatically from ``self.states.y``.
                This should have the shape ``(N_data,)``, where ``N_data`` was the
                zeroth axis of the ``X`` data provided when instantiating this object.
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
            return_full_state (bool, optional): If ``True``, return the full predicted state
                mean and covariance, rather than projecting to observation space. Default is
                ``False``, i.e. the result is projected through kernel.observation_model.
            kernel (int, optional): If specified, the index of the kernel in a
                multi-component model (for example, a sum or product of kernels)
                to extract and project (if return_full_state is False) the prediction for.

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
            assert y is not None, (
                "The GP has not been conditioned yet, and no data array `y` was given."
            )
            _llh, condGP = self.condition(y)
            return condGP.predict(
                X_test,
                return_full_state=return_full_state,
                kernel=kernel,
                return_var=return_var,
                # return_cov=return_cov,
                observation_model=observation_model,
            )
        else:
            if y is None:
                # Recall the data this GP was conditioned on
                y = self.states.y
            if X_test is None:
                # If no X_test given, predict at the data points
                if return_full_state:
                    mu = self.states.smoothed_mean
                    var = self.states.smoothed_cov
                else:
                    if kernel is None:
                        # already computed here
                        mu, var = self.loc, self.var
                    else:
                        # extract component kernel & project
                        name = kernel if isinstance(kernel, str) else kernel.name
                        H_comp = lambda X: self.kernel.observation_model(
                            X, component=name
                        )
                        mu, var = self.states.project_at_data(H_comp)
            else:
                # Predicting at new test points
                H_test = (
                    self.kernel.observation_model
                    if observation_model is None
                    else observation_model
                )
                if isinstance(
                    self.solver,
                    (IntegratedStateSpaceSolver, ParallelIntegratedStateSpaceSolver),
                ):
                    # Pass `y` along to predict for integrated solvers.
                    # Needed if X_test includes predictions with exposure lengths.
                    mean, variance = self.solver.predict(X_test, self.states(), y=y)
                else:
                    mean, variance = self.solver.predict(X_test, self.states())
                if return_full_state:
                    mu = mean
                    var = variance
                    return PredictedStates(
                        t_states=X_test, m=mu, P=var, kernel=self.kernel
                    )
                else:
                    if kernel is not None:
                        name = kernel if isinstance(kernel, str) else kernel.name
                        H_test = lambda X: self.kernel.observation_model(
                            X, component=name
                        )
                    H = jax.vmap(H_test)(X_test)
                    mu = jax.vmap(lambda H_i, m: H_i @ m)(H, mean).squeeze()
                    var = jax.vmap(lambda H_i, P: H_i @ P @ H_i.T)(
                        H, variance
                    ).squeeze()

        if return_var:
            return mu, var
        # if return_cov:
        #     return mu, var
        return mu

    def sample(
        self,
        key: jax.random.KeyArray,
        shape: Sequence[int] | None = None,
        X_test: JAXArray | None = None,
        num_test_insts: int | None = None,
    ) -> JAXArray:
        """Generate samples from the process.

        If this ``GaussianProcess`` has not been conditioned, samples are drawn
        from the prior. If it was returned by :meth:`condition`, samples are drawn
        from the posterior using Matheron's rule
        (see :meth:`~smolgp.gp.GaussianProcess._sample` and the references therein).

        By default (``X_test=None``), samples are drawn at the training coordinates.
        Passing ``X_test`` draws samples at any, possibly out-of-sample, coordinates.
        If the GP is conditioned on exposure-integrated data (``delta>0``), then
        ``X_test`` needs to be either ``(t, delta)`` or ``(t, delta, instid)``.

        For exposure-integrated sample points, ``instid`` says only which
        instrument project the result as (if the observation model depends on
        the instrument). As such, overlapping sample points are allowed to have
        the same ``instid``. Internally, where overlap would not be allowed, we
        use a separate "probe group" ID to auto-assign each exposure to a
        non-overlapping integral accumulator. If samples are requested with no
        ``instid`` given, the default is to project the result as instrument ``0``.

        Args:
            key: A ``JAX`` random number key array.
            shape (tuple, optional): The number/shape of independent *draws*
                to generate. Each single draw is a complete joint sample across
                every coordinate in ``X_test`` (or the training coordinates),
                with the correct correlations between them. Defaults to a single draw.
            X_test (JAXArray, optional): New coordinates to sample at,
                instead of the training coordinates. For integrated samples with an
                integrated kernel, this should be either ``(t, delta)`` or ``(t, delta, instid)``
                where ``t`` is the array of exposure midpoints, ``delta`` is the array of
                exposure durations, and ``instid`` is the array of instrument IDs for each exposure.
            num_test_insts (int, optional): Number of probe groups to allocate
                for exposure-integrated test points. Derived from the
                coordinates when omitted, which reads their values and so
                cannot run under ``jit``. Pass it explicitly -- it is ``1``
                for instantaneous or non-overlapping test points -- to keep
                this call jittable. Must be at least
                :func:`~smolgp.helpers.count_min_instids`, or overlapping
                windows will share a probe and the draw will be wrong.

        Returns:
            The sampled realizations from the process with shape ``(N_samples,) +
            shape`` (or plain ``(N_samples,)`` if ``shape`` is not given) where
            ``N_samples`` is the zeroth dimension of ``X_test`` or of self.X
            if ``X_test`` is not given). E.g. ``shape=(M,)`` returns ``M``
            independent draws with shape ``(N_data, M)``.
        """
        # Probe-group instid (for non-overlapping integral accumulators) is
        # always auto-assigned here; the real instrument instid (for the
        # observation model) defaults to 0 if not given. num_test_insts must
        # be concrete here since it sizes kernel_ext inside _sample.
        n_probe = 0
        instid_proj = None
        if (
            isinstance(X_test, tuple)
            and len(X_test) > 1
            and isinstance(self.solver, IntegratedStateSpaceSolver)
        ):
            t_test, delta_test = X_test[0], X_test[1]
            # Deriving the probe count reads the coordinate *values*, so it
            # cannot happen under a trace. A caller who already knows it --
            # e.g. non-overlapping exposures, or instantaneous test points,
            # both of which need exactly one probe -- can pass it in and keep
            # this whole call jittable.
            n_probe = (
                count_min_instids(t_test, delta_test)
                if num_test_insts is None
                else int(num_test_insts)
            )
            instid_group = assign_instids(t_test, delta_test, n_probe)
            instid_proj = (
                jnp.asarray(X_test[2])
                if len(X_test) > 2
                else jnp.zeros_like(instid_group)
            )
            X_test = (t_test, delta_test, instid_group)
        return self._sample(key, shape, X_test, n_probe, instid_proj)

    def numpyro_dist(self, **kwargs: Any) -> TinyDistribution:
        """Get the numpyro MultivariateNormal distribution for this process"""
        from tinygp.numpyro_support import TinyDistribution

        return TinyDistribution(self, **kwargs)

    @partial(jax.jit, static_argnums=(2, 4))
    def _sample(
        self,
        key: jax.random.KeyArray,
        shape: Sequence[int] | None,
        X_test: JAXArray | None,
        num_test_insts: int,
        instid_proj: JAXArray | None = None,
    ) -> JAXArray:
        r"""Draw samples via the residual/Matheron's-rule/Durbin-Koopman method:

        See the following references for details:
        - Doucet, "A Note on Efficient Conditional Simulation of Gaussian
          Distributions" (https://www.stats.ox.ac.uk/~doucet/doucet_simulationconditionalgaussian.pdf).
        - Durbin & Koopman (2002), "A simple and efficient simulation smoother
          for state space time series analysis," Biometrika 89(3):603-616.
        - Wilson et al. (2020/2021), "Efficiently sampling functions from
          Gaussian process posteriors" / "Pathwise Conditioning of Gaussian
          Processes."

        A prior sample is a plain forward SDE simulation projected to
        observation space. A posterior sample corrects that draw toward the
        data via Matheron's rule:

            x_prior_traj  ~ forward SDE simulation of the full latent trajectory
            prior_obs     = x_prior_traj projected to observation space
            noise_sample  ~ N(0, self.noise)
            residual      = y_obs - (prior_obs + noise_sample)
            posterior_sample = prior_obs + project(condition(residual).smoothed_mean)

        That is, an ordinary conditioning pass (the same Kalman filter + RTS smoother
        :meth:`condition` uses) run on the residual instead of on the data.

        Implementation notes:
        - The covariance/gain recursion does not depend on the residual's
          *values*, so :class:`StateSpaceSolver`/:class:`IntegratedStateSpaceSolver`
          compute it once and share it across all ``shape`` samples
          (``solver.condition_batched_mean``). The parallel solvers instead
          call ``solver.condition`` once per sample: their associative-scan
          operator fuses the covariance and mean paths at every node, so the
          same split is possible but not yet implemented (TODO: if needed).
        - With ``X_test``, the prior trajectory covers the training states
          *and* the test points in one joint pass
          (:func:`~smolgp.solvers.sample.merge_test_coords`), and the residual
          correction is propagated out to those points by the ordinary
          ``solver.predict``. Not yet batched as above (TODO: if needed).

        """
        num_samples = 1 if shape is None else math.prod(shape)
        keys = jax.random.split(key, num_samples)

        if X_test is None:
            # Sample at the training coordinates:
            state_coords = self.state_coords
            X_at_states = self.X if self.states is None else self.states.X
            N_out = self.num_data if self.states is None else self.states.y.shape[0]

            def _prior_obs_and_noise(sample_key: jax.random.KeyArray) -> JAXArray:
                key_prior, key_noise = jax.random.split(sample_key)
                x_traj = sample_prior_trajectory(self.kernel, state_coords, key_prior)
                prior_obs = project_trajectory_at_data(
                    X_at_states,
                    state_coords,
                    x_traj,
                    self.kernel.observation_model,
                    N_out,
                )
                # Independent observation-noise draw, N(0, self.noise) per point.
                # Let's us sample the GP with observation noise, e.g. y = f(X) + noise.
                # Matches tinygp's convention since there the covariance is K+noise directly.
                keys_noise = jax.random.split(key_noise, N_out)
                noise_sample = jax.vmap(
                    lambda R, k: robust_sqrt(R) @ jax.random.normal(k, (R.shape[0],))
                )(self.noise, keys_noise).squeeze()
                return prior_obs, noise_sample

            prior_obs_batch, noise_batch = jax.vmap(_prior_obs_and_noise)(keys)

            if self.states is None:
                samples_T = self.mean + prior_obs_batch + noise_batch
            else:
                # Conditioned GP: posterior samples of the latent function,
                # i.e. no extra observation noise on top, to match tinygp, whose
                #   conditioned .sample() reproduces condGP.variance (no +noise)
                residual_batch = self.states.y[None, :] - (
                    prior_obs_batch + noise_batch
                )

                # The batched-mean conditioning path is only implemented for the non-parallel solvers
                if type(self.solver) in (StateSpaceSolver, IntegratedStateSpaceSolver):
                    m_smoothed_batch = self.solver.condition_batched_mean(
                        residual_batch
                    )
                    resid_mean_obs_batch = jax.vmap(
                        lambda m: project_trajectory_at_data(
                            X_at_states,
                            state_coords,
                            m,
                            self.kernel.observation_model,
                            N_out,
                        )
                    )(m_smoothed_batch)
                else:
                    # Parallel solvers fall back to the unoptimized per-sample conditioning
                    def _one_condition(residual_i: JAXArray) -> JAXArray:
                        _, resid_conditioned_states, _ = self.solver.condition(
                            residual_i
                        )
                        _, _, (m_smoothed_resid, _P_smoothed_resid) = (
                            resid_conditioned_states
                        )
                        return project_trajectory_at_data(
                            X_at_states,
                            state_coords,
                            m_smoothed_resid,
                            self.kernel.observation_model,
                            N_out,
                        )

                    resid_mean_obs_batch = jax.vmap(_one_condition)(residual_batch)

                samples_T = prior_obs_batch + resid_mean_obs_batch

        elif self.states is None:
            # Sampling at X_test, but the GP has not been conditioned yet.
            # Still draw from the prior, but at the new test points.
            # self.noise doesn't apply to arbitrary new coordinates
            # (only defined at self.X), so this is the latent GP only
            if (
                isinstance(X_test, tuple)
                and len(X_test) > 1
                and isinstance(self.solver, IntegratedStateSpaceSolver)
            ):
                # Exposure-aware (delta=0/delta>0 possibly mixed) prior
                # sample: same merge_exposure_test_coords machinery as the
                # conditioned case, just with an empty "training" timeline
                t_test = self.kernel.coord_to_sortable(X_test)
                delta_test = X_test[1]
                instid_test = X_test[2]
                N_out = jnp.shape(jax.tree_util.tree_leaves(X_test)[0])[0]
                empty_state_coords = StateCoords(
                    t_states=jnp.zeros((0,), dtype=t_test.dtype),
                    instid=jnp.zeros((0,), dtype=int),
                    obsid=jnp.zeros((0,), dtype=int),
                    stateid=jnp.zeros((0,), dtype=int),
                )
                kernel_ext, merged_coords, _train_positions, b_positions, probe_dims = (
                    merge_exposure_test_coords(
                        self.kernel,
                        empty_state_coords,
                        t_test,
                        delta_test,
                        instid_test,
                        num_test_insts,
                    )
                )

                # delta==0 points read out via the observation model, so they
                # need the *real* instrument, not the probe-group label.
                X_test_proj = (t_test, delta_test, instid_proj)

                def _one_sample(sample_key: jax.random.KeyArray) -> JAXArray:
                    x_traj = sample_prior_trajectory(
                        kernel_ext, merged_coords, sample_key
                    )
                    return project_exposure_test_points(
                        X_test_proj,
                        kernel_ext,
                        x_traj,
                        b_positions,
                        probe_dims,
                        delta_test,
                    )

                samples_T = jax.vmap(_one_sample)(keys)
            else:
                # delta=0-only prior sample at new test points:
                t_test = self.kernel.coord_to_sortable(X_test)
                N_out = jnp.shape(jax.tree_util.tree_leaves(X_test)[0])[0]
                state_coords_test = StateCoords.instantaneous(t_test)
                positions = jnp.argsort(state_coords_test.obsid)

                def _one_sample(sample_key: jax.random.KeyArray) -> JAXArray:
                    x_traj = sample_prior_trajectory(
                        self.kernel, state_coords_test, sample_key
                    )
                    return project_trajectory_at_positions(
                        X_test, positions, x_traj, self.kernel.observation_model
                    )

                samples_T = jax.vmap(_one_sample)(keys)

        elif (
            isinstance(X_test, tuple)
            and len(X_test) > 1
            and isinstance(self.solver, IntegratedStateSpaceSolver)
        ):
            # Conditioned GP at new exposure-integrated test points.
            state_coords = self.state_coords
            # Conditioned GP may have been predicted at an X_test
            # different from X, in which case num_data would no
            # longer be be the number of training data points.
            # But we can reconstruct since we cache the data in states.y
            N_train = self.states.y.shape[0]
            t_test = self.kernel.coord_to_sortable(X_test)
            delta_test = X_test[1]
            instid_test = X_test[2]
            N_out = jnp.shape(jax.tree_util.tree_leaves(X_test)[0])[0]
            kernel_ext, merged_coords, train_positions, b_positions, probe_dims = (
                merge_exposure_test_coords(
                    self.kernel,
                    state_coords,
                    t_test,
                    delta_test,
                    instid_test,
                    num_test_insts,
                )
            )
            train_idx = data_order_indices(state_coords, N_train)
            merged_train_idx = train_positions[train_idx]
            # Coordinates carrying the *real* instrument rather than the
            # probe-group label, for every observation-model projection --
            # see sample()'s note on the two roles instid plays here.
            X_test_proj = (t_test, delta_test, instid_proj)

            def _prior_obs_and_noise(sample_key: jax.random.KeyArray) -> JAXArray:
                key_prior, key_noise = jax.random.split(sample_key)
                x_traj = sample_prior_trajectory(kernel_ext, merged_coords, key_prior)
                prior_obs_train = project_trajectory_at_positions(
                    self.states.X,
                    merged_train_idx,
                    x_traj,
                    kernel_ext.observation_model,
                )
                prior_obs_test = project_exposure_test_points(
                    X_test_proj, kernel_ext, x_traj, b_positions, probe_dims, delta_test
                )
                keys_noise = jax.random.split(key_noise, N_train)
                noise_sample = jax.vmap(
                    lambda R, k: robust_sqrt(R) @ jax.random.normal(k, (R.shape[0],))
                )(self.noise, keys_noise).squeeze()
                return prior_obs_train, prior_obs_test, noise_sample

            prior_obs_train_batch, prior_obs_test_batch, noise_batch = jax.vmap(
                _prior_obs_and_noise
            )(keys)
            residual_batch = self.states.y[None, :] - (
                prior_obs_train_batch + noise_batch
            )

            H_test = jax.vmap(self.kernel.observation_model)(X_test_proj)

            def _resid_mean_at_test(residual_i: JAXArray) -> JAXArray:
                _, resid_conditioned_states, v_S = self.solver.condition(
                    residual_i, return_v_S=True
                )
                resid_conditioned_results = (
                    state_coords,
                    resid_conditioned_states,
                    v_S,
                )
                resid_mean, _resid_var = self.solver.predict(
                    X_test_proj, resid_conditioned_results, y=residual_i
                )
                # Squeeze only the trailing D axis, not a blanket .squeeze()
                # otherwise a single test point (N_test==1) would also
                # collapse the N_test axis, breaking the shape needed to
                # add against prior_obs_test_batch (N_test, not scalar).
                return jax.vmap(lambda h, m: h @ m)(H_test, resid_mean).squeeze(-1)

            resid_mean_obs_test_batch = jax.vmap(_resid_mean_at_test)(residual_batch)
            samples_T = prior_obs_test_batch + resid_mean_obs_test_batch

        else:
            # Conditioned GP at new delta=0-only test points
            state_coords = self.state_coords
            # Conditioned GP may have been predicted at an X_test
            # different from X, in which case num_data would no
            # longer be be the number of training data points.
            # But we can reconstruct since we cache the data in states.y
            N_train = self.states.y.shape[0]
            t_test = self.kernel.coord_to_sortable(X_test)
            N_out = jnp.shape(jax.tree_util.tree_leaves(X_test)[0])[0]
            merged_coords, train_positions, test_positions = merge_test_coords(
                state_coords, t_test
            )
            train_idx = data_order_indices(state_coords, N_train)
            merged_train_idx = train_positions[train_idx]

            def _prior_obs_and_noise(sample_key: jax.random.KeyArray) -> JAXArray:
                key_prior, key_noise = jax.random.split(sample_key)
                x_traj = sample_prior_trajectory(self.kernel, merged_coords, key_prior)
                prior_obs_train = project_trajectory_at_positions(
                    self.states.X,
                    merged_train_idx,
                    x_traj,
                    self.kernel.observation_model,
                )
                prior_obs_test = project_trajectory_at_positions(
                    X_test, test_positions, x_traj, self.kernel.observation_model
                )
                keys_noise = jax.random.split(key_noise, N_train)
                noise_sample = jax.vmap(
                    lambda R, k: robust_sqrt(R) @ jax.random.normal(k, (R.shape[0],))
                )(self.noise, keys_noise).squeeze()
                return prior_obs_train, prior_obs_test, noise_sample

            prior_obs_train_batch, prior_obs_test_batch, noise_batch = jax.vmap(
                _prior_obs_and_noise
            )(keys)
            residual_batch = self.states.y[None, :] - (
                prior_obs_train_batch + noise_batch
            )

            H_test = jax.vmap(self.kernel.observation_model)(X_test)

            def _resid_mean_at_test(residual_i: JAXArray) -> JAXArray:
                # Reuse this GP's own state_coords rather than the solver's
                # (identical, but this GP may be a conditioned one whose
                # states were cached from a different solver instance).
                _, resid_conditioned_states, v_S = self.solver.condition(
                    residual_i, return_v_S=True
                )
                resid_conditioned_results = (
                    state_coords,
                    resid_conditioned_states,
                    v_S,
                )
                resid_mean, _resid_var = self.solver.predict(
                    X_test, resid_conditioned_results
                )
                # Squeeze only the trailing D axis, not a blanket .squeeze()
                # otherwise a single test point (N_test==1) would also
                # collapse the N_test axis, breaking the shape needed to
                # add against prior_obs_test_batch (N_test, not scalar).
                return jax.vmap(lambda h, m: h @ m)(H_test, resid_mean).squeeze(-1)

            resid_mean_obs_test_batch = jax.vmap(_resid_mean_at_test)(residual_batch)
            samples_T = prior_obs_test_batch + resid_mean_obs_test_batch

        samples = jnp.moveaxis(samples_T, 0, -1)
        if shape is None:
            return samples[..., 0]
        return samples.reshape((N_out,) + tuple(shape))

    @jax.jit
    def _compute_log_prob(self, v: JAXArray, S: JAXArray) -> JAXArray:
        """
        Compute the log-likelihood given v and S from the Kalman filter
        """
        ## More readable version:
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

    def get_component_mean(
        self,
        component: list | str,
        return_var: bool = False,
        **kwargs,
    ) -> Any:
        """
        Get the predictive mean (and variance) of a particular
        (or sum of) component kernel in a multi-component model
        evaluated at self.X

        Args:
            X (JAXArray, optional): The coordinates where the prediction
                should be evaluated. This should have a data type compatible
                with the ``X`` data provided when instantiating this object.
            component (list | str): The name(s) of the component kernel(s)
                to extract the mean for. If a list of names is provided,
                the joint mean and variance for that collection of kernels
                will be returned.
            return_var (bool, optional): If ``True``, also return the variances
                of each component. Default is ``False``.

        Returns:
            If ``return_var`` is ``False``:
                component_mean (JAXArray)
            If ``return_var`` is ``True``:
                component_mean (JAXArray)
                component_var (JAXArray)
        """
        if self.states is None:
            raise ValueError(
                "The GP must be conditioned before getting component means."
            )

        if isinstance(component, str):
            component = [component]

        ## Get effective observation model for the desired component(s)
        def H_comp(X):
            H = self.kernel.observation_model(X, component=component[0])
            for k, name in enumerate(component[1:]):
                H += self.kernel.observation_model(X, component=name)
            return H

        ## Project at data
        component_mean, component_var = self.states.project_at_data(H_comp)

        if return_var:
            return component_mean, component_var
        else:
            return component_mean

    def get_all_component_means(self, return_var: bool = False, **kwargs) -> Any:
        """
        Get the predictive mean (and optionally variance) of each
        component kernels individually, evaluated at self.X

        Args:
            return_var (bool, optional): If ``True``, also return the variances
                of each component. Default is ``False``.

        Returns:
            If ``return_var`` is ``False``, a list of JAX arrays containing the
            means of each component kernel evaluated at the data points.
            If ``return_var`` is ``True``, a tuple where the first element is
            the list of means as before, and the second element is a list of
            JAX arrays containing the variances of each component kernel
            evaluated at the data points.
        """
        if self.states is None:
            raise ValueError(
                "The GP must be conditioned before getting component means."
            )

        ## First, extract all kernels
        kernels = extract_leaf_kernels(self.kernel)

        ## Loop through and project each component
        results = {}
        for k, kernel in enumerate(kernels):
            mu, var = self.get_component_mean(
                component=kernel.name, return_var=True, kwargs=kwargs
            )
            if return_var:
                results[kernel.name] = (mu, var)
            else:
                results[kernel.name] = mu

        return results


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


def _default_jitter(reference: JAXArray) -> JAXArray:
    """Default to adding some amount of jitter to the diagonal, just in case,
    we use sqrt(eps) for the dtype of the mean function because that seems to
    give sensible results in general.
    """
    return jnp.sqrt(jnp.finfo(reference.dtype).eps)
