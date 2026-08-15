from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from tinygp.helpers import JAXArray

from smolgp.helpers import robust_sqrt
from smolgp.solvers.state_coords import StateCoords


def sample_prior_trajectory(
    kernel,
    state_coords: StateCoords,
    key: jax.random.KeyArray,
) -> JAXArray:
    r"""Draw one exact forward-simulation sample of the (possibly-augmented)
    latent state trajectory at every state in ``state_coords``.

    This is the "prior sampling" half of the residual/Matheron's-rule
    conditional-sampling method (see :meth:`smolgp.gp.GaussianProcess.sample`):
    a pure forward SDE simulation using the same ``transition_matrix``/
    ``process_noise``/``stationary_covariance``/``reset_matrix`` the Kalman
    filter itself uses, but with no Kalman gain/update logic.

    Mirrors :func:`~smolgp.solvers.kalman.kalman_filter`'s own conventions
    exactly: the first state is drawn directly from the stationary
    distribution (matching that filter's ``m0=zeros``/``P0=stationary_covariance()``
    initialization) rather than via a transition from a nonexistent previous
    state, and ``reset_matrix`` is applied at ``stateid==0`` states exactly as
    :func:`~smolgp.solvers.integrated.kalman.integrated_kalman_filter`'s
    ``update_start`` does. For non-integrated kernels, pass the same "unified"
    fallback used elsewhere (:meth:`StateCoords.instantaneous`, i.e.
    ``instid=zeros``, ``obsid=arange(K)``, ``stateid=ones(K)``) -- since
    ``StateSpaceModel.reset_matrix`` defaults to the identity, the
    ``stateid==0`` branch is then a harmless no-op.

    Args:
        kernel: The (possibly-augmented) state space model.
        state_coords: a :class:`~smolgp.solvers.state_coords.StateCoords`, as
            held by ``solver.state_coords`` / returned by ``solver.condition()``.
        key: A ``jax`` random number key.

    Returns:
        The sampled trajectory, shape ``(K, kernel.dimension)``, in the same
        order as ``state_coords``.
    """
    t_states, instid, obsid, stateid = (
        state_coords.t_states,
        state_coords.instid,
        state_coords.obsid,
        state_coords.stateid,
    )
    K = t_states.shape[0]
    dim = kernel.dimension

    A = kernel.transition_matrix
    Q = kernel.process_noise
    RESET = kernel.reset_matrix

    P0 = kernel.stationary_covariance()
    if not isinstance(P0, JAXArray):
        P0 = P0.to_dense()

    key0, key_steps = jax.random.split(key)
    x0 = robust_sqrt(P0) @ jax.random.normal(key0, shape=(dim,))
    keys = jax.random.split(key_steps, K)

    def step(x_prev, inputs):
        k, key_k = inputs

        # k==0 uses zero time-lag (Delta=0) to trivially step to x0
        # (drawn above from the stationary distribution above).
        Delta = jax.lax.cond(
            k > 0, lambda i: t_states[i] - t_states[i - 1], lambda _: 0.0, k
        )
        A_k = A(0, Delta)
        Q_k = Q(0, Delta)
        z = jax.random.normal(key_k, shape=(dim,))
        x_pred = A_k @ x_prev + robust_sqrt(Q_k) @ z

        def do_reset(_):
            return RESET(instid[obsid[k]]) @ x_pred

        x_k = jax.lax.cond(stateid[k] == 0, do_reset, lambda _: x_pred, operand=None)
        return x_k, x_k

    _, x_traj = jax.lax.scan(step, x0, (jnp.arange(K), keys))
    return x_traj


def data_order_indices(state_coords: StateCoords, N: int) -> JAXArray:
    """The state-array indices (into ``state_coords``' own ``K``-length arrays)
    of the ``N`` real data points, sorted into data order (``obsid`` ``0..N-1``).

    Mirrors :meth:`smolgp.gp.ConditionedStates.project_at_data`'s
    select-``stateid==1``/sort-by-``obsid`` logic exactly.
    """
    obsid, stateid = state_coords.obsid, state_coords.stateid
    ends_idx = jnp.nonzero(stateid == 1, size=N)[0]
    return ends_idx[jnp.argsort(obsid[ends_idx])]


def project_trajectory_at_positions(
    X: JAXArray,
    positions: JAXArray,
    x_traj: JAXArray,
    observation_model,
) -> JAXArray:
    """Project a full state trajectory (shape ``(K_total, dim)``) at
    explicit ``positions`` into it, applying ``observation_model`` at the
    corresponding ``X`` coordinates (``X``'s own leading dimension must
    match ``positions``)."""
    x_sel = jnp.take(x_traj, positions, axis=0)

    def project(Xi, xi):
        return observation_model(Xi) @ xi

    # Squeeze only the trailing observation-dimension axis (D, assumed 1),
    # to guard against a single test point (N==1) collapsing the N axis and breaking
    # the shape needed to add against prior_obs_test_batch (N, not scalar).
    return jax.vmap(project)(X, x_sel).squeeze(-1)


def project_trajectory_at_data(
    X: JAXArray,
    state_coords: StateCoords,
    x_traj: JAXArray,
    observation_model,
    N: int,
) -> JAXArray:
    """Project a full state trajectory (shape ``(K, dim)``) down to the ``N``
    real data points, in data order.

    For the non-integrated fallback ``state_coords`` (``stateid`` all ``1``,
    ``obsid=arange(K)``), this reduces to the identity permutation.
    """
    idx = data_order_indices(state_coords, N)
    return project_trajectory_at_positions(X, idx, x_traj, observation_model)


def merge_test_coords(
    state_coords: StateCoords,
    t_test: JAXArray,
) -> tuple[StateCoords, JAXArray, JAXArray]:
    """Merge new delta=0 (instantaneous) test times into an existing sorted
    ``state_coords`` timeline, for a joint prior-trajectory sample covering
    both the original states and the new test points.

    Test points are inserted with ``stateid=1`` (no reset) and a dummy
    in-bounds ``obsid=0`` (the reset branch is traced-but-discarded via
    ``jax.lax.cond`` even when unused, so it must not index out of bounds).
    Sorting uses the same ``(t, -stateid)`` tie-break convention
    :class:`~smolgp.solvers.integrated.solver.IntegratedStateSpaceSolver`
    already uses, so a test point exactly at an existing state's time gets a
    zero-length transition to/from it and gets identically copied.

    Args:
        state_coords: the training timeline's
            :class:`~smolgp.solvers.state_coords.StateCoords` (``K`` states).
        t_test: sortable test times to insert, length ``M``.

    Returns:
        merged_state_coords: the merged, sorted ``StateCoords``, with ``K+M``
            states. ``instid`` is passed through unchanged (length ``N``):
            test points add new *states*, not new *observations*.
        train_positions: length ``K``, the new position of each original
            state (in ``state_coords``' own order) within the merged arrays.
        test_positions: length ``M``, the new position of each ``t_test[i]``
            within the merged arrays.
    """
    t_states, instid, obsid, stateid = (
        state_coords.t_states,
        state_coords.instid,
        state_coords.obsid,
        state_coords.stateid,
    )
    K = t_states.shape[0]

    test_obsid = jnp.zeros_like(t_test, dtype=obsid.dtype)
    test_stateid = jnp.ones_like(t_test, dtype=stateid.dtype)

    t_all = jnp.concatenate([t_states, t_test])
    obsid_all = jnp.concatenate([obsid, test_obsid])
    stateid_all = jnp.concatenate([stateid, test_stateid])

    sortidx = jnp.lexsort((-stateid_all, t_all))
    inv_sortidx = jnp.argsort(sortidx)

    merged_state_coords = StateCoords(
        t_states=t_all[sortidx],
        instid=instid,
        obsid=obsid_all[sortidx],
        stateid=stateid_all[sortidx],
    )
    train_positions = inv_sortidx[:K]
    test_positions = inv_sortidx[K:]
    return merged_state_coords, train_positions, test_positions


def merge_exposure_test_coords(
    kernel, state_coords, t_test, delta_test, instid_test, num_test_insts
):
    r"""Build an extended kernel and a merged, sorted timeline for jointly
    sampling the prior at (possibly) exposure-integrated (``delta>0``) test
    points. Generalizes :func:`merge_test_coords` to include exposures.

    Each test window is simulated with a "virtual probe instrument", the same
    idea :func:`~smolgp.solvers.integrated.predict_exposure.predict_exposure`
    uses for filtering (simulation needs less machinery, since a prior draw
    has no data/update step at all). Concretely:

    - **Probe dimensions.** ``kernel_ext`` extends ``kernel`` by
      ``num_test_insts + 1`` instruments. Test points sharing an
      ``instid_test`` share one probe dimension, exactly as repeated real
      exposures from one physical instrument already share one ``z`` slot,
      enabling overlapping test exposures to be simulated simultaneously
      with correct covariances.
    - **Two states per test point.** A reset at ``a_i = t_i - delta_i/2``
      (``stateid=0``), and a readout at ``b_i = t_i + delta_i/2`` (``stateid=1``).
    - **One joint simulation.** Running :func:`sample_prior_trajectory` once
      over the merged timeline gives a single draw covering the real states
      *and* every probe, correctly correlated with each other and with any
      overlapping real observations (independent per-point simulations
      would not give).

    Extending ``num_insts`` leaves the dynamics of the original ``n``-dim
    block unchanged, so only the projection changes: use
    ``kernel_ext.observation_model`` (its extra probe columns are zero by
    construction) to match the extended trajectory's dimension.

    Args:
        kernel: the original (non-extended) kernel.
        state_coords: the training timeline's
            :class:`~smolgp.solvers.state_coords.StateCoords` (``K`` states,
            ``instid`` of length ``N``).
        t_test: exposure midpoints, length ``M``.
        delta_test: exposure widths (``0`` for instantaneous), length ``M``.
        instid_test: length ``M``, which probe group (``0..num_test_insts-1``)
            each test point belongs to.
        num_test_insts: the number of distinct probe groups, i.e.
            ``int(jnp.max(instid_test)) + 1``. Must be a static Python int
            (concrete outside any enclosing ``jax.jit``), since it determines
            ``kernel_ext``'s dimension

    Returns:
        kernel_ext: kernel with ``num_insts`` extended by ``num_test_insts + 1``
            (the ``+1`` being the shared trash dimension).
        merged_state_coords: the merged, sorted ``StateCoords``, with
            ``K + 2M`` states and an ``instid`` of length ``N + M`` (each test
            point contributes one new probe "observation").
        train_positions: length ``K``, new position of each original state.
        b_positions: length ``M``, new position of each test point's ``b_i``.
        probe_dims: length ``M``, the state dimension to read each test
            point's probe value from at ``b_positions[i]`` (equal across
            test points sharing an ``instid_test``).
    """
    t_states, instid, obsid, stateid = (
        state_coords.t_states,
        state_coords.instid,
        state_coords.obsid,
        state_coords.stateid,
    )
    K = t_states.shape[0]
    M = t_test.shape[0]
    N = instid.shape[0]
    n = kernel.dimension

    # +1 for a shared "trash" dimension (index num_test_insts) that every
    # delta==0 point's reset is redirected to -- see below.
    kernel_ext = dataclasses.replace(
        kernel, num_insts=kernel.num_insts + num_test_insts + 1
    )

    a = t_test - delta_test / 2
    b = t_test + delta_test / 2

    # Test points sharing an instid_test reuse the same dedicated instrument
    # index kernel.num_insts + instid_test[i], so its reset zeroes only that
    # group's dimension n + instid_test[i].
    #
    # A delta==0 point's reset is a no-op for itself (a_i==b_i), but if it
    # shares instid_test with a delta>0 point, its reset would spuriously
    # clobber that point's in-progress accumulation. To avoid this, every
    # delta==0 point's reset is redirected to a shared "trash" probe
    # dimension instead of its nominal group. This is harmless since
    # project_exposure_test_points never reads the probe value for
    # delta==0 points, using the ordinary observation_model-based one
    # instead.
    trash_instid = num_test_insts
    probe_instids = kernel.num_insts + jnp.where(
        delta_test > 0, instid_test, trash_instid
    )
    instid_ext = jnp.concatenate([instid, probe_instids])

    a_obsid = N + jnp.arange(M)  # resolves instid_ext[N+i] = probe_instids[i]
    a_stateid = jnp.zeros(M, dtype=stateid.dtype)
    b_obsid = jnp.zeros(M, dtype=obsid.dtype)  # dummy, unused (stateid=1)
    b_stateid = jnp.ones(M, dtype=stateid.dtype)

    t_all = jnp.concatenate([t_states, a, b])
    obsid_all = jnp.concatenate([obsid, a_obsid, b_obsid])
    stateid_all = jnp.concatenate([stateid, a_stateid, b_stateid])

    sortidx = jnp.lexsort((-stateid_all, t_all))
    inv_sortidx = jnp.argsort(sortidx)

    merged_state_coords = StateCoords(
        t_states=t_all[sortidx],
        instid=instid_ext,
        obsid=obsid_all[sortidx],
        stateid=stateid_all[sortidx],
    )
    train_positions = inv_sortidx[:K]
    b_positions = inv_sortidx[K + M : K + 2 * M]
    probe_dims = n + instid_test
    return kernel_ext, merged_state_coords, train_positions, b_positions, probe_dims


def project_exposure_test_points(
    X_test: JAXArray,
    kernel_ext,
    x_traj_ext: JAXArray,
    b_positions: JAXArray,
    probe_dims: JAXArray,
    delta_test: JAXArray,
) -> JAXArray:
    r"""Read out the prior sample at ``M`` (possibly exposure-integrated)
    test points from a trajectory produced by simulating over
    :func:`merge_exposure_test_coords`'s merged timeline.

    For ``delta_test[i] == 0``, uses the ordinary
    ``kernel_ext.observation_model``-based readout at ``X_test[i]``
    (matching :func:`project_trajectory_at_positions`). For
    ``delta_test[i] > 0``, reads ``x_traj_ext[b_positions[i], probe_dims[i]]
    / delta_test[i]`` directly -- the simulation analog of
    :func:`~smolgp.solvers.integrated.predict_exposure.predict_exposure`'s
    ``z_mean = m_final[probe_idx]`` readout. Both are computed for every
    point and combined with ``jnp.where`` (rather than a data-dependent
    partition, which JAX's static shapes don't allow); the discarded branch
    for each point is finite but otherwise meaningless.
    """
    obsmodel_readout = project_trajectory_at_positions(
        X_test, b_positions, x_traj_ext, kernel_ext.observation_model
    )

    x_at_b = jnp.take(x_traj_ext, b_positions, axis=0)
    probe_vals = jax.vmap(lambda x, d: x[d])(x_at_b, probe_dims)
    safe_delta = jnp.where(delta_test > 0, delta_test, 1.0)
    probe_readout = probe_vals / safe_delta

    return jnp.where(delta_test > 0, probe_readout, obsmodel_readout)
