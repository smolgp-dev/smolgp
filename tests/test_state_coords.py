"""Structural invariants of the unified StateCoords bookkeeping object.

Every solver, filter, smoother, and sampler indexes the same four arrays --
``t_states``, ``instid``, ``obsid``, ``stateid`` -- and their *relative*
lengths are not uniform: ``instid`` is per-observation (length ``N``) while
the rest are per-state (length ``K``). For an instantaneous kernel ``K == N``
and the distinction is invisible; for an integrated kernel ``K == 2N`` and
getting it wrong silently mis-indexes the instrument of every state.

These tests pin that contract down directly, rather than relying on the
end-to-end numerical tests to catch an off-by-one gather.
"""

import jax
import jax.numpy as jnp
import pytest

import smolgp
from smolgp.solvers.state_coords import StateCoords

jax.config.update("jax_enable_x64", True)


def _assert_core_invariants(sc, N, K, num_insts, label):
    """The invariants that must hold for ANY StateCoords, however built."""
    assert sc.t_states.shape == (K,), f"[{label}] t_states shape {sc.t_states.shape}"
    assert sc.obsid.shape == (K,), f"[{label}] obsid shape {sc.obsid.shape}"
    assert sc.stateid.shape == (K,), f"[{label}] stateid shape {sc.stateid.shape}"
    assert sc.instid.shape == (N,), (
        f"[{label}] instid must be per-OBSERVATION (length {N}), got {sc.instid.shape}"
    )
    assert sc.num_states == K, f"[{label}] num_states"
    assert sc.num_obs == N, f"[{label}] num_obs"

    # t_states is sorted (every solver's searchsorted-based predict relies on this)
    assert jnp.all(jnp.diff(sc.t_states) >= 0), f"[{label}] t_states not sorted"

    # Index ranges must be in-bounds: obsid indexes into instid/y, and
    # sample_prior_trajectory evaluates instid[obsid[k]] under lax.cond for
    # EVERY k (even non-reset ones, whose branch is traced then discarded),
    # so an out-of-range obsid would silently gather garbage.
    assert int(jnp.min(sc.obsid)) >= 0, f"[{label}] obsid has negative entries"
    assert int(jnp.max(sc.obsid)) < N, f"[{label}] obsid out of range for N={N}"
    assert set(jnp.unique(sc.stateid).tolist()) <= {0, 1}, (
        f"[{label}] stateid must be 0 (exposure start) or 1 (data-carrying)"
    )
    assert int(jnp.min(sc.instid)) >= 0, f"[{label}] instid has negative entries"
    assert int(jnp.max(sc.instid)) < num_insts, f"[{label}] instid out of range"

    # The gather must be per-state and consistent with the raw arrays
    per_state = sc.instid_per_state()
    assert per_state.shape == (K,), f"[{label}] instid_per_state shape"
    assert jnp.array_equal(per_state, sc.instid[sc.obsid]), f"[{label}] gather"

    # Exactly N data-carrying states, one per observation, covering every obsid
    ends = jnp.nonzero(sc.stateid == 1, size=K, fill_value=-1)[0]
    ends = ends[ends >= 0]
    assert ends.shape[0] == N, (
        f"[{label}] expected exactly N={N} data-carrying states, got {ends.shape[0]}"
    )
    assert jnp.array_equal(jnp.sort(sc.obsid[ends]), jnp.arange(N)), (
        f"[{label}] data-carrying states must map 1:1 onto obsid 0..N-1"
    )


def test_state_coords_instantaneous_constructor():
    """The degenerate instantaneous case: one state per observation."""
    t = jnp.array([0.0, 1.0, 2.5, 4.0, 9.0])
    sc = StateCoords.instantaneous(t)
    _assert_core_invariants(sc, N=5, K=5, num_insts=1, label="instantaneous ctor")
    assert jnp.array_equal(sc.obsid, jnp.arange(5))
    assert jnp.all(sc.stateid == 1), "every instantaneous state carries data"
    assert jnp.all(sc.instid == 0), "single implicit instrument"


@pytest.mark.parametrize("parallel", [False, True])
def test_state_coords_from_instantaneous_solver(parallel):
    """Both plain solvers must expose the same StateCoords contract."""
    kernel = smolgp.kernels.SHO(omega=0.2, quality=2.0, sigma=1.0)
    N = 6
    t = jnp.sort(jax.random.uniform(jax.random.PRNGKey(0), (N,), maxval=50.0))
    solver_cls = (
        smolgp.solvers.ParallelStateSpaceSolver
        if parallel
        else smolgp.solvers.StateSpaceSolver
    )
    gp = smolgp.GaussianProcess(kernel, X=t, noise=jnp.full(N, 0.01), solver=solver_cls)
    label = solver_cls.__name__

    sc = gp.solver.state_coords
    _assert_core_invariants(sc, N=N, K=N, num_insts=1, label=label)
    assert jnp.array_equal(sc.t_states, t), f"[{label}] t_states must be the data times"

    # gp.state_coords must agree with the solver's own, before and after conditioning
    assert jnp.array_equal(gp.state_coords.t_states, sc.t_states), f"[{label}] gp pre"
    _, condgp = gp.condition(jax.random.normal(jax.random.PRNGKey(1), (N,)))
    sc_post = condgp.state_coords
    _assert_core_invariants(
        sc_post, N=N, K=N, num_insts=1, label=f"{label} conditioned"
    )
    assert jnp.array_equal(sc_post.t_states, sc.t_states), f"[{label}] gp post"


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("Ninst", [1, 2])
def test_state_coords_from_integrated_solver(parallel, Ninst):
    """The integrated case, where K = 2N and instid stays length N."""
    S, w, Q = 2.5, 0.2, 2.0
    sigma = jnp.sqrt(S * w * Q)
    kernel = smolgp.kernels.IntegratedSHO(
        omega=w, quality=Q, sigma=sigma, num_insts=Ninst
    )
    tA = jnp.linspace(0.0, 100.0, 7)
    t, texp, instid = tA, jnp.full(7, 3.0), jnp.zeros(7, dtype=int)
    if Ninst == 2:
        tB = jnp.linspace(2.0, 98.0, 5)
        t = jnp.concatenate([t, tB])
        texp = jnp.concatenate([texp, jnp.full(5, 2.0)])
        instid = jnp.concatenate([instid, jnp.ones(5, dtype=int)])
    N = t.shape[0]

    solver_cls = (
        smolgp.solvers.ParallelIntegratedStateSpaceSolver
        if parallel
        else smolgp.solvers.IntegratedStateSpaceSolver
    )
    gp = smolgp.GaussianProcess(
        kernel, X=(t, texp, instid), noise=jnp.full(N, 0.01), solver=solver_cls
    )
    label = f"{solver_cls.__name__}, Ninst={Ninst}"

    sc = gp.solver.state_coords
    _assert_core_invariants(sc, N=N, K=2 * N, num_insts=Ninst, label=label)

    # instid must be passed through from the data, NOT gathered/expanded
    assert jnp.array_equal(sc.instid, instid), (
        f"[{label}] instid must be the data's own"
    )

    # Each observation contributes exactly one start (stateid=0) and one end
    # (stateid=1), and the start must come first in the sorted timeline.
    assert int(jnp.sum(sc.stateid == 0)) == N, f"[{label}] expected N exposure starts"
    assert int(jnp.sum(sc.stateid == 1)) == N, f"[{label}] expected N exposure ends"
    for n in range(N):
        where_n = jnp.nonzero(sc.obsid == n, size=2)[0]
        ids = sc.stateid[where_n]
        assert set(ids.tolist()) == {0, 1}, f"[{label}] obs {n} needs one start+one end"
        start_pos = int(where_n[jnp.argmin(ids)])
        end_pos = int(where_n[jnp.argmax(ids)])
        assert start_pos < end_pos, f"[{label}] obs {n} start must precede its end"
        # ...and those states must sit at the true window edges
        assert jnp.allclose(sc.t_states[start_pos], t[n] - texp[n] / 2), (
            f"[{label}] obs {n} start time"
        )
        assert jnp.allclose(sc.t_states[end_pos], t[n] + texp[n] / 2), (
            f"[{label}] obs {n} end time"
        )
        # The gather must resolve BOTH of that observation's states to its
        # own instrument -- the single most load-bearing consequence of
        # instid being per-observation rather than per-state.
        per_state = sc.instid_per_state()
        assert int(per_state[start_pos]) == int(instid[n]), (
            f"[{label}] obs {n} start inst"
        )
        assert int(per_state[end_pos]) == int(instid[n]), f"[{label}] obs {n} end inst"


def test_state_coords_data_order_indices_roundtrip():
    """data_order_indices must recover the data-carrying states in data
    order, for both the instantaneous (identity) and integrated cases."""
    from smolgp.solvers.sample import data_order_indices

    t = jnp.array([0.0, 1.0, 2.0, 3.0])
    sc = StateCoords.instantaneous(t)
    idx = data_order_indices(sc, 4)
    assert jnp.array_equal(idx, jnp.arange(4)), "instantaneous must be the identity"

    S, w, Q = 2.5, 0.2, 2.0
    kernel = smolgp.kernels.IntegratedSHO(
        omega=w, quality=Q, sigma=jnp.sqrt(S * w * Q), num_insts=2
    )
    # Deliberately interleaved/overlapping so the sorted timeline is NOT
    # simply blocked by observation
    tt = jnp.array([10.0, 12.0, 14.0, 16.0])
    texp = jnp.array([6.0, 6.0, 6.0, 6.0])
    inst = jnp.array([0, 1, 0, 1])
    N = 4
    gp = smolgp.GaussianProcess(kernel, X=(tt, texp, inst), noise=jnp.full(N, 0.01))
    sc_i = gp.solver.state_coords
    idx_i = data_order_indices(sc_i, N)
    assert idx_i.shape == (N,)
    assert jnp.array_equal(sc_i.obsid[idx_i], jnp.arange(N)), (
        "must be sorted into data order (obsid 0..N-1)"
    )
    assert jnp.all(sc_i.stateid[idx_i] == 1), "must select only data-carrying states"
    assert jnp.allclose(sc_i.t_states[idx_i], tt + texp / 2), (
        "the data-carrying state of an exposure is its END time"
    )


def test_state_coords_merge_test_coords_preserves_contract():
    """Merging instantaneous test points adds STATES, not OBSERVATIONS: the
    merged StateCoords must grow K but leave instid (length N) alone."""
    from smolgp.solvers.sample import merge_test_coords

    t = jnp.array([0.0, 2.0, 5.0])
    sc = StateCoords.instantaneous(t)
    t_test = jnp.array([-1.0, 1.0, 3.5, 9.0])
    merged, train_pos, test_pos = merge_test_coords(sc, t_test)

    K, M, N = 3, 4, 3
    assert merged.num_states == K + M
    assert merged.num_obs == N, "instid must NOT grow when adding test states"
    assert jnp.array_equal(merged.instid, sc.instid)
    assert jnp.all(jnp.diff(merged.t_states) >= 0), "merged timeline must stay sorted"
    assert int(jnp.max(merged.obsid)) < N, "obsid must stay in-bounds for the gather"

    # positions must round-trip back to the original coordinates
    assert jnp.allclose(merged.t_states[train_pos], t)
    assert jnp.allclose(merged.t_states[test_pos], t_test)
    # inserted test states must not be flagged as exposure starts (no reset)
    assert jnp.all(merged.stateid[test_pos] == 1)


def test_state_coords_merge_exposure_test_coords_preserves_contract():
    """Exposure-integrated test points add 2 states AND 1 probe observation
    each, so BOTH K and N grow -- and instid must grow to match, or the
    instid[obsid[k]] gather would go out of bounds during simulation."""
    from smolgp.solvers.sample import merge_exposure_test_coords

    S, w, Q = 2.5, 0.2, 2.0
    kernel = smolgp.kernels.IntegratedSHO(
        omega=w, quality=Q, sigma=jnp.sqrt(S * w * Q), num_insts=1
    )
    t, texp, inst = (
        jnp.array([10.0, 30.0]),
        jnp.array([4.0, 4.0]),
        jnp.array([0, 0]),
    )
    N = 2
    gp = smolgp.GaussianProcess(kernel, X=(t, texp, inst), noise=jnp.full(N, 0.01))
    sc = gp.solver.state_coords

    t_test = jnp.array([15.0, 20.0, 25.0])
    delta_test = jnp.array([2.0, 0.0, 5.0])  # mixed delta>0 and delta==0
    instid_test = jnp.array([0, 1, 0], dtype=int)
    M = 3
    num_test_insts = 2

    kernel_ext, merged, train_pos, b_pos, probe_dims = merge_exposure_test_coords(
        kernel, sc, t_test, delta_test, instid_test, num_test_insts
    )

    assert merged.num_states == sc.num_states + 2 * M
    assert merged.num_obs == N + M, "each test point adds one probe observation"
    assert jnp.all(jnp.diff(merged.t_states) >= 0), "merged timeline must stay sorted"
    # The gather is evaluated for every state under lax.cond, so obsid must
    # stay in-bounds of the (now longer) instid.
    assert int(jnp.max(merged.obsid)) < merged.num_obs
    assert int(jnp.max(merged.instid)) < kernel_ext.num_insts
    # The original observations' instids must survive unchanged at the front
    assert jnp.array_equal(merged.instid[:N], sc.instid)
    assert jnp.allclose(merged.t_states[train_pos], sc.t_states)
    assert jnp.allclose(merged.t_states[b_pos], t_test + delta_test / 2)
    assert probe_dims.shape == (M,)


# ---------------------------------------------------------------------------
# ConditionedStates.__call__() vs solver.condition(): the two producers of the
# "conditioned results" 3-tuple that solver.predict() consumes.
# ---------------------------------------------------------------------------


def _build_gp_and_y(kind, solver_cls):
    """A GP + data for each (kernel type, solver) combination."""
    if kind == "instantaneous":
        kernel = smolgp.kernels.SHO(omega=0.2, quality=2.0, sigma=1.0)
        N = 6
        t = jnp.sort(jax.random.uniform(jax.random.PRNGKey(0), (N,), maxval=50.0))
        X = t
        y = jax.random.normal(jax.random.PRNGKey(1), (N,))
    else:
        S, w, Q = 2.5, 0.2, 2.0
        kernel = smolgp.kernels.IntegratedSHO(
            omega=w, quality=Q, sigma=jnp.sqrt(S * w * Q), num_insts=2
        )
        tA = jnp.linspace(0.0, 100.0, 7)
        tB = jnp.linspace(2.0, 98.0, 5)
        t = jnp.concatenate([tA, tB])
        texp = jnp.concatenate([jnp.full(7, 3.0), jnp.full(5, 2.0)])
        instid = jnp.concatenate([jnp.zeros(7, dtype=int), jnp.ones(5, dtype=int)])
        N = t.shape[0]
        X = (t, texp, instid)
        y = jax.random.normal(jax.random.PRNGKey(2), (N,))
    gp = smolgp.GaussianProcess(kernel, X=X, noise=jnp.full(N, 0.01), solver=solver_cls)
    return gp, y


SOLVER_CASES = [
    ("instantaneous", smolgp.solvers.StateSpaceSolver),
    ("instantaneous", smolgp.solvers.ParallelStateSpaceSolver),
    ("integrated", smolgp.solvers.IntegratedStateSpaceSolver),
    ("integrated", smolgp.solvers.ParallelIntegratedStateSpaceSolver),
]


@pytest.mark.parametrize("kind,solver_cls", SOLVER_CASES)
def test_conditioned_states_call_matches_solver_condition(kind, solver_cls):
    """``ConditionedStates.__call__()`` must reproduce ``solver.condition()``'s
    output exactly -- same pytree structure, leaf shapes, dtypes, and values.

    Both feed :meth:`solver.predict`, which unpacks them identically, so any
    divergence is a latent bug. (This is the contract the plain solvers used
    to violate by returning a bare ``t_states`` array in the ``state_coords``
    slot instead of the unified :class:`StateCoords`.)
    """
    label = f"{kind}/{solver_cls.__name__}"
    gp, y = _build_gp_and_y(kind, solver_cls)
    _, condgp = gp.condition(y)

    from_states = condgp.states()
    # return_v_S=False so the v_S slot is None on both sides; the
    # return_v_S=True asymmetry is asserted separately below.
    from_solver = condgp.solver.condition(y)

    # 1. Both are the 3-tuple (StateCoords, conditioned_states, v_S)
    assert isinstance(from_states, tuple) and len(from_states) == 3, f"[{label}]"
    assert isinstance(from_solver, tuple) and len(from_solver) == 3, f"[{label}]"
    assert isinstance(from_states[0], StateCoords), (
        f"[{label}] ConditionedStates.__call__ must yield a StateCoords, "
        f"got {type(from_states[0]).__name__}"
    )
    assert isinstance(from_solver[0], StateCoords), (
        f"[{label}] solver.condition must yield a StateCoords, "
        f"got {type(from_solver[0]).__name__}"
    )

    # 2. Identical pytree structure. Catches a field added to one producer
    #    but not the other, or any reordering -- without enumerating fields,
    #    so it keeps covering StateCoords as it grows.
    struct_states = jax.tree_util.tree_structure(from_states)
    struct_solver = jax.tree_util.tree_structure(from_solver)
    assert struct_states == struct_solver, (
        f"[{label}] pytree structure differs:\n"
        f"  ConditionedStates(): {struct_states}\n"
        f"  solver.condition():  {struct_solver}"
    )

    leaves_states = jax.tree_util.tree_leaves(from_states)
    leaves_solver = jax.tree_util.tree_leaves(from_solver)
    for i, (a, b) in enumerate(zip(leaves_states, leaves_solver)):
        assert a.shape == b.shape, f"[{label}] leaf {i} shape {a.shape} vs {b.shape}"
        assert a.dtype == b.dtype, f"[{label}] leaf {i} dtype {a.dtype} vs {b.dtype}"

    # 3. Identical values -- ConditionedStates is built FROM a condition()
    #    call, so this must hold to machine precision, not merely in shape.
    for i, (a, b) in enumerate(zip(leaves_states, leaves_solver)):
        assert jnp.allclose(a, b, atol=1e-12, rtol=0), (
            f"[{label}] leaf {i} values differ: max|d|={float(jnp.max(jnp.abs(a - b))):.3e}"
        )

    # 4. The one intentional asymmetry: ConditionedStates.__call__ hardcodes
    #    None for v_S, while condition(return_v_S=True) returns (v, S). The
    #    first two elements must still agree; predict() discards the third.
    with_vs = condgp.solver.condition(y, return_v_S=True)
    assert from_states[2] is None, (
        f"[{label}] ConditionedStates v_S slot should be None"
    )
    assert with_vs[2] is not None, f"[{label}] return_v_S=True should populate v_S"
    assert jax.tree_util.tree_structure(
        from_states[:2]
    ) == jax.tree_util.tree_structure(with_vs[:2]), (
        f"[{label}] only the v_S slot may differ when return_v_S=True"
    )


@pytest.mark.parametrize("kind,solver_cls", SOLVER_CASES)
def test_conditioned_states_call_interchangeable_in_predict(kind, solver_cls):
    """The functional consequence of the above: ``solver.predict()`` must
    give identical results whichever producer built its input."""
    label = f"{kind}/{solver_cls.__name__}"
    gp, y = _build_gp_and_y(kind, solver_cls)
    _, condgp = gp.condition(y)

    if kind == "instantaneous":
        X_test = jnp.linspace(-10.0, 60.0, 11)
        kwargs = {}
    else:
        # retrodict / interpolate / extrapolate, mixed delta and instid
        X_test = (
            jnp.array([-10.0, 5.0, 50.0, 115.0]),
            jnp.array([0.0, 4.0, 2.0, 3.0]),
            jnp.array([0, 1, 0, 1], dtype=int),
        )
        kwargs = {"y": y}

    mean_a, var_a = condgp.solver.predict(X_test, condgp.states(), **kwargs)
    mean_b, var_b = condgp.solver.predict(X_test, condgp.solver.condition(y), **kwargs)

    dm = float(jnp.max(jnp.abs(mean_a - mean_b)))
    dv = float(jnp.max(jnp.abs(var_a - var_b)))
    assert dm < 1e-12, f"[{label}] predict() mean differs by producer: {dm:.3e}"
    assert dv < 1e-12, f"[{label}] predict() var differs by producer: {dv:.3e}"


if __name__ == "__main__":
    test_state_coords_instantaneous_constructor()
    for p in [False, True]:
        test_state_coords_from_instantaneous_solver(p)
        for ni in [1, 2]:
            test_state_coords_from_integrated_solver(p, ni)
    test_state_coords_data_order_indices_roundtrip()
    test_state_coords_merge_test_coords_preserves_contract()
    test_state_coords_merge_exposure_test_coords_preserves_contract()
    for kind, solver_cls in SOLVER_CASES:
        test_conditioned_states_call_matches_solver_condition(kind, solver_cls)
        test_conditioned_states_call_interchangeable_in_predict(kind, solver_cls)
    print("All StateCoords invariant tests passed.")
