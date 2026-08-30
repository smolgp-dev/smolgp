"""The dedicated likelihood-only scan agrees with the general filter path.

``GaussianProcess.log_probability`` has two routes to the same number:

- the general one, which runs the full Kalman filter, keeps every ``v_k`` and
  ``S_k``, and finishes with a batched Cholesky in ``log_prob_from_v_S``;
- a trimmed scan (``kalman_loglike``) that emits nothing and accumulates the
  log likelihood in its carry. Instantaneous kernels only: the integrated
  solver uses the general route, its dedicated scan having measured <=1.45x on
  one kernel and 1.0x on the rest, which did not justify the duplication.

The second exists purely for speed, so every test here is an equivalence test:
the fast path must reproduce the general one, and so must its gradient, since
the whole point is to use it inside an optimizer.

The parallel solvers are the trap worth pinning. Each *subclasses* its
sequential counterpart, so each inherits a ``log_probability`` whose scan is
sequential; routing them there would compute the right number the slow way and
silently discard the parallelism they exist for. An explicit override back to
``Solver._log_probability_from_filter`` is what stops that, and the two
``*_uses_generic_path`` tests are what stop the override from being quietly
dropped.
"""

import jax
import jax.numpy as jnp
import pytest
from tinygp.helpers import JAXArray

import smolgp
from smolgp.helpers import kalman_gain
from smolgp.solvers import Solver
from smolgp.solvers.kalman import kalman_loglike

jax.config.update("jax_enable_x64", True)

# Exposures must not overlap for a single instrument: the likelihood is not
# defined there, so a regular cadence is used rather than random times.
TEXP, READOUT = 140.0, 40.0


def _plain(solver=None, N=24):
    w, Q, S = 2.0 * jnp.pi / 1000.0, 2.0, 2.5
    kernel = smolgp.kernels.SHO(omega=w, quality=Q, sigma=jnp.sqrt(S * w * Q))
    t = jnp.arange(N) * (TEXP + READOUT)
    y = jnp.sin(t / 500.0)
    kwargs = {} if solver is None else {"solver": solver}
    return smolgp.GaussianProcess(kernel, t, noise=jnp.full(N, 0.09), **kwargs), y


def _integrated(solver=None, N=24, num_insts=1):
    w, Q, S = 2.0 * jnp.pi / 1000.0, 2.0, 2.5
    kernel = smolgp.kernels.IntegratedSHO(
        omega=w, quality=Q, sigma=jnp.sqrt(S * w * Q), num_insts=num_insts
    )
    t = jnp.arange(N) * (TEXP + READOUT)
    X = (t, jnp.full(N, TEXP), jnp.zeros(N, dtype=int))
    y = jnp.sin(t / 500.0)
    kwargs = {} if solver is None else {"solver": solver}
    return smolgp.GaussianProcess(kernel, X, noise=jnp.full(N, 0.09), **kwargs), y


def _via_filter(gp, y):
    """The general path, bypassing whatever ``log_probability`` would pick.

    This is ``Solver._log_probability_from_filter`` itself: run the filter, then
    reduce its innovations. Calling it directly is what makes these equivalence
    tests independent of which route ``log_probability`` chooses.
    """
    return gp.solver._log_probability_from_filter(y)


BUILDERS = [(_plain, "instantaneous"), (_integrated, "integrated")]


@pytest.mark.parametrize("build,label", BUILDERS, ids=[b[1] for b in BUILDERS])
def test_matches_filter_path(build, label, monkeypatch):
    """Whichever route a solver takes, it must agree with the general one."""
    gp, y = build()
    expect_scan = label == "instantaneous"
    assert _calls_specialized_scan(gp, y, monkeypatch) is expect_scan
    assert jnp.allclose(gp.log_probability(y), _via_filter(gp, y), rtol=1e-10)


def test_integrated_uses_generic_path(monkeypatch):
    """The integrated solver inherits the general route rather than its own scan.

    It had a dedicated ``integrated_kalman_loglike``; that measured <=1.45x on
    one kernel and 1.0x on the others, because the split it relied on only pays
    at state dimension 2 and every integrated kernel but ``IntegratedExp`` is 3
    or more. It was removed in favour of the inherited implementation, and this
    pins that choice so the duplication is not reintroduced by accident.
    """
    gp, y = _integrated()
    assert not _calls_specialized_scan(gp, y, monkeypatch)
    assert jnp.allclose(gp.log_probability(y), _via_filter(gp, y), rtol=1e-10)


@pytest.mark.parametrize("build,label", BUILDERS, ids=[b[1] for b in BUILDERS])
def test_gradient_matches_filter_path(build, label):
    gp, y = build()
    fast = jax.grad(gp.log_probability)(y)
    slow = jax.grad(lambda yy: _via_filter(gp, yy))(y)
    assert jnp.allclose(fast, slow, rtol=1e-8)


@pytest.mark.parametrize("build,label", BUILDERS, ids=[b[1] for b in BUILDERS])
def test_gradient_wrt_hyperparameter(build, label):
    """The optimizer use case: differentiate through the kernel, not just y."""

    def llh(sigma, use_fast):
        w, Q = 2.0 * jnp.pi / 1000.0, 2.0
        N = 24
        t = jnp.arange(N) * (TEXP + READOUT)
        y = jnp.sin(t / 500.0)
        noise = jnp.full(N, 0.09)
        if label == "integrated":
            kernel = smolgp.kernels.IntegratedSHO(
                omega=w, quality=Q, sigma=sigma, num_insts=1
            )
            X = (t, jnp.full(N, TEXP), jnp.zeros(N, dtype=int))
        else:
            kernel = smolgp.kernels.SHO(omega=w, quality=Q, sigma=sigma)
            X = t
        gp = smolgp.GaussianProcess(kernel, X, noise=noise)
        return gp.log_probability(y) if use_fast else _via_filter(gp, y)

    fast = jax.grad(llh)(2.2, True)
    slow = jax.grad(llh)(2.2, False)
    assert jnp.allclose(fast, slow, rtol=1e-8)


def _calls_specialized_scan(gp, y, monkeypatch):
    """Whether ``gp.log_probability`` actually reaches a likelihood-only scan.

    Asked by observation rather than by inspecting the class: what matters is
    which code runs, and a solver can decline its own scan from the inside (a
    D > 1 model does exactly that). Introspecting the method cannot see that.

    Only ``KalmanLoglike`` is watched, because it is the only likelihood-only
    scan left. The integrated solver had one and no longer does, so for
    integrated models this returns False by construction -- which is the
    behaviour ``test_integrated_uses_generic_path`` pins.
    """
    import smolgp.solvers.solver as plain_mod

    called = []
    for mod, name in ((plain_mod, "KalmanLoglike"),):
        original = getattr(mod, name)

        def spy(*args, _orig=original, _name=name, **kwargs):
            called.append(_name)
            return _orig(*args, **kwargs)

        monkeypatch.setattr(mod, name, spy)

    gp.log_probability(y)
    return bool(called)


def test_parallel_integrated_uses_generic_path(monkeypatch):
    """It inherits the sequential scan, so the override is what keeps it out."""
    gp, y = _integrated(solver=smolgp.solvers.ParallelIntegratedStateSpaceSolver)
    assert not _calls_specialized_scan(gp, y, monkeypatch)
    assert jnp.allclose(gp.log_probability(y), _via_filter(gp, y), rtol=1e-10)
    reference, _ = _integrated()
    assert jnp.allclose(gp.log_probability(y), reference.log_probability(y), rtol=1e-9)


def test_parallel_plain_uses_generic_path(monkeypatch):
    """Same trap, now that ParallelStateSpaceSolver subclasses StateSpaceSolver."""
    gp, y = _plain(solver=smolgp.solvers.ParallelStateSpaceSolver)
    assert not _calls_specialized_scan(gp, y, monkeypatch)
    assert jnp.allclose(gp.log_probability(y), _via_filter(gp, y), rtol=1e-10)
    reference, _ = _plain()
    assert jnp.allclose(gp.log_probability(y), reference.log_probability(y), rtol=1e-9)


def test_solver_hierarchy():
    """The parallel solvers must keep inheriting from their serial versions."""
    assert issubclass(smolgp.solvers.StateSpaceSolver, Solver)
    assert issubclass(smolgp.solvers.IntegratedStateSpaceSolver, Solver)
    assert issubclass(
        smolgp.solvers.ParallelStateSpaceSolver, smolgp.solvers.StateSpaceSolver
    )
    assert issubclass(
        smolgp.solvers.ParallelIntegratedStateSpaceSolver,
        smolgp.solvers.IntegratedStateSpaceSolver,
    )


def test_base_solver_requires_its_interface():
    """Solver is an interface: the parts a subclass must supply must not be
    silently absent."""
    bare = Solver.__new__(Solver)
    for name, args in (
        ("Kalman", (None,)),
        ("RTS", (None,)),
        ("smoothing_gains", (None, None)),
        ("condition", (None,)),
        ("predict", (None, None)),
    ):
        with pytest.raises(NotImplementedError):
            getattr(bare, name)(*args)


class FFprime(smolgp.kernels.Wrapper):
    """A D = 2 observable -- the state and its derivative, independently.

    Lifted from ``docs/tutorials/multivariate.ipynb``. Both the LAPACK-free
    gain and the likelihood-only scan are D == 1 only, so this is the case that
    has to fall back to the general path.
    """

    scale: JAXArray | float
    sigma: JAXArray | float
    amp1: JAXArray | float
    amp2: JAXArray | float

    def __init__(self, scale, sigma=1.0, amp1=1.0, amp2=1.0, name="FFprime"):
        self.scale = scale
        self.sigma = sigma
        self.amp1 = amp1
        self.amp2 = amp2
        self.name = name
        self.kernel = smolgp.kernels.Matern52(scale=scale, sigma=sigma)

    def observation_matrix(self, X: JAXArray) -> JAXArray:
        del X
        return jnp.array([[self.amp1, 0, 0], [0, self.amp2, 0]])


def _multivariate(N=20):
    t = jnp.linspace(0.0, 20.0, N)
    R = jnp.tile(jnp.array([[0.09, 0.0], [0.0, 0.25]]), (N, 1, 1))
    y = jnp.stack([jnp.cos(t) + jnp.sin(2 * t), -jnp.sin(t) + jnp.cos(2 * t)], axis=-1)
    kernel = FFprime(scale=jnp.pi, sigma=1.0, amp1=3.0, amp2=1.5)
    return smolgp.GaussianProcess(kernel=kernel, X=t, noise=R), y


def test_multivariate_uses_the_scan_and_agrees(monkeypatch):
    """D = 2 goes through the same split scan, with no fallback and no change in value.

    The scan splits the y-independent factorisation from the y-dependent solve,
    which is valid at any D; only the final reduction branches, deferring to
    ``log_prob_from_v_S`` when D > 1. So there is nothing for D = 2 to fall back
    to, and the answer must still match the general filter path exactly.
    """
    gp, y = _multivariate()
    assert gp.noise.shape[-1] == 2
    assert _calls_specialized_scan(gp, y, monkeypatch)
    assert jnp.allclose(gp.log_probability(y), _via_filter(gp, y), rtol=1e-12)


def test_multivariate_gradient_unaffected():
    gp, y = _multivariate()
    fast = jax.grad(gp.log_probability)(y)
    slow = jax.grad(lambda yy: _via_filter(gp, yy))(y)
    assert jnp.allclose(fast, slow, rtol=1e-10)


@pytest.mark.parametrize("D", [1, 2, 3])
def test_kalman_gain_branches_agree(D):
    """The D == 1 shortcut must equal the LU solve it replaces."""
    key = jax.random.PRNGKey(D)
    k1, k2 = jax.random.split(key)
    dim = 4
    M = jax.random.normal(k1, (D, D))
    S = M @ M.T + D * jnp.eye(D)  # symmetric positive definite, like H P H^T + R
    PHt = jax.random.normal(k2, (dim, D))
    assert jnp.allclose(kalman_gain(S, PHt), jnp.linalg.solve(S.T, PHt.T).T, rtol=1e-10)


def test_integrated_likelihood_matches_a_single_fused_scan():
    """The integrated likelihood must reproduce a hand-written one-pass recursion.

    An independent check on the exposure-aware recursion, written out here in
    full rather than reusing any solver internals. What it pins is the reset
    handling: an exposure-start state updates the covariance but contributes no
    likelihood term, so a wrong ``stateid`` branch changes the answer without
    changing any shape.

    This used to compare against a dedicated ``integrated_kalman_loglike``. That
    scan is gone -- see the module docstring -- so the reference is now checked
    against whatever route the solver actually takes.
    """
    gp, y = _integrated(N=12)
    ks, sc = gp.kernel, gp.solver.state_coords
    y_nd = y[:, None]
    R = gp.solver.noise
    m0 = jnp.zeros(ks.dimension)
    P0 = ks.stationary_covariance()
    if not isinstance(P0, jnp.ndarray):
        P0 = P0.to_dense()
    A_aug, Q_aug = ks.transition_matrix, ks.process_noise
    RESET = ks.reset_matrix
    H = jax.vmap(ks.observation_model)(gp.solver.X)
    K_states = len(sc.t_states)
    zero = jnp.zeros(())

    def fused(y_in):
        """The pre-split recursion: one scan carrying (m, P, acc)."""
        from smolgp.helpers import transition_sequence

        A_all, Q_all = transition_sequence(A_aug, Q_aug, sc.t_states)

        def step(carry, data):
            m_prev, P_prev, acc = carry
            A_prev, Q_prev, k = data
            n = sc.obsid[k]
            m_pred = A_prev @ m_prev
            P_pred = A_prev @ P_prev @ A_prev.T + Q_prev

            def at_end():
                H_k = H[n]
                v = y_in[n] - H_k @ m_pred
                PHt = P_pred @ H_k.T
                S = H_k @ PHt + R[n]
                Kk = jnp.linalg.solve(S.T, PHt.T).T
                sk = S[0, 0]
                return (m_pred + Kk @ v, P_pred - Kk @ S @ Kk.T,
                        v[0] * v[0] / sk + jnp.log(sk))

            def at_start():
                Reset = RESET(sc.instid[n])
                return Reset @ m_pred, Reset @ P_pred @ Reset.T, zero

            m_k, P_k, term = jax.lax.cond(
                sc.stateid[k] == 0, lambda _: at_start(), lambda _: at_end(),
                operand=None,
            )
            return (m_k, P_k, acc + term), None

        (_, _, acc), _ = jax.lax.scan(
            step, (m0, P0, zero), (A_all, Q_all, jnp.arange(K_states))
        )
        return -0.5 * (acc + y_in.shape[0] * jnp.log(2.0 * jnp.pi))

    got = gp.solver.log_probability(y)
    assert jnp.allclose(got, fused(y_nd), rtol=1e-10)


def test_split_scan_matches_a_single_fused_scan():
    """The two-pass form must reproduce the one-pass Kalman recursion exactly.

    Guards the refactor itself: the split is only a partitioning of the same
    arithmetic, so any divergence means the passes have got out of step.
    """
    from smolgp.helpers import transition_sequence

    gp, y = _plain(N=40)
    ks = gp.kernel
    X_s, y_s, R_s = gp.solver._to_state_order(gp.solver.X, y[:, None], gp.solver.noise)
    t_s = ks.coord_to_sortable(X_s)
    H_all = jax.vmap(ks.observation_model)(X_s)
    P0 = ks.stationary_covariance()
    if not isinstance(P0, jnp.ndarray):
        P0 = P0.to_dense()
    A_all, Q_all = transition_sequence(ks.transition_matrix, ks.process_noise, t_s)
    m0 = jnp.zeros(ks.dimension)

    def fused(y_nd):
        def step(carry, data):
            m_prev, P_prev, acc = carry
            A, Q, H, R, yk = data
            m_pred = A @ m_prev
            P_pred = A @ P_prev @ A.T + Q
            v = yk - H @ m_pred
            PHt = P_pred @ H.T
            S = H @ PHt + R
            K = jnp.linalg.solve(S.T, PHt.T).T
            s = S[0, 0]
            return (m_pred + K @ v, P_pred - K @ S @ K.T,
                    acc + v[0] * v[0] / s + jnp.log(s)), None

        (_, _, acc), _ = jax.lax.scan(
            step, (m0, P0, jnp.zeros(())), (A_all, Q_all, H_all, R_s, y_nd)
        )
        return -0.5 * (acc + len(t_s) * jnp.log(2.0 * jnp.pi))

    split = kalman_loglike(
        ks.transition_matrix, ks.process_noise, H_all, R_s, t_s, y_s, m0, P0
    )
    assert jnp.allclose(split, fused(y_s), rtol=1e-10)
