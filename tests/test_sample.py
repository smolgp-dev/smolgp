import heapq
import itertools

import jax
import jax.numpy as jnp
import tinygp

import smolgp
from smolgp.helpers import assign_min_instids
from smolgp.solvers.sample import project_trajectory_at_data, sample_prior_trajectory
from tests.test_integrated import _tied_exposure_data
from tests.test_predict import _build_dataset
from tests.utils import generate_data

jax.config.update("jax_enable_x64", True)

# NOTE on sample axis conventions: smolgp's `.sample(key, shape=(M,))` returns
# shape `(N_data,) + shape` (N first, per its docstring), whereas tinygp's own
# `.sample()` returns `shape + (N_data,)` (N *last* -- confirmed by inspecting
# `jnp.moveaxis(dot_triangular(z), 0, -1)` where `z` has shape `(N,)+shape`).
# The two are handled with the appropriate axis in each helper below.


def _mean_cov(samples_N_first):
    """samples: shape (N, M). Returns (mean (N,), cov (N,N))."""
    return jnp.mean(samples_N_first, axis=-1), jnp.cov(samples_N_first)


# ---------------------------------------------------------------------------
# 1. sample_prior_trajectory / project_trajectory_at_data, in isolation
# ---------------------------------------------------------------------------


def _check_prior_trajectory_statistics(
    kernel, X, state_coords, key, M=100_000, rtol=0.05, label=""
):
    N = jnp.shape(jax.tree_util.tree_leaves(X)[0])[0]
    keys = jax.random.split(key, M)
    samples = jax.vmap(
        lambda k: project_trajectory_at_data(
            X,
            state_coords,
            sample_prior_trajectory(kernel, state_coords, k),
            kernel.observation_model,
            N,
        )
    )(keys)
    samples = jnp.moveaxis(samples, 0, -1)  # (N, M)
    assert jnp.all(jnp.isfinite(samples)), f"[{label}] trajectory samples have NaN/Inf"

    _, cov_emp = _mean_cov(samples)
    cov_true = kernel(X, X)
    scale = float(jnp.max(jnp.diag(cov_true)))
    diff = float(jnp.max(jnp.abs(cov_emp - cov_true)))
    assert diff < rtol * scale, (
        f"[{label}] empirical covariance mismatch: {diff:.3e} (scale={scale:.3e})"
    )
    print(
        f"    ...[{label}] empirical covariance (M={M}) matches analytic to "
        f"{diff:.2e} (scale={scale:.2e})"
    )


def test_prior_trajectory_instantaneous():
    """sample_prior_trajectory's projected covariance must match kernel.evaluate(X,X)."""
    key = jax.random.PRNGKey(0)
    kernels = {
        "SHO": smolgp.kernels.SHO(omega=0.2, quality=2.0, sigma=1.3),
        "Exp": smolgp.kernels.Exp(scale=5.0, sigma=1.1),
        "Matern32": smolgp.kernels.Matern32(scale=5.0, sigma=0.9),
    }
    t = jnp.linspace(0.0, 20.0, 12)
    for name, kernel in kernels.items():
        print(f"Testing prior trajectory statistics: {name}...")
        gp = smolgp.GaussianProcess(kernel=kernel, X=t, noise=1e-6 * jnp.ones(12))
        state_coords = gp.state_coords
        _check_prior_trajectory_statistics(kernel, t, state_coords, key, label=name)


def test_prior_trajectory_integrated():
    """Same check for IntegratedSHO with num_insts in {1, 2, 3} -- exercises the
    augmented/reset machinery and (for num_insts>=2) the structurally singular
    process noise handled by robust_sqrt."""
    key = jax.random.PRNGKey(1)
    S, w, Q = 2.5, 0.2, 2.0
    sigma = jnp.sqrt(S * w * Q)
    for Ninst in (1, 2, 3):
        label = f"IntegratedSHO num_insts={Ninst}"
        print(f"Testing prior trajectory statistics: {label}...")
        kernel = smolgp.kernels.IntegratedSHO(
            omega=w, quality=Q, sigma=sigma, num_insts=Ninst
        )
        N = 4 * Ninst
        t = jnp.linspace(0.0, 40.0, N)
        texp = jnp.full(N, 2.0)
        instid = jnp.array([i % Ninst for i in range(N)])
        X = (t, texp, instid)
        gp = smolgp.GaussianProcess(kernel=kernel, X=X, noise=1e-6 * jnp.ones(N))
        state_coords = gp.state_coords
        _check_prior_trajectory_statistics(kernel, X, state_coords, key, label=label)


def test_prior_trajectory_ties_are_finite():
    """Tie scenarios (zero-length transitions) must not produce NaN/Inf, even
    though process_noise is exactly singular there -- exercises robust_sqrt.
    Regression companion to issue #3's zero-length-transition fix, reusing
    the same tie-scenario builder as test_integrated.py."""
    key = jax.random.PRNGKey(2)
    kernel = smolgp.kernels.IntegratedSHO(
        omega=0.2, quality=2.0, sigma=1.0, num_insts=2
    )
    for tie in ["start-start", "end-end", "end-start"]:
        t, texp, instid, _tied_t = _tied_exposure_data(tie)
        X = (t, texp, instid)
        gp = smolgp.GaussianProcess(kernel=kernel, X=X, noise=jnp.full(t.shape, 0.01))
        state_coords = gp.state_coords
        keys = jax.random.split(key, 500)
        traj = jax.vmap(
            lambda k, state_coords=state_coords: sample_prior_trajectory(
                kernel, state_coords, k
            )
        )(keys)
        assert jnp.all(jnp.isfinite(traj)), f"[{tie}] trajectory contains NaN/Inf"
    print("    ...tie scenarios: prior trajectory sampling finite throughout")


# ---------------------------------------------------------------------------
# 2. .sample() on a prior GP, smolgp vs tinygp (each vs its own analytic ref)
# ---------------------------------------------------------------------------


def test_prior_sample_smolgp_vs_tinygp():
    """Each library's own .sample() must match its own analytic K+noise
    covariance -- avoids doubling Monte Carlo noise by comparing two noisy
    empirical estimates directly against each other."""
    sigma, scale, omega = 2.1, 30.0, 2 * jnp.pi / 30.0
    noise_val = 0.2
    kernels = {
        "Exp": (
            smolgp.kernels.Exp(scale=scale, sigma=sigma),
            tinygp.kernels.quasisep.Exp(scale=scale, sigma=sigma),
        ),
        "SHO": (
            smolgp.kernels.SHO(omega, 2.5, sigma),
            tinygp.kernels.quasisep.SHO(omega, 2.5, sigma),
        ),
    }
    t = jnp.linspace(0.0, 100.0, 15)
    M = 50_000

    for i, (name, (ksmol, ktiny)) in enumerate(kernels.items()):
        print(f"Testing prior sample vs tinygp: {name}...")
        key_smol, key_tiny = jax.random.split(jax.random.PRNGKey(100 + i))

        gp_smol = smolgp.GaussianProcess(
            kernel=ksmol, X=t, noise=jnp.full(15, noise_val**2)
        )
        samples_smol = gp_smol.sample(key_smol, shape=(M,))  # (N, M)
        assert jnp.all(jnp.isfinite(samples_smol))
        _, cov_emp_smol = _mean_cov(samples_smol)
        cov_true = ksmol(t, t) + jnp.eye(15) * noise_val**2
        scale_ref = float(jnp.max(jnp.diag(cov_true)))
        diff_smol = float(jnp.max(jnp.abs(cov_emp_smol - cov_true)))
        assert diff_smol < 0.05 * scale_ref, (
            f"[{name} smolgp] empirical covariance mismatch: {diff_smol:.3e}"
        )
        print(f"    ...[{name} smolgp] matches its own analytic cov to {diff_smol:.2e}")

        gp_tiny = tinygp.GaussianProcess(
            kernel=ktiny, X=t, diag=jnp.full(15, noise_val**2)
        )
        samples_tiny = gp_tiny.sample(key_tiny, shape=(M,))  # (M, N) -- N last!
        assert jnp.all(jnp.isfinite(samples_tiny))
        cov_emp_tiny = jnp.cov(samples_tiny.T)
        diff_tiny = float(jnp.max(jnp.abs(cov_emp_tiny - cov_true)))
        assert diff_tiny < 0.05 * scale_ref, (
            f"[{name} tinygp] empirical covariance mismatch: {diff_tiny:.3e}"
        )
        print(f"    ...[{name} tinygp] matches its own analytic cov to {diff_tiny:.2e}")


# ---------------------------------------------------------------------------
# 3. .sample() on a conditioned GP -- the actual residual-trick target
# ---------------------------------------------------------------------------


def _check_conditioned_sample_matches_condition(gp, y, key, M=50_000, label=""):
    """Check that the mean and variance of M samples drawn from a GP conditioned
    on y matches the analytic mean and variance from condition(y)."""
    _, condgp = gp.condition(y)
    samples = condgp.sample(key, shape=(M,))  # (N, M)
    assert jnp.all(jnp.isfinite(samples)), f"[{label}] posterior samples have NaN/Inf"

    mean_emp = jnp.mean(samples, axis=-1)
    var_emp = jnp.var(samples, axis=-1, ddof=1)

    scale = float(jnp.max(condgp.variance))
    diff_mean = float(jnp.max(jnp.abs(mean_emp - condgp.loc)))
    diff_var = float(jnp.max(jnp.abs(var_emp - condgp.variance)))
    assert diff_mean < 0.05 * jnp.sqrt(scale), (
        f"[{label}] posterior sample mean mismatch: {diff_mean:.3e}"
    )
    assert diff_var < 0.1 * scale, (
        f"[{label}] posterior sample variance mismatch: {diff_var:.3e}"
    )
    print(
        f"    ...[{label}] posterior samples (M={M}) match condition(): "
        f"|dmean|={diff_mean:.2e}, |dvar|={diff_var:.2e}"
    )


def test_conditioned_sample_instantaneous():
    """Check that the mean and variance of M samples drawn from a GP conditioned
    on y matches the analytic mean and variance from condition(y), for an
    instantaneous kernel sampled at the training coordinates."""
    kernel = smolgp.kernels.SHO(omega=0.2, quality=2.0, sigma=1.3)
    ktiny = tinygp.kernels.quasisep.SHO(omega=0.2, quality=2.0, sigma=1.3)
    t, y = generate_data(20, ktiny, yerr=0.2, tmin=0, tmax=100)
    gp = smolgp.GaussianProcess(kernel=kernel, X=t, noise=jnp.full(20, 0.2**2))
    _check_conditioned_sample_matches_condition(
        gp, y, jax.random.PRNGKey(10), label="instantaneous SHO"
    )


def test_conditioned_sample_instantaneous_parallel_solver():
    """ParallelStateSpaceSolver's condition()/predict() are independently
    validated against tinygp in test_parallel.py, so this doesn't re-check
    correctness against an analytic reference. What's untested elsewhere is
    whether GaussianProcess.sample()'s dispatch -- which routes any solver
    that isn't exactly StateSpaceSolver/IntegratedStateSpaceSolver through a
    generic, unoptimized per-sample vmap(condition) fallback (see _sample())
    -- drives ParallelStateSpaceSolver correctly. Checked the same way as
    test_exposure_sample_parallel_solver: same key and data, direct
    comparison against the serial solver, to near machine precision."""
    kernel = smolgp.kernels.SHO(omega=0.2, quality=2.0, sigma=1.3)
    ktiny = tinygp.kernels.quasisep.SHO(omega=0.2, quality=2.0, sigma=1.3)
    t, y = generate_data(20, ktiny, yerr=0.2, tmin=0, tmax=100)

    gp_seq = smolgp.GaussianProcess(kernel=kernel, X=t, noise=jnp.full(20, 0.2**2))
    gp_par = smolgp.GaussianProcess(
        kernel=kernel,
        X=t,
        noise=jnp.full(20, 0.2**2),
        solver=smolgp.solvers.ParallelStateSpaceSolver,
    )

    _, condgp_seq = gp_seq.condition(y)
    _, condgp_par = gp_par.condition(y)

    key = jax.random.PRNGKey(11)
    samples_seq = condgp_seq.sample(key, shape=(2000,))
    samples_par = condgp_par.sample(key, shape=(2000,))

    diff = float(jnp.max(jnp.abs(samples_seq - samples_par)))
    assert diff < 1e-8, f"parallel vs serial solver sample() mismatch: {diff:.3e}"
    print(
        f"    ...parallel solver sample() matches serial to {diff:.2e} (same key, instantaneous)"
    )


def test_conditioned_sample_integrated_serial_and_parallel():
    """Check that the mean and variance of M samples drawn from a GP conditioned
    on y matches the analytic mean and variance from condition(y), for an
    integrated kernel sampled at the training coordinates, for Ninst in {1, 2}
    on the serial solver.

    The parallel solver is checked separately below, by direct comparison
    against the serial solver (same key/data) rather than a second
    statistical check against condition() -- see
    test_conditioned_sample_instantaneous_parallel_solver's docstring for why
    that's the more targeted check for parallel-vs-serial agreement.
    """
    for i, Ninst in enumerate((1, 2)):
        label = f"Ninst={Ninst}, serial"
        print(f"Testing conditioned sample: {label}...")
        d = _build_dataset(Ninst, jax.random.PRNGKey(20 + i))
        _check_conditioned_sample_matches_condition(
            d["gp_smol"], d["y"], jax.random.PRNGKey(30 + i), label=label
        )

    d_seq = _build_dataset(2, jax.random.PRNGKey(22))
    d_par = _build_dataset(
        2,
        jax.random.PRNGKey(22),
        solver=smolgp.solvers.ParallelIntegratedStateSpaceSolver,
    )
    y = d_seq["y"]
    assert jnp.array_equal(y, d_par["y"])  # same key -> same underlying dataset

    key = jax.random.PRNGKey(32)
    _, condgp_seq = d_seq["gp_smol"].condition(y)
    _, condgp_par = d_par["gp_smol"].condition(y)
    samples_seq = condgp_seq.sample(key, shape=(2000,))
    samples_par = condgp_par.sample(key, shape=(2000,))

    diff = float(jnp.max(jnp.abs(samples_seq - samples_par)))
    assert diff < 1e-8, f"parallel vs serial solver sample() mismatch: {diff:.3e}"
    print(
        f"    ...parallel solver sample() matches serial to {diff:.2e} (same key/data, Ninst=2)"
    )


# ---------------------------------------------------------------------------
# 4. sample() with no X_test on a GP conditioned via condition(y, X_test=...):
#    must sample at the TRAINING coordinates (self.states.X), not whatever
#    predict coordinates condition()'s own X_test happened to be -- even
#    when they have a different length than the training data.
# ---------------------------------------------------------------------------


def test_sample_with_no_X_test_uses_training_coords_not_predict_coords():
    """If no X_test is passed to sample(), it will default to sample at
    the training coordinates, even if condition() was originally called
    with an X_test argument (this is meant to be used for a successive
    call to predict after conditioning, to populate loc and variance at
    the X_test coordinates rather than at the training coordinates)."""
    kernel = smolgp.kernels.IntegratedSHO(
        omega=0.2, quality=2.0, sigma=1.0, num_insts=1
    )
    N = 8
    t = jnp.linspace(0, 20, N)
    texp = jnp.full(N, 1.0)
    instid = jnp.zeros(N, dtype=int)
    gp = smolgp.GaussianProcess(
        kernel=kernel, X=(t, texp, instid), noise=jnp.full(N, 0.01)
    )
    key = jax.random.PRNGKey(42)
    y = gp.sample(key)

    # condition() at a custom X_test with a *different* length than training
    # (combining condition+predict in one call) -- self.num_data reflects
    # this predict-coordinate count (5), not the training count (8).
    t_test = jnp.linspace(0, 20, 5)
    X_test = (t_test, jnp.zeros(5), jnp.zeros(5, dtype=int))
    _, condgp_test = gp.condition(y, X_test=X_test)
    assert condgp_test.num_data == 5
    assert condgp_test.states.y.shape[0] == N

    # sample() with no X_test must still work, giving samples at the
    # training coordinates (shape N, not 5), matching condition(y)'s own
    # (no-X_test) result there.
    samples = condgp_test.sample(key, shape=(3000,))
    assert samples.shape[0] == N, f"expected training count {N}, got {samples.shape[0]}"
    assert jnp.all(jnp.isfinite(samples))

    _, condgp_plain = gp.condition(y)
    mean_emp = jnp.mean(samples, axis=-1)
    diff = float(jnp.max(jnp.abs(mean_emp - condgp_plain.loc)))
    assert diff < 0.05, (
        f"sample() at training coords doesn't match condition(y).loc: {diff:.3e}"
    )
    print(
        f"    ...sample() with no X_test correctly uses training coords "
        f"(N={N}) despite condition()'s own X_test (M=5): |dmean|={diff:.2e}"
    )


# ---------------------------------------------------------------------------
# 5. condition_batched_mean: correctness (must exactly match condition()) and
#    performance (must be meaningfully faster than per-sample condition())
# ---------------------------------------------------------------------------


def test_condition_batched_mean_matches_condition():
    """condition_batched_mean(residual_batch) must exactly reproduce
    condition(residual)'s smoothed mean for every residual in the batch --
    proves the gains-based mean-path replay is bit-identical to the
    original recursive path, for both M=1 and M>1, plain and integrated
    solvers (incl. num_insts=2 and a zero-length-transition tie scenario)."""
    key = jax.random.PRNGKey(100)

    # Plain StateSpaceSolver
    kernel = smolgp.kernels.SHO(omega=0.3, quality=2.0, sigma=1.2)
    t = jnp.linspace(0, 30, 20)
    gp = smolgp.GaussianProcess(kernel=kernel, X=t, noise=jnp.full(20, 0.05))
    M = 5
    residual_batch = jax.random.normal(key, (M, 20))
    m_batch_new = gp.solver.condition_batched_mean(residual_batch)
    m_batch_old = jnp.stack(
        [gp.solver.condition(residual_batch[i])[1][2][0] for i in range(M)]
    )
    diff = float(jnp.max(jnp.abs(m_batch_old - m_batch_new)))
    assert diff < 1e-9, f"plain solver mismatch: {diff:.3e}"
    print(
        f"    ...[plain, M={M}] condition_batched_mean matches condition(): {diff:.2e}"
    )

    # Integrated, num_insts in {1, 2}
    for Ninst in (1, 2):
        d = _build_dataset(Ninst, jax.random.PRNGKey(200 + Ninst))
        gp_i, y_i = d["gp_smol"], d["y"]
        residual_batch_i = jax.random.normal(key, (M,) + y_i.shape)
        m_batch_new_i = gp_i.solver.condition_batched_mean(residual_batch_i)
        m_batch_old_i = jnp.stack(
            [gp_i.solver.condition(residual_batch_i[k])[1][2][0] for k in range(M)]
        )
        diff_i = float(jnp.max(jnp.abs(m_batch_old_i - m_batch_new_i)))
        assert diff_i < 1e-8, f"integrated Ninst={Ninst} mismatch: {diff_i:.3e}"
        print(
            f"    ...[integrated Ninst={Ninst}, M={M}] "
            f"condition_batched_mean matches condition(): {diff_i:.2e}"
        )

    # Tie scenario (zero-length transitions) -- exercises get_smoothing_gain's
    # singular-covariance branch inside the new gain functions specifically.
    tie_kernel = smolgp.kernels.IntegratedSHO(
        omega=0.2, quality=2.0, sigma=1.0, num_insts=2
    )
    for tie in ["start-start", "end-end", "end-start"]:
        t_tie, texp_tie, instid_tie, _tied_t = _tied_exposure_data(tie)
        gp_tie = smolgp.GaussianProcess(
            kernel=tie_kernel, X=(t_tie, texp_tie, instid_tie), noise=jnp.full(6, 0.01)
        )
        residual_tie = jnp.sin(0.1 * t_tie)
        m_old_tie = gp_tie.solver.condition(residual_tie)[1][2][0]
        m_new_tie = gp_tie.solver.condition_batched_mean(residual_tie[None, :])[0]
        assert jnp.all(jnp.isfinite(m_new_tie)), (
            f"[{tie}] batched-mean produced NaN/Inf"
        )
        diff_tie = float(jnp.max(jnp.abs(m_old_tie - m_new_tie)))
        assert diff_tie < 1e-7, f"[{tie}] mismatch: {diff_tie:.3e}"
    print(
        "    ...tie scenarios: condition_batched_mean matches condition(), finite throughout"
    )


def test_condition_batched_mean_is_faster_for_many_samples():
    """At large enough N and M, condition_batched_mean(M residuals) must be
    meaningfully faster than vmap(condition) over the same M residuals --
    the whole point of sharing the covariance/gain recursion across samples.

    At small N/M (a few hundred/thousand), fixed JAX dispatch overhead
    dominates and the two are roughly a wash (measured ratio ~0.8-1.0); the
    asymptotic O(N*dim^3) vs O(N*dim^3 + M*N*dim*D) savings only becomes
    visible once actual compute dominates -- measured ratio ~0.43-0.54 at
    N=5000-10000, M=2000 in isolation. Using N=5000, M=1000 here with a
    deliberately loose threshold (0.9, not the ~0.5 seen in isolated manual
    benchmarking): this measurement is noticeably noisier when run as part
    of the full suite (shared JIT cache/GC pressure from preceding tests),
    so the assertion only needs to catch a real regression (e.g. the
    optimization silently falling back to the slow path), not to pin down
    the exact speedup -- see the benchmark harness / demo notebook for the
    actual, more dramatic numbers at scale.
    """
    import time

    N, M = 5000, 1000
    kernel = smolgp.kernels.SHO(omega=0.2, quality=2.0, sigma=1.3)
    t = jnp.linspace(0, 200, N)
    gp = smolgp.GaussianProcess(kernel=kernel, X=t, noise=jnp.full(N, 0.04))
    key = jax.random.PRNGKey(0)
    residual_batch = jax.random.normal(key, (M, N))

    # Wrap in plain lambdas (rather than jax.jit-ing the bound methods
    # directly) -- jax.jit's caching otherwise ends up hashing the bound
    # method's __self__ (the solver, an eqx.Module with array leaves),
    # which fails since arrays aren't hashable.
    old_fn = jax.jit(lambda rb: jax.vmap(gp.solver.condition)(rb))
    new_fn = jax.jit(lambda rb: gp.solver.condition_batched_mean(rb))

    jax.block_until_ready(old_fn(residual_batch))
    jax.block_until_ready(new_fn(residual_batch))

    # Take the min over a few repeats (standard microbenchmark practice) --
    # a single measurement is too sensitive to transient system load/noise,
    # e.g. from other tests running just before this one in the full suite.
    def _time_it(fn, n_repeat=5):
        times = []
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(residual_batch))
            times.append(time.perf_counter() - t0)
        return min(times)

    t_old = _time_it(old_fn)
    t_new = _time_it(new_fn)

    print(
        f"    ...old (vmap condition): {t_old:.4f}s, new (batched mean): {t_new:.4f}s"
    )
    assert t_new < 0.9 * t_old, (
        f"expected batched-mean to be at least somewhat faster: old={t_old:.4f}s new={t_new:.4f}s"
    )


# ---------------------------------------------------------------------------
# 6. Arbitrary test-grid sampling (delta=0): statistical match to predict()'s
#    analytic mean/variance, covering retrodict/interpolate/extrapolate.
# ---------------------------------------------------------------------------


def _check_grid_sample_matches_predict(gp, y, X_test, key, M=3000, label=""):
    """Check whether the mean of M samples drawn at the given test points matches
    the analytic mean from predict(). Likewise for the variance.

    The test points must be valid instantaneous coordinates."""
    _, condgp = gp.condition(y)
    samples = condgp.sample(key, shape=(M,), X_test=X_test)
    assert jnp.all(jnp.isfinite(samples)), f"[{label}] grid samples have NaN/Inf"

    mean_emp = jnp.mean(samples, axis=-1)
    var_emp = jnp.var(samples, axis=-1, ddof=1)
    mu_true, var_true = condgp.predict(X_test, return_var=True)

    scale = float(jnp.max(var_true))
    diff_mean = float(jnp.max(jnp.abs(mean_emp - mu_true)))
    diff_var = float(jnp.max(jnp.abs(var_emp - var_true)))
    assert diff_mean < 0.05 * jnp.sqrt(scale), (
        f"[{label}] grid sample mean mismatch: {diff_mean:.3e}"
    )
    assert diff_var < 0.1 * scale, (
        f"[{label}] grid sample variance mismatch: {diff_var:.3e}"
    )
    print(
        f"    ...[{label}] grid samples (M={M}) match predict(): "
        f"|dmean|={diff_mean:.2e}, |dvar|={diff_var:.2e}"
    )


def test_grid_sample_instantaneous_retrodict_interpolate_extrapolate():
    """Check that sampling at a grid of test points with an instantaneous GP
    reproduces the analytic mean/variance from predict() for retrodiction,
    interpolation, and extrapolation."""
    kernel = smolgp.kernels.SHO(omega=0.3, quality=2.0, sigma=1.2)
    ktiny = tinygp.kernels.quasisep.SHO(omega=0.3, quality=2.0, sigma=1.2)
    t, y = generate_data(15, ktiny, yerr=0.2, tmin=0, tmax=30)
    gp = smolgp.GaussianProcess(kernel=kernel, X=t, noise=jnp.full(15, 0.2**2))

    # A grid spanning before (retrodict), among (interpolate), and after
    # (extrapolate) the training range in one call.
    X_test = jnp.linspace(-10.0, 40.0, 25)
    _check_grid_sample_matches_predict(
        gp, y, X_test, jax.random.PRNGKey(40), label="instantaneous grid"
    )


def test_grid_sample_integrated_retrodict_interpolate_extrapolate():
    """Analagous to test_grid_sample_instantaneous_retrodict_interpolate_extrapolate(), but
    now passing explicitly exposure values of delta=0"""
    d = _build_dataset(2, jax.random.PRNGKey(42))
    gp_smol, y = d["gp_smol"], d["y"]
    tmin, tmax = (
        float(jnp.min(d["t"] - d["texp"] / 2)),
        float(jnp.max(d["t"] + d["texp"] / 2)),
    )

    t_test = jnp.linspace(tmin - 20.0, tmax + 20.0, 20)
    zeros = jnp.zeros_like(t_test)
    X_test = (t_test, zeros, zeros.astype(int))
    _check_grid_sample_matches_predict(
        gp_smol, y, X_test, jax.random.PRNGKey(43), label="integrated grid"
    )


def test_grid_sample_instantaneous_reproduces_training_point():
    """Confirm grid sampling reproduces training-point sampling when
    X_test == self.X, for a plain (non-integrated) kernel."""
    kernel = smolgp.kernels.SHO(omega=0.3, quality=2.0, sigma=1.2)
    ktiny = tinygp.kernels.quasisep.SHO(omega=0.3, quality=2.0, sigma=1.2)
    t, y = generate_data(15, ktiny, yerr=0.2, tmin=0, tmax=30)
    gp = smolgp.GaussianProcess(kernel=kernel, X=t, noise=jnp.full(15, 0.2**2))

    _check_grid_sample_matches_predict(
        gp, y, t, jax.random.PRNGKey(41), label="instantaneous X_test == training X"
    )


def test_exposure_sample_reproduces_training_point():
    """The integrated-kernel analog of
    test_grid_sample_instantaneous_reproduces_training_point() above:
    X_test == self.X (the real, possibly exposure-integrated training
    coordinates -- NOT forced to delta=0) must reproduce the conditioned
    mean at that point (condgp.loc) to within Monte Carlo noise, since the
    exposure integral is exactly the same as the training point's own
    exposure."""
    d = _build_dataset(2, jax.random.PRNGKey(58))
    gp_smol, y, t, texp, instid = d["gp_smol"], d["y"], d["t"], d["texp"], d["instid"]
    _, condgp = gp_smol.condition(y)
    samples = condgp.sample(
        jax.random.PRNGKey(59), shape=(3000,), X_test=(t, texp, instid)
    )
    mean_emp = jnp.mean(samples, axis=-1)
    diff = float(jnp.max(jnp.abs(mean_emp - condgp.loc)))
    assert diff < 0.05, (
        f"exposure X_test at training points doesn't reproduce condgp.loc: {diff:.3e}"
    )
    print(
        f"    ...exposure sampling at exact training (t,texp,instid) reproduces condgp.loc: {diff:.2e}"
    )


def test_grid_sample_ties_at_exposure_boundaries():
    """A delta=0 test point exactly at a real exposure start/end must still
    give finite, correct results (zero-length transition to/from it)."""
    kernel = smolgp.kernels.IntegratedSHO(
        omega=0.2, quality=2.0, sigma=1.0, num_insts=2
    )
    for tie in ["start-start", "end-end", "end-start"]:
        t_tie, texp_tie, instid_tie, tied_t = _tied_exposure_data(tie)
        gp = smolgp.GaussianProcess(
            kernel=kernel, X=(t_tie, texp_tie, instid_tie), noise=jnp.full(6, 0.01)
        )
        y = jnp.sin(0.1 * t_tie)
        X_test = (jnp.array([tied_t, 5.0]), jnp.zeros(2), jnp.zeros(2, dtype=int))
        _check_grid_sample_matches_predict(
            gp, y, X_test, jax.random.PRNGKey(44), M=1000, label=f"tie={tie}"
        )


# ---------------------------------------------------------------------------
# 7. Exposure-integrated (delta>0) test-point sampling
# ---------------------------------------------------------------------------


def _check_exposure_sample_matches_predict(gp, y, X_test, key, M=3000, label=""):
    """Check whether the mean of M samples drawn at the given exposure-integrated
    test points matches the analytic mean from predict_exposure(), and likewise
    for the variance. The test points must be valid exposure-integrated coordinates."""
    _, condgp = gp.condition(y)
    samples = condgp.sample(key, shape=(M,), X_test=X_test)
    assert jnp.all(jnp.isfinite(samples)), f"[{label}] exposure samples have NaN/Inf"

    mean_emp = jnp.mean(samples, axis=-1)
    var_emp = jnp.var(samples, axis=-1, ddof=1)
    mu_true, var_true = condgp.predict(X_test, y=y, return_var=True)

    scale = float(jnp.max(var_true))
    diff_mean = float(jnp.max(jnp.abs(mean_emp - mu_true)))
    diff_var = float(jnp.max(jnp.abs(var_emp - var_true)))
    assert diff_mean < 0.05 * jnp.sqrt(scale), (
        f"[{label}] exposure sample mean mismatch: {diff_mean:.3e}"
    )
    assert diff_var < 0.15 * scale, (
        f"[{label}] exposure sample variance mismatch: {diff_var:.3e}"
    )
    print(
        f"    ...[{label}] exposure samples (M={M}) match predict_exposure(): "
        f"|dmean|={diff_mean:.2e}, |dvar|={diff_var:.2e}"
    )
    return samples


def test_exposure_sample_single_point():
    """Test a single delta>0 exposure-integrated test point, covering the
    simplest case of the residual-trick sampling machinery."""
    d = _build_dataset(1, jax.random.PRNGKey(50))
    gp_smol, y = d["gp_smol"], d["y"]
    X_test = (jnp.array([50.0]), jnp.array([8.0]), jnp.array([0], dtype=int))
    _check_exposure_sample_matches_predict(
        gp_smol, y, X_test, jax.random.PRNGKey(51), label="single delta>0 point"
    )


def test_exposure_sample_retrodict_interpolate_extrapolate():
    """Test a set of delta>0 exposure-integrated test points spanning before,
    among, and after the training range, covering retrodiction, interpolation,
    and extrapolation in one call."""
    d = _build_dataset(1, jax.random.PRNGKey(52))
    gp_smol, y, t, texp = d["gp_smol"], d["y"], d["t"], d["texp"]
    tmin, tmax = float(jnp.min(t - texp / 2)), float(jnp.max(t + texp / 2))
    X_test = (
        jnp.array([tmin - 15.0, 0.5 * (tmin + tmax), tmax + 15.0]),
        jnp.array([6.0, 8.0, 6.0]),
        jnp.zeros(3, dtype=int),
    )
    _check_exposure_sample_matches_predict(
        gp_smol, y, X_test, jax.random.PRNGKey(53), label="retrodict/interp/extrap"
    )


def test_exposure_sample_multiple_points_are_correlated():
    """Two overlapping/nearby delta>0 test-point samples must be mutually
    correlated (as a single coherent draw of the underlying process), not
    independently resampled. The *diagonal* (marginal variance at each
    point) is checked against predict_exposure()'s analytic value as usual;
    the cross-covariance has no equally-cheap analytic reference (it's a
    posterior, not prior, cross-covariance), so it's checked qualitatively:
    nearby points must come out far more correlated than a distant one.
    """
    d = _build_dataset(2, jax.random.PRNGKey(54))
    gp_smol, y = d["gp_smol"], d["y"]

    t_test = jnp.array(
        [30.0, 32.0, 80.0]
    )  # first two close (and overlapping), third far
    delta_test = jnp.array([5.0, 5.0, 5.0])
    instid_test = jnp.array(
        [0, 1, 0]
    )  # overlapping pair distinct; distant point reuses 0
    X_test = (t_test, delta_test, instid_test)

    samples = _check_exposure_sample_matches_predict(
        gp_smol,
        y,
        X_test,
        jax.random.PRNGKey(55),
        M=20_000,
        label="multi-probe diagonal",
    )
    cov_emp = jnp.cov(samples)

    # The key check: points 1&2 (close) must be much more correlated than
    # point 3 (far) is with either -- confirms joint (not independent) sampling.
    corr_12 = float(cov_emp[0, 1] / jnp.sqrt(cov_emp[0, 0] * cov_emp[1, 1]))
    corr_13 = float(cov_emp[0, 2] / jnp.sqrt(cov_emp[0, 0] * cov_emp[2, 2]))
    # corr_12 should come out stable around 0.67, while corr_13 should be around ±0.01
    # The 3x threshold here translates to a 28 sigma difference
    assert corr_12 > 3 * abs(corr_13), (
        f"expected nearby points far more correlated than distant ones: "
        f"corr_12={corr_12:.3f}, corr_13={corr_13:.3f}"
    )
    print(
        f"    ...multiple probes: corr(close)={corr_12:.3f} >> corr(far)={corr_13:.3f}"
    )


def test_exposure_sample_mixed_delta0_and_delta_gt0():
    """A mixed set of delta=0 and delta>0 test points must still produce
    correct results, with the delta=0 points following the usual predict()
    mean/variance and the delta>0 points following the predict_exposure()
    mean/variance."""
    d = _build_dataset(2, jax.random.PRNGKey(56))
    gp_smol, y = d["gp_smol"], d["y"]
    X_test = (
        jnp.array([10.0, 50.0, 90.0]),
        jnp.array([0.0, 8.0, 0.0]),
        jnp.array([0, 1, 0], dtype=int),
    )
    _check_exposure_sample_matches_predict(
        gp_smol, y, X_test, jax.random.PRNGKey(57), label="mixed delta=0/delta>0"
    )


def test_exposure_sample_ties_at_boundaries():
    """A delta>0 exposure-integrated test point whose start/end exactly
    matches a training point's exposure start/end must still produce finite,
    correct results (zero-length transition to/from it)."""
    kernel = smolgp.kernels.IntegratedSHO(
        omega=0.2, quality=2.0, sigma=1.0, num_insts=2
    )
    for tie in ["start-start", "end-end", "end-start"]:
        t_tie, texp_tie, instid_tie, tied_t = _tied_exposure_data(tie)
        gp = smolgp.GaussianProcess(
            kernel=kernel, X=(t_tie, texp_tie, instid_tie), noise=jnp.full(6, 0.01)
        )
        y = jnp.sin(0.1 * t_tie)
        # A delta>0 query whose start/end boundary coincides with the tie.
        X_test = (jnp.array([tied_t + 1.0]), jnp.array([2.0]), jnp.zeros(1, dtype=int))
        _check_exposure_sample_matches_predict(
            gp, y, X_test, jax.random.PRNGKey(60), M=1000, label=f"exposure tie={tie}"
        )


def test_exposure_sample_parallel_solver():
    """ParallelIntegratedStateSpaceSolver only overrides condition()'s
    Kalman/RTS recursion; it inherits predict()/predict_exposure() from
    IntegratedStateSpaceSolver. So for the exact same key and X_test, its
    sample() output must match the sequential solver's to near machine
    precision.This check essentially ensures that the parallel solver's
    condition output is equal to the sequential solver's, and that nothing
    else in between has fallen out of sync."""
    key = jax.random.PRNGKey(61)
    d_seq = _build_dataset(2, key)
    d_par = _build_dataset(
        2,
        key,
        solver=smolgp.solvers.ParallelIntegratedStateSpaceSolver,
    )
    y = d_seq["y"]
    assert jnp.array_equal(y, d_par["y"])  # same key -> same underlying dataset

    X_test = (
        jnp.array([30.0, 70.0]),
        jnp.array([6.0, 5.0]),
        jnp.array([0, 1], dtype=int),
    )
    key = jax.random.PRNGKey(62)

    _, condgp_seq = d_seq["gp_smol"].condition(y)
    _, condgp_par = d_par["gp_smol"].condition(y)

    samples_seq = condgp_seq.sample(key, shape=(2000,), X_test=X_test)
    samples_par = condgp_par.sample(key, shape=(2000,), X_test=X_test)

    diff = float(jnp.max(jnp.abs(samples_seq - samples_par)))
    assert diff < 1e-8, f"parallel vs sequential solver sample() mismatch: {diff:.3e}"
    print(
        f"    ...parallel solver sample() matches sequential to {diff:.2e} (same key/X_test)"
    )


def test_exposure_sample_matches_tinygp_directly():
    """Check that a large number of exposure-integrated samples from smolgp's
    IntegratedSHO kernel match the tinygp dense IntegratedSHOKernel's analytic
    mean/variance, for a conditioned GP. Because each draws with RNG, we can
    only compare the two distributions statistically"""
    d = _build_dataset(1, jax.random.PRNGKey(63))
    gp_smol, gp_tiny, y = d["gp_smol"], d["gp_tiny"], d["y"]

    X_test = (
        jnp.array([-10.0, 50.0, 110.0]),
        jnp.array([6.0, 8.0, 6.0]),
        jnp.zeros(3, dtype=int),
    )

    _, condgp_smol = gp_smol.condition(y)
    samples_smol = condgp_smol.sample(
        jax.random.PRNGKey(64), shape=(4000,), X_test=X_test
    )

    _, condgp_tiny_test = gp_tiny.condition(y, X_test)
    samples_tiny = condgp_tiny_test.sample(
        jax.random.PRNGKey(65), shape=(4000,)
    )  # (M, N)!

    mean_smol, var_smol = (
        jnp.mean(samples_smol, axis=-1),
        jnp.var(samples_smol, axis=-1, ddof=1),
    )
    mean_tiny, var_tiny = (
        jnp.mean(samples_tiny, axis=0),
        jnp.var(samples_tiny, axis=0, ddof=1),
    )
    offset = float(jnp.sqrt(jnp.finfo(jnp.array([0.0])).eps))

    scale = float(jnp.max(var_tiny))
    diff_mean = float(jnp.max(jnp.abs(mean_smol - mean_tiny)))
    diff_var = float(jnp.max(jnp.abs(var_smol - (var_tiny - offset))))
    assert diff_mean < 0.05 * jnp.sqrt(scale), (
        f"mismatch vs tinygp mean: {diff_mean:.3e}"
    )
    assert diff_var < 0.15 * scale, f"mismatch vs tinygp variance: {diff_var:.3e}"
    print(
        f"    ...smolgp vs tinygp condition(X_test).sample(): "
        f"|dmean|={diff_mean:.2e}, |dvar|={diff_var:.2e}"
    )


# ---------------------------------------------------------------------------
# 8. Prior sampling at delta>0 test points (X_test on an unconditioned GP)
# ---------------------------------------------------------------------------


def test_prior_sample_exposure_test_points_matches_dense_kernel():
    """Check that a large number of prior samples with delta>0 have a
    covariance that matches the dense IntegratedSHOKernel's analytic
    covariance."""
    S, w, Q = 2.5, 0.2, 2.0
    sigma = jnp.sqrt(S * w * Q)
    kernel = smolgp.kernels.IntegratedSHO(omega=w, quality=Q, sigma=sigma, num_insts=1)
    kernel_tiny = smolgp.kernels.dense.IntegratedSHOKernel(S=S, w=w, Q=Q)
    # Define dummy GP
    gp = smolgp.GaussianProcess(
        kernel=kernel,
        X=(jnp.array([0.0]), jnp.array([0.0]), jnp.array([0], dtype=int)),
        noise=jnp.array([1e-8]),
    )

    t_test = jnp.array([10.0, 30.0, 55.0])
    delta_test = jnp.array([4.0, 0.0, 6.0])  # mixed delta=0/delta>0
    instid_test = jnp.zeros(3, dtype=int)
    X_test = (t_test, delta_test, instid_test)

    M = 50_000
    samples = gp.sample(jax.random.PRNGKey(70), shape=(M,), X_test=X_test)
    assert jnp.all(jnp.isfinite(samples))

    cov_emp = jnp.cov(samples)
    cov_true = kernel_tiny(X_test, X_test)
    diff = float(jnp.max(jnp.abs(cov_emp - cov_true)))
    scale = float(jnp.max(jnp.diag(cov_true)))
    assert diff < 0.05 * scale, f"prior exposure-sample covariance mismatch: {diff:.3e}"
    print(f"    ...prior sample at delta>0 X_test matches dense kernel cov: {diff:.2e}")


# ---------------------------------------------------------------------------
# 9. Exact self-consistency: a single draw's exposure-integrated values must
#    equal the quadrature integral of that SAME draw's instantaneous curve
# ---------------------------------------------------------------------------


def test_sample_exposure_matches_quadrature_of_dense_curve_same_draw():
    """Confirm that a single draw's exposure-integrated values match the
    trapezoidal integral of that SAME draw's dense instantaneous curve over
    the corresponding windows. Should match up to quadrature error. Because
    of the way random numbers work, we have to draw both the exposure and
    dense samples in a single call to sample()
    """
    kernel = smolgp.kernels.IntegratedSHO(
        omega=0.2, quality=2.0, sigma=1.0, num_insts=1
    )
    # GP with dummy data
    gp = smolgp.GaussianProcess(
        kernel=kernel,
        X=(jnp.array([0.0]), jnp.array([0.0]), jnp.array([0], dtype=int)),
        noise=jnp.array([1e-8]),
    )
    # Sample 8 exposures from the GP, and 400 dense points in the same range
    tmid = jnp.linspace(0.0, 50.0, 8)
    texp = jnp.full(8, 4.0)
    instid = jnp.zeros(8, dtype=int)
    t_dense = jnp.linspace(-5.0, 55.0, 400)

    X_test_combined = (
        jnp.concatenate([t_dense, tmid]),
        jnp.concatenate([jnp.zeros_like(t_dense), texp]),
        jnp.concatenate([jnp.zeros_like(t_dense, dtype=int), instid]),
    )
    y_combined = gp.sample(jax.random.PRNGKey(2), X_test=X_test_combined)
    y_dense = y_combined[: len(t_dense)]
    y_exposure = y_combined[len(t_dense) :]
    assert jnp.all(jnp.isfinite(y_combined))

    def trapz_window(t_c, delta_c):
        """Integrate the dense curve over t_c +/- delta_c/2 using trapezoidal rule."""
        tt = jnp.linspace(t_c - delta_c / 2, t_c + delta_c / 2, 300)
        yy = jnp.interp(tt, t_dense, y_dense)
        return jnp.trapezoid(yy, tt) / delta_c

    y_quad = jax.vmap(trapz_window)(tmid, texp)
    diff = float(jnp.max(jnp.abs(y_exposure - y_quad)))
    assert diff < 1e-2, (
        f"exposure sample doesn't match quadrature of the same draw: {diff:.3e}"
    )
    print(
        f"    ...exposure entries of a combined draw match quadrature of its own dense entries: {diff:.2e}"
    )


# ---------------------------------------------------------------------------
# assign_min_instids: minimal non-overlapping instid_test grouping
# ---------------------------------------------------------------------------


def _conflicts(a, b, i, j):
    """Whether windows i and j must occupy different groups.

    A window needs its integral accumulator reset at ``a`` and read out at
    ``b``, so two windows can share one only if the earlier is read out before
    the later is reset. For ``a_i <= a_j`` that is ``a_j >= b_i``; they
    conflict when ``a_j < b_i``.

    Note this is deliberately *not* interval intersection. A zero-width window
    has an empty span but still needs an accumulator for an instant, so one
    landing inside another window's span conflicts with it even though the
    intersection of ``[x, x]`` with anything has measure zero. Getting this
    wrong is the whole reason the previous reference was replaced.
    """
    lo, hi = (i, j) if a[i] <= a[j] else (j, i)
    return a[hi] < b[lo]


def _min_groups_greedy(t, delta):
    """Reference minimum group count, by the sequential greedy sweep.

    This is a plain-Python implementation of the heap algorithm before
    we needed it to be jittable. Verifies the jit version is correct.

    Sweeping windows in order of start time and reusing whichever group freed
    earliest is optimal for interval graphs, so the count it returns is the
    true minimum.
    """
    a = [float(x) for x in (jnp.asarray(t) - jnp.asarray(delta) / 2)]
    b = [float(x) for x in (jnp.asarray(t) + jnp.asarray(delta) / 2)]
    order = sorted(range(len(a)), key=lambda i: a[i])

    heap = []  # (end_time, group id), min-heap on end time
    instid = [0] * len(a)
    next_id = 0
    for i in order:
        if heap and heap[0][0] <= a[i]:
            _end, gid = heapq.heappop(heap)
        else:
            gid = next_id
            next_id += 1
        instid[i] = gid
        heapq.heappush(heap, (b[i], gid))
    return instid, next_id


def _min_groups_bruteforce(t, delta, max_n=9):
    """Exhaustive chromatic number of the conflict graph, for small inputs.

    An independent check on :func:`_min_groups_greedy` itself -- it shares no
    logic with either the greedy sweep or the implementation, it just tries
    every colouring until one is valid.
    """
    a = [float(x) for x in (jnp.asarray(t) - jnp.asarray(delta) / 2)]
    b = [float(x) for x in (jnp.asarray(t) + jnp.asarray(delta) / 2)]
    n = len(a)
    assert n <= max_n, f"brute force is exponential; {n} windows is too many"
    pairs = [(i, j) for i, j in itertools.combinations(range(n), 2)
             if _conflicts(a, b, i, j)]
    for k in range(1, n + 1):
        for colouring in itertools.product(range(k), repeat=n):
            if all(colouring[i] != colouring[j] for i, j in pairs):
                return k
    return n


def _assert_groups_valid(t, delta, instid, num_groups):
    """No two conflicting windows share a group, and ids are in range."""
    a = [float(x) for x in (jnp.asarray(t) - jnp.asarray(delta) / 2)]
    b = [float(x) for x in (jnp.asarray(t) + jnp.asarray(delta) / 2)]
    ids = [int(x) for x in jnp.asarray(instid)]
    assert all(0 <= g < max(num_groups, 1) for g in ids), (
        f"group ids {ids} outside 0..{num_groups - 1}"
    )
    for i, j in itertools.combinations(range(len(a)), 2):
        if _conflicts(a, b, i, j):
            assert ids[i] != ids[j], (
                f"windows {i} [{a[i]},{b[i]}] and {j} [{a[j]},{b[j]}] conflict "
                f"but both landed in group {ids[i]}"
            )


def _assert_no_within_group_overlap(t, delta, instid, num_groups):
    a = jnp.asarray(t) - jnp.asarray(delta) / 2
    b = jnp.asarray(t) + jnp.asarray(delta) / 2
    instid = jnp.asarray(instid)
    for gid in range(num_groups):
        idx = jnp.nonzero(instid == gid)[0]
        idx = idx[jnp.argsort(a[idx])]
        for k in range(len(idx) - 1):
            assert a[idx[k + 1]] >= b[idx[k]] - 1e-9, (
                f"group {gid}: window {int(idx[k + 1])} overlaps window {int(idx[k])}"
            )


def test_assign_min_instids_no_overlap_within_group():
    """A moderately busy randomized set of windows: every group assign_min_instids
    produces must itself be free of any internal overlap."""
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    t = jax.random.uniform(k1, (60,), minval=0.0, maxval=200.0)
    delta = jax.random.uniform(k2, (60,), minval=1.0, maxval=40.0)

    instid, num_groups = assign_min_instids(t, delta)
    _assert_no_within_group_overlap(t, delta, instid, num_groups)


def test_assign_min_instids_is_minimal():
    """Independent check that assign_min_instids does return the true minimum number of groups"""
    key = jax.random.PRNGKey(1)
    k1, k2 = jax.random.split(key)
    t = jax.random.uniform(k1, (80,), minval=0.0, maxval=100.0)
    delta = jax.random.uniform(k2, (80,), minval=0.5, maxval=15.0)

    instid, num_groups = assign_min_instids(t, delta)
    _assert_groups_valid(t, delta, instid, num_groups)
    _ref_ids, expected = _min_groups_greedy(t, delta)
    assert num_groups == expected, (
        f"assign_min_instids used {num_groups} groups, but the sequential "
        f"greedy reference finds the true minimum to be {expected}"
    )


def test_assign_min_instids_reuses_freed_group():
    """The chain A=[0,10], B=[8,20], C=[18,30]: A overlaps B, B overlaps C,
    but A and C don't overlap each other. Simple check that the true minimum
    of 2 groups is achieved and C reuses A's group (A&C have id=0, B has id=1)"""
    t = jnp.array([5.0, 14.0, 24.0])
    delta = jnp.array([10.0, 12.0, 12.0])  # windows: [0,10], [8,20], [18,30]

    instid, num_groups = assign_min_instids(t, delta)
    assert num_groups == 2, f"expected the true minimum of 2 groups, got {num_groups}"
    assert instid[0] == instid[2] == 0 and instid[0] != instid[1] and instid[1] == 1, (
        f"expected A&C to share group 0, B to have group 1: got instid={instid}"
    )
    _assert_no_within_group_overlap(t, delta, instid, num_groups)


def test_assign_min_instids_touching_windows_share_a_group():
    """A window ending exactly when the next starts is "touching," not
    overlapping, should match merge_exposure_test_coords's own tie-break
    convention (an exposure end is processed before a start at equal
    times). They may safely share a group."""
    t = jnp.array([5.0, 15.0])
    delta = jnp.array([10.0, 10.0])  # windows: [0,10] and [10,20] -- touch at 10

    instid, num_groups = assign_min_instids(t, delta)
    assert num_groups == 1
    assert instid[0] == instid[1] == 0


def test_assign_min_instids_all_zero_width_share_one_group():
    """Zero-width (delta=0) windows never truly overlap anything (a==b for
    each), so a batch of them -- even at duplicate times -- should all
    collapse onto a single group."""
    t = jnp.array([1.0, 1.0, 5.0, 9.0])
    delta = jnp.zeros(4)

    _instid, num_groups = assign_min_instids(t, delta)
    assert num_groups == 1


def test_assign_min_instids_mixed_zero_and_positive_width():
    """Zero-width windows mixed in with real exposures.

    This is the case that distinguishes a correct implementation from two
    plausible-looking wrong ones:

    Windows: w0=[0,0], w1=[-1.5,3.5], w2=[2,2], w3=[0.5,5.5].

    w1, w2 and w3 mutually conflict, but w2 is an instantaneous readout sitting
    strictly inside both w1 and w3, and resetting a shared accumulator at t=2
    would destroy whichever integral was mid-flight. Hence, three groups are
    needed. w0 is free to share with w2, since t=0 lies outside [0.5, 5.5],
    which is why the answer is exactly 3 rather than 4.
    """
    t = jnp.array([0.0, 1.0, 2.0, 3.0])
    delta = jnp.array([0.0, 5.0, 0.0, 5.0])

    instid, num_groups = assign_min_instids(t, delta)
    _assert_groups_valid(t, delta, instid, num_groups)
    assert num_groups == 3, (
        f"expected 3 groups for the mixed zero/positive-width case, got "
        f"{num_groups} (instid={[int(x) for x in instid]})"
    )
    # Two independent references must agree on that 3.
    assert _min_groups_bruteforce(t, delta) == 3
    assert _min_groups_greedy(t, delta)[1] == 3


def test_assign_min_instids_zero_width_inside_one_exposure():
    """The minimal version of the same trap: a single instantaneous readout
    inside a single exposure needs its own accumulator, so 2 groups, not 1."""
    t = jnp.array([5.0, 5.0])
    delta = jnp.array([10.0, 0.0])  # [0,10] and [5,5]

    instid, num_groups = assign_min_instids(t, delta)
    _assert_groups_valid(t, delta, instid, num_groups)
    assert num_groups == 2, f"expected 2 groups, got {num_groups}"
    assert _min_groups_bruteforce(t, delta) == 2


def test_assign_min_instids_matches_greedy_reference_on_mixed_widths():
    """Fuzz the implementation against the sequential greedy reference, with a
    healthy fraction of zero-width windows in the mix.

    ``assign_min_instids`` was rewritten from that greedy heap into a numpy
    max-clique count plus a jittable ``lax.scan`` assignment; this is the
    regression test that the same number of groups is returned, though the
    ids themselves can be different as long as they are valid.
    """
    key = jax.random.PRNGKey(20260817)
    for trial in range(200):
        key, k1, k2, k3 = jax.random.split(key, 4)
        n = int(jax.random.randint(k1, (), 1, 12))
        t = jax.random.uniform(k2, (n,), minval=0.0, maxval=50.0)
        widths = jax.random.uniform(k3, (n,), minval=0.0, maxval=15.0)
        # zero out ~35% of the widths, so degenerate and real windows mix
        zero = jax.random.uniform(k3, (n,)) < 0.35
        delta = jnp.where(zero, 0.0, widths)

        instid, num_groups = assign_min_instids(t, delta)
        _assert_groups_valid(t, delta, instid, num_groups)
        _ref_ids, ref_groups = _min_groups_greedy(t, delta)
        assert num_groups == ref_groups, (
            f"trial {trial}: implementation used {num_groups} groups, greedy "
            f"reference {ref_groups}; t={t}, delta={delta}"
        )


def test_min_groups_greedy_matches_bruteforce():
    """The greedy reference is itself checked against exhaustive colouring, so
    the fuzz test above is not resting on an unverified oracle."""
    key = jax.random.PRNGKey(7)
    for _ in range(40):
        key, k1, k2, k3 = jax.random.split(key, 4)
        n = int(jax.random.randint(k1, (), 1, 8))
        t = jax.random.uniform(k2, (n,), minval=0.0, maxval=30.0)
        widths = jax.random.uniform(k3, (n,), minval=0.0, maxval=12.0)
        delta = jnp.where(jax.random.uniform(k3, (n,)) < 0.4, 0.0, widths)
        assert _min_groups_greedy(t, delta)[1] == _min_groups_bruteforce(t, delta)


# ---------------------------------------------------------------------------
# sample()'s own auto-instid entry point: X_test=(t, delta), no instid given
# ---------------------------------------------------------------------------


def test_sample_X_test_missing_instid_auto_assigns_prior():
    """sample() accepts a 2-tuple (t, delta) X_test for exposure-integrated
    test points, auto-assigning instid via assign_min_instids. Confirmed to
    give bit-identical results (same key) to manually calling
    assign_min_instids and passing its output as the 3-tuple form -- for a
    prior GP, with overlapping windows that need 2 auto-assigned groups
    (not 1), to exercise the actual grouping logic rather than a trivial
    single-group case."""
    kernel = smolgp.kernels.IntegratedSHO(omega=0.2, quality=2.0, sigma=1.0, num_insts=1)
    gp = smolgp.GaussianProcess(
        kernel=kernel,
        X=(jnp.array([0.0]), jnp.array([0.0]), jnp.array([0], dtype=int)),
        noise=jnp.array([1e-8]),
    )

    t_test = jnp.array([30.0, 32.0, 80.0])  # first two overlap
    delta_test = jnp.array([5.0, 5.0, 5.0])
    X_test_2tuple = (t_test, delta_test)
    instid_auto, num_groups = assign_min_instids(t_test, delta_test)
    assert num_groups == 2
    X_test_3tuple = (t_test, delta_test, instid_auto)

    key = jax.random.PRNGKey(80)
    y_2tuple = gp.sample(key, X_test=X_test_2tuple)
    y_3tuple = gp.sample(key, X_test=X_test_3tuple)
    diff = float(jnp.max(jnp.abs(y_2tuple - y_3tuple)))
    assert diff == 0.0, f"2-tuple X_test should exactly match its own auto-assigned 3-tuple: {diff:.3e}"
    print("    ...sample() 2-tuple X_test (prior) matches manually-replicated auto instid exactly")


def test_sample_X_test_missing_instid_auto_assigns_conditioned():
    """Same check as above, for a conditioned GP."""
    d = _build_dataset(2, jax.random.PRNGKey(81))
    gp_smol, y = d["gp_smol"], d["y"]
    _, condgp = gp_smol.condition(y)

    t_test = jnp.array([30.0, 32.0, 80.0])
    delta_test = jnp.array([5.0, 5.0, 5.0])
    X_test_2tuple = (t_test, delta_test)
    instid_auto, _num_groups = assign_min_instids(t_test, delta_test)
    X_test_3tuple = (t_test, delta_test, instid_auto)

    key = jax.random.PRNGKey(82)
    samples_2tuple = condgp.sample(key, shape=(500,), X_test=X_test_2tuple)
    samples_3tuple = condgp.sample(key, shape=(500,), X_test=X_test_3tuple)
    diff = float(jnp.max(jnp.abs(samples_2tuple - samples_3tuple)))
    assert diff == 0.0, f"2-tuple X_test should exactly match its own auto-assigned 3-tuple: {diff:.3e}"
    print("    ...sample() 2-tuple X_test (conditioned) matches manually-replicated auto instid exactly")


def test_exposure_sample_matches_predict_when_groups_exceed_num_insts():
    """Exposure-integrated posterior samples must match predict()'s analytic
    mean even when the number of probe groups exceeds the kernel's real
    instrument count.

    A test point's instid plays two roles: the probe group it accumulates
    into during the forward simulation, and the real instrument whose
    observation model projects the result. Auto-grouping (assign_min_instids)
    knows nothing about the kernel's instrument count, so overlapping test
    windows routinely need MORE groups than the kernel has instruments. When
    the two roles shared one array, the residual half was then evaluated with
    an out-of-range instrument id and the posterior sample came out silently
    wrong (by ~100% of the signal), while still agreeing with itself across
    spellings -- which is why the existing 2-tuple/3-tuple consistency tests
    could not catch it.
    """
    S, w, Q = 2.5, 0.2, 2.0
    sigma = jnp.sqrt(S * w * Q)
    kernel = smolgp.kernels.IntegratedSHO(
        omega=w, quality=Q, sigma=sigma, num_insts=1
    )
    t = jnp.linspace(0.0, 100.0, 8)
    X = (t, jnp.full(8, 3.0), jnp.zeros(8, dtype=int))
    gp = smolgp.GaussianProcess(kernel, X=X, noise=0.01)
    y = jax.random.normal(jax.random.PRNGKey(0), (8,))
    _, condgp = gp.condition(y)

    cases = {
        "1 group (== num_insts)": (jnp.array([15.0, 65.0]), jnp.full(2, 10.0)),
        "2 groups (> num_insts)": (jnp.array([30.0, 40.0]), jnp.full(2, 20.0)),
        "4 groups (>> num_insts)": (
            jnp.array([30.0, 32.0, 34.0, 36.0]),
            jnp.full(4, 20.0),
        ),
    }
    for label, (t_test, delta_test) in cases.items():
        _instid, num_groups = assign_min_instids(t_test, delta_test)
        mu_pred, var_pred = condgp.predict(
            (t_test, delta_test, jnp.zeros_like(t_test, dtype=int)),
            y=y,
            return_var=True,
        )
        samples = condgp.sample(
            jax.random.PRNGKey(7), shape=(6000,), X_test=(t_test, delta_test)
        )
        mean_emp = jnp.mean(samples, axis=-1)
        var_emp = jnp.var(samples, axis=-1, ddof=1)
        scale = float(jnp.sqrt(jnp.max(var_pred)))
        dmean = float(jnp.max(jnp.abs(mean_emp - mu_pred)))
        dvar = float(jnp.max(jnp.abs(var_emp - var_pred)))
        print(
            f"    ...[{label}] groups={num_groups}, num_insts={kernel.num_insts}: "
            f"|dmean|={dmean:.3e}, |dvar|={dvar:.3e}"
        )
        assert dmean < 0.1 * scale, f"[{label}] sample mean vs predict: {dmean:.3e}"
        assert dvar < 0.25 * float(jnp.max(var_pred)), (
            f"[{label}] sample var vs predict: {dvar:.3e}"
        )


def test_exposure_sample_explicit_instid_selects_projection():
    """An explicitly-passed instid is the *real* instrument and must drive the
    projection, matching predict() called with that same instid."""
    S, w, Q = 2.5, 0.2, 2.0
    sigma = jnp.sqrt(S * w * Q)
    kernel = smolgp.kernels.IntegratedSHO(
        omega=w, quality=Q, sigma=sigma, num_insts=2
    )
    t = jnp.linspace(0.0, 100.0, 10)
    X = (t, jnp.full(10, 3.0), jnp.array([0, 1] * 5))
    gp = smolgp.GaussianProcess(kernel, X=X, noise=0.01)
    y = jax.random.normal(jax.random.PRNGKey(0), (10,))
    _, condgp = gp.condition(y)

    t_test, delta_test = jnp.array([25.0, 65.0]), jnp.full(2, 8.0)
    for iid in [jnp.zeros(2, dtype=int), jnp.ones(2, dtype=int)]:
        X_test = (t_test, delta_test, iid)
        mu_pred, var_pred = condgp.predict(X_test, y=y, return_var=True)
        samples = condgp.sample(jax.random.PRNGKey(11), shape=(6000,), X_test=X_test)
        scale = float(jnp.sqrt(jnp.max(var_pred)))
        dmean = float(jnp.max(jnp.abs(jnp.mean(samples, axis=-1) - mu_pred)))
        assert dmean < 0.1 * scale, f"instid={iid.tolist()}: {dmean:.3e}"
        print(f"    ...explicit instid={iid.tolist()} matches predict: {dmean:.2e}")


def test_exposure_sample_explicit_instid_allows_overlap():
    """An explicit instid names the instrument to report the result as, so it
    may be repeated across *overlapping* windows -- "what would instrument i
    have seen through each of these?" is a legitimate query even though one
    instrument could not physically record them simultaneously.

    The bookkeeping that genuinely requires non-overlap (which integral
    accumulator each window uses) is assigned independently of instid, so
    this must agree with predict() called with that same instid.
    """
    S, w, Q = 2.5, 0.2, 2.0
    sigma = jnp.sqrt(S * w * Q)
    kernel = smolgp.kernels.IntegratedSHO(
        omega=w, quality=Q, sigma=sigma, num_insts=3
    )
    t = jnp.linspace(0.0, 100.0, 9)
    X = (t, jnp.full(9, 3.0), jnp.array([0, 1, 2] * 3))
    gp = smolgp.GaussianProcess(kernel, X=X, noise=0.01)
    y = jax.random.normal(jax.random.PRNGKey(0), (9,))
    _, condgp = gp.condition(y)

    # Two windows that overlap each other ([20,40] and [30,50]), both asked
    # for as instrument 2.
    t_test, delta_test = jnp.array([30.0, 40.0]), jnp.full(2, 20.0)
    instid = jnp.full(2, 2, dtype=int)
    X_test = (t_test, delta_test, instid)

    mu_pred, var_pred = condgp.predict(X_test, y=y, return_var=True)
    samples = condgp.sample(jax.random.PRNGKey(7), shape=(6000,), X_test=X_test)
    scale = float(jnp.sqrt(jnp.max(var_pred)))
    dmean = float(jnp.max(jnp.abs(jnp.mean(samples, axis=-1) - mu_pred)))
    assert dmean < 0.1 * scale, f"overlapping same-instid vs predict: {dmean:.3e}"
    print(f"    ...overlapping windows sharing instid=2 match predict: {dmean:.2e}")

    # Grouping is assigned independently, so the reported instrument does not
    # change which accumulator is used -- every instid gives the same answer
    # for an instrument-agnostic kernel.
    for other in [0, 1]:
        X_other = (t_test, delta_test, jnp.full(2, other, dtype=int))
        mu_other, _ = condgp.predict(X_other, y=y, return_var=True)
        assert jnp.allclose(mu_other, mu_pred, atol=1e-10), (
            f"instid={other} changed the predicted value for a shared model"
        )


if __name__ == "__main__":
    test_prior_trajectory_instantaneous()
    test_prior_trajectory_integrated()
    test_prior_trajectory_ties_are_finite()
    test_prior_sample_smolgp_vs_tinygp()
    test_conditioned_sample_instantaneous()
    test_conditioned_sample_instantaneous_parallel_solver()
    test_conditioned_sample_integrated_serial_and_parallel()
    test_sample_with_no_X_test_uses_training_coords_not_predict_coords()
    test_condition_batched_mean_matches_condition()
    test_condition_batched_mean_is_faster_for_many_samples()
    test_grid_sample_instantaneous_retrodict_interpolate_extrapolate()
    test_grid_sample_integrated_retrodict_interpolate_extrapolate()
    test_grid_sample_instantaneous_reproduces_training_point()
    test_exposure_sample_reproduces_training_point()
    test_grid_sample_ties_at_exposure_boundaries()
    test_exposure_sample_single_point()
    test_exposure_sample_retrodict_interpolate_extrapolate()
    test_exposure_sample_multiple_points_are_correlated()
    test_exposure_sample_mixed_delta0_and_delta_gt0()
    test_exposure_sample_ties_at_boundaries()
    test_exposure_sample_parallel_solver()
    test_exposure_sample_matches_tinygp_directly()
    test_prior_sample_exposure_test_points_matches_dense_kernel()
    test_sample_exposure_matches_quadrature_of_dense_curve_same_draw()
    test_assign_min_instids_no_overlap_within_group()
    test_assign_min_instids_is_minimal()
    test_assign_min_instids_reuses_freed_group()
    test_assign_min_instids_touching_windows_share_a_group()
    test_assign_min_instids_all_zero_width_share_one_group()
    test_assign_min_instids_mixed_zero_and_positive_width()
    test_assign_min_instids_zero_width_inside_one_exposure()
    test_assign_min_instids_matches_greedy_reference_on_mixed_widths()
    test_min_groups_greedy_matches_bruteforce()
    test_sample_X_test_missing_instid_auto_assigns_prior()
    test_sample_X_test_missing_instid_auto_assigns_conditioned()
    test_exposure_sample_matches_predict_when_groups_exceed_num_insts()
    test_exposure_sample_explicit_instid_selects_projection()
    test_exposure_sample_explicit_instid_allows_overlap()
    print("All sample() tests passed.")
