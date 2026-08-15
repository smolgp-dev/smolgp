import jax
import jax.numpy as jnp
import tinygp

import smolgp
from tests.test_kernels import (
    condition,
    kernel_function,
    likelihood,
    predict,
)
from tests.utils import generate_data

key = jax.random.PRNGKey(0)
jax.config.update("jax_enable_x64", True)


def test_instantaneous():
    """
    Baseline sanity check: a normal, non-integrated (instantaneous) dataset
    with every timestamp distinct matches tinygp's dense/quasiseparable
    solver exactly. This is the reference behavior the tied-timestamp test
    below is compared against.
    """
    w = 0.0195
    Q = 7.63
    sigma = 0.59

    kernel_smol = smolgp.kernels.SHO(omega=w, quality=Q, sigma=sigma)
    kernel_tiny = tinygp.kernels.quasisep.SHO(omega=w, quality=Q, sigma=sigma)

    N = 50
    yerr = 0.3
    t_train, y_train = generate_data(N, kernel_tiny, yerr, tmin=0, tmax=1000)
    yerr_train = jnp.full_like(t_train, yerr)

    # Test k(Delta) agrees
    kernel_function(kernel_smol, kernel_tiny, tol=1e-9, atol=1e-12)

    # Build GP objects
    gp_smol = smolgp.GaussianProcess(kernel=kernel_smol, X=t_train, noise=yerr_train**2)
    gp_tiny = tinygp.GaussianProcess(kernel=kernel_tiny, X=t_train, diag=yerr_train**2)

    # Check likelihood
    likelihood(gp_smol, gp_tiny, y_train, tol=1e-10, atol=1e-13)

    # Check conditioning
    condition(gp_smol, gp_tiny, y_train, tol=1e-9, atol=1e-12)

    # Check predictions
    predict(gp_smol, gp_tiny, y_train, tol=1e-9, atol=1e-12)


def _tied_dataset(tie, w=0.0195, Q=7.63, sigma=0.59, N=50, yerr=0.3):
    """
    Build a non-integrated (t, y) dataset with a deliberately tied pair of
    timestamps (a Delta=0 transition between two adjacent states), otherwise
    a normal sampled SHO process. Since non-integrated data has no reset
    matrix -- there's no per-instrument integral-accumulator state that ever
    gets forced to exact zero -- this Delta=0 transition is *not* expected to
    make the predicted covariance singular, unlike the integrated case in
    test_integrated.py. tie: "start", "middle", or "end" controls where in
    the (already sorted) array the tied pair falls.
    """
    kernel_tiny = tinygp.kernels.quasisep.SHO(omega=w, quality=Q, sigma=sigma)
    t_train, y_train = generate_data(N, kernel_tiny, yerr=yerr, tmin=0, tmax=1000)

    if tie == "start":
        idx = 0
    elif tie == "end":
        idx = N  # insert after the last element, duplicating the last timestamp
    else:
        idx = N // 2

    dup_val = t_train[idx] if idx < N else t_train[-1]
    t_tied = jnp.insert(t_train, idx, dup_val)
    # A distinct second measurement at the exact same timestamp (not a
    # literal duplicate observation) so the two points genuinely differ.
    y_tied = jnp.insert(y_train, idx, (y_train[idx] if idx < N else y_train[-1]) + 0.1)

    return t_tied, y_tied


def test_instantaneous_tied_timestamps():
    """
    Two data points sharing the exact same timestamp (a Delta=0 transition)
    must still condition/predict identically to tinygp, since the plain
    (non-integrated) state-space solver has no reset matrix to force a
    singular predicted covariance -- see src/smolgp/solvers/rts.py, which
    computes the smoothing gain via a raw jnp.linalg.solve with no
    singularity guard at all. This is a regression/sanity check that ties
    are safe here independent of the get_smoothing_gain fix in
    test_integrated.py.
    """
    w, Q, sigma, yerr = 0.0195, 7.63, 0.59, 0.3
    kernel_tiny = tinygp.kernels.quasisep.SHO(omega=w, quality=Q, sigma=sigma)

    for tie in ("start", "middle", "end"):
        t_tied, y_tied = _tied_dataset(tie, w=w, Q=Q, sigma=sigma, yerr=yerr)
        yerr_train = jnp.full_like(t_tied, yerr)

        kernel_smol = smolgp.kernels.SHO(omega=w, quality=Q, sigma=sigma)

        gp_smol = smolgp.GaussianProcess(
            kernel=kernel_smol, X=t_tied, noise=yerr_train**2
        )
        gp_tiny = tinygp.GaussianProcess(
            kernel=kernel_tiny, X=t_tied, diag=yerr_train**2
        )

        condition(gp_smol, gp_tiny, y_tied, tol=1e-9, atol=1e-12)
        predict(gp_smol, gp_tiny, y_tied, tol=1e-9, atol=1e-12)
        print(f"    ...tie='{tie}': finite and matches tinygp exactly")


def test_scalar_noise_matches_explicit_array():
    """``noise=<scalar>`` is shorthand for a homoscedastic ``jnp.full(N, ...)``
    and must produce a bit-identical model, for every kernel/solver type.

    A scalar is 0-D, so before this was supported it fell through the
    ndim-based normalization untouched and reached the solvers as a bare
    scalar instead of the expected ``(N, D, D)`` covariance stack.
    """
    S, w, Q = 2.5, 0.2, 2.0
    sigma = jnp.sqrt(S * w * Q)
    var = 0.037

    # --- instantaneous kernel, both solvers ---
    kernel = smolgp.kernels.SHO(omega=w, quality=Q, sigma=sigma)
    N = 8
    t = jnp.sort(jax.random.uniform(jax.random.PRNGKey(0), (N,), maxval=50.0))
    y = jax.random.normal(jax.random.PRNGKey(1), (N,))
    t_test = jnp.linspace(-5.0, 55.0, 9)

    for solver in [None, smolgp.solvers.ParallelStateSpaceSolver]:
        kw = {} if solver is None else {"solver": solver}
        gp_scalar = smolgp.GaussianProcess(kernel, X=t, noise=var, **kw)
        gp_array = smolgp.GaussianProcess(kernel, X=t, noise=jnp.full(N, var), **kw)
        label = "StateSpaceSolver" if solver is None else solver.__name__

        assert gp_scalar.noise.shape == (N, 1, 1), (
            f"[{label}] scalar noise must broadcast to (N, 1, 1), "
            f"got {gp_scalar.noise.shape}"
        )
        assert jnp.array_equal(gp_scalar.noise, gp_array.noise), f"[{label}] noise"

        llh_s, cond_s = gp_scalar.condition(y)
        llh_a, cond_a = gp_array.condition(y)
        assert jnp.array_equal(llh_s, llh_a), f"[{label}] log probability"
        mu_s, var_s = cond_s.predict(t_test, return_var=True)
        mu_a, var_a = cond_a.predict(t_test, return_var=True)
        assert jnp.array_equal(mu_s, mu_a), f"[{label}] predicted mean"
        assert jnp.array_equal(var_s, var_a), f"[{label}] predicted variance"

    # --- integrated kernel (noise still applies per observation, not per state) ---
    kernel_i = smolgp.kernels.IntegratedSHO(
        omega=w, quality=Q, sigma=sigma, num_insts=1
    )
    ti = jnp.linspace(0.0, 100.0, 6)
    Xi = (ti, jnp.full(6, 3.0), jnp.zeros(6, dtype=int))
    yi = jax.random.normal(jax.random.PRNGKey(2), (6,))
    gp_i_scalar = smolgp.GaussianProcess(kernel_i, X=Xi, noise=var)
    gp_i_array = smolgp.GaussianProcess(kernel_i, X=Xi, noise=jnp.full(6, var))
    assert gp_i_scalar.noise.shape == (6, 1, 1), (
        f"integrated scalar noise shape {gp_i_scalar.noise.shape}"
    )
    llh_s, _ = gp_i_scalar.condition(yi)
    llh_a, _ = gp_i_array.condition(yi)
    assert jnp.array_equal(llh_s, llh_a), "integrated log probability"

    # A Python float and a 0-D jnp scalar must behave identically
    gp_pyfloat = smolgp.GaussianProcess(kernel, X=t, noise=float(var))
    gp_jaxscalar = smolgp.GaussianProcess(kernel, X=t, noise=jnp.asarray(var))
    assert jnp.array_equal(gp_pyfloat.noise, gp_jaxscalar.noise), (
        "python float vs 0-D jnp scalar"
    )
    print("    ...scalar noise matches the explicit per-observation array exactly")


if __name__ == "__main__":
    test_instantaneous()
    test_instantaneous_tied_timestamps()
    test_scalar_noise_matches_explicit_array()
    print("All instantaneous kernel tests passed.")
