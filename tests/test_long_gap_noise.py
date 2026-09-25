"""Long-gap regressions with independent analytic and quadrature references."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad_vec
from scipy.linalg import expm

import smolgp
from smolgp.helpers import discretize_with_doubling

jax.config.update("jax_enable_x64", True)


def _quadrature_covariance(kernel, dt):
    """Integrate the physical impulse response, without a Van Loan block."""
    F = np.asarray(kernel.design_matrix())
    L = np.asarray(kernel.noise_effect_matrix())
    Qc = np.asarray(kernel.noise())

    def integrand(t):
        impulse = expm(F * t) @ L
        return impulse @ Qc @ impulse.T

    points = [point for point in (0.1, 1.0, 10.0, 100.0) if point < dt]
    value, _ = quad_vec(
        integrand, 0.0, dt, points=points, epsabs=1e-10, epsrel=1e-11
    )
    return value


@pytest.mark.parametrize(
    "kernel_class",
    [
        smolgp.kernels.IntegratedExp,
        smolgp.kernels.IntegratedMatern32,
        smolgp.kernels.IntegratedMatern52,
    ],
)
@pytest.mark.parametrize("dt", [0.02, 1000.0])
def test_integrated_covariance_matches_physical_quadrature(kernel_class, dt):
    kernel = kernel_class(scale=1.0, sigma=1.0)
    actual = np.asarray(kernel.process_noise(0.0, dt))
    expected = _quadrature_covariance(kernel, dt)
    np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-11)
    np.testing.assert_allclose(actual, actual.T, rtol=0, atol=1e-12)
    assert np.linalg.eigvalsh(actual).min() >= -1e-11


def test_ou_analytic_long_gap_and_gradients():
    kernel = smolgp.kernels.IntegratedExp(scale=1.0, sigma=1.0)
    expected = np.array([[1.0, 1.0], [1.0, 1997.0]])
    np.testing.assert_allclose(kernel.process_noise(0.0, 1000.0), expected, rtol=1e-11)

    def integral_variance(scale, sigma):
        kernel = smolgp.kernels.IntegratedExp(scale=scale, sigma=sigma)
        return kernel.process_noise(0.0, 1000.0)[1, 1]

    derivatives = jax.jit(jax.grad(integral_variance, argnums=(0, 1)))(1.0, 1.0)
    np.testing.assert_allclose(derivatives, (1994.0, 3994.0), rtol=1e-10)


def test_matern52_finite_but_inaccurate_long_gap():
    # The original full-gap block exponential returns finite, inaccurate
    # values here, so merely rejecting NaNs would miss this regression.
    kernel = smolgp.kernels.IntegratedMatern52(scale=1.0, sigma=1.0)
    expected = _quadrature_covariance(kernel, 300.0)
    actual = kernel.process_noise(0.0, 300.0)
    np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-11)


def test_jit_vmap_and_zero_time_derivative():
    kernel = smolgp.kernels.IntegratedExp(scale=1.0, sigma=1.0)
    covariance = lambda dt: kernel.process_noise(0.0, dt)
    batched = jax.jit(jax.vmap(covariance))(jnp.array([0.0, 0.1, 1000.0]))
    assert np.isfinite(batched).all()
    np.testing.assert_array_equal(batched[0], np.zeros((2, 2)))
    np.testing.assert_allclose(
        jax.jit(jax.jacrev(covariance))(0.0), [[2.0, 0.0], [0.0, 0.0]], atol=1e-13
    )


def test_semigroup_and_multiple_integral_states():
    kernel = smolgp.kernels.IntegratedMatern32(scale=1.0, sigma=1.0, num_insts=2)
    F, L, Qc = kernel.design_matrix(), kernel.noise_effect_matrix(), kernel.noise()
    discretize = jax.jit(lambda dt: discretize_with_doubling(F, L, Qc, dt))
    A1, Q1 = discretize(0.3)
    A2, Q2 = discretize(1000.0)
    A, Q = discretize(1000.3)
    np.testing.assert_allclose(A, A2 @ A1, rtol=2e-11, atol=2e-12)
    np.testing.assert_allclose(Q, Q2 + A2 @ Q1 @ A2.T, rtol=2e-10, atol=2e-11)
    actual = np.asarray(kernel.process_noise(0.0, 1000.3))
    np.testing.assert_allclose(actual, Q, rtol=2e-10, atol=2e-11)
    # Integral states driven by the same process are perfectly correlated.
    np.testing.assert_allclose(actual[-1], actual[-2], rtol=0, atol=1e-12)
    assert np.linalg.eigvalsh(actual).min() >= -1e-10


def test_invalid_time_and_scaling_budget_are_explicit():
    # Integer-valued systems must also produce floating-point covariances.
    F, L, Qc = jnp.array([[-1]]), jnp.array([[1]]), jnp.array([[1]])
    evaluate = jax.jit(lambda dt: discretize_with_doubling(F, L, Qc, dt, max_doublings=2))
    for dt in (-1.0, 100.0, jnp.inf, jnp.nan):
        A, Q = evaluate(dt)
        assert np.isnan(A).all() and np.isnan(Q).all()
    A, Q = evaluate(0.0)
    np.testing.assert_array_equal(A, [[1.0]])
    np.testing.assert_array_equal(Q, [[0.0]])


def test_interrupted_exposures_likelihood_matches_dense_ou():
    # Two instruments observe short exposures separated by a thousand scales.
    t = jnp.array([0.0, 0.2, 1000.0, 1000.2])
    width = 0.1
    X = (t, jnp.full(t.shape, width), jnp.array([0, 1, 0, 1]))
    y = jnp.array([0.1, -0.2, 0.3, 0.05])
    kernel = smolgp.kernels.IntegratedExp(scale=1.0, sigma=1.0, num_insts=2)
    gp = smolgp.GaussianProcess(kernel=kernel, X=X, noise=jnp.full(t.shape, 0.01))
    actual = gp.log_probability(y)

    separation = np.abs(np.asarray(t)[:, None] - np.asarray(t)[None, :])
    factor = (-np.expm1(-width) / width) ** 2
    covariance = np.exp(-(separation - width)) * factor
    variance = 2 * (width + np.expm1(-width)) / width**2
    np.fill_diagonal(covariance, variance + 0.01)
    _, logdet = np.linalg.slogdet(covariance)
    expected = -0.5 * (4 * np.log(2 * np.pi) + logdet + y @ np.linalg.solve(covariance, y))
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
