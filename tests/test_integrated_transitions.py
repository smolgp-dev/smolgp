"""Integrated-kernel transitions over short and long steps, checked against
independent analytic and quadrature references.

The augmented process noise and integrated transition matrix use Van Loan (in
rescaled units) for steps shorter than about one kernel timescale and a closed
form from the stationary covariance for longer ones (see the "Integrated process
noise" tutorial). These tests cover both regimes, the switch between them, and
units.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad_vec
from scipy.linalg import expm

import smolgp

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
    # A short step (Van Loan) followed by a long one (stationary closed form)
    # must compose exactly like a single step of the combined length.
    kernel = smolgp.kernels.IntegratedMatern32(scale=1.0, sigma=1.0, num_insts=2)
    A1, Q1 = kernel.transition_matrix(0.0, 0.3), kernel.process_noise(0.0, 0.3)
    A2, Q2 = kernel.transition_matrix(0.0, 1000.0), kernel.process_noise(0.0, 1000.0)
    A, Q = kernel.transition_matrix(0.0, 1000.3), kernel.process_noise(0.0, 1000.3)
    np.testing.assert_allclose(A, A2 @ A1, rtol=2e-11, atol=2e-12)
    np.testing.assert_allclose(Q, Q2 + A2 @ Q1 @ A2.T, rtol=2e-10, atol=2e-11)
    actual = np.asarray(Q)
    # Integral states driven by the same process are perfectly correlated.
    np.testing.assert_allclose(actual[-1], actual[-2], rtol=0, atol=1e-12)
    assert np.linalg.eigvalsh(actual).min() >= -1e-10


def test_zero_step():
    # A zero-length step changes nothing: identity transition, no process noise.
    for kernel in [
        smolgp.kernels.IntegratedExp(scale=1.0),
        smolgp.kernels.IntegratedMatern52(scale=1.0),
        smolgp.kernels.IntegratedSHO(omega=1.0, quality=0.3),
    ]:
        np.testing.assert_array_equal(
            kernel.transition_matrix(0.0, 0.0), np.eye(kernel.dimension)
        )
        np.testing.assert_array_equal(
            kernel.process_noise(0.0, 0.0), np.zeros((kernel.dimension,) * 2)
        )


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


# ---------------------------------------------------------------------------
# Short- and long-step methods, the switch between them, and units
# ---------------------------------------------------------------------------

def _all_integrated_kernels(unit=1.0):
    """Every integrated kernel (and SHO damping regime), with timescale ``unit``."""
    return {
        "Exp": smolgp.kernels.IntegratedExp(scale=unit),
        "Matern32": smolgp.kernels.IntegratedMatern32(scale=unit),
        "Matern52": smolgp.kernels.IntegratedMatern52(scale=unit),
        "Cosine": smolgp.kernels.IntegratedCosine(scale=6.0 * unit),
        "SHO underdamped": smolgp.kernels.IntegratedSHO(omega=1 / unit, quality=7.6),
        "SHO Q=1/sqrt2": smolgp.kernels.IntegratedSHO(
            omega=1 / unit, quality=1 / np.sqrt(2)
        ),
        "SHO critical": smolgp.kernels.IntegratedSHO(omega=1 / unit, quality=0.5),
        "SHO overdamped": smolgp.kernels.IntegratedSHO(omega=1 / unit, quality=0.3),
    }


def _quadrature_relative_to_rate(kernel, dt):
    """Like _quadrature_covariance, with breakpoints in units of the kernel's rate."""
    F = np.asarray(kernel.design_matrix(), dtype=float)
    L = np.asarray(kernel.noise_effect_matrix(), dtype=float)
    Qc = np.asarray(kernel.noise(), dtype=float)

    def integrand(t):
        impulse = expm(F * t) @ L
        return impulse @ Qc @ impulse.T

    scale = 1 / float(kernel.rate)
    points = [p * scale for p in (0.1, 1.0, 10.0) if p * scale < dt]
    value, _ = quad_vec(integrand, 0.0, dt, points=points, epsabs=0, epsrel=1e-12)
    return value


@pytest.mark.parametrize("name", list(_all_integrated_kernels()))
@pytest.mark.parametrize("tau", [1e-2, 0.5, 0.999, 1.001, 3.0, 30.0])
def test_process_noise_matches_quadrature_across_switch(name, tau):
    # tau = rate * dt; the switch from Van Loan to the closed form is at tau = 1.
    kernel = _all_integrated_kernels()[name]
    dt = tau / float(kernel.rate)
    expected = _quadrature_relative_to_rate(kernel, dt)
    actual = np.asarray(kernel.process_noise(0.0, dt))
    scale = np.abs(expected).max()
    if scale == 0:  # the cosine kernel is deterministic
        np.testing.assert_allclose(actual, 0.0, atol=1e-14)
        return
    assert np.abs(actual - expected).max() / scale < 1e-11
    # The integral state's own variance, which an exposure's variance depends on
    zz = abs(actual[-1, -1] - expected[-1, -1]) / abs(expected[-1, -1])
    assert zz < 1e-11, f"Q_zz relative error {zz:.1e}"


@pytest.mark.parametrize("name", list(_all_integrated_kernels()))
@pytest.mark.parametrize("tau", [1e-3, 0.5, 3.0, 1e3, 1e6])
def test_transitions_independent_of_time_units(name, tau):
    # The same kernel with times in days and in seconds must give the same
    # integral-state rows of A and Q, up to the change of units. With times in
    # seconds, derivative states scale as c**-k and z = int x dt as c (the
    # cosine's two states are a rotating pair, so they do not scale). The base
    # block of Q comes from the base kernel's own process_noise and is not
    # tested here.
    c = 86400.0
    k_days = _all_integrated_kernels(1.0)[name]
    k_secs = _all_integrated_kernels(c)[name]
    dt = tau / float(k_days.rate)
    d = k_days.d
    Dx = np.ones(d) if name == "Cosine" else c ** -np.arange(d)
    D = np.concatenate([Dx, [c]])

    A_days = np.asarray(k_days.transition_matrix(0.0, dt))
    A_secs = np.asarray(k_secs.transition_matrix(0.0, dt * c))
    # z row of A: [Phibar_x, 1]
    expected = A_days[d] * D[d] / D
    # (IntegratedSHO's hand-derived Phibar is good to ~1e-10 at rate*dt = 1e-3)
    np.testing.assert_allclose(A_secs[d], expected, rtol=1e-9, atol=0)

    Q_days = np.asarray(k_days.process_noise(0.0, dt))
    Q_secs = np.asarray(k_secs.process_noise(0.0, dt * c))
    assert np.all(np.isfinite(Q_secs))
    if name == "Cosine":  # deterministic, so Q is zero up to rounding
        np.testing.assert_allclose(Q_secs[d], 0.0, atol=1e-6 * c**2)
        return
    # z row of Q: [Qaug21, Qaug22], each entry relative to its natural scale
    expected = Q_days[d] * D[d] * D
    scale = np.sqrt(np.abs(Q_secs[d, d]) * np.abs(np.diag(Q_secs)))
    err = np.abs(Q_secs[d] - expected) / np.where(scale > 0, scale, 1.0)
    assert err.max() < 1e-9, f"z row of Q: max scaled error {err.max():.1e}"


@pytest.mark.parametrize("name", list(_all_integrated_kernels()))
def test_long_steps_finite_and_compose(name):
    # Far beyond where Van Loan overflows: finite, and composing two long steps
    # equals one step of the combined length (for Phibar and Q).
    kernel = _all_integrated_kernels()[name]
    scale = 1 / float(kernel.rate)
    for tau in [1e3, 1e5, 1e7]:
        A = np.asarray(kernel.transition_matrix(0.0, tau * scale))
        Q = np.asarray(kernel.process_noise(0.0, tau * scale))
        assert np.all(np.isfinite(A)) and np.all(np.isfinite(Q)), f"tau={tau}"
    t1, t2 = 1e4 * scale, 3e4 * scale
    A1, Q1 = kernel.transition_matrix(0.0, t1), kernel.process_noise(0.0, t1)
    A2, Q2 = kernel.transition_matrix(0.0, t2), kernel.process_noise(0.0, t2)
    A, Q = kernel.transition_matrix(0.0, t1 + t2), kernel.process_noise(0.0, t1 + t2)
    np.testing.assert_allclose(A, A2 @ A1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(Q, Q2 + A2 @ Q1 @ A2.T, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("quality", [7.6, 1 / np.sqrt(2)])
def test_hand_derived_sho_matches_closed_form(quality):
    # For long steps, the hand-derived underdamped IntegratedSHO noise and the
    # base class's stationary closed form are independent derivations of the
    # same quantity.
    from smolgp.kernels.integrated import IntegratedStateSpaceModel

    kernel = smolgp.kernels.IntegratedSHO(omega=1.0, quality=quality, sigma=1.3)
    for tau in [1.5, 30.0, 1e3, 1e5]:
        dt = tau / float(kernel.rate)
        hand = np.asarray(kernel.process_noise(0.0, dt))
        closed = np.asarray(
            IntegratedStateSpaceModel.process_noise(kernel, 0.0, dt, force_numerical=True)
        )
        np.testing.assert_allclose(hand, closed, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("quality", [0.3, 0.45])
def test_overdamped_sho_long_steps(quality):
    # The overdamped SHO's exp(-a) * cosh(x) used to overflow to inf * 0 for
    # w*dt > ~700. It must stay finite and stationary (A Pinf A^T + Q = Pinf).
    kernel = smolgp.kernels.SHO(omega=1.0, quality=quality, sigma=1.3)
    Pinf = np.asarray(kernel.stationary_covariance())
    for wdt in [10.0, 1e3, 1e5]:
        A = np.asarray(kernel.transition_matrix(jnp.zeros(()), jnp.asarray(wdt)))
        Q = np.asarray(kernel.process_noise(jnp.zeros(()), jnp.asarray(wdt)))
        assert np.all(np.isfinite(A)) and np.all(np.isfinite(Q)), f"w*dt={wdt}"
        np.testing.assert_allclose(A @ Pinf @ A.T + Q, Pinf, atol=1e-13)
