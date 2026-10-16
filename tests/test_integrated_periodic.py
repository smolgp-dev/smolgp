"""
Tests for IntegratedExpSineSquared, the exposure-averaged periodic kernel.

It is a sum of integrated periodic terms, each a deterministic rotation with a
closed-form integral, so it should reproduce the exposure-averaged covariance of
the same truncated series to round-off. That covariance is also closed form:
the exposure average of cos(w t) over a window of length d is cos(w t) sinc(w d / 2).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import smolgp

jax.config.update("jax_enable_x64", True)

GAMMA, SIGMA = 2.0, 1.7


def _terms(kernel):
    if isinstance(kernel, smolgp.kernels.Sum):
        return _terms(kernel.kernel1) + _terms(kernel.kernel2)
    return [kernel]


def _dense_exposure_averaged(period, order, t, delta):
    """Exposure-averaged covariance of the order-``order`` periodic series"""
    base = smolgp.kernels.ExpSineSquared(period=period, gamma=GAMMA, sigma=1.0, order=order)
    K = np.zeros((len(t), len(t)))
    for term in _terms(base.kernel):
        q2 = float(term.stationary_covariance()[0, 0])
        w = term.order * 2 * np.pi / period
        sinc = np.sinc(w * delta / (2 * np.pi))  # sin(w d / 2) / (w d / 2)
        K += q2 * np.cos(w * (t[:, None] - t[None, :])) * np.outer(sinc, sinc)
    return SIGMA**2 * K


@pytest.mark.parametrize("period", [1e-3, 1.0, 1e4])
def test_matches_dense_exposure_averaged(period):
    """Likelihood matches the exact dense covariance, at any period, across long gaps"""
    rng = np.random.default_rng(0)
    t = np.concatenate([n * 50 * period + 0.25 * period * np.arange(12) for n in range(3)])
    delta = rng.uniform(0.02, 0.2, len(t)) * period
    kernel = smolgp.kernels.IntegratedExpSineSquared(period=period, gamma=GAMMA, sigma=SIGMA)

    C = _dense_exposure_averaged(period, kernel.order, t, delta) + 0.05 * np.eye(len(t))
    y = np.linalg.cholesky(C) @ rng.normal(size=len(t))
    llh_dense = -0.5 * (y @ np.linalg.solve(C, y) + np.linalg.slogdet(C)[1] + len(t) * np.log(2 * np.pi))

    X = (jnp.array(t), jnp.array(delta), jnp.zeros(len(t), int))
    gp = smolgp.GaussianProcess(kernel, X, noise=jnp.full(len(t), 0.05))
    llh = float(gp.log_probability(jnp.array(y)))
    assert abs(llh - llh_dense) / abs(llh_dense) < 1e-10, (llh, llh_dense)


def test_latent_kernel_is_exp_sine_squared():
    """Without exposure times, it is the instantaneous ExpSineSquared kernel"""
    kernel = smolgp.kernels.IntegratedExpSineSquared(period=3.0, gamma=GAMMA, sigma=SIGMA)
    periodic = smolgp.kernels.ExpSineSquared(period=3.0, gamma=GAMMA, sigma=SIGMA)
    t = jnp.linspace(0, 10, 40)
    np.testing.assert_allclose(kernel(t, t), periodic(t, t), atol=1e-14)


def test_transitions_exact_for_any_step():
    """The integrated rotation is exact and finite from tiny to huge steps, including j = 0"""
    kernel = smolgp.kernels.IntegratedExpSineSquared(period=1.0, gamma=GAMMA, sigma=SIGMA)
    for term in _terms(kernel.kernel):
        w = float(term.omega)
        for dt in 10.0 ** np.arange(-12, 13, 3):
            Phibar = np.asarray(term.integrated_transition_matrix(0.0, dt))
            assert np.all(np.isfinite(Phibar)), (w, dt)
            if w == 0:
                expected = dt * np.eye(2)
            else:
                th = w * dt
                one_minus_cos = 2 * np.sin(th / 2) ** 2  # 1 - cos(th), without cancellation
                expected = np.array([[np.sin(th), -one_minus_cos], [one_minus_cos, np.sin(th)]]) / w
            np.testing.assert_allclose(Phibar, expected, rtol=1e-12, atol=1e-12 * dt)
            np.testing.assert_array_equal(term.process_noise(0.0, dt), 0.0)


def test_multiple_instruments():
    """num_insts is brought into alignment with instid through the wrapper"""
    kernel = smolgp.kernels.IntegratedExpSineSquared(period=3.0, gamma=GAMMA, sigma=SIGMA)
    t = jnp.linspace(0, 20, 30)
    X = (t, jnp.full(30, 0.4), jnp.arange(30) % 2)
    with pytest.warns(UserWarning, match="num_insts"):
        gp = smolgp.GaussianProcess(kernel, X, noise=jnp.full(30, 0.1))
    assert jnp.isfinite(gp.log_probability(jnp.sin(t)))
