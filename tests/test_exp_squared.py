"""
Tests for the state space approximation to the ExpSquared (RBF) kernel.

Two separate questions:
1. Is the state space model a faithful implementation of its own (approximate)
   kernel? Checked against a dense GP built from that same kernel, which should
   agree to round-off.
2. How close is that approximate kernel to the exact RBF? Checked against
   tinygp's dense ExpSquared, which should agree to within ``max_error``.
Both are checked across many decades of length scale, since the model is built
in units of the length scale and should not care what it is.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tinygp

import smolgp
from tests.utils import allclose, generate_data

jax.config.update("jax_enable_x64", True)

SCALES = [1e-3, 1.0, 83.3, 1e4]
ORDERS = [4, 8, 16, 24, 30]
SIGMA = 2.1


class DenseApprox(tinygp.kernels.Kernel):
    """
    A dense tinygp kernel with the same covariance as a smolgp state space
    model, k(Delta) = H A(|Delta|) Pinf H^T, to test the solver in isolation
    from the approximation
    """

    ssm: smolgp.kernels.StateSpaceModel

    def evaluate(self, X1, X2):
        H = self.ssm.observation_matrix(X1)
        A = self.ssm.transition_matrix(0.0, jnp.abs(X2 - X1))
        return (H @ A @ self.ssm.stationary_covariance() @ H.T)[0, 0]


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("order", ORDERS)
def test_kernel_function_matches_exact_rbf(scale, order):
    """The approximate kernel is within max_error of the exact RBF, at any scale"""
    ksmol = smolgp.kernels.ExpSquared(scale=scale, sigma=SIGMA, order=order)
    ktiny = SIGMA**2 * tinygp.kernels.ExpSquared(scale=scale)
    dts = jnp.linspace(0, 10 * scale, 400)
    zeros = jnp.zeros_like(dts)
    res = ksmol(zeros, dts)[0] - ktiny(zeros, dts)[0]
    err = float(jnp.max(jnp.abs(res)))
    # within the stated error, and the stated error is not a loose overestimate
    assert err <= 1.01 * float(ksmol.max_error()), (err, float(ksmol.max_error()))
    assert err >= 0.5 * float(ksmol.max_error()), (err, float(ksmol.max_error()))


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("order", ORDERS)
def test_state_space_matches_dense_approximate_kernel(scale, order):
    """
    The Kalman filter/smoother reproduces a dense GP with the same approximate
    kernel: likelihood, conditioned mean, and conditioned variance
    """
    ksmol = smolgp.kernels.ExpSquared(scale=scale, sigma=SIGMA, order=order)
    yerr = 0.3
    t, y = generate_data(
        200, SIGMA**2 * tinygp.kernels.ExpSquared(scale=scale), yerr=yerr, tmin=0, tmax=20 * scale
    )
    noise = jnp.full_like(t, yerr**2)

    gp_ssm = smolgp.GaussianProcess(ksmol, t, noise=noise)
    gp_dense = tinygp.GaussianProcess(DenseApprox(ksmol), t, diag=noise)

    llh_ssm, llh_dense = gp_ssm.log_probability(y), gp_dense.log_probability(y)
    allclose("likelihood", (llh_ssm - llh_dense) / jnp.abs(llh_dense), tol=1e-9)

    _, cond_ssm = gp_ssm.condition(y)
    _, cond_dense = gp_dense.condition(y)
    allclose("conditioned mean", cond_ssm.loc - cond_dense.loc, tol=1e-7 * SIGMA)
    allclose("conditioned variance", cond_ssm.variance - cond_dense.variance, tol=1e-7 * SIGMA**2)


@pytest.mark.parametrize("scale", SCALES)
def test_likelihood_converges_to_exact_rbf(scale):
    """With increasing order, the likelihood approaches the exact dense RBF one"""
    yerr = 0.3
    ktiny = SIGMA**2 * tinygp.kernels.ExpSquared(scale=scale)
    t, y = generate_data(200, ktiny, yerr=yerr, tmin=0, tmax=20 * scale)
    llh_exact = tinygp.GaussianProcess(ktiny, t, diag=yerr**2).log_probability(y)
    diffs = []
    for order in [4, 8, 12, 16, 20]:
        k = smolgp.kernels.ExpSquared(scale=scale, sigma=SIGMA, order=order)
        llh = smolgp.GaussianProcess(k, t, noise=jnp.full_like(t, yerr**2)).log_probability(y)
        diffs.append(float(jnp.abs(llh - llh_exact)))
    assert all(d1 > d2 for d1, d2 in zip(diffs, diffs[1:])), diffs
    assert diffs[-1] < 1e-2 * abs(float(llh_exact)), diffs


@pytest.mark.parametrize("order", ORDERS)
def test_transitions_stable_for_any_step(order):
    """
    A and Q are finite for steps from 1e-12 to 1e12 length scales; A is a
    contraction (block rotations with decay) and Q is a valid covariance
    """
    k = smolgp.kernels.ExpSquared(scale=1.0, sigma=SIGMA, order=order)
    Pinf = k.stationary_covariance()
    scale_P = float(jnp.max(jnp.diag(Pinf)))
    for dt in 10.0 ** np.arange(-12, 13, 2):
        A = k.transition_matrix(0.0, dt)
        Q = k.process_noise(0.0, dt)
        assert jnp.all(jnp.isfinite(A)) and jnp.all(jnp.isfinite(Q)), dt
        assert float(jnp.linalg.norm(A, 2)) <= 1 + 1e-12, dt
        assert float(jnp.min(jnp.linalg.eigvalsh(Q))) >= -1e-10 * scale_P, dt
    # Q -> Pinf over a very long step, and -> 0 over a very short one
    allclose("Q(long)", k.process_noise(0.0, 1e12) - Pinf, tol=1e-12 * scale_P)
    allclose("Q(short)", k.process_noise(0.0, 1e-12), tol=1e-8 * scale_P)


@pytest.mark.parametrize("scale", SCALES)
def test_jit_and_gradients(scale):
    """The likelihood is jittable and differentiable in the length scale"""
    t, y = generate_data(100, tinygp.kernels.ExpSquared(scale=scale), tmin=0, tmax=20 * scale)
    noise = jnp.full_like(t, 0.09)

    def nll(log_scale):
        k = smolgp.kernels.ExpSquared(scale=jnp.exp(log_scale), sigma=SIGMA)
        return -smolgp.GaussianProcess(k, t, noise=noise).log_probability(y)

    x0 = jnp.log(scale)
    value, grad = jax.jit(jax.value_and_grad(nll))(x0)
    allclose("jit value", value - nll(x0), tol=1e-9 * jnp.abs(value))
    fd = (nll(x0 + 1e-6) - nll(x0 - 1e-6)) / 2e-6
    allclose("gradient vs finite difference", (grad - fd) / jnp.abs(fd), tol=1e-5)


def test_order_validation():
    with pytest.raises(ValueError):
        smolgp.kernels.ExpSquared(scale=1.0, order=0)
    with pytest.raises(ValueError):
        smolgp.kernels.ExpSquared(scale=1.0, order=31)


def _exposure_averaged_rbf(t, delta, scale, sigma):
    """Exact dense covariance of exposure-averaged RBF data, (1/d_i d_j) * double integral,
    via the RBF's second antiderivative G (G'' = exp(-x^2 / 2 scale^2))"""
    from scipy.special import erf

    def G(x):
        return x * np.sqrt(np.pi / 2) * scale * erf(x / (np.sqrt(2) * scale)) + scale**2 * np.exp(
            -(x**2) / (2 * scale**2)
        )

    a, b = t - delta / 2, t + delta / 2
    K = G(b[:, None] - a) + G(a[:, None] - b) - G(b[:, None] - b) - G(a[:, None] - a)
    return sigma**2 * K / np.outer(delta, delta)


@pytest.mark.parametrize("scale", SCALES)
def test_integrated_matches_exact_exposure_averaged_rbf(scale):
    """
    IntegratedExpSquared converges to the exact dense exposure-averaged RBF,
    identically at every scale, across gaps of 1e4 length scales
    """
    rng = np.random.default_rng(3)
    t = np.concatenate([n * 1e4 * scale + np.arange(15) * 0.5 * scale for n in range(4)])
    delta = np.full(len(t), 0.3 * scale)
    C = _exposure_averaged_rbf(t, delta, scale, SIGMA) + 0.05 * np.eye(len(t))
    y = np.linalg.cholesky(C) @ rng.normal(size=len(t))
    llh_exact = -0.5 * (
        y @ np.linalg.solve(C, y) + np.linalg.slogdet(C)[1] + len(t) * np.log(2 * np.pi)
    )

    X = (jnp.array(t), jnp.array(delta), jnp.zeros(len(t), int))
    diffs = []
    for order in [8, 16, 24]:
        k = smolgp.kernels.IntegratedExpSquared(scale=scale, sigma=SIGMA, order=order)
        llh = smolgp.GaussianProcess(k, X, noise=jnp.full(len(t), 0.05)).log_probability(
            jnp.array(y)
        )
        diffs.append(abs(float(llh) - llh_exact))
    assert all(d1 > d2 for d1, d2 in zip(diffs, diffs[1:])), diffs
    assert diffs[-1] < 1e-5, diffs
