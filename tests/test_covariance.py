"""GaussianProcess.covariance and its consistency with .variance.

``covariance`` is the one place smolgp deliberately materializes an N x N
matrix, so these tests check it against a dense reference rather than against
any state-space shortcut. The conventions follow tinygp exactly:

- unconditioned: the prior covariance k(X, X) + noise, INCLUDING measurement
  noise, since an unconditioned GP models the observed y = f(X) + eps;
- conditioned:   the posterior covariance at the data, EXCLUDING noise;
- diag(covariance) == variance in both cases.

The conditioned case is built from the RTS smoother cross-covariance identity
(Sarkka & Solin 2019, Eq. 12.55), which is a genuinely different route to the
same matrix than tinygp's Schur complement -- so agreement to machine
precision is a strong check on the smoothing gains and the state-to-data
index bookkeeping, especially for integrated kernels where K = 2N.
"""

import jax
import jax.numpy as jnp
import pytest
import tinygp

import smolgp

jax.config.update("jax_enable_x64", True)

# tinygp adds sqrt(eps) jitter to its conditioned variance; subtract it to compare.
OFFSET = float(jnp.sqrt(jnp.finfo(jnp.float64).eps))

PLAIN_SOLVERS = [
    (None, "StateSpaceSolver"),
    (smolgp.solvers.ParallelStateSpaceSolver, "ParallelStateSpaceSolver"),
]
INTEGRATED_SOLVERS = [
    (None, "IntegratedStateSpaceSolver"),
    (
        smolgp.solvers.ParallelIntegratedStateSpaceSolver,
        "ParallelIntegratedStateSpaceSolver",
    ),
]


def _plain(solver=None, N=7):
    w, Q, S = 0.0195, 7.63, 2.36
    sigma = jnp.sqrt(S * w * Q)
    k_smol = smolgp.kernels.SHO(omega=w, quality=Q, sigma=sigma)
    k_tiny = tinygp.kernels.quasisep.SHO(omega=w, quality=Q, sigma=sigma)
    t = jnp.linspace(0.0, 500.0, N)
    y = jnp.sin(t / 50.0)
    noise = jnp.full(N, 0.09)
    kwargs = {} if solver is None else {"solver": solver}
    gp_smol = smolgp.GaussianProcess(k_smol, t, noise=noise, **kwargs)
    gp_tiny = tinygp.GaussianProcess(k_tiny, t, diag=noise)
    return gp_smol, gp_tiny, y, N


def _integrated(solver=None, Ninst=1, N=8):
    S, w, Q = 2.5, 0.2, 2.0
    sigma = jnp.sqrt(S * w * Q)
    k_smol = smolgp.kernels.IntegratedSHO(
        omega=w, quality=Q, sigma=sigma, num_insts=Ninst
    )
    k_dense = smolgp.kernels.dense.IntegratedSHOKernel(S=S, w=w, Q=Q)
    t = jnp.linspace(0.0, 100.0, N)
    texp = jnp.full(N, 3.0)
    instid = jnp.zeros(N, dtype=int) if Ninst == 1 else jnp.array([0, 1] * (N // 2))
    X = (t, texp, instid)
    y = jnp.sin(t / 10.0)
    noise = jnp.full(N, 0.05)
    kwargs = {} if solver is None else {"solver": solver}
    gp_smol = smolgp.GaussianProcess(k_smol, X, noise=noise, **kwargs)
    gp_tiny = tinygp.GaussianProcess(k_dense, X, diag=noise)
    return gp_smol, gp_tiny, y, N


@pytest.mark.parametrize("solver,name", PLAIN_SOLVERS)
def test_prior_covariance_matches_tinygp(solver, name):
    """Unconditioned: k(X, X) + noise, matching tinygp including the noise."""
    gp_smol, gp_tiny, _y, N = _plain(solver)
    C = gp_smol.covariance
    assert C.shape == (N, N), f"[{name}] shape {C.shape}"
    d = float(jnp.max(jnp.abs(C - gp_tiny.covariance)))
    assert d < 1e-12, f"[{name}] prior covariance vs tinygp: {d:.3e}"
    # ...and it really does include the noise (not just the bare kernel).
    bare = gp_smol.kernel(gp_smol.X, gp_smol.X)
    assert not jnp.allclose(C, bare), f"[{name}] noise missing from prior covariance"


@pytest.mark.parametrize("solver,name", PLAIN_SOLVERS)
def test_posterior_covariance_matches_tinygp(solver, name):
    """Conditioned: the Eq. 12.55 construction must reproduce tinygp's Schur
    complement, computed by a completely different route."""
    gp_smol, gp_tiny, y, N = _plain(solver)
    _, cond_smol = gp_smol.condition(y)
    _, cond_tiny = gp_tiny.condition(y)
    C = cond_smol.covariance
    ref = cond_tiny.covariance - OFFSET * jnp.eye(N)
    assert C.shape == (N, N), f"[{name}] shape {C.shape}"
    d = float(jnp.max(jnp.abs(C - ref)))
    assert d < 1e-10, f"[{name}] posterior covariance vs tinygp: {d:.3e}"
    assert jnp.allclose(C, C.T, atol=1e-12), f"[{name}] not symmetric"
    # A covariance matrix must be PSD.
    assert float(jnp.min(jnp.linalg.eigvalsh(C))) > -1e-10, f"[{name}] not PSD"


@pytest.mark.parametrize("solver,name", INTEGRATED_SOLVERS)
@pytest.mark.parametrize("Ninst", [1, 2])
def test_integrated_covariance_matches_dense(solver, name, Ninst):
    """The integrated case is the sharper test: K = 2N states, so the
    state-to-data selection and ordering have to be right, not just the
    smoother recursion."""
    gp_smol, gp_tiny, y, N = _integrated(solver, Ninst=Ninst)
    label = f"{name}, Ninst={Ninst}"

    dp = float(jnp.max(jnp.abs(gp_smol.covariance - gp_tiny.covariance)))
    assert dp < 1e-10, f"[{label}] prior covariance: {dp:.3e}"

    _, cond_smol = gp_smol.condition(y)
    _, cond_tiny = gp_tiny.condition(y)
    C = cond_smol.covariance
    ref = cond_tiny.covariance - OFFSET * jnp.eye(N)
    d = float(jnp.max(jnp.abs(C - ref)))
    assert d < 1e-9, f"[{label}] posterior covariance: {d:.3e}"
    assert jnp.allclose(C, C.T, atol=1e-10), f"[{label}] not symmetric"


@pytest.mark.parametrize("solver,name", PLAIN_SOLVERS)
def test_variance_is_the_covariance_diagonal(solver, name):
    """`variance` is computed cheaply rather than as diag(covariance), so the
    two must be pinned to agree -- prior and posterior alike."""
    gp_smol, gp_tiny, y, _N = _plain(solver)
    dv = float(jnp.max(jnp.abs(gp_smol.variance - jnp.diag(gp_smol.covariance))))
    assert dv < 1e-12, f"[{name}] prior variance != diag(covariance): {dv:.3e}"
    assert jnp.allclose(gp_smol.variance, gp_tiny.variance, atol=1e-12), (
        f"[{name}] prior variance disagrees with tinygp"
    )

    _, cond = gp_smol.condition(y)
    dv = float(jnp.max(jnp.abs(cond.variance - jnp.diag(cond.covariance))))
    assert dv < 1e-9, f"[{name}] posterior variance != diag(covariance): {dv:.3e}"


def test_prior_variance_includes_noise():
    """The unconditioned GP models observed y = f(X) + eps, so its marginal
    variance carries the measurement noise (tinygp's convention)."""
    gp_smol, _gp_tiny, _y, N = _plain()
    noise = 0.09
    bare = jax.vmap(gp_smol.kernel.evaluate)(gp_smol.X, gp_smol.X)
    assert jnp.allclose(gp_smol.variance, bare + noise, atol=1e-12), (
        "prior variance should be k(x,x) + noise"
    )


def test_covariance_unsorted_X_is_permuted():
    """Input order is a presentation detail: shuffling X must permute the
    covariance, not change it."""
    gp_smol, _gp_tiny, y, N = _plain()
    t = gp_smol.X
    noise = jnp.full(N, 0.09)
    perm = jax.random.permutation(jax.random.PRNGKey(1), N)
    k = gp_smol.kernel
    gp_u = smolgp.GaussianProcess(k, t[perm], noise=noise[perm])

    _, cond_s = gp_smol.condition(y)
    _, cond_u = gp_u.condition(y[perm])
    d = float(jnp.max(jnp.abs(cond_u.covariance - cond_s.covariance[jnp.ix_(perm, perm)])))
    assert d < 1e-10, f"covariance not permuted consistently: {d:.3e}"


def test_covariance_at_X_test_raises():
    """A GP from condition(y, X_test=...) lives at the test points, where the
    cross-covariance is not available -- it must say so rather than quietly
    returning the training-point matrix."""
    gp_smol, _gp_tiny, y, _N = _plain()
    _, cond = gp_smol.condition(y, jnp.linspace(0.0, 500.0, 13))
    with pytest.raises(NotImplementedError, match="training coordinates"):
        _ = cond.covariance


if __name__ == "__main__":
    for s, n in PLAIN_SOLVERS:
        test_prior_covariance_matches_tinygp(s, n)
        test_posterior_covariance_matches_tinygp(s, n)
        test_variance_is_the_covariance_diagonal(s, n)
    for s, n in INTEGRATED_SOLVERS:
        for ni in (1, 2):
            test_integrated_covariance_matches_dense(s, n, ni)
    test_prior_variance_includes_noise()
    test_covariance_unsorted_X_is_permuted()
    test_covariance_at_X_test_raises()
    print("All covariance tests passed.")
