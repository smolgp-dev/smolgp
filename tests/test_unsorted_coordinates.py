"""Coordinates need not arrive sorted.

Every solver steps forward through a sorted state timeline internally, but
that is an implementation detail: results are reported in the caller's input
order. The integrated solvers always did this (they lexsort in ``__init__``
and map back through ``obsid``), while the plain ones assumed pre-sorted
input and silently returned garbage otherwise -- an ``-inf`` log probability
for shuffled ``X``, and a marginal variance ~250x too large for a shuffled
``X_test`` prior sample.

These tests pin the invariant directly: shuffling the input must permute the
output and nothing else.
"""

import jax
import jax.numpy as jnp
import pytest

import smolgp

jax.config.update("jax_enable_x64", True)

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


def _plain(solver, N=40):
    kernel = smolgp.kernels.SHO(2 * jnp.pi / 60.0, 5.3, 2.1)
    t = jnp.linspace(0.0, 300.0, N)
    y = jnp.sin(t / 30.0) + 0.1 * jnp.cos(t / 7.0)
    perm = jax.random.permutation(jax.random.PRNGKey(1), N)
    kwargs = {} if solver is None else {"solver": solver}
    build = lambda tt: smolgp.GaussianProcess(kernel, X=tt, noise=0.05, **kwargs)
    return build(t), build(t[perm]), y, perm


@pytest.mark.parametrize("solver,name", PLAIN_SOLVERS)
def test_shuffled_X_log_probability(solver, name):
    """The likelihood is a property of the dataset, not of the order it was
    listed in, so it must be *identical* (not merely close)."""
    gp_s, gp_u, y, perm = _plain(solver)
    llh_s = gp_s.log_probability(y)
    llh_u = gp_u.log_probability(y[perm])
    assert jnp.isfinite(llh_u), f"[{name}] shuffled X gave a non-finite log probability"
    assert jnp.allclose(llh_s, llh_u, atol=1e-10), (
        f"[{name}] log probability changed with input order: "
        f"{float(llh_s):.10f} vs {float(llh_u):.10f}"
    )


@pytest.mark.parametrize("solver,name", PLAIN_SOLVERS)
def test_shuffled_X_condition_returns_input_order(solver, name):
    """condition() must report loc/variance in the order the data were given."""
    gp_s, gp_u, y, perm = _plain(solver)
    _, cond_s = gp_s.condition(y)
    _, cond_u = gp_u.condition(y[perm])
    dloc = float(jnp.max(jnp.abs(cond_u.loc - cond_s.loc[perm])))
    dvar = float(jnp.max(jnp.abs(cond_u.variance - cond_s.variance[perm])))
    assert dloc < 1e-10, f"[{name}] conditioned mean not in input order: {dloc:.3e}"
    assert dvar < 1e-10, f"[{name}] conditioned variance not in input order: {dvar:.3e}"


@pytest.mark.parametrize("solver,name", PLAIN_SOLVERS)
def test_shuffled_X_predict(solver, name):
    """Predictions at a fixed test grid must not depend on training order."""
    gp_s, gp_u, y, perm = _plain(solver)
    _, cond_s = gp_s.condition(y)
    _, cond_u = gp_u.condition(y[perm])
    t_test = jnp.linspace(-20.0, 320.0, 25)  # retrodict + interpolate + extrapolate
    mu_s, var_s = cond_s.predict(t_test, return_var=True)
    mu_u, var_u = cond_u.predict(t_test, return_var=True)
    assert float(jnp.max(jnp.abs(mu_s - mu_u))) < 1e-10, f"[{name}] predict mean"
    assert float(jnp.max(jnp.abs(var_s - var_u))) < 1e-10, f"[{name}] predict var"


@pytest.mark.parametrize("solver,name", PLAIN_SOLVERS)
def test_shuffled_X_test_prior_sample(solver, name):
    """A prior sample at a shuffled X_test must have the kernel's own marginal
    variance -- this is where an unsorted timeline produced negative time
    steps and a variance ~250x too large."""
    kernel = smolgp.kernels.SHO(2 * jnp.pi / 60.0, 5.3, 2.1)
    t = jnp.linspace(0.0, 300.0, 60)
    perm = jax.random.permutation(jax.random.PRNGKey(1), 60)
    kwargs = {} if solver is None else {"solver": solver}
    gp = smolgp.GaussianProcess(kernel, X=t, noise=0.0, **kwargs)

    true_var = float(kernel(t[:1], t[:1])[0, 0])
    s_sorted = gp.sample(jax.random.PRNGKey(0), shape=(4000,), X_test=t)
    s_shuf = gp.sample(jax.random.PRNGKey(0), shape=(4000,), X_test=t[perm])
    for label, s in [("sorted", s_sorted), ("shuffled", s_shuf)]:
        v = float(jnp.mean(jnp.var(s, axis=-1)))
        assert abs(v - true_var) < 0.15 * true_var, (
            f"[{name}] {label} X_test prior sample variance {v:.4f} != kernel's {true_var:.4f}"
        )


@pytest.mark.parametrize("solver,name", PLAIN_SOLVERS)
def test_shuffled_X_test_posterior_sample(solver, name):
    """A posterior sample at a shuffled X_test must be the sorted one,
    permuted -- same key, same draw."""
    gp_s, _gp_u, y, _perm = _plain(solver)
    _, cond = gp_s.condition(y)
    t_test = jnp.linspace(-20.0, 320.0, 30)
    p = jax.random.permutation(jax.random.PRNGKey(5), 30)
    s_sorted = cond.sample(jax.random.PRNGKey(2), shape=(2000,), X_test=t_test)
    s_shuf = cond.sample(jax.random.PRNGKey(2), shape=(2000,), X_test=t_test[p])
    dmean = float(
        jnp.max(jnp.abs(jnp.mean(s_shuf, axis=-1) - jnp.mean(s_sorted, axis=-1)[p]))
    )
    assert dmean < 1e-10, (
        f"[{name}] posterior sample not permuted consistently: {dmean:.3e}"
    )


@pytest.mark.parametrize("solver,name", PLAIN_SOLVERS)
def test_shuffled_X_with_tied_timestamps(solver, name):
    """The sort must be *stable*: observations sharing a timestamp keep their
    relative input order, so two distinct measurements at the same instant are
    not silently swapped."""
    kernel = smolgp.kernels.SHO(2 * jnp.pi / 60.0, 5.3, 2.1)
    t = jnp.array([0.0, 10.0, 20.0, 20.0, 30.0, 40.0])  # tie at 20.0
    y = jnp.array([0.1, 0.5, -0.2, 0.9, 0.3, -0.1])
    perm = jnp.array([4, 0, 3, 1, 5, 2])  # keeps 20.0(a) before 20.0(b)
    kwargs = {} if solver is None else {"solver": solver}
    gp_s = smolgp.GaussianProcess(kernel, X=t, noise=0.05, **kwargs)
    gp_u = smolgp.GaussianProcess(kernel, X=t[perm], noise=0.05, **kwargs)

    llh_s, cond_s = gp_s.condition(y)
    llh_u, cond_u = gp_u.condition(y[perm])
    assert jnp.isfinite(llh_u), (
        f"[{name}] tied+shuffled gave non-finite log probability"
    )
    assert jnp.allclose(llh_s, llh_u, atol=1e-10), f"[{name}] tied log probability"
    dloc = float(jnp.max(jnp.abs(cond_u.loc - cond_s.loc[perm])))
    assert dloc < 1e-10, (
        f"[{name}] tied conditioned mean not in input order: {dloc:.3e}"
    )


@pytest.mark.parametrize("solver,name", INTEGRATED_SOLVERS)
def test_shuffled_X_integrated_unchanged(solver, name):
    """The integrated solvers already sorted internally; guard against the
    plain-solver change regressing that."""
    S, w, Q = 2.5, 0.2, 2.0
    kernel = smolgp.kernels.IntegratedSHO(
        omega=w, quality=Q, sigma=jnp.sqrt(S * w * Q), num_insts=2
    )
    t = jnp.linspace(0.0, 100.0, 12)
    texp = jnp.full(12, 2.0)
    instid = jnp.array([0, 1] * 6)
    y = jnp.sin(t / 10.0)
    perm = jax.random.permutation(jax.random.PRNGKey(4), 12)
    kwargs = {} if solver is None else {"solver": solver}
    gp_s = smolgp.GaussianProcess(kernel, X=(t, texp, instid), noise=0.05, **kwargs)
    gp_u = smolgp.GaussianProcess(
        kernel, X=(t[perm], texp[perm], instid[perm]), noise=0.05, **kwargs
    )
    llh_s, cond_s = gp_s.condition(y)
    llh_u, cond_u = gp_u.condition(y[perm])
    assert jnp.allclose(llh_s, llh_u, atol=1e-9), f"[{name}] log probability"
    dloc = float(jnp.max(jnp.abs(cond_u.loc - cond_s.loc[perm])))
    assert dloc < 1e-9, f"[{name}] conditioned mean not in input order: {dloc:.3e}"


if __name__ == "__main__":
    for s, n in PLAIN_SOLVERS:
        test_shuffled_X_log_probability(s, n)
        test_shuffled_X_condition_returns_input_order(s, n)
        test_shuffled_X_predict(s, n)
        test_shuffled_X_test_prior_sample(s, n)
        test_shuffled_X_test_posterior_sample(s, n)
        test_shuffled_X_with_tied_timestamps(s, n)
    for s, n in INTEGRATED_SOLVERS:
        test_shuffled_X_integrated_unchanged(s, n)
    print("All unsorted-coordinate tests passed.")
