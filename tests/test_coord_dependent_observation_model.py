"""Solvers must hand the observation model the FULL coordinates, not just
the sortable scalar timeline.

A kernel's ``H`` is allowed to depend on any channel of ``X``, not only the
sort key: the motivating case is an asynchronous multi-output model, where
``X = (t, texp, outputid)`` and ``H`` selects a per-output amplitude by
``outputid``. Every *built-in* non-integrated kernel ignores ``X`` entirely
(all ten do a literal ``del X`` and return a constant matrix), so
``H(t[k])`` and ``H(X[k])`` were indistinguishable everywhere in the test
suite -- which is exactly why the plain solvers passing the reduced timeline
went unnoticed until a user hit it. These tests use a kernel whose ``H``
genuinely reads ``X``, which is the only way to tell the two apart.

The integrated solvers were always correct here (they evaluate
``jax.vmap(H)(X)`` over the full coordinates), so this covers the plain
sequential and parallel solvers.
"""

import jax
import jax.numpy as jnp
import pytest
from tinygp.helpers import JAXArray

import smolgp

jax.config.update("jax_enable_x64", True)

AMP1, AMP2 = 3.0, 1.5


class ScaledAmpAsync(smolgp.kernels.Wrapper):
    """Two asynchronous output series sharing one latent Matern52 process,
    each observed with its own amplitude, selected by ``X``'s id channel."""

    kernel: smolgp.kernels.StateSpaceModel
    amp1: float | JAXArray
    amp2: float | JAXArray

    def __init__(self, scale, sigma=1.0, amp1=AMP1, amp2=AMP2, name="ScaledAmpAsync"):
        self.kernel = smolgp.kernels.Matern52(scale=scale, sigma=sigma)
        self.amp1 = amp1
        self.amp2 = amp2
        self.name = name

    def observation_model(self, X, component=None):
        _t, _texp, outputid = X
        return jnp.where(
            outputid == 0,
            jnp.array([[self.amp1, 0.0, 0.0]]),
            jnp.array([[self.amp2, 0.0, 0.0]]),
        )


def _latent(t):
    return jnp.sin(0.7 * t) + 0.3 * jnp.cos(1.9 * t)


def _build(solver=None, n_per=25, noise_var=1e-6):
    """Two interleaved output series generated from ONE latent curve, so the
    exact posterior ratio between them is known analytically (amp2/amp1)."""
    t1 = jnp.linspace(0.0, 20.0, n_per)
    t2 = jnp.linspace(0.5, 20.5, n_per)
    t_all = jnp.concatenate([t1, t2])
    outputid = jnp.concatenate(
        [jnp.zeros(n_per, dtype=int), jnp.ones(n_per, dtype=int)]
    )
    idx = jnp.argsort(t_all, stable=True)
    t_all, outputid = t_all[idx], outputid[idx]

    y = jnp.where(outputid == 0, AMP1 * _latent(t_all), AMP2 * _latent(t_all))
    X = (t_all, jnp.zeros_like(t_all), outputid)
    noise = jnp.full(t_all.shape, noise_var)

    kwargs = {} if solver is None else {"solver": solver}
    gp = smolgp.GaussianProcess(
        kernel=ScaledAmpAsync(scale=1.0, sigma=2.0), X=X, noise=noise, **kwargs
    )
    return gp, y


PLAIN_SOLVERS = [
    (None, "StateSpaceSolver"),
    (smolgp.solvers.ParallelStateSpaceSolver, "ParallelStateSpaceSolver"),
]


@pytest.mark.parametrize("solver,name", PLAIN_SOLVERS)
def test_coord_dependent_H_conditions_and_predicts(solver, name):
    """condition()/predict() must run at all, and must respect the
    per-output amplitude encoded in X's id channel."""
    gp, y = _build(solver)
    llh, condgp = gp.condition(y)
    assert jnp.isfinite(llh), f"[{name}] non-finite log probability"

    ts = jnp.linspace(1.0, 19.0, 40)
    zeros = jnp.zeros_like(ts)
    mu0 = condgp.predict((ts, zeros, zeros.astype(int)))
    mu1 = condgp.predict((ts, zeros, jnp.ones_like(ts, dtype=int)))

    assert jnp.all(jnp.isfinite(mu0)) and jnp.all(jnp.isfinite(mu1)), (
        f"[{name}] NaN/Inf"
    )

    # Both outputs view the SAME latent state through different H, so the
    # posterior means must differ by exactly amp2/amp1 -- an exact algebraic
    # identity, independent of how well the GP fits the data.
    ratio_err = float(jnp.max(jnp.abs(mu1 - (AMP2 / AMP1) * mu0)))
    assert ratio_err < 1e-10, (
        f"[{name}] outputs must differ by exactly amp2/amp1; max err {ratio_err:.3e}"
    )

    # ...and the id channel must genuinely be doing something (guard against
    # an H that accidentally ignores it and makes the above trivially true).
    assert float(jnp.max(jnp.abs(mu1 - mu0))) > 1e-3, (
        f"[{name}] outputs are identical -- the id channel is being ignored"
    )

    # With near-noiseless data the posterior should track the true curve
    err = float(jnp.max(jnp.abs(mu0 - AMP1 * _latent(ts))))
    assert err < 0.5, f"[{name}] posterior does not track the latent curve: {err:.3e}"


def test_coord_dependent_H_serial_and_parallel_agree():
    """The two plain solvers are the same computation by different means, so
    they must agree to near machine precision on this kernel too."""
    gp_seq, y = _build(None)
    gp_par, _ = _build(smolgp.solvers.ParallelStateSpaceSolver)

    llh_seq, cond_seq = gp_seq.condition(y)
    llh_par, cond_par = gp_par.condition(y)
    dllh = float(jnp.abs(llh_seq - llh_par))
    assert dllh < 1e-8, f"log probability mismatch: {dllh:.3e}"

    ts = jnp.linspace(-3.0, 24.0, 60)  # retrodict + interpolate + extrapolate
    zeros = jnp.zeros_like(ts)
    for oid in [0, 1]:
        X_test = (ts, zeros, jnp.full(ts.shape, oid, dtype=int))
        mu_seq, var_seq = cond_seq.predict(X_test, return_var=True)
        mu_par, var_par = cond_par.predict(X_test, return_var=True)
        dm = float(jnp.max(jnp.abs(mu_seq - mu_par)))
        dv = float(jnp.max(jnp.abs(var_seq - var_par)))
        assert dm < 1e-8, f"[outputid={oid}] mean mismatch: {dm:.3e}"
        assert dv < 1e-8, f"[outputid={oid}] var mismatch: {dv:.3e}"


def test_coord_dependent_H_sample_matches_predict():
    """sample() drives the same observation model through a different code
    path (sample_prior_trajectory + condition_batched_mean), so check its
    empirical moments against predict()'s analytic ones."""
    gp, y = _build(None, noise_var=0.01)
    _, condgp = gp.condition(y)

    ts = jnp.linspace(2.0, 18.0, 12)
    zeros = jnp.zeros_like(ts)
    X_test = (ts, zeros, jnp.ones_like(ts, dtype=int))  # the amp2 output

    samples = condgp.sample(jax.random.PRNGKey(0), shape=(4000,), X_test=X_test)
    mu_pred, var_pred = condgp.predict(X_test, return_var=True)

    scale = float(jnp.sqrt(jnp.max(var_pred)))
    dmean = float(jnp.max(jnp.abs(jnp.mean(samples, axis=-1) - mu_pred)))
    dvar = float(jnp.max(jnp.abs(jnp.var(samples, axis=-1) - var_pred)))
    assert dmean < 0.15 * scale, f"sample mean vs predict mean: {dmean:.3e}"
    assert dvar < 0.25 * float(jnp.max(var_pred)), (
        f"sample var vs predict var: {dvar:.3e}"
    )


if __name__ == "__main__":
    for s, n in PLAIN_SOLVERS:
        test_coord_dependent_H_conditions_and_predicts(s, n)
    test_coord_dependent_H_serial_and_parallel_agree()
    test_coord_dependent_H_sample_matches_predict()
    print("All coordinate-dependent observation model tests passed.")
