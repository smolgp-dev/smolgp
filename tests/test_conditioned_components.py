"""``get_component(..., conditioned_on=...)``: conditional component variances vs a dense GP.

For y = f_A + f_B + f_C + noise (independent GP components), the dense joint posterior of the
components at times t, t' is

    Cov(f_a(t), f_b(t') | y) = δ_ab k_a(t, t') − k_a(t, X) C⁻¹ k_b(X, t'),   C = Σ_i k_i(X, X) + N.

Conditioning a target sum T on variables L (each a sum of components) at the same time t is then the
Schur complement Var(T) − Cov(T, L) Cov(L, L)⁻¹ Cov(L, T). smolgp must match it at the data
(``get_component_mean``) and at test points (``PredictedStates.get_component``).
"""

import jax
import jax.numpy as jnp
import numpy as np
import tinygp

import smolgp
from tests.utils import generate_data

jax.config.update("jax_enable_x64", True)

PARAMS = dict(
    A=dict(omega=0.0195, quality=7.63, sigma=0.59),
    B=dict(omega=0.0023, quality=1 / np.sqrt(2), sigma=0.329),
)
M52 = dict(scale=600.0, sigma=5.3)


def _kernels():
    smol = {
        "A": smolgp.kernels.SHO(**PARAMS["A"], name="A"),
        "B": smolgp.kernels.SHO(**PARAMS["B"], name="B"),
        "C": smolgp.kernels.Matern52(**M52, name="C"),
    }
    dense = {
        "A": tinygp.kernels.quasisep.SHO(**PARAMS["A"]),
        "B": tinygp.kernels.quasisep.SHO(**PARAMS["B"]),
        "C": tinygp.kernels.quasisep.Matern52(**M52),
    }
    return smol, dense


def _dense_conditional_var(dense, X, yerr, t, target, conds):
    """Dense marginal and conditional variance of the sum ``target`` given ``conds`` at times ``t``."""
    names = list(dense)
    C = sum(np.asarray(dense[n](X, X)) for n in names) + np.diag(yerr**2)
    Cinv = np.linalg.inv(C)

    def cov(a, b):  # Cov(f_a(t_i), f_b(t_i) | y) at each t_i
        out = -np.einsum("ij,jk,ik->i", np.asarray(dense[a](t, X)), Cinv, np.asarray(dense[b](t, X)))
        if a == b:
            out = out + np.diag(np.asarray(dense[a](t, t)))
        return out

    def cov_sum(sa, sb):  # Cov(Σ_{a∈sa} f_a, Σ_{b∈sb} f_b)
        return sum(cov(a, b) for a in sa for b in sb)

    as_set = lambda v: (v,) if isinstance(v, str) else tuple(v)
    T = as_set(target)
    L = [as_set(v) for v in conds]
    var = cov_sum(T, T)
    c = np.stack([cov_sum(T, l) for l in L], axis=-1)                               # (M, k)
    S = np.stack([np.stack([cov_sum(li, lj) for lj in L], -1) for li in L], -2)    # (M, k, k)
    cvar = var - np.einsum("mj,mj->m", c, np.linalg.solve(S, c[..., None])[..., 0])
    return var, cvar


CASES = [
    ("A", ["B"]),                    # single variable
    ("A", [("B", "C")]),             # conditional on the sum B + C
    ("A", ["B", "C"]),               # conditional on B and C jointly
    (["A", "B"], ["C"]),             # summed target
    ("C", [("A", "B")]),
]


def test_conditioned_on_matches_dense():
    smol, dense = _kernels()
    ksmol = smol["A"] + smol["B"] + smol["C"]
    kdense = dense["A"] + dense["B"] + dense["C"]
    N, yerr = 60, 0.3
    X, y = generate_data(N, kdense, yerr, tmin=0, tmax=1000)
    yerr = np.full(N, yerr)
    gp = smolgp.GaussianProcess(kernel=ksmol, X=X, noise=yerr**2)
    _, cg = gp.condition(y)
    t_test = jnp.linspace(-50.0, 1050.0, 200)
    ps = cg.predict(t_test, return_full_state=True, return_var=True)

    for target, conds in CASES:
        # at test points
        mu, var = ps.get_component(target, return_var=True)
        mu_c, cvar = ps.get_component(target, return_var=True, conditioned_on=conds)
        dvar, dcvar = _dense_conditional_var(dense, np.asarray(X), yerr, np.asarray(t_test), target, conds)
        np.testing.assert_allclose(mu_c, mu)                                   # mean unchanged
        np.testing.assert_allclose(var, dvar, rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(cvar, dcvar, rtol=1e-6, atol=1e-9)
        assert np.all(np.asarray(cvar) <= np.asarray(var) + 1e-12)

        # at the data
        _, dvar_X = cg.get_component_mean(target, return_var=True)
        _, cvar_X = cg.get_component_mean(target, return_var=True, conditioned_on=conds)
        dv, dcv = _dense_conditional_var(dense, np.asarray(X), yerr, np.asarray(X), target, conds)
        np.testing.assert_allclose(dvar_X, dv, rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(cvar_X, dcv, rtol=1e-6, atol=1e-9)


def test_conditioned_on_forms():
    """A bare name / bare tuple is one variable; joint conditioning is at least as tight as either alone."""
    smol, dense = _kernels()
    ksmol = smol["A"] + smol["B"] + smol["C"]
    X, y = generate_data(40, dense["A"] + dense["B"] + dense["C"], 0.3, tmin=0, tmax=1000)
    _, cg = smolgp.GaussianProcess(kernel=ksmol, X=X, noise=0.09).condition(y)
    ps = cg.predict(jnp.linspace(0.0, 1000.0, 50), return_full_state=True, return_var=True)
    v = lambda c: np.asarray(ps.get_component("A", return_var=True, conditioned_on=c)[1])
    np.testing.assert_allclose(v("B"), v(["B"]))
    np.testing.assert_allclose(v(("B", "C")), v([("B", "C")]))
    joint = v(["B", "C"])
    assert np.all(joint <= v(["B"]) + 1e-12) and np.all(joint <= v(["C"]) + 1e-12)
