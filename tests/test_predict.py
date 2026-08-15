import jax
import jax.numpy as jnp
import tinygp

import smolgp

key = jax.random.PRNGKey(0)
jax.config.update("jax_enable_x64", True)

OFFSET = float(jnp.sqrt(jnp.finfo(jnp.array([0.0])).eps))  # tinygp variance jitter


def _build_dataset(Ninst, key, solver=None, Nobs=None):
    """
    A realistic (real sampled SHO process, exposure-averaged) multi-exposure
    dataset for Ninst in {1, 2, 3} instruments, plus matching smolgp/tinygp
    GaussianProcess objects. Mirrors tests/test_integrated.py's
    _generic_tie_dataset/build_comparison_dataset, but without any
    deliberately-tied timestamps. Can specify number of observations
    per instrument via Nobs (list of length Ninst, default [10, 8, 6]).
    """
    S, w, Q = 2.5, 0.2, 2.0
    sigma = jnp.sqrt(S * w * Q)
    true_kernel = tinygp.kernels.quasisep.SHO(omega=w, quality=Q, sigma=sigma)
    kernel_smol = smolgp.kernels.IntegratedSHO(
        omega=w, quality=Q, sigma=sigma, num_insts=Ninst
    )
    kernel_tiny = smolgp.kernels.dense.IntegratedSHOKernel(S=S, w=w, Q=Q)

    if Nobs is None:
        Nobs = [10, 8, 6]

    nA = Nobs[0]
    tA = jnp.linspace(0.0, 100.0, nA)
    texpA = jnp.full(nA, 3.0)
    instA = jnp.zeros(nA, dtype=int)
    t, texp, instid = tA, texpA, instA
    if Ninst >= 2:
        nB = Nobs[1]
        tB = jnp.linspace(2.0, 98.0, nB)
        texpB = jnp.full(nB, 2.0)
        instB = jnp.ones(nB, dtype=int)
        t = jnp.concatenate([t, tB])
        texp = jnp.concatenate([texp, texpB])
        instid = jnp.concatenate([instid, instB])
    if Ninst >= 3:
        nC = Nobs[2]
        tC = jnp.linspace(5.0, 95.0, nC)
        texpC = jnp.full(nC, 1.0)
        instC = jnp.full(nC, 2, dtype=int)
        t = jnp.concatenate([t, tC])
        texp = jnp.concatenate([texp, texpC])
        instid = jnp.concatenate([instid, instC])

    yerr = 0.05
    tmin, tmax = float(jnp.min(t - texp / 2)), float(jnp.max(t + texp / 2))
    buf = 0.1 * (tmax - tmin)
    t_grid = jnp.arange(tmin - buf, tmax + buf, 0.01)
    true_gp = tinygp.GaussianProcess(true_kernel, t_grid)
    y_grid = true_gp.sample(key)

    def make_exposure(m, d):
        tt = jnp.linspace(m - d / 2, m + d / 2, 200)
        return jnp.mean(jnp.interp(tt, t_grid, y_grid))

    y_true = jax.vmap(make_exposure)(t, texp)
    y = y_true + yerr * jax.random.normal(jax.random.split(key)[1], shape=y_true.shape)

    X_train = (t, texp, instid)
    kwargs = {} if solver is None else {"solver": solver}
    gp_smol = smolgp.GaussianProcess(
        kernel=kernel_smol, X=X_train, noise=jnp.full(t.shape, yerr**2), **kwargs
    )
    gp_tiny = tinygp.GaussianProcess(
        kernel=kernel_tiny, X=X_train, diag=jnp.full(t.shape, yerr**2)
    )
    return {
        "t": t,
        "texp": texp,
        "instid": instid,
        "y": y,
        "yerr": yerr,
        "gp_smol": gp_smol,
        "gp_tiny": gp_tiny,
    }


def _assert_matches_tiny(d, X_test, tol=1e-8, label=""):
    """Compare smolgp's exposure-aware predict() against tinygp's dense kernel prediction."""
    gp_smol, gp_tiny, y = d["gp_smol"], d["gp_tiny"], d["y"]
    _, condgp = gp_smol.condition(y)
    mu_smol, var_smol = condgp.predict(X_test, y=y, return_var=True)

    mu_tiny, var_tiny = gp_tiny.predict(y, X_test, return_var=True)
    var_tiny = var_tiny - OFFSET

    assert jnp.all(jnp.isfinite(mu_smol)), f"[{label}] predicted mean has NaN/Inf"
    assert jnp.all(jnp.isfinite(var_smol)), f"[{label}] predicted variance has NaN/Inf"

    diff_m = float(jnp.max(jnp.abs(mu_smol - mu_tiny)))
    diff_v = float(jnp.max(jnp.abs(var_smol - var_tiny)))
    assert diff_m < tol, f"[{label}] mean mismatch vs tinygp: {diff_m:.3e}"
    assert diff_v < tol, f"[{label}] var mismatch vs tinygp: {diff_v:.3e}"
    print(
        f"    ...[{label}] matches tinygp: max|dmean|={diff_m:.2e}, max|dvar|={diff_v:.2e}"
    )
    return mu_smol, var_smol


def test_predict_exposure_within_one_gap():
    """A single nonzero-texp query strictly inside a gap between training data
    (i.e. no intervening real states)."""
    # One instrument by default will generate points from
    # t=0 to t=100 in 10 sec gaps with 3 sec exposures
    # We will query at t=5 with a 1 sec exposure, which is within the first gap:
    # [-1.5, 1.5], <test point: [4, 5]>, [8.5, 11.5], [18.5, 21.5], ..., [98.5, 101.5]
    d = _build_dataset(Ninst=1, key=jax.random.PRNGKey(1))
    X_test = (jnp.array([5.0]), jnp.array([1.0]), jnp.array([0], dtype=int))
    _assert_matches_tiny(d, X_test, label="within one gap")


def test_predict_exposure_spans_multiple_states():
    """
    A query spanning multiple real states across multiple instruments,
    including one where the intervening real events are on a *different*
    instrument than instid_test (to exercise the shared cross-instrument
    process-noise coupling).
    """
    d = _build_dataset(Ninst=2, key=jax.random.PRNGKey(2))
    for instid_star, label in [(0, "same inst as gap owner"), (1, "different inst")]:
        X_test = (
            jnp.array([10.0]),
            jnp.array([8.0]),
            jnp.array([instid_star], dtype=int),
        )
        _assert_matches_tiny(d, X_test, label=f"spans multiple states, {label}")


def test_predict_exposure_ties():
    """Query boundary exactly coincides with a real exposure start/end."""
    d = _build_dataset(Ninst=1, key=jax.random.PRNGKey(3))
    # Instrument 0's exposure at tmid=0 spans [-1.5, 1.5] (texp=3.0)
    cases = {
        "start tie": (0.5, 4.0),  # a = 0.5 - 2.0 = -1.5 (real start)
        "end tie": (0.0, 3.0),  # b = 0.0 + 1.5 = 1.5 (real end)
    }
    for label, (t_star, delta_star) in cases.items():
        X_test = (
            jnp.array([t_star]),
            jnp.array([delta_star]),
            jnp.array([0], dtype=int),
        )
        _assert_matches_tiny(d, X_test, label=label)


def test_predict_exposure_instids_noeffect_in_shared_model():
    """
    For a model with a shared underlying process, the predicted mean and
    variance should be independent of instid_star. This is a simple check
    that instid_star does not affect such a prediction.
    """
    d = _build_dataset(Ninst=2, key=jax.random.PRNGKey(4))
    gp_smol, y = d["gp_smol"], d["y"]
    _, condgp = gp_smol.condition(y)

    t_star, delta_star = 30.0, 6.0
    mus, varss = [], []
    for instid_star in (0, 1):
        X_test = (
            jnp.array([t_star]),
            jnp.array([delta_star]),
            jnp.array([instid_star], dtype=int),
        )
        mu, var = condgp.predict(X_test, y=y, return_var=True)
        mus.append(mu)
        varss.append(var)
    assert jnp.allclose(mus[0], mus[1], atol=1e-12), (
        "instid_star changed the predicted mean"
    )
    assert jnp.allclose(varss[0], varss[1], atol=1e-12), (
        "instid_star changed the predicted variance"
    )
    print("    ...instid_star does not impact the prediction for a shared model.")


def test_predict_exposure_retrodict_extrapolate_everything():
    """Tests retrodict (entirely before all data), extrapolate (entirely after),
    and a test point spanning the *entire* dataset (both fallbacks at once)."""
    d = _build_dataset(Ninst=2, key=jax.random.PRNGKey(5))
    t, texp = d["t"], d["texp"]
    tmin, tmax = float(jnp.min(t - texp / 2)), float(jnp.max(t + texp / 2))

    cases = {
        "true retrodict": (tmin - 15.0, 4.0),
        "true extrapolate": (tmax + 15.0, 4.0),
        "spans entire dataset": ((tmin + tmax) / 2, 2 * (tmax - tmin)),
    }
    for label, (t_star, delta_star) in cases.items():
        X_test = (
            jnp.array([t_star]),
            jnp.array([delta_star]),
            jnp.array([0], dtype=int),
        )
        _assert_matches_tiny(d, X_test, label=label)


def test_predict_exposure_reproduces_training_point():
    """
    Predicting at the training points (t, delta, instid) must reproduce
    the output of condition()
    """
    d = _build_dataset(Ninst=2, key=jax.random.PRNGKey(6))
    gp_smol, y, t, texp, instid = d["gp_smol"], d["y"], d["t"], d["texp"], d["instid"]
    _, condgp = gp_smol.condition(y)

    for idx in range(len(t)):
        X_test = (t[idx : idx + 1], texp[idx : idx + 1], instid[idx : idx + 1])
        mu_pred, var_pred = condgp.predict(X_test, y=y, return_var=True)

        mu_at_data, var_at_data = condgp.loc, condgp.variance
        assert jnp.allclose(mu_pred, mu_at_data[idx], atol=1e-8), (
            "doesn't reproduce training mean"
        )
        assert jnp.allclose(var_pred, var_at_data[idx], atol=1e-8), (
            "doesn't reproduce training variance"
        )
    print("    ...predicting at the training points reproduces condition()'s result")


def test_predict_exposure_zero_delta_matches_instantaneous():
    """
    predict_exposure, in the delta_star -> 0 limit, must recover
    the same answer as the instantaneous predict path. Use a
    tiny (but-nonzero) delta for this check as delta_star==0 itself
    is explicitly routed to the instantaneous predict.
    """
    d = _build_dataset(Ninst=2, key=jax.random.PRNGKey(7))
    gp_smol, y = d["gp_smol"], d["y"]
    _, condgp = gp_smol.condition(y)

    t_star = 42.0
    zeros = jnp.zeros(1)

    # existing instantaneous path (delta_test == 0 exactly)
    X_inst = (jnp.array([t_star]), zeros, zeros.astype(int))
    mu_inst, var_inst = condgp.predict(X_inst, return_var=True)

    # exposure path with a tiny (but nonzero) delta
    for tiny_delta in (1e-2, 1e-4, 1e-6):
        X_exp = (
            jnp.array([t_star]),
            jnp.array([tiny_delta]),
            jnp.array([0], dtype=int),
        )
        mu_exp, var_exp = condgp.predict(X_exp, y=y, return_var=True)
        diff_m = float(jnp.abs(mu_exp - mu_inst))
        diff_v = float(jnp.abs(var_exp - var_inst))
        print(
            f"    ...delta={tiny_delta:.0e}: |dmean|={diff_m:.2e}, |dvar|={diff_v:.2e}"
        )
        assert diff_m < 10 * tiny_delta, (
            f"mean doesn't converge to instantaneous as delta->0 ({diff_m:.3e})"
        )
        assert diff_v < 10 * tiny_delta, (
            f"var doesn't converge to instantaneous as delta->0 ({diff_v:.3e})"
        )


def test_predict_exposure_recalls_y_automatically():
    """
    Once conditioned, a delta>0 (exposure-integrated) should use the
    cached y from GaussianProcess.condition() as needed.
    """
    d = _build_dataset(Ninst=1, key=jax.random.PRNGKey(8))
    gp_smol, y = d["gp_smol"], d["y"]
    _, condgp = gp_smol.condition(y)

    X_test = (jnp.array([5.0]), jnp.array([1.0]), jnp.array([0], dtype=int))
    mu_explicit, var_explicit = condgp.predict(X_test, y=y, return_var=True)
    mu_recalled, var_recalled = condgp.predict(X_test, return_var=True)  # no y

    assert jnp.all(jnp.isfinite(mu_recalled)), "recalled y prediction has NaN/Inf"
    assert jnp.array_equal(mu_explicit, mu_recalled), "recalled y gave a different mean"
    assert jnp.array_equal(var_explicit, var_recalled), (
        "recalled y gave a different variance"
    )
    print("    ...predict() correctly recalls y from ConditionedStates")


def test_predict_exposure_matches_tiny():
    """
    A single X_test (mixing delta==0 and delta>0 points) with several
    instid_star values to compare against tinygp in one call.
    """
    d = _build_dataset(Ninst=3, key=jax.random.PRNGKey(9))
    t_stars = jnp.array([5.0, 10.0, 10.0, 50.0, 2.0, 99.0, -10.0, 115.0, 20.0, 30.0])
    delta_stars = jnp.array([1.0, 8.0, 8.0, 20.0, 6.0, 10.0, 4.0, 4.0, 0.0, 0.0])
    instid_stars = jnp.array([0, 0, 1, 2, 0, 0, 0, 0, 0, 2], dtype=int)
    X_test = (t_stars, delta_stars, instid_stars)
    _assert_matches_tiny(d, X_test, label="mixed batch, Ninst=3")


def test_predict_exposure_parallel_solver():
    """Repeat the test with ParallelIntegratedStateSpaceSolver."""
    d = _build_dataset(
        Ninst=2,
        key=jax.random.PRNGKey(10),
        solver=smolgp.solvers.ParallelIntegratedStateSpaceSolver,
    )
    t_stars = jnp.array([5.0, 10.0, 10.0, -10.0, 115.0, 20.0])
    delta_stars = jnp.array([1.0, 8.0, 8.0, 4.0, 4.0, 0.0])
    instid_stars = jnp.array([0, 0, 1, 0, 0, 1], dtype=int)
    X_test = (t_stars, delta_stars, instid_stars)
    _assert_matches_tiny(d, X_test, label="parallel solver, mixed batch")


def test_predict_exposure_is_integral():
    """
    Independent check that the prediction represents the exposure-average of the
    underlying process. Computes the exposure-integrated prediction and compares
    it to a dense-grid instantaneous prediction over [t-delta/2, t+delta/2],
    which is then trapezoidally integrated and divided by delta_star.
    """
    d = _build_dataset(Ninst=2, key=jax.random.PRNGKey(12))
    gp_smol, y = d["gp_smol"], d["y"]
    _, condgp = gp_smol.condition(y)

    t_star, delta_star, instid_star = 10.0, 8.0, 0
    a, b = t_star - delta_star / 2, t_star + delta_star / 2

    X_exp = (
        jnp.array([t_star]),
        jnp.array([delta_star]),
        jnp.array([instid_star], dtype=int),
    )
    mu_exp, _ = condgp.predict(X_exp, y=y, return_var=True)

    s_grid = jnp.linspace(a, b, 400)
    zeros = jnp.zeros_like(s_grid)
    X_inst = (s_grid, zeros, zeros.astype(int))
    mu_inst = condgp.predict(X_inst, return_var=False)
    mu_quad = jnp.trapezoid(mu_inst, s_grid) / delta_star

    diff = float(jnp.abs(jnp.asarray(mu_exp).reshape(()) - mu_quad))
    print(f"    ...predict_exposure is the integral of predict: |dmean|={diff:.2e}")
    assert diff < 1e-6, (
        f"predict_exposure does not match the integral of predict: {diff:.3e}"
    )


def test_condition_with_X_test_matches_condition_then_predict():
    """``gp.condition(y, X_test)`` must agree exactly with the two-step
    ``gp.condition(y)`` then ``condgp.predict(X_test)``.

    These are two spellings of the same computation -- the inline form just
    fuses the predict into the condition call -- so they should agree to
    machine precision, for every solver type.

    Regression test for a shape mismatch in the inline path: the plain
    (non-integrated) solvers' ``condition()`` returned a bare ``t_states``
    array as ``conditioned_results[0]``, while ``predict()`` unpacks that
    slot as a 4-field ``state_coords``. Every other call site rebuilt the
    proper value via ``GaussianProcess.state_coords`` first, so only the
    inline ``condition(y, X_test=...)`` path (gp.py) hit it -- raising
    ``ValueError: too many values to unpack`` for N != 4 training points,
    and (worse) silently unpacking 4 scalar times as the four fields when
    N == 4. The N == 4 case is covered explicitly below.
    """
    S, w, Q = 2.5, 0.2, 2.0
    sigma = jnp.sqrt(S * w * Q)

    # --- plain / instantaneous kernel, both solver types ---
    kernel_plain = smolgp.kernels.SHO(omega=w, quality=Q, sigma=sigma)
    for solver, sname in [
        (None, "StateSpaceSolver"),
        (smolgp.solvers.ParallelStateSpaceSolver, "ParallelStateSpaceSolver"),
    ]:
        # N=4 is the silent-corruption case (a bare length-4 t_states array
        # unpacks "successfully" into the four state_coords fields); the
        # others raise outright. Cover both failure modes.
        for N in [4, 7]:
            k1, k2 = jax.random.split(jax.random.PRNGKey(N))
            t = jnp.sort(jax.random.uniform(k1, (N,), minval=0.0, maxval=50.0))
            y = jax.random.normal(k2, (N,))
            kwargs = {} if solver is None else {"solver": solver}
            gp = smolgp.GaussianProcess(
                kernel_plain, X=t, noise=jnp.full(N, 0.01), **kwargs
            )
            t_test = jnp.linspace(-10.0, 60.0, 13)

            _, condgp_two_step = gp.condition(y)
            mu_two, var_two = condgp_two_step.predict(t_test, return_var=True)

            _, condgp_inline = gp.condition(y, X_test=t_test)
            mu_in, var_in = condgp_inline.loc, condgp_inline.variance

            dm = float(jnp.max(jnp.abs(mu_in - mu_two)))
            dv = float(jnp.max(jnp.abs(var_in - var_two)))
            print(f"    ...[{sname}, N={N}] max|dmean|={dm:.2e}, max|dvar|={dv:.2e}")
            assert dm < 1e-10, f"[{sname}, N={N}] inline vs two-step mean: {dm:.3e}"
            assert dv < 1e-10, f"[{sname}, N={N}] inline vs two-step var: {dv:.3e}"

    # --- integrated kernel, both solver types (already consistent today) ---
    for solver, sname in [
        (None, "IntegratedStateSpaceSolver"),
        (
            smolgp.solvers.ParallelIntegratedStateSpaceSolver,
            "ParallelIntegratedStateSpaceSolver",
        ),
    ]:
        d = _build_dataset(Ninst=2, key=jax.random.PRNGKey(21), solver=solver)
        gp_smol, y = d["gp_smol"], d["y"]
        X_test = (
            jnp.array([5.0, 10.0, -10.0, 115.0]),
            jnp.array([1.0, 8.0, 4.0, 0.0]),
            jnp.array([0, 1, 0, 1], dtype=int),
        )

        _, condgp_two_step = gp_smol.condition(y)
        mu_two, var_two = condgp_two_step.predict(X_test, y=y, return_var=True)

        _, condgp_inline = gp_smol.condition(y, X_test=X_test)
        mu_in, var_in = condgp_inline.loc, condgp_inline.variance

        dm = float(jnp.max(jnp.abs(mu_in - mu_two)))
        dv = float(jnp.max(jnp.abs(var_in - var_two)))
        print(f"    ...[{sname}] max|dmean|={dm:.2e}, max|dvar|={dv:.2e}")
        assert dm < 1e-10, f"[{sname}] inline vs two-step mean: {dm:.3e}"
        assert dv < 1e-10, f"[{sname}] inline vs two-step var: {dv:.3e}"


if __name__ == "__main__":
    test_predict_exposure_within_one_gap()
    test_predict_exposure_spans_multiple_states()
    test_predict_exposure_ties()
    test_predict_exposure_instids_noeffect_in_shared_model()
    test_predict_exposure_retrodict_extrapolate_everything()
    test_predict_exposure_reproduces_training_point()
    test_predict_exposure_zero_delta_matches_instantaneous()
    test_predict_exposure_recalls_y_automatically()
    test_predict_exposure_matches_tiny()
    test_predict_exposure_parallel_solver()
    test_predict_exposure_is_integral()
    test_condition_with_X_test_matches_condition_then_predict()
    print("All predict() exposure tests passed.")
