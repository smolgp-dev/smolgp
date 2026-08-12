import warnings

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
from tests.utils import generate_integrated_data

key = jax.random.PRNGKey(0)
jax.config.update("jax_enable_x64", True)


def test_integrated():
    ## SHO Kernel
    S = 2.36
    w = 0.0195
    Q = 7.63
    sigma = jnp.sqrt(S * w * Q)

    ## Instantaneous kernel for generating data
    true_kernel = tinygp.kernels.quasisep.SHO(omega=w, quality=Q, sigma=sigma)

    ## Generate mock data
    Ninst = 2  # 3
    Ns = [30, 50, 80]
    yerr = [0.3, 0.6, 0.24]
    texps = [140.0, 55, 12]
    readouts = [40.0, 28.0, 40.0]
    t_train = []
    y_train = []
    texp_train = []
    yerr_train = []
    instid = []
    for n in range(Ninst):
        t, y = generate_integrated_data(
            Ns[n], true_kernel, texp=texps[n], readout=readouts[n], yerr=yerr[n]
        )
        t_train.append(t)
        y_train.append(y)
        texp_train.append(jnp.full_like(t, texps[n]))
        yerr_train.append(jnp.full_like(t, yerr[n]))
        instid.append(jnp.full_like(t, n).astype(int))  # has to be integer
    t_train = jnp.concatenate(t_train)
    y_train = jnp.concatenate(y_train)
    texp_train = jnp.concatenate(texp_train)
    yerr_train = jnp.concatenate(yerr_train)
    instid = jnp.concatenate(instid)
    X_train = (t_train, texp_train, instid)

    # Integrated kernels
    kernel_smol = smolgp.kernels.integrated.IntegratedSHO(
        omega=w, quality=Q, sigma=sigma, num_insts=Ninst
    )
    kernel_tiny = smolgp.kernels.dense.IntegratedSHOKernel(S=S, w=w, Q=Q)

    print("Testing IntegratedSHO kernel...")

    # Test k(Delta) agrees
    kernel_function(kernel_smol, kernel_tiny, tol=1e-9, atol=1e-12)

    # Build GP objects
    gp_smol = smolgp.GaussianProcess(kernel=kernel_smol, X=X_train, noise=yerr_train**2)
    gp_tiny = tinygp.GaussianProcess(kernel=kernel_tiny, X=X_train, diag=yerr_train**2)

    # Check likelihood
    likelihood(gp_smol, gp_tiny, y_train, tol=1e-10, atol=1e-13)

    # Check conditioning
    condition(gp_smol, gp_tiny, y_train, tol=1e-9, atol=1e-12)

    # Check predictions
    predict(gp_smol, gp_tiny, y_train, tol=1e-9, atol=1e-12)

    # TODO: test predict with exposure times


def test_integrated_evaluate():
    """
    kernel.evaluate(X1, X2) must match the dense implementation with tinygp

    Ensures the algorithm implemented to solve (H2 @ Pinf @ Phi.T @ H1.T)
    for the integral state (IntegratedStateSpaceModel.evaluate) is correct.
    """
    w0 = 2.0 * jnp.pi / 300
    Q = 2.0
    sigma = 1.0
    S0 = sigma**2 / (w0 * Q)

    kernel_smol = smolgp.kernels.IntegratedSHO(sigma=sigma, omega=w0, quality=Q)
    kernel_tiny = smolgp.kernels.dense.IntegratedSHOKernel(S=S0, w=w0, Q=Q)

    # Grid check: fixed (matching) exposure widths on both sides, varying separation.
    dts = jnp.linspace(0, 1000, 50)
    zeros = jnp.zeros_like(dts)
    instids = jnp.zeros_like(dts, dtype=int)
    for exptime in [0.0, 10.0, 30.0, 100.0, 300.0, 1000.0]:
        texp = jnp.full_like(dts, exptime)
        X0 = (zeros, texp, instids)
        X1 = (dts, texp, instids)
        cov_smol = kernel_smol(X0, X1)[0, :]
        cov_tiny = kernel_tiny(X0, X1)[0, :]
        diff = float(jnp.max(jnp.abs(cov_smol - cov_tiny)))
        assert diff < 1e-9, f"exptime={exptime}: max|diff|={diff:.3e}"
    print(
        "    ...evaluate() grid (matching widths, varying separation): matches dense kernel"
    )

    # Pairwise edge cases: asymmetric widths, mixed zero/nonzero, overlap,
    # nesting, reversed order, exact ties, and self-variance (X1==X2).
    cases = {
        "asymmetric widths, overlapping": (0.0, 50.0, 20.0, 30.0),
        "asymmetric widths, non-overlapping": (0.0, 20.0, 100.0, 80.0),
        "mixed: window1 point, window2 wide": (0.0, 0.0, 50.0, 200.0),
        "mixed: window1 wide, window2 point": (0.0, 200.0, 300.0, 0.0),
        "one nested inside other": (0.0, 100.0, 10.0, 20.0),
        "reverse order (t2 < t1)": (100.0, 40.0, 20.0, 60.0),
        "identical windows (X1==X2)": (50.0, 30.0, 50.0, 30.0),
        "exact tie: b1==a2 (touching)": (0.0, 40.0, 40.0, 40.0),
    }
    for label, (t1, d1, t2, d2) in cases.items():
        X1 = (jnp.array([t1]), jnp.array([d1]), jnp.array([0]))
        X2 = (jnp.array([t2]), jnp.array([d2]), jnp.array([0]))
        cov_smol = kernel_smol(X1, X2)[0, 0]
        cov_tiny = kernel_tiny(X1, X2)[0, 0]
        diff = float(jnp.abs(cov_smol - cov_tiny))
        assert diff < 1e-9, f"[{label}] max|diff|={diff:.3e}"
    print("    ...evaluate() pairwise edge cases: matches dense kernel")

    # evaluate_diag (the variance, X1==X2) must also match.
    for exptime in [0.0, 10.0, 100.0, 1000.0]:
        X = (jnp.array([0.0]), jnp.array([exptime]), jnp.array([0]))
        var_smol = kernel_smol(X)[0]
        var_tiny = kernel_tiny(X)[0]
        diff = float(jnp.abs(var_smol - var_tiny))
        assert diff < 1e-9, f"[variance, exptime={exptime}] diff={diff:.3e}"
    print("    ...evaluate_diag(): matches dense kernel")


def _make_instid_data(insts):
    """A minimal X = (t, texp, instid) tuple with a given instid array."""
    t = jnp.arange(len(insts), dtype=float)
    texp = jnp.full(len(insts), 0.1)
    instid = jnp.array(insts, dtype=int)
    return t, texp, instid


def test_num_insts_mismatch_reinit():
    """
    A kernel constructed with the default num_insts=1 should be automatically
    reinitialized (with a warning) to match the number of instruments implied
    by the data's instid array.
    """
    kernel = smolgp.kernels.IntegratedExp(scale=1.0)  # default num_insts=1
    X = _make_instid_data([0, 1, 2, 0, 1, 2])  # 3 instruments

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        gp = smolgp.GaussianProcess(kernel=kernel, X=X)
        assert any("Reinitializing" in str(wi.message) for wi in w), (
            "Expected a warning about reinitializing num_insts"
        )

    assert gp.kernel.num_insts == 3
    assert gp.solver.kernel.num_insts == 3
    print("    ...num_insts mismatch: auto-reinitialized to 3 with a warning")


def test_num_insts_wrapped_kernel():
    """
    An integrated kernel wrapped in a Scale (e.g. via scalar multiplication)
    should still be found and reinitialized.
    """
    kernel = 2.0 * smolgp.kernels.IntegratedExp(scale=1.0)
    X = _make_instid_data([0, 1, 2, 0, 1, 2])  # 3 instruments

    gp = smolgp.GaussianProcess(kernel=kernel, X=X)

    assert gp.kernel.kernel.num_insts == 3
    assert gp.solver.kernel.kernel.num_insts == 3
    print("    ...Scale-wrapped integrated kernel: num_insts fixed to 3")


def test_instid_validation():
    """
    Malformed instid arrays should raise a clear ValueError.
    """
    t, texp, instid = _make_instid_data([0, 1, 2, 0, 1, 2])
    bad_cases = {
        "non-integer dtype": (t, texp, instid.astype(float)),
        "wrong length": (t, texp, instid[:3]),
        "non-dense values": (t, texp, jnp.array([0, 2, 0, 2, 0, 2])),
    }
    for name, badX in bad_cases.items():
        try:
            smolgp.GaussianProcess(
                kernel=smolgp.kernels.IntegratedExp(scale=1.0), X=badX
            )
            raise AssertionError(f"Expected ValueError for {name}, but none was raised")
        except ValueError:
            print(f"    ...instid validation: correctly rejected {name}")


def test_num_insts_preserved_on_subset_predict():
    """
    Predicting at test points that only cover a subset of instruments must
    not shrink num_insts (or otherwise desync the kernel from the solver
    it was already conditioned with).
    """
    kernel = smolgp.kernels.IntegratedExp(scale=1.0, num_insts=3)
    X = _make_instid_data([0, 1, 2, 0, 1, 2])
    y = jnp.sin(X[0])

    gp = smolgp.GaussianProcess(kernel=kernel, X=X, noise=jnp.full(6, 1e-4))
    _, condgp = gp.condition(y)
    assert condgp.kernel.num_insts == 3

    X_test = _make_instid_data([0, 0, 0])  # only instrument 0
    mu = condgp.predict(X_test, y=y)

    assert condgp.kernel.num_insts == 3
    assert condgp.solver.kernel.num_insts == 3
    assert mu.shape == (3,)
    print("    ...num_insts preserved (3) after predicting on an instrument subset")


def _tied_exposure_data(tie):
    """
    Build a 2-instrument integrated dataset with one deliberately tied
    (zero-length-transition) pair of states at t=tied_t, embedded among
    otherwise normal (non-tied) exposures. Returns (t, texp, instid, tied_t).

    tie: one of "start-start", "end-end", "end-start"
    """
    # Instrument 0: exposures at t=0,10,20, texp=2 (spans [t-1, t+1])
    tA = jnp.array([0.0, 10.0, 20.0])
    texpA = jnp.full(3, 2.0)

    # Instrument 1: one exposure engineered to tie with instrument 0's
    # t=10 exposure (start=9, end=11), plus two untied exposures so there's
    # a realistic multi-exposure dataset to fit.
    if tie == "start-start":
        # tmid=11, texp=4 -> start = 11-2 = 9 (ties with instrument 0's start)
        tB, texpB, tied_t = (
            jnp.array([-15.0, 11.0, 25.0]),
            jnp.array([2.0, 4.0, 2.0]),
            9.0,
        )
    elif tie == "end-end":
        # tmid=9, texp=4 -> end = 9+2 = 11 (ties with instrument 0's end)
        tB, texpB, tied_t = (
            jnp.array([-15.0, 9.0, 25.0]),
            jnp.array([2.0, 4.0, 2.0]),
            11.0,
        )
    elif tie == "end-start":
        # tmid=13, texp=4 -> start = 13-2 = 11 (ties with instrument 0's end)
        tB, texpB, tied_t = (
            jnp.array([-15.0, 13.0, 25.0]),
            jnp.array([2.0, 4.0, 2.0]),
            11.0,
        )
    else:
        raise ValueError(tie)

    t = jnp.concatenate([tA, tB])
    texp = jnp.concatenate([texpA, texpB])
    instid = jnp.concatenate([jnp.zeros(3, dtype=int), jnp.ones(3, dtype=int)])
    return t, texp, instid, tied_t


def _run_tie_scenario(tie, solver=None):
    """
    Build the dataset for a given tie scenario and confirm conditioning and
    prediction remain finite throughout -- a regression test for
    https://github.com/smolgp-dev/smolgp/issues/3 (NaN in the RTS smoother
    when the transition between two adjacent states is exactly zero-length).
    """
    t, texp, instid, tied_t = _tied_exposure_data(tie)
    y = jnp.sin(0.1 * t)

    kernel = smolgp.kernels.IntegratedSHO(
        omega=0.2, quality=2.0, sigma=1.0, num_insts=2
    )
    kwargs = {} if solver is None else {"solver": solver}
    gp = smolgp.GaussianProcess(
        kernel=kernel, X=(t, texp, instid), noise=jnp.full(t.shape, 0.1**2), **kwargs
    )

    _, condgp = gp.condition(y)
    assert jnp.all(jnp.isfinite(condgp.loc)), f"[{tie}] smoothed mean contains NaN/Inf"
    assert jnp.all(jnp.isfinite(condgp.variance)), (
        f"[{tie}] smoothed variance contains NaN/Inf"
    )

    # Also predict directly at the tied timestamp (covers solver.py's smooth())
    X_test = (jnp.array([tied_t]), jnp.array([0.0]), jnp.array([0], dtype=int))
    mu_test, var_test = condgp.predict(X_test, return_var=True)
    assert jnp.all(jnp.isfinite(mu_test)), f"[{tie}] predict mean at tie is NaN/Inf"
    assert jnp.all(jnp.isfinite(var_test)), (
        f"[{tie}] predict variance at tie is NaN/Inf"
    )

    print(f"    ...tie='{tie}' ({type(gp.solver).__name__}): finite throughout")


def test_zero_length_transitions_serial():
    """
    Regression test for https://github.com/smolgp-dev/smolgp/issues/3, using
    the serial IntegratedStateSpaceSolver, for all three possible tie types.
    """
    for tie in ["start-start", "end-end", "end-start"]:
        _run_tie_scenario(tie)


def test_zero_length_transitions_parallel():
    """Same as test_zero_length_transitions_serial, but for the parallel solver."""
    for tie in ["start-start", "end-end", "end-start"]:
        _run_tie_scenario(tie, solver=smolgp.solvers.ParallelIntegratedStateSpaceSolver)


def test_smoothing_gain_singular_input():
    """
    get_smoothing_gain should detect a singular P_pred_next (e.g. from a
    reset-zeroed row/column) via its logdet-based check and automatically
    fall back to the lstsq minimum-norm solution, matching a manual lstsq
    call on the same system exactly.
    """
    from smolgp.helpers import get_smoothing_gain

    key1 = jax.random.PRNGKey(1)
    n = 4
    reset_idx = 1

    A = jax.random.normal(key1, (n, n))
    P = A @ A.T + n * jnp.eye(n)  # random SPD matrix
    diag = jnp.ones(n).at[reset_idx].set(0.0)
    Reset = jnp.diag(diag)

    P_pred_next = Reset @ P @ Reset.T  # exactly singular at reset_idx
    numerator = P @ Reset.T  # matches the AR/A_k.T structure at Delta=0

    G = get_smoothing_gain(P_pred_next, numerator)
    G_lstsq, *_ = jnp.linalg.lstsq(P_pred_next.T, numerator.T)
    G_lstsq = G_lstsq.T

    assert jnp.all(jnp.isfinite(G)), (
        "get_smoothing_gain produced NaN/Inf on a singular input"
    )
    assert jnp.allclose(G, G_lstsq, atol=1e-10), (
        f"get_smoothing_gain did not fall back to the lstsq solution:\n{G}\nvs\n{G_lstsq}"
    )
    print(
        "    ...get_smoothing_gain: correctly falls back to lstsq on a singular P_pred_next"
    )


def _generic_tie_dataset(Ninst, tie_type, key):
    """
    Build a tied-exposure geometry for any of the four zero-length-transition
    types, for Ninst in {1, 2, 3} instruments, and generate real synthetic
    data by sampling a true SHO process and exposure-averaging it. Mirrors
    tests/nan_diagnostic.ipynb's build_comparison_dataset.

    tie_type: one of "starts", "ends", "endstart", "startend" -- see
    build_comparison_dataset in the notebook for the distinction between them.
    """
    if tie_type not in ("starts", "ends", "endstart", "startend"):
        raise ValueError(f"Unknown tie_type: {tie_type!r}")

    tA = jnp.array([0.0, 10.0, 20.0])
    texpA = jnp.full(3, 2.0)
    if tie_type == "starts":
        tB, texpB = jnp.array([-15.0, 11.0, 25.0]), jnp.array([2.0, 4.0, 2.0])
    elif tie_type == "ends":
        tB, texpB = jnp.array([-15.0, 9.0, 25.0]), jnp.array([2.0, 4.0, 2.0])
    elif tie_type == "endstart":
        tB, texpB = jnp.array([-15.0, 13.0, 25.0]), jnp.array([2.0, 4.0, 2.0])
    else:  # "startend"
        tB, texpB = jnp.array([-15.0, 7.0, 25.0]), jnp.array([2.0, 4.0, 2.0])

    t = jnp.concatenate([tA, tB])
    texp = jnp.concatenate([texpA, texpB])
    exp_inst = ["A"] * 3 + ["B"] * 3

    if Ninst == 3:
        tC = jnp.array([9.5, 17.0, 30.0])
        texpC = jnp.array([0.5, 2.0, 2.0])
        t = jnp.concatenate([t, tC])
        texp = jnp.concatenate([texp, texpC])
        exp_inst = exp_inst + ["C"] * 3

    if Ninst == 1:
        instid = jnp.zeros(len(t), dtype=int)
    else:
        label_to_id = {lbl: i for i, lbl in enumerate(sorted(set(exp_inst)))}
        instid = jnp.array([label_to_id[lbl] for lbl in exp_inst], dtype=int)

    S, w, Q = 2.5, 0.2, 2.0
    sigma = jnp.sqrt(S * w * Q)
    true_kernel = tinygp.kernels.quasisep.SHO(omega=w, quality=Q, sigma=sigma)

    tmin, tmax = float(jnp.min(t - texp / 2)), float(jnp.max(t + texp / 2))
    buffer = 0.1 * (tmax - tmin)
    t_grid = jnp.arange(tmin - buffer, tmax + buffer, 0.02)
    true_gp = tinygp.GaussianProcess(true_kernel, t_grid)
    y_grid = true_gp.sample(key)

    def make_exposure(tmid, delta):
        t_in_exp = jnp.linspace(tmid - delta / 2, tmid + delta / 2, 50)
        return jnp.mean(jnp.interp(t_in_exp, t_grid, y_grid))

    y_true = jax.vmap(make_exposure)(t, texp)
    yerr = 0.05
    y = y_true + yerr * jax.random.normal(jax.random.split(key)[1], shape=y_true.shape)

    return {
        "t": t,
        "texp": texp,
        "instid": instid,
        "num_insts": Ninst,
        "y": y,
        "yerr": yerr,
        "S": S,
        "w": w,
        "Q": Q,
        "sigma": sigma,
    }


def _assert_smol_matches_tiny(Ninst, tie_type, key, tol=1e-8):
    """
    Compare smolgp (state-space) against tinygp (dense quasiseparable) on a
    dataset containing a deliberate zero-length transition of the given
    tie_type, both at the data points and at a dense prediction grid. This is
    the definitive regression test for
    https://github.com/smolgp-dev/smolgp/issues/3: tinygp's dense solver
    never builds an augmented/reset state and so cannot hit this class of
    singularity, making it the ground truth smolgp must match exactly.
    """
    d = _generic_tie_dataset(Ninst, tie_type, key)
    t, texp, instid, y, yerr = d["t"], d["texp"], d["instid"], d["y"], d["yerr"]
    S, w, Q, sigma = d["S"], d["w"], d["Q"], d["sigma"]
    X_train = (t, texp, instid)

    kernel_smol = smolgp.kernels.IntegratedSHO(
        omega=w, quality=Q, sigma=sigma, num_insts=Ninst
    )
    kernel_tiny = smolgp.kernels.dense.IntegratedSHOKernel(S=S, w=w, Q=Q)

    gp_smol = smolgp.GaussianProcess(
        kernel=kernel_smol, X=X_train, noise=jnp.full(t.shape, yerr**2)
    )
    gp_tiny = tinygp.GaussianProcess(
        kernel=kernel_tiny, X=X_train, diag=jnp.full(t.shape, yerr**2)
    )

    # tinygp adds a machine-epsilon jitter to variances that smolgp doesn't
    offset = jnp.sqrt(jnp.finfo(jnp.array([0.0])).eps)

    _, condgp_smol = gp_smol.condition(y)
    _, condgp_tiny = gp_tiny.condition(y, gp_tiny.X)
    mu_smol_data, var_smol_data = condgp_smol.loc, condgp_smol.variance
    mu_tiny_data, var_tiny_data = condgp_tiny.loc, condgp_tiny.variance - offset

    tmin, tmax = float(jnp.min(t - texp / 2)), float(jnp.max(t + texp / 2))
    pad = 0.1 * (tmax - tmin)
    t_test = jnp.linspace(tmin - pad, tmax + pad, 200)
    zeros = jnp.zeros_like(t_test)
    X_test = (t_test, zeros, zeros.astype(int))

    mu_smol_pred, var_smol_pred = gp_smol.predict(X_test, y, return_var=True)
    mu_tiny_pred, var_tiny_pred = gp_tiny.predict(y, X_test, return_var=True)
    var_tiny_pred = var_tiny_pred - offset

    label = f"Ninst={Ninst}, tie_type={tie_type!r}"
    for name, a, b in [
        ("data mean", mu_smol_data, mu_tiny_data),
        ("data var", var_smol_data, var_tiny_data),
        ("pred mean", mu_smol_pred, mu_tiny_pred),
        ("pred var", var_smol_pred, var_tiny_pred),
    ]:
        assert jnp.all(jnp.isfinite(a)), f"[{label}] smol {name} contains NaN/Inf"
        assert jnp.allclose(a, b, atol=tol), (
            f"[{label}] smol vs tiny {name} mismatch: "
            f"max diff = {float(jnp.max(jnp.abs(a - b))):.3e}"
        )
    print(f"    ...{label}: smolgp matches tinygp to within {tol:.0e}")


def test_smol_matches_tiny_all_tie_types():
    """
    Definitive regression test for https://github.com/smolgp-dev/smolgp/issues/3:
    for every physically realizable combination of Ninst in {1, 2, 3} and
    tie_type in {starts, ends, endstart, startend}, smolgp's state-space
    solver must remain finite and match tinygp's dense solver to near
    machine precision, even though the geometry deliberately contains a
    zero-length transition (including the Ninst>=2 simultaneous-reset case
    that a Delta==0 check alone cannot catch).

    tie_type="starts"/"ends" require two *different* exposures to start (or
    end) at the exact same instant -- only realizable with Ninst>=2, since a
    single instrument cannot be mid-integration of two overlapping exposures
    at once. For Ninst=1 only "endstart"/"startend" (back-to-back,
    non-overlapping exposures) are tested.
    """
    key = jax.random.PRNGKey(42)
    for Ninst in (1, 2, 3):
        tie_types = (
            ("endstart", "startend")
            if Ninst == 1
            else ("starts", "ends", "endstart", "startend")
        )
        for tie_type in tie_types:
            key, subkey = jax.random.split(key)
            _assert_smol_matches_tiny(Ninst, tie_type, subkey)


if __name__ == "__main__":
    test_integrated()
    test_integrated_evaluate()
    test_num_insts_mismatch_reinit()
    test_num_insts_wrapped_kernel()
    test_instid_validation()
    test_num_insts_preserved_on_subset_predict()
    test_zero_length_transitions_serial()
    test_zero_length_transitions_parallel()
    test_smoothing_gain_singular_input()
    test_smol_matches_tiny_all_tie_types()
    print("All integrated kernel tests passed.")
