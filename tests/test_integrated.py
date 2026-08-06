import warnings

import jax
import jax.numpy as jnp
import tinygp
import smolgp

from tests.utils import generate_integrated_data
from tests.test_kernels import (
    kernel_function,
    likelihood,
    condition,
    predict,
)

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
            smolgp.GaussianProcess(kernel=smolgp.kernels.IntegratedExp(scale=1.0), X=badX)
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
    mu = condgp.predict(X_test)

    assert condgp.kernel.num_insts == 3
    assert condgp.solver.kernel.num_insts == 3
    assert mu.shape == (3,)
    print("    ...num_insts preserved (3) after predicting on an instrument subset")


if __name__ == "__main__":
    test_integrated()
    test_num_insts_mismatch_reinit()
    test_num_insts_wrapped_kernel()
    test_instid_validation()
    test_num_insts_preserved_on_subset_predict()
    print("All integrated kernel tests passed.")
