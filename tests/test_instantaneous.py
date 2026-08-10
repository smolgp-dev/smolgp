import jax
import jax.numpy as jnp
import tinygp
import smolgp

from tests.utils import generate_data
from tests.test_kernels import (
    kernel_function,
    likelihood,
    condition,
    predict,
)

key = jax.random.PRNGKey(0)
jax.config.update("jax_enable_x64", True)


def test_instantaneous():
    """
    Baseline sanity check: a normal, non-integrated (instantaneous) dataset
    with every timestamp distinct matches tinygp's dense/quasiseparable
    solver exactly. This is the reference behavior the tied-timestamp test
    below is compared against.
    """
    w = 0.0195
    Q = 7.63
    sigma = 0.59

    kernel_smol = smolgp.kernels.SHO(omega=w, quality=Q, sigma=sigma)
    kernel_tiny = tinygp.kernels.quasisep.SHO(omega=w, quality=Q, sigma=sigma)

    N = 50
    yerr = 0.3
    t_train, y_train = generate_data(N, kernel_tiny, yerr, tmin=0, tmax=1000)
    yerr_train = jnp.full_like(t_train, yerr)

    # Test k(Delta) agrees
    kernel_function(kernel_smol, kernel_tiny, tol=1e-9, atol=1e-12)

    # Build GP objects
    gp_smol = smolgp.GaussianProcess(kernel=kernel_smol, X=t_train, noise=yerr_train**2)
    gp_tiny = tinygp.GaussianProcess(kernel=kernel_tiny, X=t_train, diag=yerr_train**2)

    # Check likelihood
    likelihood(gp_smol, gp_tiny, y_train, tol=1e-10, atol=1e-13)

    # Check conditioning
    condition(gp_smol, gp_tiny, y_train, tol=1e-9, atol=1e-12)

    # Check predictions
    predict(gp_smol, gp_tiny, y_train, tol=1e-9, atol=1e-12)


def _tied_dataset(tie, w=0.0195, Q=7.63, sigma=0.59, N=50, yerr=0.3):
    """
    Build a non-integrated (t, y) dataset with a deliberately tied pair of
    timestamps (a Delta=0 transition between two adjacent states), otherwise
    a normal sampled SHO process. Since non-integrated data has no reset
    matrix -- there's no per-instrument integral-accumulator state that ever
    gets forced to exact zero -- this Delta=0 transition is *not* expected to
    make the predicted covariance singular, unlike the integrated case in
    test_integrated.py. tie: "start", "middle", or "end" controls where in
    the (already sorted) array the tied pair falls.
    """
    kernel_tiny = tinygp.kernels.quasisep.SHO(omega=w, quality=Q, sigma=sigma)
    t_train, y_train = generate_data(N, kernel_tiny, yerr=yerr, tmin=0, tmax=1000)

    if tie == "start":
        idx = 0
    elif tie == "end":
        idx = N  # insert after the last element, duplicating the last timestamp
    else:
        idx = N // 2

    dup_val = t_train[idx] if idx < N else t_train[-1]
    t_tied = jnp.insert(t_train, idx, dup_val)
    # A distinct second measurement at the exact same timestamp (not a
    # literal duplicate observation) so the two points genuinely differ.
    y_tied = jnp.insert(y_train, idx, (y_train[idx] if idx < N else y_train[-1]) + 0.1)

    return t_tied, y_tied


def test_instantaneous_tied_timestamps():
    """
    Two data points sharing the exact same timestamp (a Delta=0 transition)
    must still condition/predict identically to tinygp, since the plain
    (non-integrated) state-space solver has no reset matrix to force a
    singular predicted covariance -- see src/smolgp/solvers/rts.py, which
    computes the smoothing gain via a raw jnp.linalg.solve with no
    singularity guard at all. This is a regression/sanity check that ties
    are safe here independent of the get_smoothing_gain fix in
    test_integrated.py.
    """
    w, Q, sigma, yerr = 0.0195, 7.63, 0.59, 0.3
    kernel_tiny = tinygp.kernels.quasisep.SHO(omega=w, quality=Q, sigma=sigma)

    for tie in ("start", "middle", "end"):
        t_tied, y_tied = _tied_dataset(tie, w=w, Q=Q, sigma=sigma, yerr=yerr)
        yerr_train = jnp.full_like(t_tied, yerr)

        kernel_smol = smolgp.kernels.SHO(omega=w, quality=Q, sigma=sigma)

        gp_smol = smolgp.GaussianProcess(kernel=kernel_smol, X=t_tied, noise=yerr_train**2)
        gp_tiny = tinygp.GaussianProcess(kernel=kernel_tiny, X=t_tied, diag=yerr_train**2)

        condition(gp_smol, gp_tiny, y_tied, tol=1e-9, atol=1e-12)
        predict(gp_smol, gp_tiny, y_tied, tol=1e-9, atol=1e-12)
        print(f"    ...tie='{tie}': finite and matches tinygp exactly")


if __name__ == "__main__":
    test_instantaneous()
    test_instantaneous_tied_timestamps()
    print("All instantaneous kernel tests passed.")
