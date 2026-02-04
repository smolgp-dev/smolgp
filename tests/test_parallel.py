import jax
import jax.numpy as jnp
import tinygp
import smolgp

from tests.utils import generate_data, generate_integrated_data
from tests.test_kernels import (
    likelihood,
    condition,
    predict,
)

key = jax.random.PRNGKey(0)
jax.config.update("jax_enable_x64", True)


def test_parallel():
    ## SHO Kernel
    S = 2.36
    w = 0.0195
    Q = 7.63
    sigma = jnp.sqrt(S * w * Q)

    ## Base kernels
    kernel_smol = smolgp.kernels.SHO(omega=w, quality=Q, sigma=sigma)
    kernel_tiny = tinygp.kernels.quasisep.SHO(omega=w, quality=Q, sigma=sigma)

    print("Testing ParallelStateSpaceSolver...")
    ## Generate mock data
    N = 50
    yerr = 0.3
    t_train, y_train = generate_data(N, kernel_tiny, yerr=yerr)
    yerr_train = jnp.full_like(t_train, yerr)

    # Build GP objects
    gp_smol = smolgp.GaussianProcess(
        kernel=kernel_smol,
        X=t_train,
        diag=yerr_train**2,
        solver=smolgp.solvers.ParallelStateSpaceSolver,
    )
    gp_tiny = tinygp.GaussianProcess(kernel=kernel_tiny, X=t_train, diag=yerr_train**2)

    # Check likelihood, condition, predict
    likelihood(gp_smol, gp_tiny, y_train, tol=1e-10, atol=1e-13)
    condition(gp_smol, gp_tiny, y_train, tol=1e-9, atol=1e-12)
    predict(gp_smol, gp_tiny, y_train, tol=1e-9, atol=1e-12)

    print()
    print("Testing ParallelIntegratedStateSpaceSolver...")

    ## Mock integrated data
    texp, readout = 140.0, 40.0
    t_train, y_train = generate_integrated_data(
        N, kernel_tiny, texp=texp, readout=readout, yerr=yerr
    )
    texp_train = jnp.full_like(t_train, texp)
    yerr_train = jnp.full_like(t_train, yerr)
    instid = jnp.full_like(t_train, 0).astype(int)  # has to be integer
    X_train = (t_train, texp_train, instid)

    # Integrated kernels
    ikernel_smol = smolgp.kernels.integrated.IntegratedSHO(
        omega=w, quality=Q, sigma=sigma, num_insts=1
    )
    ikernel_tiny = smolgp.kernels.dense.IntegratedSHOKernel(S=S, w=w, Q=Q)

    # Build GP objects
    gp_smol = smolgp.GaussianProcess(
        kernel=ikernel_smol,
        X=X_train,
        diag=yerr_train**2,
        solver=smolgp.solvers.ParallelIntegratedStateSpaceSolver,
    )
    gp_tiny = tinygp.GaussianProcess(kernel=ikernel_tiny, X=X_train, diag=yerr_train**2)

    # Check likelihood, condition, predict
    likelihood(gp_smol, gp_tiny, y_train, tol=1e-10, atol=1e-13)
    condition(gp_smol, gp_tiny, y_train, tol=1e-9, atol=1e-12)
    predict(gp_smol, gp_tiny, y_train, tol=1e-9, atol=1e-12)


if __name__ == "__main__":
    test_parallel()
    print("All parallel solver tests passed.")
