import jax
import jax.numpy as jnp
import tinygp
import smolgp

import benchmark.kernels as testgp
from benchmark.benchmark import generate_integrated_data
from test_kernels import (
    test_kernel_function,
    test_likelihood,
    test_condition,
    test_predict,
)

key = jax.random.PRNGKey(0)
jax.config.update("jax_enable_x64", True)

if __name__ == "__main__":
    ## SHO Kernel
    S = 2.36
    w = 0.0195
    Q = 7.63
    sigma = jnp.sqrt(S * w * Q)

    ## Instantaneous kernel for generating data
    true_kernel = tinygp.kernels.quasisep.SHO(omega=w, quality=Q, sigma=sigma)

    # Integrated kernels
    kernel_smol = smolgp.kernels.integrated.IntegratedSHO(
        omega=w, quality=Q, sigma=sigma, num_insts=1
    )
    kernel_tiny = testgp.IntegratedSHOKernel(S=S, w=w, Q=Q)

    # Test k(Delta) agrees
    test_kernel_function(kernel_smol, kernel_tiny, tol=1e-9, atol=1e-12)

    ## Generate mock data
    N = 50
    yerr = 0.3
    texp, readout = 140.0, 40.0
    t_train, y_train = generate_integrated_data(
        N, true_kernel, texp=texp, readout=readout, yerr=yerr
    )
    texp_train = jnp.full_like(t_train, texp)
    yerr_train = jnp.full_like(t_train, yerr)
    instid = jnp.full_like(t_train, 0)
    X_train = (t_train, texp_train, instid)

    # Build GP objects
    gp_smol = smolgp.GaussianProcess(kernel=kernel_smol, X=X_train, diag=yerr_train**2)
    gp_tiny = tinygp.GaussianProcess(kernel=kernel_tiny, X=X_train, diag=yerr_train**2)

    # Check likelihood
    test_likelihood(gp_smol, gp_tiny, y_train, tol=1e-10, atol=1e-13)

    # Check conditioning
    test_condition(gp_smol, gp_tiny, y_train, tol=1e-9, atol=1e-12)

    # Check predictions
    test_predict(gp_smol, gp_tiny, y_train, tol=1e-9, atol=1e-12)

    # TODO: test predict with exposure times

    print("All tests passed.")
