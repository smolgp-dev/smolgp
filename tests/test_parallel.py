import jax
import jax.numpy as jnp
import tinygp

import smolgp
from tests.test_kernels import (
    condition,
    likelihood,
    predict,
)
from tests.utils import allclose, generate_data, generate_integrated_data

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
        noise=yerr_train**2,
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
        N, kernel_smol, texp=texp, readout=readout, yerr=yerr
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
        noise=yerr_train**2,
        solver=smolgp.solvers.ParallelIntegratedStateSpaceSolver,
    )
    gp_tiny = tinygp.GaussianProcess(kernel=ikernel_tiny, X=X_train, diag=yerr_train**2)

    # Check likelihood, condition, predict
    likelihood(gp_smol, gp_tiny, y_train, tol=1e-10, atol=1e-13)
    condition(gp_smol, gp_tiny, y_train, tol=1e-9, atol=1e-12)
    predict(gp_smol, gp_tiny, y_train, tol=1e-9, atol=1e-12)


def test_parallel_1d_state():
    """Kernels with 1 state dimension (e.g., Exp) must match the sequential solver"""
    t = jnp.sort(jax.random.uniform(key, (50,), maxval=20.0))
    y = jnp.sin(t)
    kernels = {
        "Exp": smolgp.kernels.Exp(scale=1.0, sigma=1.2),
        "2*Exp": 2.0 * smolgp.kernels.Exp(scale=1.0),
        "Constant": smolgp.kernels.Constant(sigma=1.5),
    }
    for name, kernel in kernels.items():
        gp_seq = smolgp.GaussianProcess(kernel, t, noise=0.1)
        gp_par = smolgp.GaussianProcess(
            kernel,
            t,
            noise=0.1,
            solver=smolgp.solvers.ParallelStateSpaceSolver,
        )
        allclose(
            f"{name} likelihood",
            gp_par.log_probability(y) - gp_seq.log_probability(y),
            tol=1e-10,
            atol=1e-13,
        )
        _, cond_seq = gp_seq.condition(y)
        _, cond_par = gp_par.condition(y)
        allclose(f"{name} conditioned mean", cond_par.loc - cond_seq.loc, tol=1e-10)
        allclose(
            f"{name} conditioned variance",
            cond_par.variance - cond_seq.variance,
            tol=1e-10,
        )


if __name__ == "__main__":
    test_parallel()
    test_parallel_1d_state()
    print("All parallel solver tests passed.")
