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
    gp_smol = smolgp.GaussianProcess(kernel=kernel_smol, X=X_train, diag=yerr_train**2)
    gp_tiny = tinygp.GaussianProcess(kernel=kernel_tiny, X=X_train, diag=yerr_train**2)

    # Check likelihood
    likelihood(gp_smol, gp_tiny, y_train, tol=1e-10, atol=1e-13)

    # Check conditioning
    condition(gp_smol, gp_tiny, y_train, tol=1e-9, atol=1e-12)

    # Check predictions
    predict(gp_smol, gp_tiny, y_train, tol=1e-9, atol=1e-12)

    # TODO: test predict with exposure times


if __name__ == "__main__":
    test_integrated()
    print("All integrated kernel tests passed.")
