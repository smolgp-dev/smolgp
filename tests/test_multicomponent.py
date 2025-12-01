import jax
import jax.numpy as jnp
import tinygp
import smolgp
from smolgp.kernels.base import extract_leaf_kernels

from utils import generate_data
import testgp
from test_kernels import allclose, offset

key = jax.random.PRNGKey(0)
jax.config.update("jax_enable_x64", True)


def test_multicomponent():
    # Kernel parameters
    ## SHO Kernel
    S = 2.36
    w = 0.0195
    Q = 7.63
    sigma = jnp.sqrt(S * w * Q)
    ## Matern-5/2 Kernel
    scale = 600.0
    sigma = 5.3

    # State space version
    ssm1 = smolgp.kernels.SHO(omega=w, quality=Q, sigma=sigma)
    ssm2 = smolgp.kernels.Matern52(scale=scale, sigma=sigma)
    ssm_sum = ssm1 + ssm2
    ssm_prod = ssm1 * ssm2

    # Quasisep version
    qsm1 = tinygp.kernels.quasisep.SHO(omega=w, quality=Q, sigma=sigma)
    qsm2 = tinygp.kernels.quasisep.Matern52(scale=scale, sigma=sigma)
    qsm_sum = qsm1 + qsm2
    qsm_prod = qsm1 * qsm2

    kernels = {"Sum": (ssm_sum, qsm_sum), "Product": (ssm_prod, qsm_prod)}

    from test_kernels import (
        likelihood,
        condition,
        predict,
    )

    for name in kernels:
        ksmol, ktiny = kernels[name]

        ## Generate mock data
        N = 50
        yerr = 0.3
        t_train, y_train = generate_data(N, ktiny, yerr, tmin=0, tmax=1000)
        yerr_train = jnp.full_like(t_train, yerr)

        print(f"Testing {name}...")
        # Build GP objects
        gp_smol = smolgp.GaussianProcess(kernel=ksmol, X=t_train, diag=yerr_train**2)
        gp_tiny = tinygp.GaussianProcess(kernel=ktiny, X=t_train, diag=yerr_train**2)

        # Check likelihood
        likelihood(gp_smol, gp_tiny, y_train, tol=1e-10, atol=1e-11)

        # Check conditioning
        condition(gp_smol, gp_tiny, y_train, tol=1e-9, atol=1e-11)

        # Check predictions
        predict(gp_smol, gp_tiny, y_train, tol=1e-9, atol=1e-11)
        print()

        ## Model decomposition tests for sums of GPs
        if name == "Sum":
            ## Verify component means and predict_component_means agree
            tmin, tmax = t_train.min(), t_train.max()
            dt = 0.1 * (tmax - tmin)
            t_test = jnp.linspace(tmin - dt, tmax + dt, 1000)

            print("  Testing component means...")
            # smolgp version of component means prediction
            llh, condGP = gp_smol.condition(y_train)
            ys_ssm, yvars_ssm = condGP.component_means(return_var=True)
            mus_ssm, vars_ssm = condGP.predict_component_means(t_test, return_var=True)

            # tinygp version of component means prediction
            mus_qsm = []
            vars_qsm = []
            for k in testgp.extract_leaf_kernels(gp_tiny.kernel):
                yk, var_k = gp_tiny.predict(y_train, t_test, return_var=True, kernel=k)
                mus_qsm.append(yk)
                vars_qsm.append(var_k)

            ys_qsm = []
            yvars_qsm = []
            for k in testgp.extract_leaf_kernels(gp_tiny.kernel):
                yk, var_k = gp_tiny.predict(y_train, t_train, return_var=True, kernel=k)
                ys_qsm.append(yk)
                yvars_qsm.append(var_k)

            # Compare
            for i in range(len(mus_ssm)):
                y_res = ys_qsm[i] - ys_ssm[i]
                yvar_res = (yvars_qsm[i] - offset) - yvars_ssm[i]
                allclose(f"component means {i}", y_res, tol=1e-9, atol=1e-12)
                allclose(f"component vars {i}", yvar_res, tol=1e-9, atol=1e-12)

                mu_res = mus_qsm[i] - mus_ssm[i]
                var_res = (vars_qsm[i] - offset) - vars_ssm[i]
                allclose(f"predicted component means {i}", mu_res, tol=1e-9, atol=1e-12)
                allclose(f"predicted component vars {i}", var_res, tol=1e-9, atol=1e-12)

            print()

    # Confirm we don't decomposed beyond Sums
    print("Testing kernel decomposition is only for sums...")
    testkernel = ssm1 + (ssm2 * ssm1)
    components = extract_leaf_kernels(testkernel)
    assert components == [ssm1, (ssm2 * ssm1)], (
        "Kernel was over-decomposed! Recall we cannot decompose products into individual GPs, only sums."
    )
    # print("    Checking names are unique & correct...")
    # # kernel = assign_unique_kernel_names(kernel) # TODO: add this check if it isn't covered naturally by other tests


if __name__ == "__main__":
    test_multicomponent()
    print("All multicomponent kernel tests passed.")
