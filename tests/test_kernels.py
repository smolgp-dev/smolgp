import jax
import jax.numpy as jnp
import tinygp
import smolgp
from utils import generate_data

jax.config.update("jax_enable_x64", True)

# The tinygp variances include a default noise of machine epsilon
offset = jnp.sqrt(jnp.finfo(jnp.array([0.0])).eps)


def allclose(name, residuals, tol, atol=1e-14):
    """
    Check all residuals are < tol
    if they are, but aren't < atol, print a warning
    """
    maxres = jnp.max(jnp.abs(residuals))
    assert maxres < tol, (
        f"{name} did not agree to within desired tolerance."
        f" Maximum absolute deviation is {maxres:.3e} "
    )
    if maxres < atol:
        print(f"    ...{name}: agrees exactly (<{maxres:.0e})")
    else:
        print(f"    ...{name}: agrees (WARNING: only to < {maxres:.1e})")


def test_kernel_function(ksmol, ktiny, tol=1e-14, atol=1e-14):
    """
    Check the kernel function k(Delta) is the same
    """
    dts = jnp.linspace(0, 1000, 500)
    zeros = jnp.zeros_like(dts)
    cov_tiny = ksmol(zeros, dts)  # [0, :]
    cov_smol = ktiny(zeros, dts)  # [0, :]
    res = cov_tiny - cov_smol
    allclose("kernel function", res, tol=tol, atol=atol)


def test_likelihood(gp_smol, gp_tiny, y_train, tol=1e-12, atol=1e-12):
    """
    Check the likelihoods are the same
    """
    llh_smol = gp_smol.log_probability(y_train)
    llh_tiny = gp_tiny.log_probability(y_train)
    res = llh_smol - llh_tiny
    allclose("likelihood", res, tol=tol, atol=atol)


def test_condition(gp_smol, gp_tiny, y_train, tol=1e-12, atol=1e-12):
    """
    Check the conditioned means/vars are the same
    """
    llh_smol, condGP_smol = gp_smol.condition(y_train)
    llh_tiny, condGP_tiny = gp_tiny.condition(y_train, gp_tiny.X)
    mu_res = condGP_tiny.loc - condGP_smol.loc
    var_res = (condGP_tiny.variance - offset) - condGP_smol.variance
    allclose("conditioned means", mu_res, tol=tol, atol=atol)
    allclose("conditioned variances", var_res, tol=tol, atol=atol)


def test_predict(gp_smol, gp_tiny, y_train, tol=1e-12, atol=1e-12):
    """
    Check the predicted means/vars are the same
    """
    t_train = gp_smol.kernel.coord_to_sortable(gp_smol.X)
    tmin, tmax = t_train.min(), t_train.max()
    dt = 0.1 * (tmax - tmin)  # include a retrodict/extrapolate
    t_test = jnp.linspace(tmin - dt, tmax + dt, 1000)
    if isinstance(gp_smol.X, tuple):
        zeros = jnp.zeros_like(t_test)
        # no exposure time, single instrument
        X_test = (t_test, zeros, zeros.astype(int))
    else:
        X_test = t_test
    mu_tiny, var_tiny = gp_tiny.predict(y_train, X_test, return_var=True)
    mu_smol, var_smol = gp_smol.predict(X_test, y_train, return_var=True)
    mu_res = mu_tiny - mu_smol
    var_res = (var_tiny - offset) - var_smol
    allclose("predicted means", mu_res, tol=tol, atol=atol)
    allclose("predicted variances", var_res, tol=tol, atol=atol)


def test_kernel(kernel_smol, kernel_tiny):
    """
    Check the smolgp kernel produces the same
    results as tinygp for all the above tests
    """
    if kernel_smol.name in ["ExpSineSquared"]:
        covtol, covatol = 1e-3, 1e-6
        llhtol, llhatol = 0.2, 1e-3
        condtol, condatol = 1e-1, 1e-3
        predtol, predatol = 1e-1, 1e-3
    else:
        covtol, covatol = 1e-9, 1e-14
        llhtol, llhatol = 1e-9, 1e-12
        condtol, condatol = 1e-9, 1e-13
        predtol, predatol = 1e-9, 1e-13

    # Test k(Delta) agrees
    test_kernel_function(kernel_smol, kernel_tiny, tol=covtol, atol=covatol)

    ## Generate mock data
    N = 50
    yerr = 0.3
    t_train, y_train = generate_data(N, kernel_tiny, yerr, tmin=0, tmax=1000)
    yerr_train = jnp.full_like(t_train, yerr)

    # Build GP objects
    gp_smol = smolgp.GaussianProcess(kernel=kernel_smol, X=t_train, diag=yerr_train**2)
    gp_tiny = tinygp.GaussianProcess(kernel=kernel_tiny, X=t_train, diag=yerr_train**2)

    # Check likelihood
    test_likelihood(gp_smol, gp_tiny, y_train, tol=llhtol, atol=llhatol)

    # Check conditioning
    test_condition(gp_smol, gp_tiny, y_train, tol=condtol, atol=condatol)

    # Check predictions
    test_predict(gp_smol, gp_tiny, y_train, tol=predtol, atol=predatol)


if __name__ == "__main__":
    kernels = {}

    # Kernel parameters to use for testing
    sigma = 2.1  # amplitude
    scale = 83.3  # length scale for Exp/Matern
    omega = 2 * jnp.pi / 83.3  # freq for SHO/periodic

    ################ Exp ################
    ksmol = smolgp.kernels.Exp(scale=scale, sigma=sigma)
    ktiny = tinygp.kernels.quasisep.Exp(scale=scale, sigma=sigma)
    kernels["Exp"] = [ksmol, ktiny]

    ################ Matern-3/2 ################
    ksmol = smolgp.kernels.Matern32(scale=scale, sigma=sigma)
    ktiny = tinygp.kernels.quasisep.Matern32(scale=scale, sigma=sigma)
    kernels["Matern32"] = [ksmol, ktiny]

    ################ Matern-5/2 ################
    ksmol = smolgp.kernels.Matern52(scale=scale, sigma=sigma)
    ktiny = tinygp.kernels.quasisep.Matern52(scale=scale, sigma=sigma)
    kernels["Matern52"] = [ksmol, ktiny]

    ################ Cosine ################
    ksmol = smolgp.kernels.Cosine(scale=scale, sigma=sigma)
    ktiny = tinygp.kernels.quasisep.Cosine(scale=scale, sigma=sigma)
    kernels["Cosine"] = [ksmol, ktiny]

    ################ SHO ################
    quality = {"underdamped": 5.3, "critical": 0.5, "overdamped": 0.1}
    for shotype in quality:
        q = quality[shotype]
        ksmol = smolgp.kernels.SHO(omega, q, sigma)
        ktiny = tinygp.kernels.quasisep.SHO(omega, q, sigma)
        kernels[f"SHO_{shotype}"] = [ksmol, ktiny]

    ################ ExpSineSquared ################
    ell = 1.164  # 0.7
    gamma = 2 / ell**2
    period = 163.3
    # we have to hardcode sigma=1 to match the definition in tinygp
    ksmol = smolgp.kernels.ExpSineSquared(gamma=gamma, period=period, sigma=1.0)
    ktiny = tinygp.kernels.ExpSineSquared(gamma=gamma, scale=period)
    kernels["ExpSineSquared"] = [ksmol, ktiny]

    ## Test them
    for name in kernels:
        ksmol, ktiny = kernels[name]
        print(f"Testing {name}...")
        test_kernel(ksmol, ktiny)
        print()

    print("All tests passed.")
