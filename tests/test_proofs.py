import jax
import jax.numpy as jnp
import smolgp
from jax.scipy.linalg import expm
from tests.utils import allclose

key = jax.random.PRNGKey(0)
jax.config.update("jax_enable_x64", True)


def base(kernel, dts):
    """
    Confirm all analytic versions of matrices
    agree with their numerical versions--
    we're assuming `kernel` has analytic A and Q defined
    """

    # For (underdamped) SHO, these are analytic functions
    transition_matrix_analytic = lambda dt: kernel.transition_matrix(0, dt)
    process_noise_analytic = lambda dt: kernel.process_noise(0, dt)

    # Numerical versions
    # transition_matrix_numerical = lambda dt: expm(F * dt)
    transition_matrix_numerical = lambda dt: super(
        type(kernel), kernel
    ).transition_matrix(0, dt)
    process_noise_vanloan = lambda dt: super(type(kernel), kernel).process_noise(
        0, dt, use_van_loan=True
    )
    process_noise_fromPinf = lambda dt: super(type(kernel), kernel).process_noise(
        0, dt, use_van_loan=False
    )

    # Test all dts
    print("Comparing analytic Phi vs. expm(F*dt)...")
    A_analytic = jax.vmap(transition_matrix_analytic)(dts)
    A_numerical = jax.vmap(transition_matrix_numerical)(dts)
    allclose(
        "Transition matrix",
        A_analytic - A_numerical,
        tol=1e-9,
        atol=1e-12,
    )
    print("Comparing analytic Q vs. VanLoan Q...")
    Q_analytic = jax.vmap(process_noise_analytic)(dts)
    Q_vanloan = jax.vmap(process_noise_vanloan)(dts)
    allclose(
        "Process noise (VanLoan)",
        Q_analytic - Q_vanloan,
        tol=1e-9,
        atol=1e-12,
    )
    print("Comparing analytic Q vs. Q from Pinf...")
    Q_fromPinf = jax.vmap(process_noise_fromPinf)(dts)
    allclose(
        "Process noise (from Pinf)",
        Q_analytic - Q_fromPinf,
        tol=1e-9,
        atol=1e-12,
    )
    print("All base matrix tests passed.")


def integrated_transition(kernel, dts):
    """
    Confirm all analytic versions of matrices and their
    augmented forms agree with their numerical versions--
    we're assuming `kernel` has analytic Phibar defined
    """

    ### Integrated transition matrix
    Phibar_analytic = lambda dt: kernel.integrated_transition_matrix(0, dt)
    Phibar_from_VanLoan = lambda dt: super(
        type(kernel), kernel
    ).integrated_transition_matrix(0, dt)
    print("Comparing analytic Phibar vs. Van Loan...")
    A_analytic = jax.vmap(Phibar_analytic)(dts)
    A_numerical = jax.vmap(Phibar_from_VanLoan)(dts)
    allclose(
        "Integrated transition matrix",
        A_analytic - A_numerical,
        tol=1e-8,
        atol=1e-12,
    )

    ### Augmented transition matrix
    F_aug = kernel.design_matrix()
    Phiaug_numerical = lambda dt: expm(F_aug * dt)
    Phiaug_analytic = lambda dt: kernel.transition_matrix(0, dt)
    print("Comparing analytic Phiaug vs. expm(Faug*dt)...")
    Aaug_analytic = jax.vmap(Phiaug_analytic)(dts)
    Aaug_numerical = jax.vmap(Phiaug_numerical)(dts)
    allclose(
        "Augmented transition matrix",
        Aaug_analytic - Aaug_numerical,
        tol=1e-5,
        atol=1e-12,
    )


def integrated_process_noise(kernel, dts, tol=1e-8):
    # Augmented process noise
    Qaug_implemented = lambda dt: kernel.process_noise(0, dt)
    Qaug_from_VanLoan = lambda dt: kernel.process_noise(0, dt, force_numerical=True)
    print("Comparing implemented Qaug vs. VanLoan...")
    Qaug_analytic = jax.vmap(Qaug_implemented)(dts)
    Qaug_vanloan = jax.vmap(Qaug_from_VanLoan)(dts)
    allclose(
        "Augmented process noise",
        Qaug_analytic - Qaug_vanloan,
        tol=tol,
        atol=1e-12,
    )
    print("All integrated/augmented matrix tests passed.")


def test_proofs():
    ## TODO: do we also want to test shapes here?

    ## This test is done with the SHO Kernel, since
    ## we have analytic versions to compare against
    S = 2.36
    w = 0.0195
    Q = 7.63
    sigma = jnp.sqrt(S * w * Q)
    sho = smolgp.kernels.SHO(omega=w, quality=Q, sigma=sigma)
    isho = smolgp.kernels.IntegratedSHO(omega=w, quality=Q, sigma=sigma)

    ## Test a wide dynamic range of Deltas
    ## NOTE: expm breaks down around ~1e6
    dts = jnp.logspace(-6, 5, 1000)

    print("Testing base (instantaneous) kernel matrices...")
    base(sho, dts)
    print()
    print("Testing integrated kernel matrices...")
    integrated_transition(isho, dts)

    ## Separately test small Delta and large Delta for process noise
    dtsmall = dts[dts < 1e3]
    dtlarge = dts[dts >= 1e3]
    integrated_process_noise(isho, dtsmall, tol=1e-6)
    integrated_process_noise(isho, dtlarge, tol=1e-3)

    ## and verify dt=0 is correct
    assert jnp.all(isho.process_noise(0.0, 0.0) == 0), (
        "Process noise at dt=0 should be zero."
    )
    assert jnp.all(isho.transition_matrix(0.0, 0.0) == jnp.eye(isho.dimension)), (
        "Transition matrix at dt=0 should be the identity."
    )


if __name__ == "__main__":
    test_proofs()
    print("All proof tests passed.")
