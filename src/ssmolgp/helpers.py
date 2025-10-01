import jax.numpy as jnp
from jax.scipy.linalg import expm
from tinygp.helpers import JAXArray

def Q_from_VanLoan(F: JAXArray, L: JAXArray, Qc: JAXArray, dt: JAXArray) -> JAXArray:
    """
    Van Loan method to compute Q = ∫0^dt exp(F (dt-s)) L Qc L^T exp(F^T (dt-s)) ds

    Parameters:
        F: StateSpaceModel.design_matrix
        L: StateSpaceModel.noise_effect_matrix
        Qc: StateSpaceModel.process_noise
        dt: time step between measurements (dt = X2 - X1)

    Returns:
        Q: Process noise covariance matrix over time step dt

    See Van Loan (1978) "Computing Integrals Involving the Matrix Exponential"
    PDF at https://www.olemartin.no/artikler/vanloan.pdf
    https://ecommons.cornell.edu/items/cba38b2e-6ad4-45e6-8109-0a019fe5114c
    """
    QL = L*Qc@L.T
    b = len(F) # block size
    Z = jnp.zeros_like(F)
    C = jnp.block([[-F, QL],
                   [Z, F.T]])
    VanLoanBlock = expm(C*dt)
    G2 = VanLoanBlock[:b,b:]
    F3 = VanLoanBlock[b:,b:]
    return F3.T @ G2