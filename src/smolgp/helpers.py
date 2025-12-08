import jax.numpy as jnp
from jax.scipy.linalg import expm
from tinygp.helpers import JAXArray


def block_view(A, b):
    Nb, Mb = A.shape
    assert Nb % b == 0 and Mb % b == 0
    N = Nb // b
    M = Mb // b
    return A.reshape(N, b, M, b).transpose(0, 2, 1, 3)


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
    QL = L @ Qc @ L.T
    b = len(F)  # block size
    Z = jnp.zeros_like(F)
    C = jnp.block([[-F, QL], [Z, F.T]])
    VanLoanBlock = expm(C * dt)
    G2 = VanLoanBlock[:b, b:]
    F3 = VanLoanBlock[b:, b:]
    return F3.T @ G2


def Phibar_from_VanLoan(F: JAXArray, dt: JAXArray) -> JAXArray:
    """
    Van Loan method to compute Phibar = ∫0^dt exp(F s) ds

    Parameters:
        F: StateSpaceModel.design_matrix
        dt: time step between measurements (dt = X2 - X1)

    Returns:
        Phibar: The integrated transition matrix over time step dt

    See Van Loan (1978) "Computing Integrals Involving the Matrix Exponential"
    PDF at https://www.olemartin.no/artikler/vanloan.pdf
    https://ecommons.cornell.edu/items/cba38b2e-6ad4-45e6-8109-0a019fe5114c
    """
    b = len(F)  # block size
    Z = jnp.zeros((b, b))
    I = jnp.eye(b)
    C = jnp.block([[F, I], [Z, Z]])
    VanLoanBlock = expm(C * dt)
    G3 = VanLoanBlock[:b, b:]
    return G3


def VanLoan(
    F: JAXArray, L: JAXArray, Qc: JAXArray, dt: JAXArray
) -> tuple[JAXArray, JAXArray]:
    """
    Constructs the full matrix C and returns its matrix exponential

    Parameters:
        F: StateSpaceModel.design_matrix
        L: StateSpaceModel.noise_effect_matrix
        Qc: StateSpaceModel.process_noise
        dt: time step between measurements (dt = X2 - X1)

    Returns:
        dict : Dictionary of the submatrices of the Van Loan exponential, from which
                                various integrals (e.g. Q, Phibar) can be computed.

    See Van Loan (1978) "Computing Integrals Involving the Matrix Exponential"
    PDF at https://www.olemartin.no/artikler/vanloan.pdf
    https://ecommons.cornell.edu/items/cba38b2e-6ad4-45e6-8109-0a019fe5114c
    """
    QL = L @ Qc @ L.T
    b = len(F)  # block size
    I = jnp.eye(b)
    Z = jnp.zeros_like(F)
    C = jnp.block(
        [
            [-F, I, Z, Z],
            [Z, -F, QL, Z],
            [Z, Z, F.T, I],
            [Z, Z, Z, Z],
        ]
    )
    VanLoanBlock = block_view(expm(C * dt), b)

    F1 = VanLoanBlock[0, 0]
    G1 = VanLoanBlock[0, 1]
    H1 = VanLoanBlock[0, 2]
    K1 = VanLoanBlock[0, 3]
    F2 = VanLoanBlock[1, 1]
    G2 = VanLoanBlock[1, 2]
    H2 = VanLoanBlock[1, 3]
    F3 = VanLoanBlock[2, 2]
    G3 = VanLoanBlock[2, 3]
    F4 = VanLoanBlock[3, 3]

    return {
        "F1": F1,
        "F2": F2,
        "F3": F3,
        "F4": F4,
        "G1": G1,
        "G2": G2,
        "G3": G3,
        "H1": H1,
        "H2": H2,
        "K1": K1,
    }
