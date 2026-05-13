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
    r"""Compute the process noise covariance via the Van Loan method.

    Evaluates

    .. math::

        Q_k = \int_0^{\Delta t} e^{F(\Delta t - s)}\, L\, Q_c\, L^T\, e^{F^T(\Delta t - s)}\, ds

    See `Van Loan (1978) <https://ecommons.cornell.edu/items/cba38b2e-6ad4-45e6-8109-0a019fe5114c>`_,
    "Computing Integrals Involving the Matrix Exponential" (`PDF <https://www.olemartin.no/artikler/vanloan.pdf>`_).

    Args:
        F: Feedback (design) matrix :math:`F` from :meth:`~smolgp.kernels.StateSpaceModel.design_matrix`.
        L: Noise effect matrix :math:`L` from :meth:`~smolgp.kernels.StateSpaceModel.noise_effect_matrix`.
        Qc: Spectral density :math:`Q_c` from :meth:`~smolgp.kernels.StateSpaceModel.noise`.
        dt: Time step :math:`\Delta t = X_2 - X_1`.

    Returns:
        Process noise covariance matrix :math:`Q_k` over time step :math:`\Delta t`.
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
    r"""Compute the integrated transition matrix via the Van Loan method.

    Evaluates

    .. math::

        \bar{\Phi} = \int_0^{\Delta t} e^{F s}\, ds

    See `Van Loan (1978) <https://ecommons.cornell.edu/items/cba38b2e-6ad4-45e6-8109-0a019fe5114c>`_,
    "Computing Integrals Involving the Matrix Exponential" (`PDF <https://www.olemartin.no/artikler/vanloan.pdf>`_).

    Args:
        F: Feedback (design) matrix :math:`F` from :meth:`~smolgp.kernels.StateSpaceModel.design_matrix`.
        dt: Time step :math:`\Delta t = X_2 - X_1`.

    Returns:
        Integrated transition matrix :math:`\bar{\Phi}` over time step :math:`\Delta t`.
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
) -> dict[str, JAXArray]:
    r"""Compute all submatrices of the Van Loan matrix exponential.

    Assembles the block matrix :math:`C` and returns its matrix exponential,
    partitioned into the submatrices ``F1``-``F4``, ``G1``-``G3``, ``H1``-``H2``,
    ``K1`` (see Van Loan 1978 for notation), from which various integrals such as
    :func:`Q_from_VanLoan` and :func:`Phibar_from_VanLoan` can be derived.

    See `Van Loan (1978) <https://ecommons.cornell.edu/items/cba38b2e-6ad4-45e6-8109-0a019fe5114c>`_,
    "Computing Integrals Involving the Matrix Exponential" (`PDF <https://www.olemartin.no/artikler/vanloan.pdf>`_).

    Args:
        F: Feedback (design) matrix :math:`F`.
        L: Noise effect matrix :math:`L`.
        Qc: Spectral density :math:`Q_c`.
        dt: Time step :math:`\Delta t = X_2 - X_1`.

    Returns:
        Dictionary of named submatrices of the Van Loan exponential.
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
