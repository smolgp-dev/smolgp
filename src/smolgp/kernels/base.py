"""
The kernels implemented in this subpackage are defined similarly to
:class: `tinygp.kernels.quasisep.Quasisep` but are modified to:
1. Include the `noise_effect_matrix` and `process_noise` matrix
2. Treat the observation model as a column vector and the transition matrix
   in its usual form (compared to transposed forms in tinygp.kernels.quasisep)
3. Handle integrated versions of each kernel (in integrated.py)
These kernels are compatible with :class:`smolgp.solvers.StateSpaceSolver`
which use Bayesian filtering and smoothing algorithms to perform scalable GP
inference. (see :ref:`api-solvers-statespace` for more technical details).

Like the quasisep kernels, these methods are experimental, so you may find
the documentation patchy in places. You are encouraged to `open issues or
pull requests <https://github.com/rrubenza/smolgp/issues>`_ as you find gaps.
"""

from __future__ import annotations

import warnings
from abc import abstractmethod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
from jax.scipy.special import gammaln

from tinygp.helpers import JAXArray
from tinygp.kernels.base import Kernel
from tinygp.solvers.quasisep.block import Block

from smolgp.helpers import Q_from_VanLoan


def extract_leaf_kernels(kernel, all=False):
    """
    Recursively extract leaf kernels from a sum or product of kernels

    If all==True, extract from both Sum and Product nodes
        (useful for returning all kernel elements).
    Default is False, which only extracts from Sum nodes.
        (as used in decomposing multi-component models, only valid for sums)
    """
    leaf_level = (Sum, Product) if all else (Sum)
    if isinstance(kernel, leaf_level):
        return extract_leaf_kernels(kernel.kernel1) + extract_leaf_kernels(
            kernel.kernel2
        )
    else:
        return [kernel]


class StateSpaceModel(Kernel):
    """
    The base class for an instantaneous linear Gaussian state space model

    Has attributes
        `name` (str) : used for unique identification in multicomponent models
        `dimension` (int) : dimensionality $d$ of the state space model

    The components of a state space model are:
    1. design_matrix         : The feedback matrix, F
    2. stationary_covariance : The stationary covariance, Pinf
    3. observation_model     : The observation model, H
    4. noise                 : The spectral density of the white noise process, Qc
    5. noise_effect_matrix   : The noise effect matrix, L
    6. transition_matrix     : The transition matrix, A_k
        (optional, default uses jax.scipy.linalg.expm)
    7. process_noise        : The process noise, Q_k
        (optional, default uses Pinf and A or (alternatively) Van Loan matrix exponential)

    As a child of :class:`tinygp.kernels.Kernel`, this class also implements
    addition and multiplication with other kernels, as well as evaluation

    """

    name: str = eqx.field(static=True)

    def coord_to_sortable(self, X: JAXArray) -> JAXArray:
        """A helper function used to convert coordinates to sortable 1-D values

        By default, this is the identity, but in cases where ``X`` is structured
        (e.g. multivariate inputs), this can be used to appropriately unwrap
        that structure.
        """
        if isinstance(X, tuple):
            return X[0]
        return X

    @property
    def dimension(self) -> int:
        """The dimension of the state space model, d"""
        return self.design_matrix().shape[0]

    @abstractmethod
    def design_matrix(self) -> JAXArray:
        """The design (also called the feedback) matrix for the process, $F$"""
        raise NotImplementedError

    @abstractmethod
    def stationary_covariance(self) -> JAXArray:
        """The stationary covariance of the process, Pinf"""
        raise NotImplementedError

    def observation_model(self, X: JAXArray, component: str | None = None) -> JAXArray:
        """The observation model for the process, $H$"""
        keep = (component is None) or (self.name == component)
        return self.observation_matrix(X) * int(keep)

    @abstractmethod
    def observation_matrix(self, X: JAXArray) -> JAXArray:
        """The observation matrix for the process, $H$"""
        raise NotImplementedError

    @abstractmethod
    def noise(self) -> JAXArray:
        """The spectral density of the white noise process, $Q_c$"""
        raise NotImplementedError

    @abstractmethod
    def noise_effect_matrix(self) -> JAXArray:
        """The noise effect matrix, $L$"""
        raise NotImplementedError

    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        """
        The transition matrix between two states at coordinates X1 and X2, $A_k$

        Default behavior uses jax.scipy.linalg.expm(self.design_matrix() * (X2 - X1)),
        which is appropriate for stationary kernels defined by a linear Gaussian SSM.

        Overload this method if you have a more general model or simply wish to
        define the transition matrix analytically.
        """
        F = self.design_matrix()
        t1 = self.coord_to_sortable(X1)
        t2 = self.coord_to_sortable(X2)
        dt = t2 - t1
        return expm(F * dt)

    def process_noise(self, X1: JAXArray, X2: JAXArray, use_van_loan=False) -> JAXArray:
        """
        The process noise matrix $Q_k$

        Default behavior computes Q from Pinf - A @ Pinf @ A
        (see Eq. 7 in Solin & Sarkka 2014). Alternatively,
        give use_van_loan=True to use the Van Loan method to compute
        Q from the matrix exponential involving the F, L, and Qc

        Overload this method if you have a more general model or simply wish to
        define the process noise analytically.
        """
        if use_van_loan:
            # use Van Loan matrix exponential given F, L, Qc
            t1 = self.coord_to_sortable(X1)
            t2 = self.coord_to_sortable(X2)
            dt = t2 - t1
            F = self.design_matrix()
            L = self.noise_effect_matrix()
            Qc = self.noise()
            return Q_from_VanLoan(F, L, Qc, dt)
        else:
            # See Eq. 7 in Solin & Sarkka 2014
            # https://users.aalto.fi/~ssarkka/pub/solin_mlsp_2014.pdf
            Pinf = self.stationary_covariance()
            A = self.transition_matrix(X1, X2)
            return Pinf - A @ Pinf @ A.T

    def reset_matrix(self, instid: int = 0) -> JAXArray:
        """
        The reset matrix for an instantaneous state
        space model is trivially the identity matrix.
        """
        return jnp.eye(self.dimension)

    def __add__(self, other: Kernel | JAXArray) -> Kernel:
        if not isinstance(other, StateSpaceModel):
            raise ValueError(
                "StateSpaceModel kernels can only be added to other StateSpaceModel kernels"
            )
        return Sum(self, other)

    def __radd__(self, other: Any) -> Kernel:
        # We'll hit this first branch when using the `sum` function
        if other == 0:
            return self
        if not isinstance(other, StateSpaceModel):
            raise ValueError(
                "StateSpaceModel kernels can only be added to other StateSpaceModel kernels"
            )
        return Sum(other, self)

    def __mul__(self, other: Kernel | JAXArray) -> Kernel:
        if isinstance(other, StateSpaceModel):
            return Product(self, other)
        if isinstance(other, Kernel) or jnp.ndim(other) != 0:
            raise ValueError(
                "StateSpaceModel kernels can only be multiplied by scalars and other "
                "StateSpaceModel kernels"
            )
        return Scale(kernel=self, scale=other)

    def __rmul__(self, other: Any) -> Kernel:
        if isinstance(other, StateSpaceModel):
            return Product(other, self)
        if isinstance(other, Kernel) or jnp.ndim(other) != 0:
            raise ValueError(
                "StateSpaceModel kernels can only be multiplied by scalars and other "
                "StateSpaceModel kernels"
            )
        return Scale(kernel=self, scale=other)

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        """
        The kernel evaluated via the state space representatio

        See Eq. 4 in Hartikainen & Särkkä 2010
        https://users.aalto.fi/~ssarkka/pub/gp-ts-kfrts.pdf
        """
        Pinf = self.stationary_covariance()
        h1 = self.observation_model(X1)
        h2 = self.observation_model(X2)
        return jnp.where(
            self.coord_to_sortable(X1) < self.coord_to_sortable(X2),
            (h2 @ Pinf @ self.transition_matrix(X1, X2).T @ h1.T).squeeze(),
            (h1 @ self.transition_matrix(X2, X1) @ Pinf @ h2.T).squeeze(),
        )

    def evaluate_diag(self, X: JAXArray) -> JAXArray:
        """For state space kernels, the variance is simple to compute"""
        h = self.observation_model(X)
        return h @ self.stationary_covariance() @ h.T

    def psd(self, omega: JAXArray) -> JAXArray:
        """
        The power spectral density (PSD) of the kernel

        See Eq. 8 in Solin & Sarkka 2014
        https://users.aalto.fi/~ssarkka/pub/solin_mlsp_2014.pdf
        """
        F = self.design_matrix()
        L = self.noise_effect_matrix()
        Qc = self.noise()
        H = self.observation_matrix(0)  # PSD is stationary, so X doesn't matter
        I = jnp.eye(self.dimension)

        def compute_psd(w: JAXArray) -> JAXArray:
            M = jnp.linalg.inv(F + 1j * w * I)
            S = H @ M.conj() @ L @ Qc @ L.T @ M.T @ H.T
            return S.squeeze().real / (2 * jnp.pi)

        return jax.vmap(compute_psd)(omega)


class Sum(StateSpaceModel):
    """
    A helper to represent the sum of two quasiseparable kernels

    The state dimension becomes d = d1 + d2
    """

    kernel1: StateSpaceModel
    kernel2: StateSpaceModel

    def __init__(self, kernel1, kernel2):
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.name = f"Sum({kernel1.name}, {kernel2.name})"

    @property
    def dimension(self) -> int:
        """The dimension of the summed state space model"""
        return self.kernel1.dimension + self.kernel2.dimension

    def coord_to_sortable(self, X: JAXArray) -> JAXArray:
        """We assume that both kernels use the same coordinates"""
        return self.kernel1.coord_to_sortable(X)

    def design_matrix(self) -> JAXArray:
        """F = BlockDiag(F1, F2)"""
        return Block(
            self.kernel1.design_matrix(), self.kernel2.design_matrix()
        ).to_dense()

    def noise_effect_matrix(self) -> JAXArray:
        """L = BlockDiag(L1, L2)"""
        return Block(
            self.kernel1.noise_effect_matrix(), self.kernel2.noise_effect_matrix()
        ).to_dense()

    def stationary_covariance(self) -> JAXArray:
        """Pinf = BlockDiag(Pinf1, Pinf2)"""
        return Block(
            self.kernel1.stationary_covariance(),
            self.kernel2.stationary_covariance(),
        ).to_dense()

    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        """A = BlockDiag(A1, A2)"""
        return Block(
            self.kernel1.transition_matrix(X1, X2),
            self.kernel2.transition_matrix(X1, X2),
        ).to_dense()

    def process_noise(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        """Q = BlockDiag(Q1, Q2)"""
        return Block(
            self.kernel1.process_noise(X1, X2), self.kernel2.process_noise(X1, X2)
        ).to_dense()

    def observation_model(self, X: JAXArray, component=None) -> JAXArray:
        """H = [H1, H2] with component extraction"""
        return jnp.hstack(
            (
                self.kernel1.observation_model(X, component=component),
                self.kernel2.observation_model(X, component=component),
            )
        )

    def observation_matrix(self, X: JAXArray) -> JAXArray:
        """H = [H1, H2]"""
        return jnp.hstack(
            (
                self.kernel1.observation_matrix(X),
                self.kernel2.observation_matrix(X),
            )
        )

    def reset_matrix(self, instid: int = 0) -> JAXArray:
        """RESET = BlockDiag(RESET1, RESET2)"""
        return Block(
            self.kernel1.reset_matrix(instid), self.kernel2.reset_matrix(instid)
        ).to_dense()

    def noise(self) -> JAXArray:
        """Qc = BlockDiag(Qc1, Qc2)"""
        return Block(self.kernel1.noise(), self.kernel2.noise()).to_dense()


class Product(StateSpaceModel):
    """
    A helper to represent the product of two StateSpaceModel kernels

    The state dimension becomes d = d1 * d2
    """

    kernel1: StateSpaceModel
    kernel2: StateSpaceModel

    def __init__(self, kernel1, kernel2):
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.name = f"Product({kernel1.name}, {kernel2.name})"

    @property
    def dimension(self) -> int:
        """The dimension of the product state space model"""
        return self.kernel1.dimension * self.kernel2.dimension

    def coord_to_sortable(self, X: JAXArray) -> JAXArray:
        """We assume that both kernels use the same coordinates"""
        return self.kernel1.coord_to_sortable(X)

    def design_matrix(self) -> JAXArray:
        """F = F1 ⊗ I + I ⊗ F2"""
        F1 = self.kernel1.design_matrix()
        F2 = self.kernel2.design_matrix()
        I1 = jnp.eye(F1.shape[0])
        I2 = jnp.eye(F2.shape[0])
        return jnp.kron(F1, I2) + jnp.kron(I1, F2)

    def noise_effect_matrix(self) -> JAXArray:
        """
        L for products is not uniquely defined!
        We choose a convenient form here where
        L is simply the identity, and choose
        Qc such that we get the correct L@Qc@L^T
        """
        return jnp.eye(self.dimension)

    def stationary_covariance(self) -> JAXArray:
        """Pinf = Pinf1 ⊗ Pinf2"""
        Pinf1 = self.kernel1.stationary_covariance()
        Pinf2 = self.kernel2.stationary_covariance()
        return jnp.kron(Pinf1, Pinf2)

    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        """A = A1 ⊗ A2"""
        A1 = self.kernel1.transition_matrix(X1, X2)
        A2 = self.kernel2.transition_matrix(X1, X2)
        return jnp.kron(A1, A2)

    def process_noise(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        """
        Q for a product is best determined via the identity
        Q = Pinf - A Pinf A^T
        """
        Q1 = self.kernel1.process_noise(X1, X2)
        Q2 = self.kernel2.process_noise(X1, X2)
        Pinf1 = self.kernel1.stationary_covariance()
        Pinf2 = self.kernel2.stationary_covariance()
        return jnp.kron(Pinf1, Q2) + jnp.kron(Q1, Pinf2) - jnp.kron(Q1, Q2)

    def observation_model(self, X: JAXArray, component=None) -> JAXArray:
        """H = H1 ⊗ H2 with component extraction"""
        H1 = self.kernel1.observation_model(X, component=component)
        H2 = self.kernel2.observation_model(X, component=component)
        return jnp.kron(H1, H2)

    def observation_matrix(self, X: JAXArray) -> JAXArray:
        """H = H1 ⊗ H2"""
        H1 = self.kernel1.observation_matrix(X)
        H2 = self.kernel2.observation_matrix(X)
        return jnp.kron(H1, H2)

    def reset_matrix(self, instid: int = 0) -> JAXArray:
        """RESET = RESET1 ⊗ RESET2"""
        Reset1 = self.kernel1.reset_matrix(instid)
        Reset2 = self.kernel2.reset_matrix(instid)
        return jnp.kron(Reset1, Reset2)

    def noise(self) -> JAXArray:
        """
        Qc for products is not uniquely defined!
        Here we choose a convenient form where
        Qc = L1 Qc1 L1^T ⊗ Pinf2 + Pinf1 ⊗ L2 Qc2 L2^T
        and L is the identity, so that L Qc L^T gives the correct process noise
        """
        Qc1 = self.kernel1.noise()
        Qc2 = self.kernel2.noise()
        L1 = self.kernel1.noise_effect_matrix()
        L2 = self.kernel2.noise_effect_matrix()
        Pinf1 = self.kernel1.stationary_covariance()
        Pinf2 = self.kernel2.stationary_covariance()
        B1 = L1 @ Qc1 @ L1.T
        B2 = L2 @ Qc2 @ L2.T
        B = jnp.kron(B1, Pinf2) + jnp.kron(Pinf1, B2)
        return B


class Wrapper(StateSpaceModel):
    """A base class for wrapping kernels with some custom implementations"""

    kernel: StateSpaceModel

    @property
    def dimension(self) -> int:
        return self.kernel.dimension

    def coord_to_sortable(self, X: JAXArray) -> JAXArray:
        return self.kernel.coord_to_sortable(X)

    def design_matrix(self) -> JAXArray:
        return self.kernel.design_matrix()

    def noise_effect_matrix(self) -> JAXArray:
        return self.kernel.noise_effect_matrix()

    def noise(self) -> JAXArray:
        return self.kernel.noise()

    def stationary_covariance(self) -> JAXArray:
        return self.kernel.stationary_covariance()

    def observation_matrix(self, X: JAXArray) -> JAXArray:
        return self.kernel.observation_matrix(X)

    def observation_model(self, X: JAXArray, component=None) -> JAXArray:
        return self.kernel.observation_model(X, component=component)

    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return self.kernel.transition_matrix(X1, X2)

    def process_noise(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return self.kernel.process_noise(X1, X2)

    def reset_matrix(self, instid=0):
        return self.kernel.reset_matrix(instid)


class Scale(Wrapper):
    """The product of a scalar and a quasiseparable kernel"""

    scale: JAXArray | float

    def stationary_covariance(self) -> JAXArray:
        return self.scale * self.kernel.stationary_covariance()

    # TODO: also scale Qc?
    def noise(self) -> JAXArray:
        return self.scale * self.kernel.noise()


################ GP KERNEL DEFINITIONS ################
## TODO: tinygp base kernels not yet implemented
##       some will require approximations
# Polynomial
# ExpSquared (RBF), will need approx. via Taylor expannsion
#     see Hartikainen and Särkkä, 2010; Särkkä et al., 2013
# ExpSineSquared (will need approx)
# RationalQuadratic
# Quasiperiodic (not explicitly in tinygp, but is: ExpSquared * ExpSineSquared
##    some also define ExpSquared * ExpSineSquared * ExpCosineSquared for P/2 term
## RotationTerm (sum of SHO as in celerite)


class Constant(StateSpaceModel):
    r"""A constant kernel

    .. math::

        k(\mathbf{x}_i,\,\mathbf{x}_j) = \sigma^2

    where :math:`\sigma` is a parameter.

    Args:
        sigma: The parameter :math:`\sigma` in the above equation.
    """

    sigma: JAXArray | float

    def __init__(
        self,
        sigma: JAXArray | float = jnp.ones(()),
        name: str = "Constant",
        **kwargs,
    ):
        self.sigma = sigma
        self.name = name

    def design_matrix(self) -> JAXArray:
        """The design (also called the feedback) matrix for a Constant process, F"""
        return jnp.array([[0]])

    def stationary_covariance(self) -> JAXArray:
        """The stationary covariance of a Constant process, Pinf"""
        return jnp.array([[jnp.square(self.sigma)]])

    def noise(self) -> JAXArray:
        """The scalar Qc for a Constant process"""
        return jnp.array([[0]])

    def observation_matrix(self, X: JAXArray) -> JAXArray:
        """The observation model H for a Constant process"""
        del X
        return jnp.array([[1]])

    def noise_effect_matrix(self) -> JAXArray:
        """The noise effect matrix L for a Constant process"""
        return jnp.array([[1]])

    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        """The transition matrix A_k for a Constant process"""
        del X1, X2
        return self.eye(self.dimension)

    def process_noise(self, X1, X2, use_van_loan=False):
        """The process noise Q_k for a Constant process"""
        return jnp.array([[0]])


class SHO(StateSpaceModel):
    r"""The damped, driven simple harmonic oscillator kernel

    This form of the kernel was introduced by `Foreman-Mackey et al. (2017)
    <https://arxiv.org/abs/1703.09710>`_, and it takes the form:

    .. math::

        k(\Delta) = \sigma^2\,\exp\left(-\frac{\omega_0\,\Delta}{2\,Q}\right)
        \left\{\begin{array}{ll}
            1 + \omega_0\,\Delta & \mbox{for } Q = 1/2 \\
            \cosh(\eta\,\omega_0\,\Delta) + \fra{1}{2\eta Q} \sinh(\eta\,\omega_0\,\Delta)
                & \mbox{for } Q < 1/2 \\
            \frac{1}{2\eta Q}\cos(\eta\,\omega_0\,\Delta) + \sin(\eta\,\omega_0\,\Delta)
                & \mbox{for } Q > 1/2
        \end{array}\right.

    for :math:`\Delta = |x_i - x_j|`, :math:`\eta = \sqrt{|1 - 1/(4Q^2)|}`.

    Args:
        omega: The parameter :math:`\omega_0`.
        quality: The parameter :math:`Q`.
        sigma (optional): The parameter :math:`\sigma`. Defaults to a value of
            1. Specifying the explicit value here provides a slight performance
            boost compared to independently multiplying the kernel with a
            prefactor.
    """

    omega: JAXArray | float
    quality: JAXArray | float
    sigma: JAXArray | float = eqx.field(default_factory=lambda: jnp.ones(()))
    eta: JAXArray | float

    def __init__(
        self,
        omega: JAXArray | float,
        quality: JAXArray | float,
        sigma: JAXArray | float = jnp.ones(()),
        name: str = "SHO",
        **kwargs,
    ):
        # SHO parameters
        self.omega = omega
        self.quality = quality
        self.sigma = sigma
        self.name = name
        self.eta = jnp.sqrt(jnp.abs(1 - 1 / (4 * self.quality**2)))

    def design_matrix(self) -> JAXArray:
        """The design (also called the feedback) matrix for the SHO process, F"""
        return jnp.array(
            [[0, 1], [-jnp.square(self.omega), -self.omega / self.quality]]
        )

    def stationary_covariance(self) -> JAXArray:
        """The stationary covariance of the SHO process, Pinf"""
        return jnp.square(self.sigma) * jnp.diag(jnp.array([1, jnp.square(self.omega)]))

    def noise(self) -> JAXArray:
        """The scalar Qc for the SHO process"""
        omega3 = jnp.power(self.omega, 3)
        return jnp.array([[2 * omega3 * jnp.square(self.sigma) / self.quality]])

    def observation_matrix(self, X: JAXArray) -> JAXArray:
        """The observation model H for the SHO process"""
        del X
        return jnp.array([[1, 0]])

    def noise_effect_matrix(self) -> JAXArray:
        """The noise effect matrix L for the SHO process"""
        return jnp.array([[0], [1]])

    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        """The transition matrix A_k for the SHO process"""
        t1 = self.coord_to_sortable(X1)
        t2 = self.coord_to_sortable(X2)
        dt = t2 - t1
        n = self.eta
        w = self.omega
        q = self.quality

        def critical(dt: JAXArray) -> JAXArray:
            return jnp.exp(-w * dt) * jnp.array(
                [[1 + w * dt, dt], [-jnp.square(w) * dt, 1 - w * dt]]
            )

        def underdamped(dt: JAXArray) -> JAXArray:
            f = 2 * n * q
            x = n * w * dt
            sin = jnp.sin(x)
            cos = jnp.cos(x)
            return jnp.exp(-0.5 * w * dt / q) * jnp.array(
                [[cos + sin / f, sin / (w * n)], [-w * sin / n, cos - sin / f]]
            )

        def overdamped(dt: JAXArray) -> JAXArray:
            f = 2 * n * q
            x = n * w * dt
            sinh = jnp.sinh(x)
            cosh = jnp.cosh(x)
            return jnp.exp(-0.5 * w * dt / q) * jnp.array(
                [[cosh + sinh / f, sinh / (w * n)], [-w * sinh / n, cosh - sinh / f]]
            )

        return jax.lax.cond(
            jnp.allclose(q, 0.5),
            critical,
            lambda dt: jax.lax.cond(q > 0.5, underdamped, overdamped, dt),
            dt,
        )

    def process_noise(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        """The process noise Q_k for the SHO process"""
        t1 = self.coord_to_sortable(X1)
        t2 = self.coord_to_sortable(X2)
        dt = t2 - t1
        n = self.eta
        w = self.omega
        q = self.quality

        def critical(dt: JAXArray) -> JAXArray:
            Pinf = self.stationary_covariance()
            A = self.transition_matrix(0, dt)
            return Pinf - A @ Pinf @ A.T

        def underdamped(dt: JAXArray) -> JAXArray:
            f = 2 * n * q
            q2 = jnp.square(q)
            n2 = jnp.square(n)
            w2 = jnp.square(w)
            a = w * dt / q  # argument in exponential
            x = n * w * dt  # argument in sin/cos
            exp = jnp.exp(a)
            sin = jnp.sin(x)
            sin2 = jnp.sin(2 * x)
            sinsq = jnp.square(sin)
            Q11 = exp - 1 - sin2 / f - sinsq / (2 * n2 * q2)
            Q12 = Q21 = w * sinsq / (n2 * q)
            Q22 = w2 * (exp - 1 + sin2 / f - sinsq / (2 * n2 * q2))
            return (
                jnp.square(self.sigma)
                * jnp.exp(-a)
                * jnp.array([[Q11, Q12], [Q21, Q22]])
            )

        def overdamped(dt: JAXArray) -> JAXArray:
            Pinf = self.stationary_covariance()
            A = self.transition_matrix(0, dt)
            return Pinf - A @ Pinf @ A.T

        return jax.lax.cond(
            jnp.allclose(q, 0.5),
            critical,
            lambda dt: jax.lax.cond(q > 0.5, underdamped, overdamped, dt),
            dt,
        )


class Exp(StateSpaceModel):
    r"""A state space implementation of :class:`tinygp.kernels.quasisep.Exp`

    This kernel takes the form:

    .. math::

        k(\Delta)=\sigma^2\,\exp\left(-\frac{\Delta}{\ell}\right)

    for :math:`\Delta = |x_i - x_j|`. Also known as the "Ornstein-Uhlenbeck" kernel,
    and is also equivalent to a Matérn-1/2 kernel.

    Args:
        scale: The parameter :math:`\ell`.
        sigma (optional): The parameter :math:`\sigma`. Defaults to a value of 1.
    """

    scale: JAXArray | float
    sigma: JAXArray | float = eqx.field(default_factory=lambda: jnp.ones(()))
    lam: JAXArray | float

    def __init__(
        self,
        scale: JAXArray | float,
        sigma: JAXArray | float = jnp.ones(()),
        name: str = "Exp",
        **kwargs,
    ):
        # Exp parameters
        self.scale = scale
        self.sigma = sigma
        self.name = name
        self.lam = 1 / self.scale

    def design_matrix(self) -> JAXArray:
        """The design (also called the feedback) matrix for the Exp process, F"""
        return jnp.array([[-self.lam]])

    def stationary_covariance(self) -> JAXArray:
        """The stationary covariance of the Exp process, Pinf"""
        return jnp.square(self.sigma) * jnp.ones((1, 1))

    def observation_matrix(self, X: JAXArray) -> JAXArray:
        """The observation model H for the Exp process"""
        del X
        return jnp.array([[1]])

    def noise_effect_matrix(self) -> JAXArray:
        """The noise effect matrix L for the Exp process"""
        return jnp.array([[1]])

    def noise(self) -> JAXArray:
        """The scalar Qc for the Exp process"""
        return jnp.array([[2 * self.lam * jnp.square(self.sigma)]])

    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        """The transition matrix A_k for the Exp process"""
        t1 = self.coord_to_sortable(X1)
        t2 = self.coord_to_sortable(X2)
        dt = t2 - t1
        return jnp.exp(-dt[None, None] / self.scale)

    # def process_noise(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
    #     """The process noise Q_k for the Exp process"""
    #     dt = X2 - X1
    #     TODO: can implement here the analytic version, but by
    #         default the parent class will generate w/ Pinf - A Pinf A^T


class Matern32(StateSpaceModel):
    r"""A state space implementation of :class:`tinygp.kernels.quasisep.Matern32`

    This kernel takes the form:

    .. math::

        k(\Delta)=\sigma^2\,\left(1+f\,\Delta\right)\,\exp(-f\,\Delta)

    for :math:`\Delta = |x_i - x_j|` and :math:`f = \sqrt{3} / \ell`.

    Args:
        scale: The parameter :math:`\ell`.
        sigma (optional): The parameter :math:`\sigma`. Defaults to a value of 1.
    """

    scale: JAXArray | float
    sigma: JAXArray | float
    lam: JAXArray | float

    def __init__(
        self,
        scale: JAXArray | float,
        sigma: JAXArray | float = jnp.ones(()),
        name: str = "Matern32",
        **kwargs,
    ):
        # Matern-3/2 parameters
        self.scale = scale
        self.sigma = sigma
        self.name = name
        self.lam = jnp.sqrt(3) / self.scale

    def design_matrix(self) -> JAXArray:
        """The design (also called the feedback) matrix for the Matern-3/2 process, F"""
        lam2 = jnp.square(self.lam)
        return jnp.array([[0, 1], [-lam2, -2 * self.lam]])

    def stationary_covariance(self) -> JAXArray:
        """The stationary covariance of the Matern-3/2 process, Pinf"""
        return jnp.square(self.sigma) * jnp.diag(jnp.array([1, jnp.square(self.lam)]))

    def observation_matrix(self, X: JAXArray) -> JAXArray:
        """The observation model H for the Matern-3/2 process"""
        del X
        return jnp.array([[1, 0]])

    def noise_effect_matrix(self) -> JAXArray:
        """The noise effect matrix L for the Matern-3/2 process"""
        return jnp.array([[0], [1]])

    def noise(self) -> JAXArray:
        """The scalar Qc for the Matern-3/2 process"""
        lam3 = jnp.power(self.lam, 3)
        return jnp.array([[4 * lam3 * jnp.square(self.sigma)]])

    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        """The transition matrix A_k for the Matern-3/2 process"""
        t1 = self.coord_to_sortable(X1)
        t2 = self.coord_to_sortable(X2)
        dt = t2 - t1
        lam = self.lam
        return jnp.exp(-lam * dt) * jnp.array(
            [[1 + lam * dt, dt], [-jnp.square(lam) * dt, 1 - lam * dt]]
        )

    # def process_noise(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
    #     """The process noise Q_k for the Matern-3/2 process"""
    #     dt = X2 - X1
    #     TODO: can implement here the analytic version, but by
    #         default the parent class will generate w/ Pinf - A Pinf A^T


class Matern52(StateSpaceModel):
    r"""A state space implementation of :class:`tinygp.kernels.quasisep.Matern52`

    This kernel takes the form:

    .. math::

        k(\Delta)=\sigma^2\,\left(1+f\,\Delta + \frac{f^2\,\Delta^2}{3}\right)
            \,\exp(-f\,\Delta)

    for :math:`\Delta = |x_i - x_j|` and :math:`f = \sqrt{5} / \ell`.

    Args:
        scale: The parameter :math:`\ell`.
        sigma (optional): The parameter :math:`\sigma`. Defaults to a value of 1.
    """

    scale: JAXArray | float
    sigma: JAXArray | float
    lam: JAXArray | float

    def __init__(
        self,
        scale: JAXArray | float,
        sigma: JAXArray | float = jnp.ones(()),
        name: str = "Matern52",
        **kwargs,
    ):
        # Matern-5/2 parameters
        self.scale = scale
        self.sigma = sigma
        self.name = name
        self.lam = jnp.sqrt(5) / self.scale

    def design_matrix(self) -> JAXArray:
        """The design (also called the feedback) matrix for the Matern-5/2 process, F"""
        lam2 = jnp.square(self.lam)
        lam3 = lam2 * self.lam
        return jnp.array([[0, 1, 0], [0, 0, 1], [-lam3, -3 * lam2, -3 * self.lam]])

    def stationary_covariance(self) -> JAXArray:
        """The stationary covariance of the Matern-5/2 process, Pinf"""
        lam2 = jnp.square(self.lam)
        lam2o3 = lam2 / 3
        return jnp.square(self.sigma) * jnp.array(
            [[1, 0, -lam2o3], [0, lam2o3, 0], [-lam2o3, 0, jnp.square(lam2)]]
        )

    def observation_matrix(self, X: JAXArray) -> JAXArray:
        """The observation model H for the Matern-5/2 process"""
        del X
        return jnp.array([[1, 0, 0]])

    def noise_effect_matrix(self) -> JAXArray:
        """The noise effect matrix L for the Matern-5/2 process"""
        return jnp.array([[0], [0], [1]])

    def noise(self) -> JAXArray:
        """The scalar Qc for the Matern-5/2 process"""
        lam5 = jnp.power(self.lam, 5)
        return jnp.array([[16 * lam5 * jnp.square(self.sigma) / 3]])

    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        """The transition matrix A_k for the Matern-5/2 process"""
        t1 = self.coord_to_sortable(X1)
        t2 = self.coord_to_sortable(X2)
        dt = t2 - t1
        lam = self.lam
        lam2 = jnp.square(lam)
        d2 = jnp.square(dt)
        a11 = 0.5 * lam2 * d2 + lam * dt + 1
        a12 = dt * (lam * dt + 1)
        a13 = 0.5 * d2
        a21 = -0.5 * lam * lam2 * d2
        a22 = -lam2 * d2 + lam * dt + 1
        a23 = 0.5 * dt * (2 - lam * dt)
        a31 = 0.5 * lam2 * lam * dt * (lam * dt - 2)
        a32 = lam2 * dt * (lam * dt - 3)
        a33 = 0.5 * lam2 * d2 - 2 * lam * dt + 1
        return jnp.exp(-lam * dt) * jnp.array(
            [
                [a11, a12, a13],
                [a21, a22, a23],
                [a31, a32, a33],
            ]
        )

    # def process_noise(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
    #     """The process noise Q_k for the Matern-5/2 process"""
    #     dt = X2 - X1
    #     TODO: can implement here the analytic version, but by
    #           default the parent class will generate w/ Pinf - A Pinf A^T


class Cosine(StateSpaceModel):
    r"""A state space implementation of :class:`tinygp.kernels.quasisep.Cosine`

    This kernel takes the form:

    .. math::

        k(\Delta)=\sigma^2\,\cos(-2\,\pi\,\Delta/\ell)

    for :math:`\Delta = |x_i - x_j|`.

    Args:
        scale: The parameter :math:`\ell`.
        sigma (optional): The parameter :math:`\sigma`. Defaults to a value of 1.
    """

    scale: JAXArray | float
    sigma: JAXArray | float = eqx.field(default_factory=lambda: jnp.ones(()))
    omega: JAXArray | float

    def __init__(self, scale, sigma=jnp.ones(()), name="Cosine", **kwargs):
        # Cosine parameters
        self.scale = scale
        self.sigma = sigma
        self.name = name
        self.omega = 2 * jnp.pi / self.scale

    def design_matrix(self) -> JAXArray:
        """The design (also called the feedback) matrix for the Cosine process, F"""
        return jnp.array([[0, -self.omega], [self.omega, 0]])

    def stationary_covariance(self) -> JAXArray:
        """The stationary covariance of the Cosine process, Pinf"""
        return jnp.square(self.sigma) * jnp.eye(2)

    def observation_matrix(self, X: JAXArray) -> JAXArray:
        """The observation model H for the Cosine process"""
        del X
        return jnp.array([[1, 0]])

    def noise_effect_matrix(self) -> JAXArray:
        """The noise effect matrix L for the Cosine process"""
        return jnp.array([[0], [1]])

    def noise(self) -> JAXArray:
        """The scalar Qc for the Cosine process"""
        Qc = 0.0  # Does not have a white noise driving process
        return jnp.array([[Qc]])

    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        """The transition matrix A_k for the Cosine process"""
        t1 = self.coord_to_sortable(X1)
        t2 = self.coord_to_sortable(X2)
        dt = t2 - t1
        arg = self.omega * dt
        cos = jnp.cos(arg)
        sin = jnp.sin(arg)
        return jnp.array([[cos, -sin], [sin, cos]])

    # def process_noise(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
    #     """The process noise Q_k for the Cosine process"""
    #     dt = X2 - X1
    #     TODO: can implement here the analytic version, but by
    #           default the parent class will generate w/ Pinf - A Pinf A^T


# class ExpSquared(StateSpaceModel):
#     r"""
#     A state space implementation of the exponential squared
#     (also called the radial basis function, or RBF) kernel

#     .. math::

#         k(\mathbf{x}_i,\,\mathbf{x}_j) = \exp(-r^2 / 2)

#     where, by default,

#     .. math::

#         r^2 = ||(\mathbf{x}_i - \mathbf{x}_j) / \ell||_2^2

#     Args:
#         scale: The parameter :math:`\ell`.
#     """


class ExpSineSquared(Wrapper):
    r"""The exponential sine squared or "periodic" kernel

    .. math::

        k(\mathbf{x}_i,\,\mathbf{x}_j) = \sigma^2 \exp(-\Gamma\,\sin^2 \pi r)

    where, by default,

    .. math::

        r = ||(\mathbf{x}_i - \mathbf{x}_j) / P||_1

    In the state space representation, this kernel is approximated using
    a finite number of basis functions. The method was introduced by
    `Solin & Särkkä (2014) <https://proceedings.mlr.press/v33/solin14.html`_.
    See their Figure 2 for number of basis functions to reach desired accuracy.
    Default behavior will automatically select the number of basis functions
    based on the length scale :math:`\ell`.

    Args:
        period: The parameter :math:`P`.
        gamma: The parameter :math:`\Gamma`.
        sigma: The parameter :math:`\sigma`. Defaults to a value of 1.
    """

    gamma: JAXArray | float | None = None
    period: JAXArray | float
    scale: JAXArray | float
    omega: JAXArray | float
    sigma: JAXArray | float = eqx.field(default_factory=lambda: jnp.ones(()))
    order: int | None
    kernel: StateSpaceModel

    def __init__(
        self,
        period: JAXArray | float,
        gamma: JAXArray | float = jnp.ones(()),
        sigma: JAXArray | float = jnp.ones(()),
        name: str = "ExpSineSquared",
        order: int | None = None,
        **kwargs,
    ):
        self.period = period  # P
        self.gamma = gamma  # \Gamma
        self.sigma = sigma  # \sigma
        self.name = name

        self.scale = 2 / self.gamma  # \ell^2 in Solin & Särkkä (2014)
        self.omega = 2 * jnp.pi / self.period  # \omega_0 in Solin & Särkkä (2014)

        # Auto-select order (J) using Fig 2 of
        # Solin & Särkkä (2014) as a
        ell = jnp.sqrt(self.scale)
        if order is None:
            if ell >= 1:
                self.order = 4
            elif ell >= 0.5:
                self.order = 6
            elif ell >= 0.25:
                self.order = 8
            else:
                self.order = 16
                warnings.warn(
                    "ExpSineSquared kernel with scale < 0.25 (gamma > 16) may require a high order approximation; "
                    "it may be worthwhile to change units to a more compatible scale (recommended) "
                    "or specify the 'order' parameter explicitly."
                )
        else:
            self.order = order

        # Construct the kernel as a sum of PeriodicTerms
        kernel = self.PeriodicTerm(0, self.omega, self.scale)
        for j in range(1, self.order):
            kernel += self.PeriodicTerm(j, self.omega, self.scale)
        self.kernel = kernel

    def error_bound(self):
        """
        An upper bound on the error in the covariance
        from the Taylor series approximation.

        See Sec 3.4 of Solin & Särkkä (2014).
        """
        J = self.order
        return jnp.exp(1 - self.scale) / jax.scipy.special.factorial(J + 1)

    def stationary_covariance(self) -> JAXArray:
        return jnp.square(self.sigma) * self.kernel.stationary_covariance()

    # Class for each periodic term in the Taylor series expansion
    class PeriodicTerm(StateSpaceModel):
        """
        A single term in the state space representation of the
        exponential sine squared or "periodic" kernel

        See Solin & Särkkä (2014) for details.
        """

        order: int
        omega: JAXArray | float
        scale: JAXArray | float

        def __init__(self, order, omega, scale, **kwargs):
            self.order = order
            self.omega = omega  # \omega_0
            self.scale = scale  # \ell^2
            self.name = f"PeriodicTerm_{order}"

        def design_matrix(self) -> JAXArray:
            """The design (also called the feedback) matrix for the process, $F$"""
            j = self.order
            w = self.omega
            return jnp.array([[0, -w * j], [w * j, 0]])

        def stationary_covariance(self) -> JAXArray:
            """The stationary covariance of the process, Pinf"""
            j = self.order
            arg = 1 / self.scale
            coeff = jax.lax.cond(j == 0, lambda _: 1.0, lambda _: 2.0, j)
            qj2 = coeff * self.Ij(j, arg) / jnp.exp(arg)
            return qj2 * jnp.eye(self.dimension)

        def observation_matrix(self, X: JAXArray) -> JAXArray:
            """The observation matrix for the process, $H$"""
            return jnp.array([[1, 0]])

        def noise(self) -> JAXArray:
            """The spectral density of the white noise process, $Q_c$"""
            return jnp.zeros((self.dimension, self.dimension))

        def noise_effect_matrix(self) -> JAXArray:
            """The noise effect matrix, $L$"""
            return jnp.eye(self.dimension)

        def Ij(self, j, x, terms=50) -> JAXArray:
            """
            The modified Bessel function of the first kind, order j, at x.
            Approximated via a truncated Taylor series expansion.
            """
            k = jnp.arange(terms)
            log_terms = (
                -gammaln(k + 1) - gammaln(k + j + 1) + (2 * k + j) * jnp.log(x / 2)
            )
            return jnp.sum(jnp.exp(log_terms))
