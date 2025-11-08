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

__all__ = [
    "StateSpaceModel",
    "Wrapper",
    "Sum",
    "Product",
    "Scale",
    "SHO",
    "Exp",
    "Matern32",
    "Matern52",
    "Cosine",
]

from abc import abstractmethod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import expm

from tinygp.helpers import JAXArray
from tinygp.kernels.base import Kernel
from tinygp.solvers.quasisep.block import Block

from smolgp.helpers import Q_from_VanLoan


def extract_leaf_kernels(kernel):
    """Recursively extract all leaf kernels from a sum or product of kernels"""
    if isinstance(kernel, (Sum, Product)):
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
    dimension: JAXArray | float = eqx.field(static=True)

    def coord_to_sortable(self, X: JAXArray) -> JAXArray:
        """A helper function used to convert coordinates to sortable 1-D values

        By default, this is the identity, but in cases where ``X`` is structured
        (e.g. multivariate inputs), this can be used to appropriately unwrap
        that structure.
        """
        return X

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
        dt = X2 - X1
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
            dt = X2 - X1
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
        """The kernel evaluated via the state space representation"""
        Pinf = self.stationary_covariance()
        h1 = self.observation_model(X1)
        h2 = self.observation_model(X2)
        return jnp.where(
            self.coord_to_sortable(X1) < self.coord_to_sortable(X2),
            h2 @ Pinf @ self.transition_matrix(X1, X2) @ h1,
            h1 @ Pinf @ self.transition_matrix(X2, X1) @ h2,
        )

    def evaluate_diag(self, X: JAXArray) -> JAXArray:
        """For state space kernels, the variance is simple to compute"""
        h = self.observation_model(X)
        return h @ self.stationary_covariance() @ h


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
        self.dimension = self.kernel1.dimension + self.kernel2.dimension

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
        """TODO: is it just Qc = Qc1 + Qc2 ??"""
        ## TODO: we might not need to even implement this
        ##  since practically we only need process_noise
        ##  which can be has each component process_noise
        ##  defined by its own Qc already
        raise NotImplementedError


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
        self.dimension = self.kernel1.dimension * self.kernel2.dimension

    def coord_to_sortable(self, X: JAXArray) -> JAXArray:
        """We assume that both kernels use the same coordinates"""
        return self.kernel1.coord_to_sortable(X)

    def design_matrix(self) -> JAXArray:
        """F = F1 ⊗ I + I ⊗ F2"""
        F1 = self.kernel1.design_matrix()
        F2 = self.kernel2.design_matrix()
        return _prod_helper(F1, jnp.eye(F2.shape[0])) + _prod_helper(
            jnp.eye(F2.shape[0]), F2
        )

    def noise_effect_matrix(self) -> JAXArray:
        """L = L1 ⊗ L2"""
        return _prod_helper(
            self.kernel1.noise_effect_matrix(),
            self.kernel2.noise_effect_matrix(),
        )

    def stationary_covariance(self) -> JAXArray:
        """Pinf = Pinf1 ⊗ Pinf2"""
        return _prod_helper(
            self.kernel1.stationary_covariance(),
            self.kernel2.stationary_covariance(),
        )

    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        """A = A1 ⊗ A2"""
        return _prod_helper(
            self.kernel1.transition_matrix(X1, X2),
            self.kernel2.transition_matrix(X1, X2),
        )

    def process_noise(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        """Q = Q1 ⊗ Q2"""
        return _prod_helper(
            self.kernel1.process_noise(X1, X2), self.kernel2.process_noise(X1, X2)
        )

    def observation_model(self, X: JAXArray, component=None) -> JAXArray:
        """H = H1 ⊗ H2 with component extraction"""
        return jnp.array(
            [
                _prod_helper(
                    self.kernel1.observation_model(X, component=component),
                    self.kernel2.observation_model(X, component=component),
                )
            ]
        )

    def observation_matrix(self, X: JAXArray) -> JAXArray:
        """H = H1 ⊗ H2"""
        return jnp.array(
            [
                _prod_helper(
                    self.kernel1.observation_matrix(X),
                    self.kernel2.observation_matrix(X),
                )
            ]
        )

    def reset_matrix(self, instid: int = 0) -> JAXArray:
        """RESET = RESET1 ⊗ RESET2"""
        return jnp.array(
            [
                _prod_helper(
                    self.kernel1.reset_matrix(instid),
                    self.kernel2.reset_matrix(instid),
                )
            ]
        )

    def noise(self) -> JAXArray:
        """TODO: is it Qc = Qc1 * Qc2 ??"""
        ## TODO: we might not need to even implement this
        ##  since practically we only need process_noise
        ##  which can be has each component process_noise
        ##  defined by its own Qc already
        raise NotImplementedError


class Wrapper(StateSpaceModel):
    """A base class for wrapping kernels with some custom implementations"""

    kernel: StateSpaceModel

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

    def observation_model(self, X: JAXArray) -> JAXArray:
        return self.kernel.observation_model(self.coord_to_sortable(X))

    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return self.kernel.transition_matrix(
            self.coord_to_sortable(X1), self.coord_to_sortable(X2)
        )

    def process_noise(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return self.kernel.process_noise(
            self.coord_to_sortable(X1), self.coord_to_sortable(X2)
        )


class Scale(Wrapper):
    """The product of a scalar and a quasiseparable kernel"""

    scale: JAXArray | float

    def stationary_covariance(self) -> JAXArray:
        return self.scale * self.kernel.stationary_covariance()

    # TODO: also scale Qc?


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
        self.dimension = 2
        self.eta = jnp.sqrt(jnp.abs(1 - 1 / (4 * self.quality**2)))

    def design_matrix(self) -> JAXArray:
        """The design (also called the feedback) matrix for the SHO process, F"""
        return jnp.array(
            [[0, 1], [-jnp.square(self.omega), -self.omega / self.quality]]
        )

    def stationary_covariance(self) -> JAXArray:
        """The stationary covariance of the SHO process, Pinf"""
        return jnp.diag(jnp.square(self.sigma) * jnp.array([1, jnp.square(self.omega)]))

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
        dt = X2 - X1
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
        dt = X2 - X1
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


# class Exp(StateSpaceModel):
#     r"""A state space implementation of :class:`tinygp.kernels.quasisep.Exp`

#     This kernel takes the form:

#     .. math::

#         k(\tau)=\sigma^2\,\exp\left(-\frac{\tau}{\ell}\right)

#     for :math:`\tau = |x_i - x_j|`.

#     Args:
#         scale: The parameter :math:`\ell`.
#         sigma (optional): The parameter :math:`\sigma`. Defaults to a value of
#             1. Specifying the explicit value here provides a slight performance
#             boost compared to independently multiplying the kernel with a
#             prefactor.
#     """

#     scale: JAXArray | float
#     sigma: JAXArray | float = eqx.field(default_factory=lambda: jnp.ones(()))

#     def design_matrix(self) -> JAXArray:
#         return jnp.array([[-1 / self.scale]])

#     def stationary_covariance(self) -> JAXArray:
#         return jnp.ones((1, 1))

#     def observation_model(self, X: JAXArray) -> JAXArray:
#         del X
#         return jnp.array([self.sigma])

#     def noise_effect_matrix(self) -> JAXArray:
#         """ The noise effect matrix L for the process """
#         return jnp.array([[0], [1]])

#     def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
#         dt = X2 - X1
#         return jnp.exp(-dt[None, None] / self.scale)


# class Matern32(StateSpaceModel):
#     r"""A state space implementation of :class:`tinygp.kernels.quasisep.Matern32`

#     This kernel takes the form:

#     .. math::

#         k(\tau)=\sigma^2\,\left(1+f\,\tau\right)\,\exp(-f\,\tau)

#     for :math:`\tau = |x_i - x_j|` and :math:`f = \sqrt{3} / \ell`.

#     Args:
#         scale: The parameter :math:`\ell`.
#         sigma (optional): The parameter :math:`\sigma`. Defaults to a value of
#             1. Specifying the explicit value here provides a slight performance
#             boost compared to independently multiplying the kernel with a
#             prefactor.
#     """

#     scale: JAXArray | float
#     sigma: JAXArray | float = eqx.field(default_factory=lambda: jnp.ones(()))

#     def noise(self) -> JAXArray:
#         f = np.sqrt(3) / self.scale
#         return 4 * f**3

#     def design_matrix(self) -> JAXArray:
#         f = np.sqrt(3) / self.scale
#         return jnp.array([[0, 1], [-jnp.square(f), -2 * f]])

#     def stationary_covariance(self) -> JAXArray:
#         return jnp.diag(jnp.array([1, 3 / jnp.square(self.scale)]))

#     def observation_model(self, X: JAXArray) -> JAXArray:
#         return jnp.array([self.sigma, 0])

#     def noise_effect_matrix(self) -> JAXArray:
#         """ The noise effect matrix L for the process """
#         return jnp.array([[0], [1]])

#     def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
#         dt = X2 - X1
#         f = np.sqrt(3) / self.scale
#         return jnp.exp(-f * dt) * jnp.array(
#             [[1 + f * dt, -jnp.square(f) * dt], [dt, 1 - f * dt]]
#         )


class Matern52(StateSpaceModel):
    r"""A state space implementation of :class:`tinygp.kernels.quasisep.Matern52`

    This kernel takes the form:

    .. math::

        k(\tau)=\sigma^2\,\left(1+f\,\tau + \frac{f^2\,\tau^2}{3}\right)
            \,\exp(-f\,\tau)

    for :math:`\tau = |x_i - x_j|` and :math:`f = \sqrt{5} / \ell`.

    Args:
        scale: The parameter :math:`\ell`.
        sigma (optional): The parameter :math:`\sigma`. Defaults to a value of
            1. Specifying the explicit value here provides a slight performance
            boost compared to independently multiplying the kernel with a
            prefactor.
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
        self.dimension = 3
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
        return jnp.array(
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
        dt = X2 - X1
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
    #           default the parent class will generate w/ Van Loan


# class Cosine(StateSpaceModel):
#     r"""A state space implementation of :class:`tinygp.kernels.quasisep.Cosine`

#     This kernel takes the form:

#     .. math::

#         k(\tau)=\sigma^2\,\cos(-2\,\pi\,\tau/\ell)

#     for :math:`\tau = |x_i - x_j|`.

#     Args:
#         scale: The parameter :math:`\ell`.
#         sigma (optional): The parameter :math:`\sigma`. Defaults to a value of
#             1. Specifying the explicit value here provides a slight performance
#             boost compared to independently multiplying the kernel with a
#             prefactor.
#     """

#     scale: JAXArray | float
#     sigma: JAXArray | float = eqx.field(default_factory=lambda: jnp.ones(()))

#     def design_matrix(self) -> JAXArray:
#         f = 2 * np.pi / self.scale
#         return jnp.array([[0, -f], [f, 0]])

#     def stationary_covariance(self) -> JAXArray:
#         return jnp.eye(2)

#     def observation_model(self, X: JAXArray) -> JAXArray:
#         return jnp.array([self.sigma, 0])

#     def noise_effect_matrix(self) -> JAXArray:
#         """ The noise effect matrix L for the process """
#         return jnp.array([[0], [1]])

#     def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
#         dt = X2 - X1
#         f = 2 * np.pi / self.scale
#         cos = jnp.cos(f * dt)
#         sin = jnp.sin(f * dt)
#         return jnp.array([[cos, sin], [-sin, cos]])


def _prod_helper(a1: JAXArray, a2: JAXArray) -> JAXArray:
    i, j = np.meshgrid(np.arange(a1.shape[0]), np.arange(a2.shape[0]))
    i = i.flatten()
    j = j.flatten()
    if a1.ndim == 1:
        return a1[i] * a2[j]
    elif a1.ndim == 2:
        return a1[i[:, None], i[None, :]] * a2[j[:, None], j[None, :]]
    else:
        raise NotImplementedError
