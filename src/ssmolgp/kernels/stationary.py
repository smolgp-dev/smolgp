"""
The kernels implemented in this subpackage are defined similarly to 
:class: `tinygp.kernels.quasisep.Quasisep` but are modified to:
1. Include the `noise_effect_matrix` and `process_noise` matrix
2. Treat the observation model as a column vector and the transition matrix
   in its usual form (compared to transposed forms in quasisep)
3. Handle integrated versions of each kernel
These kernels are compatible with :class:`ssmolgp.solvers.StateSpaceSolver`
which use Bayesian filtering and smoothing algorithms to perform scalable GP
inference. (see :ref:`api-solvers-statespace` for more technical details). 

Like the quasisep kernels, these methods are experimental, so you may find 
the documentation patchy in places. You are encouraged to `open issues or 
pull requests <https://github.com/rrubenza/ssmolgp/issues>`_ as you find gaps.
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

from ssmolgp.helpers import Q_from_VanLoan

class StateSpaceModel(Kernel):
    """
    The base class for all stationary linear Gaussian state space models

    The components of a state space model are:
    1. design_matrix         : The feedback matrix, F
    2. stationary_covariance : The stationary covariance, Pinf
    3. observation_model     : The observation model, H
    4. noise                 : The spectral density of the white noise process, Qc
    5. noise_effect_matrix   : The noise effect matrix, L 
        (defaults to [0, 1] if not provided)
    6. transition_matrix     : The transition matrix, A_k
        (optional, default uses jax.scipy.linalg.expm)
    7. process_noise        : The process noise, Q_k
        (optional, default uses Van Loan matrix exponential)

    As a child of :class:`tinygp.kernels.Kernel`, this class also implements
    addition and multiplication with other kernels, as well as evaluation
    
    """

    dimension: JAXArray | float = eqx.field(static=True) # dimensionality $d$ of the state vector

    @abstractmethod
    def design_matrix(self) -> JAXArray:
        """The design (also called the feedback) matrix for the process, $F$"""
        raise NotImplementedError

    @abstractmethod
    def stationary_covariance(self) -> JAXArray:
        """The stationary covariance of the process, $P_\infty$"""
        raise NotImplementedError

    @abstractmethod
    def observation_model(self, X: JAXArray) -> JAXArray:
        """The observation model for the process, $H$"""
        raise NotImplementedError

    @abstractmethod
    def noise(self) -> JAXArray:
        ''' The spectral density of the white noise process, $Q_c$ '''
        raise NotImplementedError
    
    @abstractmethod
    def noise_effect_matrix(self) -> JAXArray:
        ''' The noise effect matrix L, by default this is [0,1]'''
        raise NotImplementedError
    
    @abstractmethod
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

    @abstractmethod    
    def process_noise(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        """
        The process noise matrix $Q_k$
        
        Default behavior computes Q from Pinf - A @ Pinf @ A, if Pinf and A 
        are both implemented. Otherwise, uses the Van Loan method to compute
        Q from the matrix exponential involving the F, L, and Qc

        Overload this method if you have a more general model or simply wish to
        define the process noise analytically.
        """
        try:
            # See Eq. 7 in Solin & Sarkka 2014
            # https://users.aalto.fi/~ssarkka/pub/solin_mlsp_2014.pdf
            Pinf = self.stationary_covariance()
            A = self.transition_matrix(X1, X2)
            return Pinf - A @ Pinf @ A.T 
        except NotImplementedError:
            # use Van Loan matrix exponential given F, L, Qc
            dt = X2 - X1
            F = self.design_matrix()
            L = self.noise_effect_matrix()
            Qc = self.noise()
            return Q_from_VanLoan(F,L,Qc,dt)


    def coord_to_sortable(self, X: JAXArray) -> JAXArray:
        """A helper function used to convert coordinates to sortable 1-D values

        By default, this is the identity, but in cases where ``X`` is structured
        (e.g. multivariate inputs), this can be used to appropriately unwrap
        that structure.
        """
        return X


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

    dimension: JAXArray | float = eqx.field(
        init=False, default_factory=lambda self: self.kernel1.dimension + self.kernel2.dimension
    )

    def coord_to_sortable(self, X: JAXArray) -> JAXArray:
        """We assume that both kernels use the same coordinates"""
        return self.kernel1.coord_to_sortable(X)

    def design_matrix(self) -> JAXArray:
        """F = BlockDiag(F1, F2)"""
        return Block(self.kernel1.design_matrix(), self.kernel2.design_matrix())

    def noise_effect_matrix(self) -> JAXArray:
        """L = BlockDiag(L1, L2)"""
        return Block(self.kernel1.noise_effect_matrix(), self.kernel2.noise_effect_matrix())

    def stationary_covariance(self) -> JAXArray:
        """Pinf = BlockDiag(Pinf1, Pinf2)"""
        return Block(self.kernel1.stationary_covariance(),
                     self.kernel2.stationary_covariance(),
        )

    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        """A = BlockDiag(A1, A2)"""
        return Block(
            self.kernel1.transition_matrix(X1, X2),
            self.kernel2.transition_matrix(X1, X2),
        )

    def process_noise(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        """Q = BlockDiag(Q1, Q2)"""
        return Block(self.kernel1.process_noise(X1, X2), 
                     self.kernel2.process_noise(X1, X2))

    def observation_model(self, X: JAXArray) -> JAXArray:
        """H = [H1, H2]"""
        return jnp.hstack(
            (
                self.kernel1.observation_model(X),
                self.kernel2.observation_model(X),
            )
        )

class Product(StateSpaceModel):
    """
    A helper to represent the product of two StateSpaceModel kernels
    
    The state dimension becomes d = d1 * d2
    """

    kernel1: StateSpaceModel
    kernel2: StateSpaceModel

    dimension: JAXArray | float = eqx.field(
        init=False, default_factory=lambda self: self.kernel1.dimension * self.kernel2.dimension
    )

    def coord_to_sortable(self, X: JAXArray) -> JAXArray:
        """We assume that both kernels use the same coordinates"""
        return self.kernel1.coord_to_sortable(X)

    def design_matrix(self) -> JAXArray:
        """F = F1 ⊗ I + I ⊗ F2"""
        F1 = self.kernel1.design_matrix()
        F2 = self.kernel2.design_matrix()
        return _prod_helper(F1, jnp.eye(F2.shape[0])) +\
               _prod_helper(jnp.eye(F1.shape[0]), F2)
    
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
    
    def process_noise(self,  X1: JAXArray, X2: JAXArray) -> JAXArray:
        """Q = Q1 ⊗ Q2"""
        return _prod_helper(
            self.kernel1.process_noise(X1, X2), 
            self.kernel2.process_noise(X1, X2)
        )

    def observation_model(self, X: JAXArray) -> JAXArray:
        """H = H1 ⊗ H2"""
        return _prod_helper(
            self.kernel1.observation_model(X),
            self.kernel2.observation_model(X),
        )

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
            self.coord_to_sortable(X1),
            self.coord_to_sortable(X2)
        )

    def process_noise(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return self.kernel.process_noise(
            self.coord_to_sortable(X1),
            self.coord_to_sortable(X2)
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

    def design_matrix(self) -> JAXArray:
        return jnp.array(
            [[0, 1], [-jnp.square(self.omega), -self.omega / self.quality]]
        )

    def stationary_covariance(self) -> JAXArray:
        return jnp.diag(jnp.array([1, jnp.square(self.omega)]))

    def observation_model(self, X: JAXArray) -> JAXArray:
        del X
        return jnp.array([self.sigma, 0])

    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        dt = X2 - X1
        w = self.omega
        q = self.quality

        def critical(dt: JAXArray) -> JAXArray:
            return jnp.exp(-w * dt) * jnp.array(
                [[1 + w * dt, -jnp.square(w) * dt], [dt, 1 - w * dt]]
            )

        def underdamped(dt: JAXArray) -> JAXArray:
            f = jnp.sqrt(jnp.maximum(4 * jnp.square(q) - 1, 0))
            arg = 0.5 * f * w * dt / q
            sin = jnp.sin(arg)
            cos = jnp.cos(arg)
            return jnp.exp(-0.5 * w * dt / q) * jnp.array(
                [
                    [cos + sin / f, -2 * q * w * sin / f],
                    [2 * q * sin / (w * f), cos - sin / f],
                ]
            )

        def overdamped(dt: JAXArray) -> JAXArray:
            f = jnp.sqrt(jnp.maximum(1 - 4 * jnp.square(q), 0))
            arg = 0.5 * f * w * dt / q
            sinh = jnp.sinh(arg)
            cosh = jnp.cosh(arg)
            return jnp.exp(-0.5 * w * dt / q) * jnp.array(
                [
                    [cosh + sinh / f, -2 * q * w * sinh / f],
                    [2 * q * sinh / (w * f), cosh - sinh / f],
                ]
            )

        return jax.lax.cond(
            jnp.allclose(q, 0.5),
            critical,
            lambda dt: jax.lax.cond(q > 0.5, underdamped, overdamped, dt),
            dt,
        )


class Exp(Quasisep):
    r"""A scalable implementation of :class:`tinygp.kernels.stationary.Exp`

    This kernel takes the form:

    .. math::

        k(\tau)=\sigma^2\,\exp\left(-\frac{\tau}{\ell}\right)

    for :math:`\tau = |x_i - x_j|`.

    Args:
        scale: The parameter :math:`\ell`.
        sigma (optional): The parameter :math:`\sigma`. Defaults to a value of
            1. Specifying the explicit value here provides a slight performance
            boost compared to independently multiplying the kernel with a
            prefactor.
    """

    scale: JAXArray | float
    sigma: JAXArray | float = eqx.field(default_factory=lambda: jnp.ones(()))

    def design_matrix(self) -> JAXArray:
        return jnp.array([[-1 / self.scale]])

    def stationary_covariance(self) -> JAXArray:
        return jnp.ones((1, 1))

    def observation_model(self, X: JAXArray) -> JAXArray:
        del X
        return jnp.array([self.sigma])

    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        dt = X2 - X1
        return jnp.exp(-dt[None, None] / self.scale)


class Matern32(Quasisep):
    r"""A scalable implementation of :class:`tinygp.kernels.stationary.Matern32`

    This kernel takes the form:

    .. math::

        k(\tau)=\sigma^2\,\left(1+f\,\tau\right)\,\exp(-f\,\tau)

    for :math:`\tau = |x_i - x_j|` and :math:`f = \sqrt{3} / \ell`.

    Args:
        scale: The parameter :math:`\ell`.
        sigma (optional): The parameter :math:`\sigma`. Defaults to a value of
            1. Specifying the explicit value here provides a slight performance
            boost compared to independently multiplying the kernel with a
            prefactor.
    """

    scale: JAXArray | float
    sigma: JAXArray | float = eqx.field(default_factory=lambda: jnp.ones(()))

    def noise(self) -> JAXArray:
        f = np.sqrt(3) / self.scale
        return 4 * f**3

    def design_matrix(self) -> JAXArray:
        f = np.sqrt(3) / self.scale
        return jnp.array([[0, 1], [-jnp.square(f), -2 * f]])

    def stationary_covariance(self) -> JAXArray:
        return jnp.diag(jnp.array([1, 3 / jnp.square(self.scale)]))

    def observation_model(self, X: JAXArray) -> JAXArray:
        return jnp.array([self.sigma, 0])

    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        dt = X2 - X1
        f = np.sqrt(3) / self.scale
        return jnp.exp(-f * dt) * jnp.array(
            [[1 + f * dt, -jnp.square(f) * dt], [dt, 1 - f * dt]]
        )


class Matern52(Quasisep):
    r"""A scalable implementation of :class:`tinygp.kernels.stationary.Matern52`

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
    sigma: JAXArray | float = eqx.field(default_factory=lambda: jnp.ones(()))

    def design_matrix(self) -> JAXArray:
        f = np.sqrt(5) / self.scale
        f2 = jnp.square(f)
        return jnp.array([[0, 1, 0], [0, 0, 1], [-f2 * f, -3 * f2, -3 * f]])

    def stationary_covariance(self) -> JAXArray:
        f = np.sqrt(5) / self.scale
        f2 = jnp.square(f)
        f2o3 = f2 / 3
        return jnp.array([[1, 0, -f2o3], [0, f2o3, 0], [-f2o3, 0, jnp.square(f2)]])

    def observation_model(self, X: JAXArray) -> JAXArray:
        del X
        return jnp.array([self.sigma, 0, 0])

    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        dt = X2 - X1
        f = np.sqrt(5) / self.scale
        f2 = jnp.square(f)
        d2 = jnp.square(dt)
        return jnp.exp(-f * dt) * jnp.array(
            [
                [
                    0.5 * f2 * d2 + f * dt + 1,
                    -0.5 * f * f2 * d2,
                    0.5 * f2 * f * dt * (f * dt - 2),
                ],
                [
                    dt * (f * dt + 1),
                    -f2 * d2 + f * dt + 1,
                    f2 * dt * (f * dt - 3),
                ],
                [
                    0.5 * d2,
                    0.5 * dt * (2 - f * dt),
                    0.5 * f2 * d2 - 2 * f * dt + 1,
                ],
            ]
        )


class Cosine(Quasisep):
    r"""A scalable implementation of :class:`tinygp.kernels.stationary.Cosine`

    This kernel takes the form:

    .. math::

        k(\tau)=\sigma^2\,\cos(-2\,\pi\,\tau/\ell)

    for :math:`\tau = |x_i - x_j|`.

    Args:
        scale: The parameter :math:`\ell`.
        sigma (optional): The parameter :math:`\sigma`. Defaults to a value of
            1. Specifying the explicit value here provides a slight performance
            boost compared to independently multiplying the kernel with a
            prefactor.
    """

    scale: JAXArray | float
    sigma: JAXArray | float = eqx.field(default_factory=lambda: jnp.ones(()))

    def design_matrix(self) -> JAXArray:
        f = 2 * np.pi / self.scale
        return jnp.array([[0, -f], [f, 0]])

    def stationary_covariance(self) -> JAXArray:
        return jnp.eye(2)

    def observation_model(self, X: JAXArray) -> JAXArray:
        return jnp.array([self.sigma, 0])

    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        dt = X2 - X1
        f = 2 * np.pi / self.scale
        cos = jnp.cos(f * dt)
        sin = jnp.sin(f * dt)
        return jnp.array([[cos, sin], [-sin, cos]])


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


class CARMA(Quasisep):
    r"""A continuous-time autoregressive moving average (CARMA) process kernel

    This process has the power spectrum density (PSD)

    .. math::

        P(\omega) = \sigma^2\,\frac{|\sum_{q} \beta_q\,(i\,\omega)^q|^2}{|\sum_{p}
            \alpha_p\,(i\,\omega)^p|^2}

    defined following Equation 1 in `Kelly et al. (2014)
    <https://arxiv.org/abs/1402.5978>`_, where :math:`\alpha_p` and :math:`\beta_0`
    are set to 1. In this implementation, we absorb :math:`\sigma` into the
    definition of :math:`\beta` parameters. That is :math:`\beta_{new}` =
    :math:`\beta * \sigma`.

    .. note::
        To construct a stationary CARMA kernel/process, the roots of the
        characteristic polynomials for Equation 1 in `Kelly et al. (2014)` must
        have negative real parts. This condition can be met automatically by
        requiring positive input parameters when instantiating the kernel using
        the :func:`init` method for CARMA(1,0), CARMA(2,0), and CARMA(2,1)
        models or by requiring positive input parameters when instantiating the
        kernel using the :func:`from_quads` method.

    .. note:: Implementation details

        The logic behind this implementation is simple---finding the correct
        combination of real/complex exponential kernels that resembles the
        autocovariance function of the CARMA model. Note that the order also
        matters. This task is achieved using the `acvf` method. Then the rest
        is copied from the `Exp` and `Celerite` kernel.

        Given the requirement of negative roots for stationarity, the
        `from_quads` method is implemented to facilitate consturcting
        stationary higher-order CARMA models beyond CARMA(2,1). The inputs for
        `from_quads` are the coefficients of the quadratic equations factorized
        out of the full characteristic polynomial. `poly2quads` is used to
        factorize a polynomial into a product of said quadractic equations, and
        `quads2poly` is used for the reverse process.

        One last trick is the use of `_real_mask`, `_complex_mask`, and
        `complex_select`, which are arrays of 0s and 1s. They are implemented
        to avoid control flows. More specifically, some intermediate quantities
        are computed regardless, but are only used if there is a matching real
        or complex exponential kernel for the specific CARMA kernel.

    Args:
        alpha: The parameter :math:`\alpha` in the definition above, exlcuding
            :math:`\alpha_p`. This should be an array of length `p`.
        beta: The product of parameters :math:`\beta` and parameter :math:`\sigma`
            in the definition above. This should be an array of length `q+1`,
            where `q+1 <= p`.
    """

    alpha: JAXArray
    beta: JAXArray
    sigma: JAXArray
    arroots: JAXArray
    acf: JAXArray
    _real_mask: JAXArray
    _complex_mask: JAXArray
    _complex_select: JAXArray
    obsmodel: JAXArray

    def __init__(self, alpha: Any, beta: Any):
        sigma = jnp.ones(())
        alpha = jnp.atleast_1d(jnp.asarray(alpha))
        beta = jnp.atleast_1d(jnp.asarray(beta))
        assert alpha.ndim == 1
        assert beta.ndim == 1
        p = alpha.shape[0]
        assert beta.shape[0] <= p

        # Find acvf using Eqn. 4 in Kelly+14, giving the correct combination of
        # real/complex exponential kernels
        arroots = carma_roots(jnp.append(alpha, 1.0))
        acf = carma_acvf(arroots, alpha, beta * sigma)

        # Mask for real/complex exponential kernels
        _real_mask = jnp.abs(arroots.imag) < 10 * jnp.finfo(arroots.imag.dtype).eps
        _complex_mask = ~_real_mask
        complex_idx = jnp.cumsum(_complex_mask) * _complex_mask
        _complex_select = _complex_mask * complex_idx % 2

        # Construct the obsservation model => real + complex
        om_real = jnp.sqrt(jnp.abs(acf.real))

        a, b, c, d = (
            2 * acf.real,
            2 * acf.imag,
            -arroots.real,
            -arroots.imag,
        )
        c2 = jnp.square(c)
        d2 = jnp.square(d)
        s2 = c2 + d2
        denom = jnp.where(_real_mask, 1.0, 2 * c * s2)
        h2_2 = d2 * (a * c - b * d) / denom
        h2 = jnp.sqrt(h2_2)
        denom = jnp.where(_real_mask, 1.0, d)
        h1 = (c * h2 - jnp.sqrt(a * d2 - s2 * h2_2)) / denom
        om_complex = jnp.array([h1, h2])

        # for complex roots, every conjugate pair match one full celerite term,
        # so, every other entry from om_complex is used.
        # same logic as for _complex_select
        self.obsmodel = jnp.where(_real_mask, om_real, jnp.ravel(om_complex)[::2])

        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.arroots = arroots
        self.acf = acf
        self._real_mask = _real_mask
        self._complex_mask = _complex_mask
        self._complex_select = _complex_select

    @classmethod
    def init(cls, alpha: JAXArray, beta: JAXArray) -> CARMA:
        return cls(alpha, beta)

    @classmethod
    def from_quads(
        cls, alpha_quads: JAXArray, beta_quads: JAXArray, beta_mult: JAXArray
    ) -> CARMA:
        r"""Construct a CARMA kernel using the roots of its characteristic polynomials

        The roots can be parameterized as the 0th and 1st order coefficients of a set
        of quadratic equations (2nd order coefficient equals 1). The product of
        those quadratic equations gives the characteristic polynomials of CARMA.
        The input of this method are said coefficients of the quadratic equations.
        See Equation 30 in `Kelly et al. (2014) <https://arxiv.org/abs/1402.5978>`_.
        for more detail.

        Args:
            alpha_quads: Coefficients of the auto-regressive (AR) quadratic
                equations corresponding to the :math:`\alpha` parameters. This should
                be an array of length `p`.
            beta_quads: Coefficients of the moving-average (MA) quadratic
                equations corresponding to the :math:`\beta` parameters. This should
                be an array of length `q`.
            beta_mult: A multiplier of the MA coefficients, equivalent to
                :math:`\beta_q`---the last entry of the :math:`\beta` parameters input
                to the :func:`init` method.
        """

        alpha_quads = jnp.atleast_1d(alpha_quads)
        beta_quads = jnp.atleast_1d(beta_quads)
        beta_mult = jnp.atleast_1d(beta_mult)

        alpha = carma_quads2poly(jnp.append(alpha_quads, jnp.array([1.0])))[:-1]
        beta = carma_quads2poly(jnp.append(beta_quads, beta_mult))

        return cls(alpha, beta)

    def design_matrix(self) -> JAXArray:
        # for real exponential components
        dm_real = jnp.diag(self.arroots.real * self._real_mask)

        # for complex exponential components
        dm_complex_diag = jnp.diag(self.arroots.real * self._complex_mask)

        # upper triangle entries
        dm_complex_u = jnp.diag((self.arroots.imag * self._complex_select)[:-1], k=1)

        return dm_real + dm_complex_diag + -dm_complex_u.T + dm_complex_u

    def stationary_covariance(self) -> JAXArray:
        p = self.acf.shape[0]

        # for real exponential components
        diag = jnp.diag(jnp.where(self.acf.real > 0, jnp.ones(p), -jnp.ones(p)))

        # for complex exponential components
        denom = jnp.where(self._real_mask, 1.0, self.arroots.imag)
        diag_complex = jnp.diag(
            2
            * jnp.square(
                self.arroots.real
                / denom
                * jnp.roll(self._complex_select, 1)
                * self._complex_mask
            )
        )
        c_over_d = self.arroots.real / denom

        # upper triangular entries
        sc_complex_u = jnp.diag((-c_over_d * self._complex_select)[:-1], k=1)

        return diag + diag_complex + sc_complex_u + sc_complex_u.T

    def observation_model(self, X: JAXArray) -> JAXArray:
        del X
        return self.obsmodel

    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        dt = X2 - X1
        c = -self.arroots.real
        d = -self.arroots.imag
        decay = jnp.exp(-c * dt)
        sin = jnp.sin(d * dt)

        tm_real = jnp.diag(decay * self._real_mask)
        tm_complex_diag = jnp.diag(decay * jnp.cos(d * dt) * self._complex_mask)
        tm_complex_u = jnp.diag(
            (decay * sin * self._complex_select)[:-1],
            k=1,
        )

        return tm_real + tm_complex_diag + -tm_complex_u.T + tm_complex_u


@jax.jit
def carma_roots(poly_coeffs: JAXArray) -> JAXArray:
    roots = jnp.roots(poly_coeffs[::-1], strip_zeros=False)
    return roots[jnp.argsort(roots.real)]


@jax.jit
def carma_quads2poly(quads_coeffs: JAXArray) -> JAXArray:
    """Expand a product of quadractic equations into a polynomial

    Args:
        quads_coeffs: The 0th and 1st order coefficients of the quadractic
            equations. The last entry is a multiplier, which corresponds
            to the coefficient of the highest order term in the output full
            polynomial.

    Returns:
        Coefficients of the full polynomial. The first entry corresponds to
        the lowest order term.
    """

    size = quads_coeffs.shape[0] - 1
    remain = size % 2
    nPair = size // 2
    mult_f = quads_coeffs[-1:]  # The coeff of highest order term in the output

    poly = jax.lax.cond(
        remain == 1,
        lambda x: jnp.array([1.0, x]),
        lambda _: jnp.array([0.0, 1.0]),
        quads_coeffs[-2],
    )
    poly = poly[-remain + 1 :]

    for p in jnp.arange(nPair):
        poly = jnp.convolve(
            poly,
            jnp.append(
                jnp.array([quads_coeffs[p * 2], quads_coeffs[p * 2 + 1]]),
                jnp.ones((1,)),
            )[::-1],
        )

    # the returned is low->high following Kelly+14
    return poly[::-1] * mult_f


def carma_poly2quads(poly_coeffs: JAXArray) -> JAXArray:
    """Factorize a polynomial into a product of quadratic equations

    Args:
        poly_coeffs: Coefficients of the input characteristic polynomial. The
            first entry corresponds to the lowest order term.

    Returns:
        The 0th and 1st order coefficients of the quadractic equations. The last
        entry is a multiplier, which corresponds to the coefficient of the highest
        order term in the full polynomial.
    """

    quads = jnp.empty(0)
    mult_f = poly_coeffs[-1]
    roots = carma_roots(poly_coeffs / mult_f)
    odd = bool(len(roots) & 0x1)

    rootsComp = roots[roots.imag != 0]
    rootsReal = roots[roots.imag == 0]
    nCompPair = len(rootsComp) // 2
    nRealPair = len(rootsReal) // 2

    for i in range(nCompPair):
        root1 = rootsComp[i]
        root2 = rootsComp[i + 1]
        quads = jnp.append(quads, (root1 * root2).real)
        quads = jnp.append(quads, -(root1.real + root2.real))

    for i in range(nRealPair):
        root1 = rootsReal[i]
        root2 = rootsReal[i + 1]
        quads = jnp.append(quads, (root1 * root2).real)
        quads = jnp.append(quads, -(root1.real + root2.real))

    if odd:
        quads = jnp.append(quads, -rootsReal[-1].real)

    return jnp.append(quads, jnp.array(mult_f))


def carma_acvf(arroots: JAXArray, arparam: JAXArray, maparam: JAXArray) -> JAXArray:
    r"""Compute the coefficients of the autocovariance function (ACVF)

    Args:
        arroots: The roots of the autoregressive characteristic polynomial.
        arparam: :math:`\alpha` parameters
        maparam: :math:`\beta` parameters

    Returns:
        ACVF coefficients, each entry corresponds to one root.
    """
    from jax._src import dtypes  # type: ignore

    arparam = jnp.atleast_1d(arparam)
    maparam = jnp.atleast_1d(maparam)

    complex_dtype = dtypes.to_complex_dtype(arparam.dtype)

    p = arparam.shape[0]
    q = maparam.shape[0] - 1
    sigma = maparam[0]

    # normalize beta_0 to 1
    maparam = maparam / sigma

    # init acf product terms
    num_left = jnp.zeros(p, dtype=complex_dtype)
    num_right = jnp.zeros(p, dtype=complex_dtype)
    denom = -2 * arroots.real + jnp.zeros_like(arroots) * 1j

    for k in range(q + 1):
        num_left += maparam[k] * jnp.power(arroots, k)
        num_right += maparam[k] * jnp.power(jnp.negative(arroots), k)

    root_idx = jnp.arange(p)
    for j in range(1, p):
        root_k = arroots[jnp.roll(root_idx, j)]
        denom *= (root_k - arroots) * (jnp.conj(root_k) + arroots)

    return sigma**2 * num_left * num_right / denom