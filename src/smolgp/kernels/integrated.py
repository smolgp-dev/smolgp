# TODO: 
# 1. copy base.py kernel object style
# 2. add integrated_transition_matrix and integrated_process_noise
# 3. add attribute/property for num_insts 
# 4. define each of the usual matrix components to be the augmented version
#    e.g. stationary_covariance --> BlockDiag(sho.stationary_covariance, identity)

# in the solver, user will have passed t, texp, instid, and y
# from there, stateid will get auto-created according to t and texp


"""
TODO: docstring for integrated kernel
"""

from __future__ import annotations

__all__ = [
    "IntegratedSHO",
]

from abc import abstractmethod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import expm

from tinygp.helpers import JAXArray

from smolgp.kernels import StateSpaceModel
from smolgp.helpers import Q_from_VanLoan

class IntegratedStateSpaceModel(StateSpaceModel):
    """
    A generic class that augments a :class:`StateSpaceModel` 
    obejct (which has state `x`) with an integral state `z`, 
    to model the joint state `X = [x; z]`.
    """

    ## Same functions as StateSpaceModel
    ## put into augmented form, plus:
    ##   - integrated_transition_matrix
    ##   - integrated_process_noise
    def __init__(self):
        pass



    dimension: JAXArray | float = eqx.field(static=True) # dimensionality $d$ of the state vector

    @abstractmethod
    def design_matrix(self) -> JAXArray:
        """The design (also called the feedback) matrix for the process, $F$"""
        raise NotImplementedError

    @abstractmethod
    def stationary_covariance(self) -> JAXArray:
        """The stationary covariance of the process, Pinf"""
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





    


class IntegratedSHO(StateSpaceModel):
    r"""The damped, driven simple harmonic oscillator kernel 
        integrated over a finite time range, :math:`\delta`.

    This form of the kernel was introduced by `Luhn et al. (in prep)
    <LINK>`_, and it takes the form:

    Args:
        omega: The parameter :math:`\omega_0`.
        quality: The parameter :math:`Q`.
        sigma (optional): The parameter :math:`\sigma`. Defaults to a value of
            1. Specifying the explicit value here provides a slight performance
            boost compared to independently multiplying the kernel with a
            prefactor.
    """

    omega  : JAXArray | float
    quality: JAXArray | float
    sigma  : JAXArray | float = eqx.field(default_factory=lambda: jnp.ones(()))
    dimension: JAXArray | float = eqx.field(init=False, default=2)
    eta : JAXArray | float

    def __init__(self, omega: JAXArray | float,
                 quality: JAXArray | float,
                 sigma: JAXArray | float = jnp.ones(()),
                 **kwargs):
        
        # SHO parameters
        self.omega   = omega
        self.quality = quality
        self.sigma   = sigma


        self.eta = jnp.sqrt(jnp.abs(1-1/(4*self.quality**2)))


    # def design_matrix(self) -> JAXArray:
    #     """The design (also called the feedback) matrix for the SHO process, F"""
    #     return jnp.array(
    #         [[0, 1], [-jnp.square(self.omega), -self.omega / self.quality]]
    #     )
    
    # def stationary_covariance(self) -> JAXArray:
    #     """The stationary covariance of the SHO process, Pinf"""
    #     return jnp.diag(jnp.square(self.sigma) * jnp.array([1, jnp.square(self.omega)]))

    # def noise(self) -> JAXArray:
    #     """The scalar Qc for the SHO process"""
    #     omega3 = jnp.power(self.omega,3)
    #     return jnp.array([[2*omega3*jnp.square(self.sigma)/self.quality]])

    # def observation_model(self, X: JAXArray) -> JAXArray:
    #     """ The observation model H for the SHO process """
    #     del X
    #     return jnp.array([[1, 0]])

    # def noise_effect(self) -> JAXArray:
    #     """ The noise effect matrix L for the SHO process """
    #     return jnp.array([[0], [1]])
    
    # def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
    #     """The transition matrix A_k for the SHO process"""
    #     dt = X2 - X1
    #     n = self.eta
    #     w = self.omega
    #     q = self.quality

    #     def critical(dt: JAXArray) -> JAXArray:
    #         return jnp.exp(-w * dt) * jnp.array(
    #             [[1 + w * dt, dt], [-jnp.square(w) * dt, 1 - w * dt]]
    #         )

    #     def underdamped(dt: JAXArray) -> JAXArray:
    #         f = 2*n*q
    #         x = n*w*dt
    #         sin = jnp.sin(x)
    #         cos = jnp.cos(x)
    #         return jnp.exp(-0.5*w*dt/q) * jnp.array(
    #             [
    #                 [cos+sin/f, sin/(w*n)],
    #                 [-w*sin/n , cos-sin/f]
    #             ]
    #         )

    #     def overdamped(dt: JAXArray) -> JAXArray:
    #         f = 2*n*q
    #         x = n*w*dt
    #         sinh = jnp.sinh(x)
    #         cosh = jnp.cosh(x)
    #         return jnp.exp(-0.5*w*dt/q) * jnp.array(
    #             [
    #                 [cosh+sinh/f, sinh/(w*n)],
    #                 [-w*sinh/n  , cosh-sinh/f]
    #             ]
    #         )
        
    #     return jax.lax.cond(
    #         jnp.allclose(q, 0.5),
    #         critical,
    #         lambda dt: jax.lax.cond(q > 0.5, underdamped, overdamped, dt),
    #         dt,
    #     )

    # def process_noise(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
    #     """The process noise Q_k for the SHO process"""
    #     dt = X2 - X1
    #     n = self.eta
    #     w = self.omega
    #     q = self.quality
    #     assert q>0.5 # TODO: currently only for Q>1/2

    #     def critical(dt: JAXArray) -> JAXArray:
    #         Pinf = self.stationary_covariance()
    #         A = self.transition_matrix(0,dt)
    #         return Pinf - A @ Pinf @ A.T

    #     def underdamped(dt: JAXArray) -> JAXArray:
    #         f = 2*n*q; q2 = jnp.square(q); n2 = jnp.square(n); w2 = jnp.square(w)
    #         a = w*dt/q # argument in exponential
    #         x = n*w*dt # argument in sin/cos
    #         exp = jnp.exp(a)
    #         sin = jnp.sin(x)
    #         sin2 = jnp.sin(2*x)
    #         sinsq = jnp.square(sin)
    #         Q11 = exp - 1 - sin2/f - sinsq/(2*n2*q2)
    #         Q12 = Q21 = w*sinsq/(n2*q)
    #         Q22 = w2 * (exp - 1 + sin2/f - sinsq/(2*n2*q2) )
    #         return jnp.square(self.sigma) * jnp.exp(-a) * jnp.array(
    #             [
    #                 [Q11, Q12],
    #                 [Q21, Q22]
    #             ]
    #         ) 

    #     def overdamped(dt: JAXArray) -> JAXArray:
    #         Pinf = self.stationary_covariance()
    #         A = self.transition_matrix(0,dt)
    #         return Pinf - A @ Pinf @ A.T
            
    #     return jax.lax.cond(
    #         jnp.allclose(q, 0.5),
    #         critical,
    #         lambda dt: jax.lax.cond(q > 0.5, underdamped, overdamped, dt),
    #         dt,
    #     )