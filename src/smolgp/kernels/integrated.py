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

import smolgp.kernels as base_kernels
from smolgp.kernels import StateSpaceModel
from smolgp.helpers import Q_from_VanLoan, Phibar_from_VanLoan


class IntegratedStateSpaceModel(StateSpaceModel):
    """
    A generic class that augments a :class:`StateSpaceModel` 
    object (which has state `x`) with an integral state `z`, 
    to model the joint state `X = [x; z]`.

    The coordinates for an integrated model should be a tuple of
        X = (t, delta, instid), 
    where `t` is the usual coordinate (e.g. time) at the measurements (midpoints),
    `delta` is the integration range (e.g. exposure time) for each measurement,
    and `instid` is an index encoding which instrument the measurement corresponds to.
    """

    dimension: float = eqx.field(static=True) # dimensionality of the augmented space
    num_insts: int   = eqx.field(static=True) # number of integral states
    base_model: StateSpaceModel  # the underlying (non-integrated) SSM (dimension=d)
    d: int = eqx.field(static=True)      # dimension of the base model
    I: JAXArray  # identity matrix of size dxd
    Z: JAXArray  # zero matrix of size dxd

    def __init__(self, base_model: StateSpaceModel, num_insts: int = 1):
        """
        Augment the base_model (:class:`StateSpaceModel`) with `num_insts` integral states.
        """
        self.base_model = base_model
        self.num_insts  = num_insts

        # set dimension of integrated model from num_insts and base model
        self.d = self.base_model.dimension 
        self.dimension = self.d * (1 + self.num_insts)

        # Useful identity and zero matrices
        self.I = jnp.eye(self.d)
        self.Z = jnp.zeros((self.d,self.d))

    def coord_to_sortable(self, X: JAXArray) -> JAXArray:
        """A helper function used to convert coordinates to sortable 1-D values

        If X is a tuple, e.g. of (time, delta, instid), this assumes the first coordinate is the sortable one
        """
        if isinstance(X, tuple) or isinstance(X, list):
            return X[0]
        else:
            return X

    def design_matrix(self) -> JAXArray:
        """The augmented design (also called the feedback) matrix for the process, $F$"""
        F = self.base_model.design_matrix()
        F_aug = [[F] + [self.Z]*self.num_insts]
        for _ in range(self.num_insts):
            F_aug.append([self.I] + [self.Z]*self.num_insts)
        F_aug = jnp.block(F_aug)
        return F_aug

    def stationary_covariance(self) -> JAXArray:
        """The augmented stationary covariance of the process, Pinf"""
        Pinf = self.base_model.stationary_covariance()
        Pinf_aug = jnp.diag(jnp.ones(self.dimension)).at[:self.d,:self.d].set(Pinf)
        return Pinf_aug

    def observation_model(self, X: JAXArray) -> JAXArray:
        """The augmented observation model for the process, $H$"""
        if isinstance(X, tuple) or isinstance(X, list):
            # Observing integral state (z) with exposure time (delta)
            t, delta, instid = X
            
            ## TODO: make sure this works for multivariate data
            # H_base = self.base_model.observation_model(t)
            # H_z = H_base/delta # observe the average value over exposure
            # H_aug = jnp.zeros((H_base.shape[0], self.dimension))
            # H_aug = jax.lax.dynamic_update_slice(H_aug, H_z, (self.d*(1+instid),))
            # return H_aug
            
            # Hardcoded 1-D version for now
            H_z = jnp.zeros(self.d).at[0].set(1)/delta
            H_aug = jnp.zeros(self.dimension)
            H_aug = jax.lax.dynamic_update_slice(H_aug, H_z, (self.d*(1+instid),))
            return jnp.array([H_aug])
        else:
            # Observing only the base state (x)
            # H_x = self.base_model.observation_model(X) # TODO: use this and get the shapes riht
            H_x = jnp.zeros(self.d).at[0].set(1)  # hardcoded 1-D version for now
            H_aug = jnp.zeros(self.dimension)
            H_aug = jax.lax.dynamic_update_slice(H_aug, H_x, (0,))
            return jnp.array([H_aug])

    def noise(self) -> JAXArray:
        ''' The spectral density of the white noise process, $Q_c$ '''
        return self.base_model.noise()
    
    def noise_effect_matrix(self) -> JAXArray:
        ''' The augmented noise effect matrix, $L$'''
        L = self.base_model.noise_effect_matrix()
        L_aug = jnp.vstack([L] + [jnp.zeros_like(L)]*self.num_insts)
        return L_aug
    
    def integrated_transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        """
        The integrated transition matrix between two states at coordinates X1 and X2, $A_k$

        By default uses the Van Loan method to compute Phibar = ∫0^dt exp(F s) ds

        Overload this method if you wish to define the integrated transition matrix analytically.
        """
        F = self.base_model.design_matrix()
        t1 = self.coord_to_sortable(X1)
        t2 = self.coord_to_sortable(X2)
        dt = t2 - t1
        return Phibar_from_VanLoan(F, dt)
        
    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        """
        The augmented transition matrix between two states at coordinates X1 and X2, $A_k$
        """
        t1 = self.coord_to_sortable(X1)
        t2 = self.coord_to_sortable(X2)
        PHI    = self.base_model.transition_matrix(t1, t2)
        INTPHI = self.integrated_transition_matrix(t1, t2)
        PHIAUG = jnp.eye(self.dimension)
        PHIAUG = PHIAUG.at[:self.d, :self.d].set(PHI)
        for i in range(self.num_insts):
            PHIAUG = PHIAUG.at[(1+i)*self.d: (2+i)*self.d, :self.d].set(INTPHI)
        return PHIAUG
 
    def process_noise(self, X1: JAXArray, X2: JAXArray, use_van_loan=False) -> JAXArray:
        """
        The augmented process noise matrix $Q_k$

        Default behavior computes Q from the Van Loan 
        matrix exponential involving F, L, and Qc

        Overload this method if you wish to define the process noise analytically.
        """
        t1 = self.coord_to_sortable(X1)
        t2 = self.coord_to_sortable(X2)
        dt = t2 - t1
        F_aug = self.design_matrix()
        L_aug = self.noise_effect_matrix()
        Qc = self.noise()
        return Q_from_VanLoan(F_aug,L_aug,Qc,dt)

    def reset_matrix(self, instid: int = 0) -> JAXArray:
        """
        The reset matrix, RESET_k,for instrument `instid` (0-indexed)

        By default, resets only the integral states to zero.
        Overload this method if you wish to define a different reset behavior.
        """
        diag = jnp.ones(self.d*(self.num_insts+1))
        diag = jax.lax.dynamic_update_slice(diag, jnp.zeros(self.d), (self.d*(1+instid),))
        return jnp.diag(diag)




class IntegratedSHO(IntegratedStateSpaceModel):
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
    eta    : JAXArray | float

    def __init__(self, omega: JAXArray | float,
                 quality: JAXArray | float,
                 sigma: JAXArray | float = jnp.ones(()),
                 num_insts: int = 1,
                 ):
        
        # SHO parameters
        self.omega   = omega
        self.quality = quality
        self.sigma   = sigma
        self.eta = jnp.sqrt(jnp.abs(1-1/(4*self.quality**2)))

        base_model = base_kernels.SHO(omega=self.omega, quality=self.quality, sigma=self.sigma)
        super().__init__(base_model=base_model, num_insts=num_insts)

    def integrated_transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        """The integrated transition matrix Phibar for the SHO process"""

        # Shorthand notations
        n = self.eta
        w = self.omega
        q = self.quality
        a = -0.5*w/q
        b = n*w
        a2plusb2 = jnp.square(a) + jnp.square(b)
        A = 1/(2*n*q)
        B = 1/(n*w) # = 1/b
        C = -w/n

        def critical(t1: JAXArray, t2: JAXArray) -> JAXArray:
            ## TODO: returning numerical result until we do this integral by hand
            F = self.base_model.design_matrix()
            return Phibar_from_VanLoan(F, t2-t1)

        def underdamped(t1: JAXArray, t2: JAXArray) -> JAXArray:
     
            def Int_ecos(t):
                return jnp.exp(a*t) * (a*jnp.cos(b*t) + b*jnp.sin(b*t))

            def Int_esin(t):
                return jnp.exp(a*t) * (a*jnp.sin(b*t) - b*jnp.cos(b*t))
            
            Ic = Int_ecos(t2) - Int_ecos(t1)
            Is = Int_esin(t2) - Int_esin(t1)
            Phibar11 = Ic + A*Is
            Phibar12 = B * Is
            Phibar21 = C * Is
            Phibar22 = Ic - A*Is
 
            return jnp.array([
                            [Phibar11, Phibar12],
                            [Phibar21, Phibar22]
                        ]) / a2plusb2

        def overdamped(t1: JAXArray, t2: JAXArray) -> JAXArray:
            ## TODO: returning numerical result until we do this integral by hand
            F = self.base_model.design_matrix()
            return Phibar_from_VanLoan(F, t2-t1)
        
        # Return the appropriate form based on quality factor
        t1 = self.coord_to_sortable(X1)
        t2 = self.coord_to_sortable(X2)

        return jax.lax.cond(
            jnp.allclose(q, 0.5),
            critical,
            lambda t1,t2: jax.lax.cond(q > 0.5, underdamped, overdamped, t1,t2),
            t1,t2,
        )
    


### TODO: is it just these for all the named kernels?
# class IntegratedMatern32(IntegratedStateSpaceModel):
#     def __init__(self, **kwargs):
#         base_model = StateSpaceModel.Matern32(**kwargs)
#         super().__init__(base_model=base_model, **kwargs)

