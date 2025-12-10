# TODO:
# 1. copy base.py kernel object style
# 2. add integrated_transition_matrix and integrated_process_noise
# 3. add attribute/property for num_insts
# 4. define each of the usual matrix components to be the augmented version
#    e.g. stationary_covariance --> BlockDiag(sho.stationary_covariance, identity)

# in the solver, user will have passed t, texp, instid, and y
# from there, stateid will get auto-created according to t and texp

"""
These kernels are compatible with :class:`smolgp.solvers.integrated.IntegratedStateSpaceSolver`,
which uses Bayesian filtering and smoothing algorithms to perform scalable GP
inference. (see :ref:`api-solvers-statespace` for more technical details).
On GPU, a performance boost may be observed for large datasets by using the
:class:`smolgp.solvers.parallel.ParallelStateSpaceSolver` class.

Like the quasisep kernels, these methods are experimental, so you may find
the documentation patchy in places. You are encouraged to `open issues or
pull requests <https://github.com/rrubenza/smolgp/issues>`_ as you find gaps.
"""

from __future__ import annotations

__all__ = [
    "IntegratedSHO",
    "IntegratedExp",
    "IntegratedMatern32",
    "IntegratedMatern52",
    "IntegratedCosine",
]

import equinox as eqx
import jax
import jax.numpy as jnp
from functools import partial

from tinygp.helpers import JAXArray

import smolgp.kernels
from smolgp.kernels import StateSpaceModel
from smolgp.helpers import Phibar_from_VanLoan


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

    base_model: StateSpaceModel  # the base (non-integrated) SSM
    num_insts: int = eqx.field(static=True)  # number of integral states

    @property
    def d(self) -> int:
        """The dimension of the base (non-integrated) state space model"""
        return self.base_model.dimension

    @property
    def dimension(self) -> int:
        """The dimension of the augmented state space model"""
        return self.d + self.num_insts

    def coord_to_sortable(self, X: JAXArray) -> JAXArray:
        """
        A helper function used to convert coordinates to sortable 1-D values

        If X is a tuple, e.g. of (time, delta, instid), this assumes the first coordinate is the sortable one
        """
        if isinstance(X, tuple):
            return X[0]
        else:
            return X

    def design_matrix(self) -> JAXArray:
        """The augmented design (also called the feedback) matrix for the process, $F$"""
        F = self.base_model.design_matrix()
        F_aug = jnp.zeros((self.dimension, self.dimension))
        F_aug = F_aug.at[: self.d, : self.d].set(F)
        for i in range(self.num_insts):
            F_aug = F_aug.at[self.d + i, 0].set(1.0)
        return F_aug

    def stationary_covariance(self) -> JAXArray:
        """The augmented stationary covariance of the process, Pinf"""
        Pinf = self.base_model.stationary_covariance()
        Pinf_aug = jnp.diag(jnp.ones(self.dimension)).at[: self.d, : self.d].set(Pinf)
        return Pinf_aug

    def observation_matrix(self, X: JAXArray) -> JAXArray:
        """The augmented observation model for the process, $H$"""

        ## TODO: make sure this works for multivariate data, e.g. like:
        # H_base = self.base_model.observation_model(t)
        # H_z = H_base/delta # observe the average value over exposure
        # H_aug = jnp.zeros((H_base.shape[0], self.dimension))
        # H_aug = jax.lax.dynamic_update_slice(H_aug, H_z, (self.d*(1+instid),))
        ## Below is hardcoded for 1-D data

        def H_integral(t: JAXArray, delta: JAXArray, instid: int) -> JAXArray:
            """Observation model for integral state"""
            H_z = jnp.array([1.0 / delta])
            H_aug = jnp.zeros(self.dimension)
            H_aug = jax.lax.dynamic_update_slice(H_aug, H_z, (self.d + instid,))
            return H_aug

        def H_latent(t: JAXArray, instid: int) -> JAXArray:
            """Observation model for latent (non-integral) state"""
            # H_x = self.base_model.observation_model(X) # TODO: use this to get the shapes right
            H_x = jnp.zeros(self.d).at[0].set(1)  # hardcoded 1-D version for now
            H_aug = jnp.zeros(self.dimension)
            H_aug = jax.lax.dynamic_update_slice(H_aug, H_x, (0,))
            return H_aug

        if isinstance(X, tuple) or isinstance(X, list):
            # Observing integral state (z) with exposure time (delta)
            t, delta, instid = X
            H_aug = jax.lax.cond(
                delta > 0,
                lambda _: H_integral(t, delta, instid),
                lambda _: H_latent(t, instid),
                operand=None,
            )
        else:
            # default to latent state if no exposure time provided
            H_aug = H_latent(X, instid=0)

        return jnp.array([H_aug])

    def noise(self) -> JAXArray:
        """The spectral density of the white noise process, $Q_c$"""
        return self.base_model.noise()

    def noise_effect_matrix(self) -> JAXArray:
        """The augmented noise effect matrix, $L$"""
        L = self.base_model.noise_effect_matrix()
        L_aug = jnp.vstack([L] + [0.0] * self.num_insts)
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
        PHI = self.base_model.transition_matrix(t1, t2)
        INTPHI = self.integrated_transition_matrix(t1, t2)[0, :]
        PHIAUG = jnp.eye(self.dimension)
        PHIAUG = PHIAUG.at[: self.d, : self.d].set(PHI)
        for i in range(self.num_insts):
            PHIAUG = PHIAUG.at[self.d + i : self.d + i + 1, : self.d].set(INTPHI)
        return PHIAUG

    def integrated_process_noise(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        """
        Computes the submatrices Qaug12, Qaug21, and Qaug22
        needed to assemble the augmented process noise matrix.

        By default uses the Van Loan method to compute these submatrices.
        Overload this method if you wish to define these submatrices analytically.
        """

        t1 = self.coord_to_sortable(X1)
        t2 = self.coord_to_sortable(X2)
        dt = t2 - t1
        F = self.base_model.design_matrix()
        L = self.base_model.noise_effect_matrix()
        Qc = self.base_model.noise()

        vanloan = smolgp.helpers.VanLoan(F, L, Qc, dt)
        F3 = vanloan["F3"]
        H2 = vanloan["H2"]
        K1 = vanloan["K1"]

        M = F3.T @ H2
        W = (F3.T @ K1) + (F3.T @ K1).T

        Qaug12 = M[:, :1]
        Qaug21 = Qaug12.T
        Qaug22 = W[:1, :1]
        return Qaug12, Qaug21, Qaug22

    # @partial(
    #     jax.jit,
    #     static_argnames=("force_numerical"),
    # )
    def process_noise(
        self, X1: JAXArray, X2: JAXArray, force_numerical: bool = False
    ) -> JAXArray:
        """
        The augmented process noise matrix $Q_k$

        Default behavior computes Q from the Van Loan
        matrix exponential involving F, L, and Qc

        Overload this method if you wish to define the
        integrated process noise analytically.
        """

        t1 = self.coord_to_sortable(X1)
        t2 = self.coord_to_sortable(X2)
        dt = t2 - t1
        if force_numerical:
            Qaug12, Qaug21, Qaug22 = super(type(self), self).integrated_process_noise(
                X1, X2
            )
        else:
            Qaug12, Qaug21, Qaug22 = self.integrated_process_noise(X1, X2)
        Qbase = self.base_model.process_noise(0, dt)
        QAUG = jnp.tile(Qaug22, (self.dimension, self.dimension))
        QAUG = QAUG.at[: self.d, : self.d].set(Qbase)
        for i in range(self.num_insts):
            QAUG = QAUG.at[: self.d, self.d + i : self.d + i + 1].set(Qaug12)
            QAUG = QAUG.at[self.d + i : self.d + i + 1, : self.d].set(Qaug21)
        return QAUG

    def reset_matrix(self, instid: int = 0) -> JAXArray:
        """
        The reset matrix, RESET_k,for instrument `instid` (0-indexed)

        By default, resets only the integral states to zero.
        Overload this method if you wish to define a different reset behavior.
        """
        diag = jnp.ones(self.dimension)
        diag = jax.lax.dynamic_update_slice(diag, jnp.array([0.0]), (self.d + instid,))
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

    omega: JAXArray | float
    quality: JAXArray | float
    sigma: JAXArray | float = eqx.field(default_factory=lambda: jnp.ones(()))
    eta: JAXArray | float

    def __init__(
        self,
        omega: JAXArray | float,
        quality: JAXArray | float,
        sigma: JAXArray | float = jnp.ones(()),
        num_insts: int = 1,
        name: str = "IntegratedSHO",
        **kwargs,
    ):
        self.num_insts = num_insts
        self.name = name

        # SHO parameters
        self.omega = omega
        self.quality = quality
        self.sigma = sigma
        self.eta = jnp.sqrt(jnp.abs(1 - 1 / (4 * self.quality**2)))

        # Base model
        self.base_model = smolgp.kernels.SHO(
            omega=self.omega, quality=self.quality, sigma=self.sigma
        )

    def integrated_transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        """The integrated transition matrix Phibar for the SHO process"""

        # Shorthand notations
        n = self.eta
        w = self.omega
        q = self.quality
        a = -0.5 * w / q
        b = n * w
        a2plusb2 = jnp.square(a) + jnp.square(b)
        A = 1 / (2 * n * q)
        B = 1 / (n * w)  # = 1/b
        C = -w / n

        def critical(t1: JAXArray, t2: JAXArray) -> JAXArray:
            ## TODO: returning numerical result until we do this integral by hand
            F = self.base_model.design_matrix()
            return Phibar_from_VanLoan(F, t2 - t1)

        def underdamped(t1: JAXArray, t2: JAXArray) -> JAXArray:
            ## General integral from t1->t2:
            # def Int_ecos(t):
            #     return jnp.exp(a * t) * (a * jnp.cos(b * t) + b * jnp.sin(b * t))
            # def Int_esin(t):
            #     return jnp.exp(a * t) * (a * jnp.sin(b * t) - b * jnp.cos(b * t))
            # Ic = Int_ecos(t2) - Int_ecos(t1)
            # Is = Int_esin(t2) - Int_esin(t1)
            # Phibar11 = Ic + A * Is
            # Phibar12 = B * Is
            # Phibar21 = C * Is
            # Phibar22 = Ic - A * Is

            ## Paper version: hardcoded for t1=0, dt=t2-t1
            dt = t2 - t1
            arg = b * dt
            exp = jnp.exp(a * dt)
            sin = jnp.sin(arg)
            cos = jnp.cos(arg)
            asin = a * sin
            bsin = b * sin
            acos = a * cos
            bcos = b * cos
            Ic = acos + bsin
            Is = asin - bcos
            Phibar11 = exp * (Ic + A * Is) - (a - A * b)
            Phibar12 = B * (Is * exp + b)
            Phibar21 = C * (Is * exp + b)
            Phibar22 = exp * (Ic - A * Is) - (a + A * b)
            return jnp.array([[Phibar11, Phibar12], [Phibar21, Phibar22]]) / a2plusb2

        def overdamped(t1: JAXArray, t2: JAXArray) -> JAXArray:
            ## TODO: returning numerical result until we do this integral by hand
            F = self.base_model.design_matrix()
            return Phibar_from_VanLoan(F, t2 - t1)

        # Return the appropriate form based on quality factor
        t1 = self.coord_to_sortable(X1)
        t2 = self.coord_to_sortable(X2)

        return jax.lax.cond(
            jnp.allclose(q, 0.5),
            critical,
            lambda t1, t2: jax.lax.cond(q > 0.5, underdamped, overdamped, t1, t2),
            t1,
            t2,
        )

    def integrated_process_noise(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        """The integrated process noise submatrices for the SHO process"""
        t1 = self.coord_to_sortable(X1)
        t2 = self.coord_to_sortable(X2)
        dt = t2 - t1
        n = self.eta
        w = self.omega
        q = self.quality
        a = -0.5 * w / q
        b = n * w
        sigma2 = jnp.square(self.sigma)
        A = 1 / (2 * n * q)

        def critical(dt: JAXArray) -> JAXArray:
            # TODO: returning numerical result until we do this integral by hand
            return super(type(self), self).integrated_process_noise(0, dt)

        def underdamped(dt: JAXArray) -> JAXArray:
            x = a * dt
            arg = b * dt
            w2 = jnp.square(w)
            q2 = jnp.square(q)
            q4 = jnp.square(q2)
            exp = jnp.exp(x)
            exp2 = jnp.exp(2 * x)
            exp2m1 = jnp.expm1(2 * x)
            sin = jnp.sin(arg)
            cos = jnp.cos(arg)
            sinsq = jnp.square(jnp.sin(arg))
            sin2 = jnp.sin(2 * arg)
            cos2 = jnp.cos(2 * arg)
            A2 = jnp.square(A)
            iQ12_1 = jnp.square(exp * (cos + A * sin) - 1) / (q * w)
            iQ12_2 = A * exp * (4 * sin - exp * sin2) - 2 * A2 * exp2 * sinsq + exp2m1

            part1 = 8 * q * w * dt + 4 * q2 - 12
            part2 = A2 * exp2 * (cos2 - 16 * q4)
            part3_1 = 16 * exp * (cos + (1 - 2 * q2) * A * sin)
            part3_2 = exp2 * ((1 - 3 * A2) / A * sin2 - 3 * cos2)
            part3 = part3_1 + part3_2
            iQ22 = 1 / (4 * q2 * w2) * (part1 + part2 + part3)
            iQ22 = jnp.maximum(iQ22, 0.0)  # prevent underflows at dt=0

            Qaug12 = sigma2 * jnp.array([[iQ12_1], [iQ12_2]])
            Qaug21 = Qaug12.T
            Qaug22 = sigma2 * jnp.array([[iQ22]])
            return Qaug12, Qaug21, Qaug22

        def overdamped(dt: JAXArray) -> JAXArray:
            ## TODO: returning numerical result until we do this integral by hand
            return super(type(self), self).integrated_process_noise(0, dt)

        return jax.lax.cond(
            jnp.allclose(q, 0.5),
            critical,
            lambda dt: jax.lax.cond(q > 0.5, underdamped, overdamped, dt),
            dt,
        )


## TODO: is there a way to automate this? aka make a generic IntegratedKernel class...
## Default constructions for all kernels in smolgp.kernels.base
## IntegratedStateSpaceModel parent class will handle the augmentation
## All component matrices will be auto-generated numerically (e.g. A via expm, Q via Van Loan)
class IntegratedExp(IntegratedStateSpaceModel):
    scale: JAXArray | float
    sigma: JAXArray | float = eqx.field(default_factory=lambda: jnp.ones(()))
    lam: JAXArray | float

    def __init__(
        self,
        scale: JAXArray | float,
        sigma: JAXArray | float = jnp.ones(()),
        num_insts: int = 1,
        name: str = "IntegratedExp",
        **kwargs,
    ):
        self.scale = scale
        self.sigma = sigma
        self.name = name
        self.num_insts = num_insts
        self.base_model = smolgp.kernels.Exp(scale=self.scale, sigma=self.sigma)
        self.lam = self.base_model.lam


class IntegratedMatern32(IntegratedStateSpaceModel):
    scale: JAXArray | float
    sigma: JAXArray | float = eqx.field(default_factory=lambda: jnp.ones(()))
    lam: JAXArray | float

    def __init__(
        self,
        scale: JAXArray | float,
        sigma: JAXArray | float = jnp.ones(()),
        num_insts: int = 1,
        name: str = "IntegratedMatern32",
        **kwargs,
    ):
        self.scale = scale
        self.sigma = sigma
        self.name = name
        self.num_insts = num_insts
        self.base_model = smolgp.kernels.Matern32(scale=self.scale, sigma=self.sigma)
        self.lam = self.base_model.lam


class IntegratedMatern52(IntegratedStateSpaceModel):
    scale: JAXArray | float
    sigma: JAXArray | float = eqx.field(default_factory=lambda: jnp.ones(()))
    lam: JAXArray | float

    def __init__(
        self,
        scale: JAXArray | float,
        sigma: JAXArray | float = jnp.ones(()),
        num_insts: int = 1,
        name: str = "IntegratedMatern52",
        **kwargs,
    ):
        self.scale = scale
        self.sigma = sigma
        self.name = name
        self.num_insts = num_insts
        self.base_model = smolgp.kernels.Matern52(scale=self.scale, sigma=self.sigma)
        self.lam = self.base_model.lam


class IntegratedCosine(IntegratedStateSpaceModel):
    scale: JAXArray | float
    sigma: JAXArray | float = eqx.field(default_factory=lambda: jnp.ones(()))
    omega: JAXArray | float

    def __init__(
        self,
        scale: JAXArray | float,
        sigma: JAXArray | float = jnp.ones(()),
        num_insts: int = 1,
        name: str = "IntegratedCosine",
        **kwargs,
    ):
        self.scale = scale
        self.sigma = sigma
        self.name = name
        self.num_insts = num_insts
        self.base_model = smolgp.kernels.Cosine(scale=self.scale, sigma=self.sigma)
        self.omega = self.base_model.omega
