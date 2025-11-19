import jax
import jax.numpy as jnp
import equinox as eqx
import tinygp

from tinygp.kernels.base import Sum, Product
from tinygp.kernels.quasisep import Sum as qsSum, Product as qsProduct

def extract_leaf_kernels(kernel):
    """Recursively extract all leaf kernels from a sum or product of kernels"""
    if isinstance(kernel, (Sum, Product, qsSum, qsProduct)):
        return extract_leaf_kernels(kernel.kernel1) + extract_leaf_kernels(
            kernel.kernel2
        )
    else:
        return [kernel]

def unpack_coordinates(X1, X2):
    """
    Unpack the input coordinates X1 and X2 into time, instrument ID, and exposure time.

    X1 and X2 can either be
        t1 and t2 for single instrument, no exposure times
    or
        (t1, instid1) and (t2, instid2) for multiple instruments, no exposure times
    or
        (t1, instid1, delta1) and (t2, instid2, delta2) for exposure times, also requires instrument IDs

    Returns
        (t1, instid1, delta1), (t2, instid2, delta2)
        where instid and delta are arrays of zeros if not provided
    """
    if not type(X1) is tuple:
        # Single instrument, no exposure times
        t1 = X1
        t2 = X2
        instid1 = delta1 = jnp.zeros_like(t1)
        instid2 = delta2 = jnp.zeros_like(t2)
    elif len(X1) == 2:
        # Multiple instruments, no exposure times
        t1, instid1 = X1
        t2, instid2 = X2
        delta1 = delta2 = jnp.zeros_like(t1)
    elif len(X1) == 3:
        # When using and exposure times
        t1, delta1, instid1 = X1
        t2, delta2, instid2 = X2
    else:
        raise ValueError("X1 and X2 must be tuples of length 1, 2 or 3.")
    return (t1, delta1, instid1), (t2, delta2, instid2)

################## Full/dense Matern-5/2 #################
class Matern52Kernel(tinygp.kernels.Kernel):
    amp: jax.Array | float
    lam: jax.Array | float
    instid: jax.Array = eqx.field(static=True)

    def __init__(self, amp=None, lam=None, instid=0):
        """
        Matern-5/2 kernel for smooth instrumental drift.

        Parameters
        ----------
        amp : float
            Amplitude (m/s)
        lam : float
            Timescale (days)
        instid : float
            Instrument ID this kernel describes
        """
        self.instid = instid

        self.amp = 1.0 if amp is None else amp
        self.lam = 3 / 24 if lam is None else lam

    def evaluate(self, X1, X2):
        """
        Calculate the kernel for given pair of times X1 and X2.

        X1 and X2 should be a unxt.Quantity with units
            can also be a tuple consisting of (t1, instid)
        """
        (t1, delta1, instid1), (t2, delta2, instid2) = unpack_coordinates(X1, X2)

        # time between pairs of observations in units the kernel is defined in
        # Delta = jnp.abs((t1 - t2).to(self.tunit).value)
        Delta = jnp.abs(t1 - t2)

        # Matern 5/2 kernel
        R = jnp.sqrt(5) / self.lam
        k = jnp.exp(-R * Delta) * (1 + R * Delta + R**2 * Delta**2 / 3)

        # Decorrelate different instruments
        dij_inst = ((instid1 == self.instid) & (instid2 == self.instid)).astype(int)

        return self.amp**2 * k * dij_inst
    

################# Full (dense) SHO kernel  #################
class SHOKernel(tinygp.kernels.Kernel):
    S: jax.Array | float
    w: jax.Array | float
    Q: jax.Array | float
    rho: jax.Array | float
    tau: jax.Array | float
    sig: jax.Array | float

    def __init__(
        self,
        S=None,
        w=None,
        Q=None,
        rho=None,
        tau=None,
        sig=None,
        #  unit=u.unit('m')**2 / u.unit('s')**2,
        #  tunit=u.unit('s'),
    ):
        """
        A simple harmonic oscillator (SHO)
        kernel that is aware of time units

        Parameters
        ----------
        S : jax.Array | float
            Power at characteristic frequency (m^2/rad/s)
        w : jax.Array | float
            Characteristic frequency (rad/s)
        Q : jax.Array | float
            Quality factor (dimensionless).
        unit: unxt.Unit
            Covariance unit (default: m^2/s^2)
        tunit: unxt.Unit
            Time unit (default: s)

        Alternatively, one can give a more physical parameterization
        ----------
            rho : jax.Array | float
                Undamped period of the oscillator (s).
            tau : jax.Array | float
                Damping timescale of the process (s).
            sig : jax.Array | float
                Standard deviation of the process (m/s).
        """

        # Extract parameterization and assign values
        param1 = (S is not None) and (w is not None) and (Q is not None)
        param2 = (rho is not None) and (tau is not None) and (sig is not None)
        if param1:
            self.S = S
            self.w = w
            self.Q = Q
            self.rho = 2 * jnp.pi / self.w
            self.tau = 2 * self.Q / self.w
            self.sig = jnp.sqrt(self.S * self.w * self.Q)
        elif param2:
            self.rho = rho
            self.tau = tau
            self.sig = sig
            self.w = 2 * jnp.pi / self.rho
            self.Q = self.tau * self.w / 2
            self.S = self.sig**2 / (self.w * self.Q)
        else:
            raise ValueError("Must specifiy parameter values!")

    def evaluate(self, X1, X2):
        """
        Calculate the kernel for given pair of times X1 and X2.

        X1 and X2 should be a unxt.Quantity with units
            can also be a tuple consisting of (t1, instid)
        """

        (t1, delta1, instid1), (t2, delta2, instid2) = unpack_coordinates(X1, X2)

        # time between pairs of observations in units the kernel is defined in
        # Delta = jnp.abs((t1 - t2).to(self.tunit).value)
        Delta = jnp.abs(t1 - t2)

        # Calculate the kernel
        eta = jnp.sqrt(jnp.abs(1 - 1 / (4 * self.Q**2)))  # damping factor
        k = jnp.cos(eta * self.w * Delta) + 1 / (2 * eta * self.Q) * jnp.sin(
            eta * self.w * Delta
        )
        return self.sig**2 * jnp.exp(-Delta / self.tau) * k

    def __repr__(self):
        s = f"{type(self)}(\n"
        s += f" S={self.S},\n"
        s += f" w={self.w}\n"
        s += f" Q={self.Q}\n"
        s += f" rho={self.rho}\n"
        s += f" tau={self.tau}\n"
        s += f" sig={self.sig}\n"
        s += ")"
        return s



################# Full/dense Integrated SHO kernel  #################
class IntegratedSHOKernel(tinygp.kernels.Kernel):
    S: jax.Array | float
    w: jax.Array | float
    Q: jax.Array | float
    rho: jax.Array | float
    tau: jax.Array | float
    sig: jax.Array | float
    eta: jax.Array | float
    a: jax.Array | float

    def __init__(
        self,
        S=None,
        w=None,
        Q=None,
        rho=None,
        tau=None,
        sig=None,
    ):
        """
        A simple harmonic oscillator (SHO)
        kernel integrated over an exposure

        Details in Luhn et al. 2025 (in prep)

        Parameters
        ----------
        S : jax.Array | float
            Power at characteristic frequency (m^2/rad/s)
        w : jax.Array | float
            Characteristic frequency (rad/s)
        Q : jax.Array | float
            Quality factor (dimensionless).
        unit: unxt.Unit
            Covariance unit (default: m^2/s^2)
        tunit: unxt.Unit
            Time unit (default: s)

        Alternatively, one can give a more physical parameterization
        ----------
            rho : jax.Array | float
                Undamped period of the oscillator (s).
            tau : jax.Array | float
                Damping timescale of the process (s).
            sig : jax.Array | float
                Standard deviation of the process (m/s).
        """

        # Extract parameterization and assign values
        param1 = (S is not None) and (w is not None) and (Q is not None)
        param2 = (rho is not None) and (tau is not None) and (sig is not None)
        if param1:
            self.S = S
            self.w = w
            self.Q = Q
            self.rho = 2 * jnp.pi / self.w
            self.tau = 2 * self.Q / self.w
            self.sig = jnp.sqrt(self.S * self.w * self.Q)
        elif param2:
            self.rho = rho
            self.tau = tau
            self.sig = sig
            self.w = 2 * jnp.pi / self.rho
            self.Q = self.tau * self.w / 2
            self.S = self.sig**2 / (self.w * self.Q)
        else:
            raise ValueError("Must specifiy parameter values!")

        self.eta = jnp.sqrt(jnp.abs(1 - 1 / (4 * self.Q**2)))  # damping factor
        self.a = 1 / (2 * self.Q * self.eta)  # common quantity in integrals

    def __repr__(self):
        s = f"{type(self)}(\n"
        s += f" S={self.S},\n"
        s += f" w={self.w}\n"
        s += f" Q={self.Q}\n"
        s += f" rho={self.rho}\n"
        s += f" tau={self.tau}\n"
        s += f" sig={self.sig}\n"
        s += ")"
        return s

    def Delta(self, X1, X2):
        """
        Time difference (absolute value)
        between pairs of observations X1, X2.

        Delta = | t1 - t2 |
        """
        # Extract t1, t2 from X1, X2
        t1 = X1[0] if type(X1) == tuple else X1
        t2 = X2[0] if type(X2) == tuple else X2

        # time between pairs of observations in units the kernel is defined in
        # Delta = jnp.abs((t1 - t2).to(self.tunit).value)
        Delta = jnp.abs(t1 - t2)
        return Delta

    def latent_process(self, Delta):
        """
        Latent kernel for given time difference Delta
        """
        k = jnp.cos(self.eta * self.w * Delta) + 1 / (2 * self.eta * self.Q) * jnp.sin(
            self.eta * self.w * Delta
        )
        return self.sig**2 * jnp.exp(-Delta / self.tau) * k

    # def latent_process(self, X1, X2):
    #     '''
    #     Latent kernel for a given pair of observations X1 and X2.

    #     X1 and X2 can either be
    #         (t1, texp1) and (t2, texp2) for single instrument
    #     or
    #         (t1, instid1, texp1) and (t2, instid2, texp2) for multiple instruments

    #     t1 and t2 can be a unxt.Quantity with units
    #     '''
    #     return self.latent_process(self.Delta(X1, X2))

    def I0(self, y):
        """Eq. 5 - helper function for single integral (Eq. 4)"""
        return jnp.exp(-self.a * y) * (
            (1 - self.a**2) * jnp.sin(y) - 2 * self.a * jnp.cos(y)
        )

    def I1(self, lower, upper):
        """Eq. 8 - helper function for double integrals (Eq. 7, 11)"""

        def f1(y):
            num = (
                jnp.exp(-self.a * y)
                * (1 - self.a**2)
                * (jnp.cos(y) + self.a * jnp.sin(y))
            )
            den = self.eta * self.w * (1 + self.a**2)
            return num / den

        return f1(upper) - f1(lower)

    def I2(self, lower, upper):
        """Eq. 9 - helper function for double integrals (Eq. 7, 11)"""

        def f2(y):
            num = (
                -2 * self.a * jnp.exp(-self.a * y) * (jnp.sin(y) - self.a * jnp.cos(y))
            )
            den = self.eta * self.w * (1 + self.a**2)
            return num / den

        return f2(upper) - f2(lower)

    def integrated_separate(self, Delta, delta1, delta2):
        """
        The double integral for two non-overlapping observations.
        Depends on the time-lag (Delta) and the two exposure times (delta1, delta2)

        Eq. 11 in Luhn et al. (2025, in prep)
        """
        # Bounds of integrals
        y1 = self.eta * self.w * ((delta1 + delta2) / 2 + Delta)  # Eq. 12
        y2 = self.eta * self.w * ((delta1 + delta2) / 2 + Delta - delta1)  # Eq. 12
        y3 = self.eta * self.w * ((delta1 - delta2) / 2 + Delta)  # Eq. 12
        y4 = self.eta * self.w * ((delta1 - delta2) / 2 + Delta - delta1)  # Eq. 12

        pre = (self.S * self.Q) / (delta1 * delta2 * self.eta * (1 + self.a**2))
        return pre * (
            self.I1(y1, y2) - self.I2(y1, y2) - self.I1(y3, y4) + self.I2(y3, y4)
        )

    def integrated_overlap(self, delta):
        """
        The double integral for two perfectly overlapping
        observations (i.e. zero time-lag). As such, this
        only depends on the exposure time (delta)

        Eq. 7 in Luhn et al. (2025, in prep)
        """
        # Bounds of integrals
        y1, y2 = self.eta * self.w * delta, 0.0

        pre = (2 * self.S * self.Q) / (delta * delta * self.eta * (1 + self.a**2))
        return pre * (self.I1(y1, y2) - self.I2(y1, y2) + 2 * self.a * delta)

    def integrated_single(self, Delta, delta):
        """
        The single integral for when one observation
        has zero exposure time (e.g. latent curve)

        Eq. 4 in Luhn et al. (2025, in prep)
        """
        # Bounds of integrals
        y1 = self.eta * self.w * (delta / 2 + Delta)
        y2 = self.eta * self.w * (Delta - delta / 2)

        pre = -(self.S * self.Q) / (delta * self.eta)
        return pre / (1 + self.a**2) * (self.I0(y2) - self.I0(y1))

    def full_single_integral(self, Delta, delta):
        """
        This helper function handles the logic for the single integral case.

        For pairs of observation coordinates separated by Delta, such that
        one is the 0-exposure test point (i.e., the latent curve) and the
        other has exposure time delta, this function handles the logic to
        break up the integral and call self.integrated_single() for the
        sub-exposures where appropriate
        """

        # Define our coordinate system:
        # - Delta must be positive
        # - obs1 starts at t=0
        # - obs1 is the finite exposure, obs2 is instantaneous
        # these are the begin and end times for obs1
        # obs1 spans time p1 to p2
        # obs2 is a single point at p3
        p1 = 0
        p2 = delta
        p3 = delta / 2 + Delta

        # Initalize output
        result = jnp.zeros_like(Delta)

        # Check if we need to use the instantaneous kernel
        # This will be true if the finite exposure (obs1) is zero and/or
        # significantly less than the SHO timescale to be functionally zero
        timescale = 2 * jnp.pi / self.w
        use_latent = jnp.abs(delta / timescale) < 1e-4
        result = jnp.where(use_latent, self.latent_process(Delta), result)

        ## CASE 1
        # Check if obs2 is outside the exposure of obs1
        # If so, can use the single integral directly
        notothers = ~use_latent
        case1 = p2 <= p3
        result = jnp.where(
            case1 & notothers, self.integrated_single(Delta, delta), result
        )

        ## CASE 2
        # If obs2 did occur during exposure of obs1
        # we need to split the integral into a left and right side
        # result = jnp.where(p3 < p2,
        #         (self.integrated_single(p3/2, p3)*p3
        #         + self.integrated_single(p3/2, delta-p3 )*(delta-p3))/delta, result)
        def calc_integral(p3, delta):
            delta_left = p3
            Delta_left = delta_left / 2
            int = self.integrated_single(Delta_left, delta_left) * delta_left

            delta_right = delta - p3  # or p2-p3
            Delta_right = delta_right / 2
            int += self.integrated_single(Delta_right, delta_right) * delta_right

            return int / delta

        notothers &= ~case1
        case2 = p3 < p2
        result = jnp.where(case2 & notothers, calc_integral(p3, delta), result)
        return result

    def evaluate(self, X1, X2):
        """
        Compute the integrated SHO kernel for a pair of observations X1 and X2.

        X1 and X2 can either be
            (t1, texp1) and (t2, texp2) for single instrument
        or
            (t1, instid1, texp1) and (t2, instid2, texp2) for multiple instruments

        t1 and t2 can be a unxt.Quantity with units
        """
        assert len(X1) > 1, "X1 and X2 must include timestamps and exposure times"
        (t1, delta1, instid1), (t2, delta2, instid2) = unpack_coordinates(X1, X2)

        # Time difference and exposure times in units the kernel is defined in
        # Delta = jnp.abs((t1 - t2).to(self.tunit).value)
        # delta1 = delta1.to(self.tunit).value
        # delta2 = delta2.to(self.tunit).value
        Delta = jnp.abs(t1 - t2)

        # Define our coordinate system:
        # - Delta must be positive
        # - obs1 starts at t=0
        # - obs1 is the longer exposure
        delta1, delta2 = jnp.maximum(delta1, delta2), jnp.minimum(delta1, delta2)

        # In our coordinate system, these are the
        # begin and end times for both observations
        # obs1 spans time p1 to p2
        # obs2 spans time p3 to p4
        p1 = 0
        p2 = delta1
        p3 = (delta1 - delta2) / 2 + Delta
        p4 = p3 + delta2

        # Initialize output kernel
        k = jnp.zeros_like(Delta)

        ##### Zero-exposure cases #####
        # If the "longer" exposure (obs1) is zero, or is
        # significantly less than the SHO timescale to be
        # functionally zero, then both obs are zero-exposure
        # so we can simply use the latent kernel with Delta
        timescale = 2 * jnp.pi / self.w
        use_latent = jnp.abs(delta1 / timescale) < 1e-8
        k = jnp.where(use_latent, self.latent_process(Delta), k)

        # We might also have a finite exposure (obs1) but
        # the second is zero or functionally zero.
        # This can come up when conditioning the GP to
        # determine the "true" latent curve
        notothers = ~use_latent
        case0 = jnp.abs(delta2 / timescale) < 1e-8
        k = jnp.where(case0 & notothers, self.full_single_integral(Delta, delta1), k)

        ##### Finite-exposure cases #####

        ##### CASE 1: obs1 and obs2 completely overlap (e.g., the diagonal)
        notothers &= ~case0
        case1 = (Delta == 0) & (delta1 == delta2)
        k = jnp.where(case1 & notothers, self.integrated_overlap(delta1), k)

        ##### CASE 2: obs1 and obs2 are completely separated
        notothers &= ~case1
        case2 = Delta >= (delta1 + delta2) / 2
        k = jnp.where(
            case2 & notothers, self.integrated_separate(Delta, delta1, delta2), k
        )

        ##### CASE 3: obs 1 and obs 2 share mutual partial overlap
        #     |----------|
        #           |-------|
        #     p1    p3   p2 p4
        notothers &= ~case2
        overlap = Delta < (delta1 + delta2) / 2

        case3 = overlap & (p4 > p2)

        def calc_case3(p1, p2, p3, p4, delta1, delta2):
            # Break it up into 3 integrals

            # Int 1 (non-overlap 1)
            # |------|
            #        |------|
            # p1     p3     p4
            # delta1_1 = delta1
            # delta2_1 = (delta1+delta2)/2+Delta - t2
            # Delta_1  = t2+delta2_1/2. - t2/2
            delta1_1 = p3 - p1
            delta2_1 = p4 - p3
            Delta_1 = (p3 + delta2_1 / 2) - (p1 + delta1_1 / 2)
            int_case3 = self.integrated_separate(Delta_1, delta1_1, delta2_1) * (
                delta1_1 * delta2_1
            )
            # Note we multiply by (delta1_1*delta2_1) to unnormalize the integral,
            # so we can renormalize using the full 1/(d1*d2) at the end

            # Int 2 (non-overlap 2)
            #        |----|
            #             |---|
            #        p3   p2  p4
            # delta1_2 = t1
            # delta2_2 = t2-t1
            # Delta_2  = t2-delta2_2/2. - delta1_2/2
            delta1_2 = p2 - p3
            delta2_2 = p4 - p2
            Delta_2 = (p2 + delta2_2 / 2) - (p3 + delta1_2 / 2)
            ## AVOID THIS INTEGRAL IF p2==p3
            int_case3 = jnp.where(
                p2 == p3,
                int_case3,
                int_case3
                + self.integrated_separate(Delta_2, delta1_2, delta2_2)
                * (delta1_2 * delta2_2),
            )

            # Int 3 (overlap)
            #        |----|
            #        |----|
            #        p3   p2
            # delta_overlap = (t2-t1)
            delta_overlap = p2 - p3
            ## AVOID THIS INTEGRAL IF p2==p3
            int_case3 = jnp.where(
                p2 == p3,
                int_case3,
                int_case3 + self.integrated_overlap(delta_overlap) * (delta_overlap**2),
            )

            return int_case3 / (delta1 * delta2)

        k = jnp.where(case3 & notothers, calc_case3(p1, p2, p3, p4, delta1, delta2), k)

        ##### CASE 4: obs 2 is completely within obs 1
        #      |------------|
        #        |-------|
        #     p1 p3     p4 p2
        notothers &= ~case3
        case4 = overlap & (p4 <= p2)  # & (p2!=p4)

        def calc_case4(p1, p2, p3, p4, delta1, delta2):
            # Int 1 (overlap)
            #    |-------|
            #    |-------|
            #    p3      p4
            delta_overlap = delta2
            int_case4 = self.integrated_overlap(delta_overlap) * (delta_overlap**2)

            # Int 2 (non-overlap 1)
            #  |--|
            #     |------|
            #  p1 p3    p4
            delta1_1 = p3 - p1
            delta2_1 = p4 - p3
            Delta_1 = (p3 + delta2_1 / 2) - (p1 + delta1_1 / 2)
            ## AVOID THIS INTEGRAL IF p1==p3
            int_case4 = jnp.where(
                p1 == p3,
                int_case4,
                int_case4
                + self.integrated_separate(Delta_1, delta1_1, delta2_1)
                * (delta1_1 * delta2_1),
            )
            # int_case4 += self.integrated_separate(Delta_1, delta1_1, delta2_1)*(delta1_1*delta2_1)

            # Int 3 (non-overlap 2)
            #          |--|
            #  |-------|
            #  p3     p4  p2
            delta1_2 = p2 - p4
            delta2_2 = p4 - p3
            Delta_2 = (p4 + delta1_2 / 2) - (p3 + delta2_2 / 2)
            ## AVOID THIS INTEGRAL IF p2==p4
            int_case4 = jnp.where(
                p2 == p4,
                int_case4,
                int_case4
                + self.integrated_separate(Delta_2, delta1_2, delta2_2)
                * (delta1_2 * delta2_2),
            )
            # int_case4 += self.integrated_separate(Delta_2, delta1_2, delta2_2)*(delta1_2*delta2_2)

            return int_case4 / (delta1 * delta2)

        k = jnp.where(case4 & notothers, calc_case4(p1, p2, p3, p4, delta1, delta2), k)

        # notothers &= ~case4
        # print( jnp.all( ~notothers  ) ) # check we got all points

        return k
