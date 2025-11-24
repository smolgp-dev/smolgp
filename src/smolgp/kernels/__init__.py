"""
The primary model building interface in ``smolgp`` is via "kernels", which are
typically constructed as sums and products of objects defined in this
subpackage, or by subclassing :class:`Kernel` as discussed in the :ref:`kernels`
tutorial. The kernels implemented here are extensions of those defined in
``tinygp.kernels.quasisep`` to be fully-compatible with the state space solvers.

For modeling integrated measurements, use the kernels in ``smolgp.kernels.integrated``.

Sums and Products of ``smolgp`` kernels are also ``smolgp`` kernels.

For mixed kernels containing some integrated and some instantaneous components,
the integrated solver will be used to handle the entire kernel.
"""

# Model class and utilities
from smolgp.kernels.base import (
    StateSpaceModel as StateSpaceModel,
    Sum as Sum,
    Product as Product,
    Wrapper as Wrapper,
    Scale as Scale,
)

# Specific kernels
from smolgp.kernels.base import (
    Constant as Constant,
    SHO as SHO,
    Exp as Exp,
    Matern32 as Matern32,
    Matern52 as Matern52,
    Cosine as Cosine,
    ExpSineSquared as ExpSineSquared,
)

# Integrated kernels
from smolgp.kernels.integrated import (
    IntegratedSHO as IntegratedSHO,
    IntegratedExp as IntegratedExp,
    IntegratedMatern32 as IntegratedMatern32,
    IntegratedMatern52 as IntegratedMatern52,
    IntegratedCosine as IntegratedCosine,
)
