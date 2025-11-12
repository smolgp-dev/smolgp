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

__all__ = [
    "StateSpaceModel",
    "Sum",
    "Product",
    "extract_leaf_kernels",
    "Wrapper",
    "Scale",
    "Constant",
    "SHO",
    "Exp",
    "Matern32",
    "Matern52",
    "Cosine",
    "IntegratedSHO",
    "IntegratedExp",
    "IntegratedMatern32",
    "IntegratedMatern52",
    "IntegratedCosine",
]

# Model class and utilities
from smolgp.kernels.base import (
    StateSpaceModel,
    Sum,
    Product,
    Wrapper,
    Scale,
)

# Specific kernels
from smolgp.kernels.base import (
    Constant,
    SHO,
    Exp,
    Matern32,
    Matern52,
    Cosine,
)

# Integrated kernels
from smolgp.kernels.integrated import (
    IntegratedSHO,
    IntegratedExp,
    IntegratedMatern32,
    IntegratedMatern52,
    IntegratedCosine,
)
