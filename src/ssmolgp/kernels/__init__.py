"""
The primary model building interface in ``ssmolgp`` is via "kernels", which are
typically constructed as sums and products of objects defined in this
subpackage, or by subclassing :class:`Kernel` as discussed in the :ref:`kernels`
tutorial. The kernels implemented here are extensions of those defined in
``tinygp.kernels.quasisep`` to be fully-compatible with the state space solvers.

For modeling integrated measurements, use the kernels in ``ssmolgp.kernels.integrated``.

Sums and Products of ``ssmolgp`` kernels are also ``ssmolgp`` kernels. 

For mixed kernels containing some integrated and some instantaneous components, 
the integrated solver will be used to handle the entire kernel.
"""

__all__ = [
    "StateSpaceModel",
    "Sum",
    "Product",
    "Scale",
    "SHO",
    # "Exp",
    # "Matern32",
    # "Matern52",
    # "Cosine",
    "IntegratedSHO",
]

from ssmolgp.kernels import StateSpaceModel
from ssmolgp.kernels.base import (
    Product,
    Sum,
)
from ssmolgp.kernels.base import (
    SHO,
    # Exp,
    # Matern32,
    # Matern52,
    # Cosine,
)
from ssmolgp.kernels.integrated import (
    IntegratedSHO,
)
