"""
The primary model building interface in ``smolgp`` is via "kernels", which are
typically constructed as sums and products of objects defined in this
subpackage, or by subclassing :class:`Kernel` as discussed in the :ref:`kernels`
tutorial. The kernels implemented here are extensions of those defined in
``tinygp.kernels.quasisep`` to be fully-compatible with the state space solvers.
The key differences are that `smolgp` kernels:
1. Include the `noise_effect_matrix` and `process_noise` matrix
2. Treat the observation model as a row vector that can be selected on/off via the kernel's name
3. The transition matrix is in its usual form (compared to transposed forms in tinygp.kernels.quasisep)

For modeling integrated measurements, use the kernels in ``smolgp.kernels.integrated``.

Sums and Products of ``smolgp`` kernels are also ``smolgp`` kernels.
Their conditioned and predictive distributions can be decomposed per-component
using GaussianProcess.component_means and predict_component_means, respectively.

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
