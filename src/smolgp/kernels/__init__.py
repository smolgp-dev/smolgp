"""
The primary model-building interface in ``smolgp`` is via "kernels", typically
constructed as sums and products of the objects defined in this subpackage.
The kernels here are extensions of those in ``tinygp.kernels.quasisep`` and are
fully compatible with the state space solvers.  Key differences from
``tinygp.kernels.quasisep``:

1. Each kernel exposes :meth:`~smolgp.kernels.StateSpaceModel.noise_effect_matrix`
   (:math:`L`) and :meth:`~smolgp.kernels.StateSpaceModel.process_noise` (:math:`Q_k`).
2. The observation model is a row vector that can be selected per-component via
   the kernel's ``name`` attribute (used for multi-component decomposition).
3. The transition matrix is in its conventional form (not transposed as in
   ``tinygp.kernels.quasisep``).

For modeling time-averaged (integrated) measurements, use the kernels in
:mod:`smolgp.kernels.integrated`.

Sums and products of ``smolgp`` kernels are themselves ``smolgp`` kernels.
Their conditioned and predictive distributions can be decomposed per-component
via :meth:`~smolgp.GaussianProcess.get_component_mean` and
:meth:`~smolgp.GaussianProcess.get_all_component_means`.

For mixed kernels with both integrated and instantaneous components, the
:class:`~smolgp.solvers.IntegratedStateSpaceSolver` is used automatically.
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

# Dense kernels (for testing)
import smolgp.kernels.dense
