"""
``smolgp`` is designed to be a drop-in extension of the `tinygp <https://github.com/dfm/tinygp>`_
library for building Gaussian Process (GP) models in Python. As such, it is also built on top of
`jax <https://github.com/google/jax>`_. The driving design philosophy is to match the API of ``tinygp``
as closely as possible. With only a few exceptions, any existing code you have that uses ``tinygp`` should
work with ``smolgp`` by simply by finding-and-replacing ``tiny`` with ``smol``.


``smolgp`` uses the state space
representations of Gaussian Processes to implement linear-time (or up to logN with
parallelization on GPU) solvers for GP regression and forecasting. It also implements
"integrated" kernels that can model time-averaged measurements, such as those from
long-exposure astronomical observations, which can also be solved in linear time and are
also compatible with the parallel methods.

The primary way that you interact with ``smolgp`` is to construct
"kernel" functions using the building blocks provided in the ``kernels``
subpackage (see :mod:`smolgp.kernels`), and then passing that to a
:class:`GaussianProcess` object to do all the computations. Check out the
:ref:`tutorials` for a more complete introduction.
"""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

from smolgp import (
    kernels as kernels,
)
from smolgp import (
    solvers as solvers,
)
from smolgp.gp import GaussianProcess as GaussianProcess

try:
    __version__ = _version("smolgp")
except _PackageNotFoundError:
    __version__ = "unknown"


def enable_x64(use_x64: bool = True) -> None:
    """Enable (or disable) 64-bit precision in JAX.

    State space GPs can lose accuracy in 32-bit precision, especially with
    integrated kernels or long gaps between observations, so 64-bit is
    recommended. This is a shorthand for
    ``jax.config.update("jax_enable_x64", use_x64)``. It is process-wide, so
    it also affects any other JAX code, and should be called before creating
    any arrays.

    Args:
        use_x64: If ``True`` (default), use 64-bit precision; if ``False``,
            revert to JAX's default 32-bit precision.
    """
    import jax

    jax.config.update("jax_enable_x64", use_x64)
