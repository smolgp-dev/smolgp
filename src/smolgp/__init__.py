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
subpackage (see :ref:`api-kernels`), and then passing that to a
:class:`GaussianProcess` object to do all the computations. Check out the
:ref:`tutorials` for a more complete introduction.
"""

from smolgp import (
    kernels as kernels,
    solvers as solvers,
)
from smolgp.gp import GaussianProcess as GaussianProcess
# from smolgp.smolgp_version import __version__ as __version__
