"""
``ssmolgp`` is designed to be a drop-in extension of the `tinygp <https://github.com/dfm/tinygp>`_ 
library for building Gaussian Process (GP) models in Python. It is also built 
on top of `jax <https://github.com/google/jax>`_. ``ssmolgp`` uses state space
representations of Gaussian Processes to implement linear-time solvers for
GP regression and forecasting. Uniquely it also implements "integrated" kernels
that can model time-averaged measurements, such as those from long-exposure
astronomical observations, and can solve these in linear time as well.

The primary way that you will use to interact with ``ssmolgp`` is by constructing
"kernel" functions using the building blocks provided in the ``kernels``
subpackage (see :ref:`api-kernels`), and then passing that to a
:class:`GaussianProcess` object to do all the computations. Check out the
:ref:`tutorials` for a more complete introduction.
"""

from ssmolgp import (
    kernels as kernels,
    solvers as solvers,
)
from ssmolgp.gp import GaussianProcess as GaussianProcess
# from ssmolgp.ssmolgp_version import __version__ as __version__
