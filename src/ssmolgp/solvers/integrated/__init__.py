r"""
This subpackage implements the Kalman filter and RTS smoother for an augmented
state space model which includes an integral state as a 
:class:`IntegratedStateSpaceSolver`. This is intended to be used with
:class:`smolgp.kernels.integrated` state space models to properly account
for integrated (e.g. exposure-averaged) measurements. 

TODO: add documentation from the paper as a refresher/explainer of the augmented approach
        see this file in tinygp for example doc style
        
"""

__all__ = ["IntegratedStateSpaceSolver"]

from smolgp.solvers.integrated.solver import IntegratedStateSpaceSolver
