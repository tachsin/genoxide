"""Surrogate models: cheap approximations of an expensive function, fitted to its evaluations.

:mod:`genoxide.model.gp` has Gaussian process regression, the model of :class:`genoxide.Bo`,
which can also be fitted and queried on its own.
"""

from . import gp

__all__ = ["gp"]
