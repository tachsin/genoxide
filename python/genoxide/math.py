"""genoxide's portable math: the same bits on every platform.

The platform's ``math.sin``, ``numpy.exp`` and the like come from the operating system's C library
or from numpy's own code, and their last bit can differ between Linux, macOS and Windows. A fitness
function that calls them can then rank two solutions differently on different systems, and a
seeded run drifts apart. These are genoxide's ``genoxide::math`` functions, within 1 ulp of the true
value, which use only basic floating point operations: the same results everywhere, and the same
as a Rust program that calls them. Each takes a number or an array of numbers, and returns a
``float`` or a ``float64`` array of the same shape::

    import numpy as np
    import genoxide as gx

    gx.math.cos(0.0)  # 1.0
    x = np.linspace(-5.12, 5.12, 5)
    rastrigin = 10 * len(x) + np.sum(x * x - 10 * gx.math.cos(2 * np.pi * x))

:mod:`genoxide.nn`'s networks use them, as do the test problems of :mod:`genoxide.problems`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import _genoxide

__all__ = [
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "atan2",
    "sinh",
    "cosh",
    "tanh",
    "exp",
    "exp2",
    "expm1",
    "log",
    "log1p",
    "log2",
    "log10",
    "pow",
    "cbrt",
    "hypot",
]

def _apply(function: str, *arguments: Any) -> Any:
    """``function`` of each value of the arguments, broadcast together: a ``float`` for numbers,
    an array for arrays."""
    arrays = np.broadcast_arrays(*(np.asarray(argument, dtype=np.float64) for argument in arguments))
    shape = arrays[0].shape
    flat = [np.ascontiguousarray(array.reshape(-1)) for array in arrays]
    result = _genoxide.portable_math(function, *flat).reshape(shape)
    if shape == ():
        return float(result)
    return result


def sin(x: Any) -> Any:
    """The sine of ``x``, in radians."""
    return _apply("sin", x)


def cos(x: Any) -> Any:
    """The cosine of ``x``, in radians."""
    return _apply("cos", x)


def tan(x: Any) -> Any:
    """The tangent of ``x``, in radians."""
    return _apply("tan", x)


def asin(x: Any) -> Any:
    """The arcsine of ``x``, in [-pi/2, pi/2]; NaN outside [-1, 1]."""
    return _apply("asin", x)


def acos(x: Any) -> Any:
    """The arccosine of ``x``, in [0, pi]; NaN outside [-1, 1]."""
    return _apply("acos", x)


def atan(x: Any) -> Any:
    """The arctangent of ``x``, in [-pi/2, pi/2]."""
    return _apply("atan", x)


def atan2(y: Any, x: Any) -> Any:
    """The four-quadrant arctangent of ``y / x``, in [-pi, pi]."""
    return _apply("atan2", y, x)


def sinh(x: Any) -> Any:
    """The hyperbolic sine of ``x``."""
    return _apply("sinh", x)


def cosh(x: Any) -> Any:
    """The hyperbolic cosine of ``x``."""
    return _apply("cosh", x)


def tanh(x: Any) -> Any:
    """The hyperbolic tangent of ``x``: the activation of :mod:`genoxide.nn`'s "tanh" units."""
    return _apply("tanh", x)


def exp(x: Any) -> Any:
    """``e`` to the power ``x``."""
    return _apply("exp", x)


def exp2(x: Any) -> Any:
    """2 to the power ``x``."""
    return _apply("exp2", x)


def expm1(x: Any) -> Any:
    """``e^x - 1``, accurate near 0."""
    return _apply("expm1", x)


def log(x: Any) -> Any:
    """The natural logarithm of ``x``: -inf for 0, NaN for negative ``x``."""
    return _apply("log", x)


def log1p(x: Any) -> Any:
    """``log(1 + x)``, accurate near 0."""
    return _apply("log1p", x)


def log2(x: Any) -> Any:
    """The base-2 logarithm of ``x``."""
    return _apply("log2", x)


def log10(x: Any) -> Any:
    """The base-10 logarithm of ``x``."""
    return _apply("log10", x)


def pow(base: Any, exponent: Any) -> Any:  # noqa: A001
    """``base`` to the power ``exponent``, with the special cases of C's ``pow``."""
    return _apply("pow", base, exponent)


def cbrt(x: Any) -> Any:
    """The cube root of ``x``."""
    return _apply("cbrt", x)


def hypot(x: Any, y: Any) -> Any:
    """``sqrt(x**2 + y**2)`` without overflow or underflow in between."""
    return _apply("hypot", x, y)
