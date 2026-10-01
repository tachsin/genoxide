"""Test problems of symbolic regression, each with its paper's target, sampling of training and
test points, and function set, evaluated in Rust.

Each problem is a fitness function: :class:`genoxide.gp.regression.Regression` on its dataset
with its primitives, the RMSE after linear scaling. It gives its ``primitives()`` (build the
:class:`genoxide.gp.Gp` from them), ``dataset()``, ``regression(metric=..., linear_scaling=...)``
(the fitness function with other settings), ``target(point)``, ``formula``, ``name`` and
``reference``; :func:`all` lists them::

    import genoxide as gx

    problem = gx.gp.regression.problems.Nguyen1()
    tree = problem.primitives().parse("add(mul(x, add(x, mul(x, x))), x)")
    print(problem(tree))  # x(x + x^2) + x, the target: an error at the level of rounding

Random points come from a fixed seed of genoxide's portable random numbers, so the data are the
same on every platform and as Rust's. The optimum is exact recovery: an error at the level of
rounding (at most 1e-10 of the targets' ``deviation()``) on the training and the test points.

Koza's set (Koza-1 to 3) and Nguyen's (Nguyen-1 to 12) are ``add``, ``sub``, ``mul``, the
protected ``pdiv``, ``sin``, ``cos``, ``exp`` and the protected ``plog``, with the variable ``x``
(and ``y`` for Nguyen-9 to 12), without constants. Koza's problems have 20 training points uniform
in [-1, 1] and 101 evenly spaced test points; Nguyen's, training points as the paper samples them
and five times as many test points from another seed in the same range. The names, targets,
sampling and sets are as McDermott et al. (2012) restate them, not yet checked against the
originals (https://github.com/tachsin/genoxide/issues/168):

- McDermott, J. et al. (2012). Genetic programming needs better benchmarks. GECCO 2012: 791-798.
  doi:10.1145/2330163.2330273
"""

from __future__ import annotations

import builtins
from collections.abc import Callable

from ... import _genoxide

__all__ = [
    "RegressionProblem",
    "Koza1",
    "Koza2",
    "Koza3",
    "Nguyen1",
    "Nguyen2",
    "Nguyen3",
    "Nguyen4",
    "Nguyen5",
    "Nguyen6",
    "Nguyen7",
    "Nguyen8",
    "Nguyen9",
    "Nguyen10",
    "Nguyen11",
    "Nguyen12",
    "all",
]

RegressionProblem = _genoxide.RegressionProblem


class Koza1(RegressionProblem):
    """Koza-1, Koza's quartic: x^4 + x^3 + x^2 + x (Koza 1992), the classic first problem of
    genetic programming, and the toy problem McDermott et al. (2012) found the most overused. 20
    training points uniform in [-1, 1], 101 evenly spaced test points."""

    def __new__(cls) -> Koza1:
        return super().__new__(cls, "Koza-1")


class Koza2(RegressionProblem):
    """Koza-2: x^5 - 2x^3 + x (Koza 1994), with Koza-1's function set and sampling."""

    def __new__(cls) -> Koza2:
        return super().__new__(cls, "Koza-2")


class Koza3(RegressionProblem):
    """Koza-3: x^6 - 2x^4 + x^2 (Koza 1994), with Koza-1's function set and sampling."""

    def __new__(cls) -> Koza3:
        return super().__new__(cls, "Koza-3")


class Nguyen1(RegressionProblem):
    """Nguyen-1: x^3 + x^2 + x, from 20 points uniform in [-1, 1] (Uy et al. 2011)."""

    def __new__(cls) -> Nguyen1:
        return super().__new__(cls, "Nguyen-1")


class Nguyen2(RegressionProblem):
    """Nguyen-2: x^4 + x^3 + x^2 + x, from 20 points uniform in [-1, 1] (Uy et al. 2011)."""

    def __new__(cls) -> Nguyen2:
        return super().__new__(cls, "Nguyen-2")


class Nguyen3(RegressionProblem):
    """Nguyen-3: x^5 + x^4 + x^3 + x^2 + x, from 20 points uniform in [-1, 1] (Uy et al.
    2011)."""

    def __new__(cls) -> Nguyen3:
        return super().__new__(cls, "Nguyen-3")


class Nguyen4(RegressionProblem):
    """Nguyen-4: x^6 + x^5 + x^4 + x^3 + x^2 + x, from 20 points uniform in [-1, 1] (Uy et al.
    2011)."""

    def __new__(cls) -> Nguyen4:
        return super().__new__(cls, "Nguyen-4")


class Nguyen5(RegressionProblem):
    """Nguyen-5: sin(x^2) cos(x) - 1, from 20 points uniform in [-1, 1] (Uy et al. 2011)."""

    def __new__(cls) -> Nguyen5:
        return super().__new__(cls, "Nguyen-5")


class Nguyen6(RegressionProblem):
    """Nguyen-6: sin(x) + sin(x + x^2), from 20 points uniform in [-1, 1] (Uy et al. 2011)."""

    def __new__(cls) -> Nguyen6:
        return super().__new__(cls, "Nguyen-6")


class Nguyen7(RegressionProblem):
    """Nguyen-7: ln(x + 1) + ln(x^2 + 1), from 20 points uniform in [0, 2] (Uy et al. 2011)."""

    def __new__(cls) -> Nguyen7:
        return super().__new__(cls, "Nguyen-7")


class Nguyen8(RegressionProblem):
    """Nguyen-8: sqrt(x), from 20 points uniform in [0, 4] (Uy et al. 2011)."""

    def __new__(cls) -> Nguyen8:
        return super().__new__(cls, "Nguyen-8")


class Nguyen9(RegressionProblem):
    """Nguyen-9: sin(x) + sin(y^2), from 100 points uniform in [-1, 1]^2 (Uy et al. 2011)."""

    def __new__(cls) -> Nguyen9:
        return super().__new__(cls, "Nguyen-9")


class Nguyen10(RegressionProblem):
    """Nguyen-10: 2 sin(x) cos(y), from 100 points uniform in [-1, 1]^2 (Uy et al. 2011)."""

    def __new__(cls) -> Nguyen10:
        return super().__new__(cls, "Nguyen-10")


class Nguyen11(RegressionProblem):
    """Nguyen-11: x^y, from 100 points uniform in [0, 1]^2 (Uy et al. 2011)."""

    def __new__(cls) -> Nguyen11:
        return super().__new__(cls, "Nguyen-11")


class Nguyen12(RegressionProblem):
    """Nguyen-12: x^4 - x^3 + y^2/2 - y, from 100 points uniform in [-1, 1]^2 (Uy et al.
    2011)."""

    def __new__(cls) -> Nguyen12:
        return super().__new__(cls, "Nguyen-12")


def all() -> builtins.list[RegressionProblem]:
    """Every problem, in the order of this module, as Rust's ``problems::all()``."""
    classes: builtins.list[Callable[[], RegressionProblem]] = [
        Koza1,
        Koza2,
        Koza3,
        Nguyen1,
        Nguyen2,
        Nguyen3,
        Nguyen4,
        Nguyen5,
        Nguyen6,
        Nguyen7,
        Nguyen8,
        Nguyen9,
        Nguyen10,
        Nguyen11,
        Nguyen12,
    ]
    return [problem() for problem in classes]
