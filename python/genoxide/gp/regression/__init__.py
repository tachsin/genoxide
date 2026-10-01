"""Symbolic regression: trees of mathematical functions fitted to data, evaluated in Rust.

:func:`primitives` makes a set of the functions, by name, and the variables, with optional
:class:`genoxide.gp.Constants`; :class:`Sample` holds points and their targets and
:class:`Dataset` a training sample with an optional test sample; :class:`Regression` is the
fitness function, the error of a tree on the training sample (the RMSE, after linear scaling, by
default), evaluated on all the points at once in Rust without the GIL::

    import numpy as np
    import genoxide as gx

    # 3x^2 + 2 from 21 points: linear scaling finds the 3 and the 2, the search only x^2
    x = np.linspace(-1.0, 1.0, 21)
    dataset = gx.gp.regression.Dataset(gx.gp.regression.Sample(x, 3.0 * x * x + 2.0))
    primitives = gx.gp.regression.primitives(["add", "sub", "mul"], ["x"])
    regression = gx.gp.regression.Regression(primitives, dataset)
    ga = gx.Ga(
        gx.gp.Gp(primitives),
        population_size=100,
        select=gx.Tournament(3),
        crossover=gx.gp.SubtreeCrossover(),
        mutation=gx.gp.SubtreeMutation(),
        mutation_rate=0.1,
        objective="minimize",
        seed=1,
    )
    result = ga.run(regression, target=1e-12, generations=50)
    print(regression.display(result.best_genome))  # e.g. 2 + 3 * (mul(x, x))

**Linear scaling** (Keijzer 2003): the error is the one of ``a + b f(x)``, with ``a`` and ``b``
fitted by least squares on the training sample, so the search looks for the shape of the target
and not its scale and offset. ``Regression(..., linear_scaling=False)`` measures the tree
itself; ``scaling(tree)`` gives ``(a, b)`` and ``display(tree)`` the scaled expression.

**Invalid trees.** A tree whose value isn't finite at a training point is invalid: its fitness
is None, worse than any error. The analytic quotient ``aq``, ``a / sqrt(1 + b^2)``, is defined
everywhere and is the recommended division (Ni et al. 2013); Koza's protected ``pdiv``, ``plog``
and ``psqrt`` replicate papers that use them.

The functions are those of :data:`FUNCTIONS`, computed with genoxide's portable math, so an
error is the same bits on every platform and as a Rust program's. :mod:`.problems` has the test
problems of Koza and Nguyen, each with its paper's data and function set.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from ... import _genoxide
from .. import Constants, PrimitiveSet

__all__ = ["FUNCTIONS", "primitives", "Sample", "Dataset", "Regression", "problems"]

Sample = _genoxide.Sample
Dataset = _genoxide.Dataset
Regression = _genoxide.Regression

FUNCTIONS: tuple[str, ...] = (
    "add",
    "sub",
    "mul",
    "div",
    "aq",
    "neg",
    "inv",
    "square",
    "cube",
    "sin",
    "cos",
    "exp",
    "log",
    "sqrt",
    "tanh",
    "abs",
    "pdiv",
    "plog",
    "psqrt",
)
"""The functions, by name: ``add``, ``sub``, ``mul``, ``div`` (IEEE division: a tree that divides
by 0 is invalid), ``aq`` (the analytic quotient ``a / sqrt(1 + b^2)``), ``neg``, ``inv`` (``1 /
a``), ``square``, ``cube``, ``sin``, ``cos``, ``exp``, ``log``, ``sqrt``, ``tanh``, ``abs``, and
Koza's protected ``pdiv`` (1 for a division by 0), ``plog`` (``ln |a|``, 0 at 0) and ``psqrt``
(``sqrt |a|``)."""


def primitives(
    functions: Sequence[str],
    variables: Sequence[str],
    constants: Constants | None = None,
) -> PrimitiveSet:
    """A primitive set of symbolic regression: ``functions`` by name (see :data:`FUNCTIONS`), the
    ``variables``, named in the order of the data's columns, and ``constants`` if given, e.g.
    ``gx.gp.Constants.normal(0.0, 5.0)``.

    Raises a ``ValueError`` for an unknown function, no variables or more than 2^16, a name used
    twice or that isn't a name (empty, with whitespace, ``(``, ``)`` or ``,``, or a number), or
    wrong constants.
    """
    if isinstance(functions, str) or isinstance(variables, str):
        raise ValueError("functions and variables are sequences of names, not a string")
    described = None
    if constants is not None:
        if not isinstance(constants, Constants):
            raise ValueError(f"constants is a gx.gp.Constants, not {constants!r}")
        described = json.dumps(constants._describe())
    return _genoxide.regression_primitives(
        [str(name) for name in functions], [str(name) for name in variables], described
    )


from . import problems  # noqa: E402
