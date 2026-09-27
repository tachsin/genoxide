"""The constrained problems of the CEC 2006 special session, g01 to g06, evaluated in Rust.

Liang, J. J., Runarsson, T. P., Mezura-Montes, E., Clerc, M., Suganthan, P. N., Coello Coello,
C. A. and Deb, K. (2006). Problem Definitions and Evaluation Criteria for the CEC 2006 Special
Session on Constrained Real-Parameter Optimization. Technical report, Nanyang Technological
University, Singapore. The definitions, bounds, solutions and values are the report's. Each class
names the report's source for it; those sources are still to be read (#168).

Every problem is minimized, and its fitness is ``(score, violation)``::

    import genoxide as gx

    problem = gx.problems.cec2006.G06()
    shade = gx.De(problem.genome, objective=problem.objective, seed=1)
    result = shade.run(problem, evaluations=20_000)
    print(result.best_fitness - problem.optimum.value, result.violation)

The violation adds up ``max(0, g(x))`` over the inequalities and ``max(0, |h(x)| − δ)`` over the
equalities: the report counts an equality as met when ``|h(x)| <= δ``, with δ =
:data:`EQUALITY_TOLERANCE`. ``constraints(x)`` gives the inequalities ``g(x) <= 0``, then the
equalities ``h(x) = 0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from . import Problem

__all__ = ["EQUALITY_TOLERANCE", "G01", "G02", "G03", "G04", "G05", "G06"]

EQUALITY_TOLERANCE = 0.0001
"""The report's tolerance δ of an equality constraint: ``|h(x)| <= 0.0001`` counts as met."""


class _WithTolerance(Problem):
    """A problem with equality constraints and their tolerance δ."""

    tolerance: float | None

    def _describe(self) -> dict[str, Any]:
        description: dict[str, Any] = {"type": self._type}
        if self.tolerance is not None:
            description["tolerance"] = float(self.tolerance)
        return description


@dataclass(frozen=True)
class G01(Problem):
    """g01: ``5 Σᵢ₌₁⁴ xᵢ − 5 Σᵢ₌₁⁴ xᵢ² − Σᵢ₌₅¹³ xᵢ``, a quadratic in 13 dimensions with 9 linear
    inequalities.

    Bounds x₁…x₉ in [0, 1], x₁₀…x₁₂ in [0, 100], x₁₃ in [0, 1]; minimum −15 at
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1), with six constraints active.

    The report's eqs. 4-5, after Floudas, C. A. and Pardalos, P. M. (1990). A Collection of Test
    Problems for Constrained Global Optimization Algorithms. LNCS 455, Springer.
    """

    _type: ClassVar[str] = "g01"


@dataclass(frozen=True)
class G02(Problem):
    """g02: ``−|Σ cos⁴ xᵢ − 2 Π cos² xᵢ| / √(Σ i xᵢ²)`` in 20 dimensions, subject to
    ``Π xᵢ >= 0.75`` and ``Σ xᵢ <= 7.5n``: a maximization, negated.

    Bounds (0, 10]²⁰, closed at 1.5e-154, the smallest positive value whose square doesn't
    underflow; best known −0.80361910412559, not proven optimal.

    The report's eqs. 6-7, after Koziel, S. and Michalewicz, Z. (1999). Evolutionary algorithms,
    homomorphous mappings, and constrained parameter optimization. Evolutionary Computation 7(1):
    19-44.
    """

    _type: ClassVar[str] = "g02"


@dataclass(frozen=True)
class G03(_WithTolerance):
    """g03: ``−(√n)ⁿ Π xᵢ`` in 10 dimensions, subject to ``Σ xᵢ² = 1``: a maximization, negated.

    Bounds [0, 1]¹⁰; minimum ``−(1 + δ)⁵`` at xᵢ = √((1 + δ)/10), derived from the definition:
    −1.00050010001 for the report's δ. ``tolerance`` is δ, :data:`EQUALITY_TOLERANCE` if None.

    The report's eqs. 8-9, after Michalewicz, Z., Nazhiyath, G. and Michalewicz, M. (1996). A note
    on usefulness of geometrical crossover for numerical optimization problems. Proceedings of the
    5th Annual Conference on Evolutionary Programming: 305-312.
    """

    tolerance: float | None = None
    _type: ClassVar[str] = "g03"


@dataclass(frozen=True)
class G04(Problem):
    """g04: ``5.3578547x₃² + 0.8356891x₁x₅ + 37.293239x₁ − 40792.141``, a quadratic in 5
    dimensions with 6 nonlinear inequalities.

    Bounds x₁ in [78, 102], x₂ in [33, 45], x₃…x₅ in [27, 45]; minimum −30665.53867178332 at
    (78, 33, 29.9952560256815985, 45, 36.7758129057882073).

    The report's eqs. 10-11, after Himmelblau, D. M. (1972). Applied Nonlinear Programming.
    McGraw-Hill.
    """

    _type: ClassVar[str] = "g04"


@dataclass(frozen=True)
class G05(_WithTolerance):
    """g05: ``3x₁ + 0.000001x₁³ + 2x₂ + (0.000002/3)x₂³``, a cubic in 4 dimensions with 2 linear
    inequalities and 3 nonlinear equalities.

    Bounds x₁, x₂ in [0, 1200], x₃, x₄ in [−0.55, 0.55]; best known 5126.4967140071, for the
    report's δ only (``optimum`` is None for another). ``tolerance`` is δ,
    :data:`EQUALITY_TOLERANCE` if None.

    The report's eqs. 12-13, after Hock, W. and Schittkowski, K. (1981). Test Examples for
    Nonlinear Programming Codes. Lecture Notes in Economics and Mathematical Systems 187,
    Springer.
    """

    tolerance: float | None = None
    _type: ClassVar[str] = "g05"


@dataclass(frozen=True)
class G06(Problem):
    """g06: ``(x₁ − 10)³ + (x₂ − 20)³`` in 2 dimensions, inside a thin crescent between two
    circles.

    Bounds x₁ in [13, 100], x₂ in [0, 100]; minimum −6961.81387558015 at
    (14.095, 0.8429607892154795668), where both constraints are active.

    The report's eqs. 14-15, after Floudas and Pardalos (1990).
    """

    _type: ClassVar[str] = "g06"
