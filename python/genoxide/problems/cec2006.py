"""The constrained problems of the CEC 2006 special session, g01 to g24, evaluated in Rust.

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

__all__ = [
    "EQUALITY_TOLERANCE",
    "G01",
    "G02",
    "G03",
    "G04",
    "G05",
    "G06",
    "G07",
    "G08",
    "G09",
    "G10",
    "G11",
    "G12",
    "G13",
    "G14",
    "G15",
    "G16",
    "G17",
    "G18",
    "G19",
    "G20",
    "G21",
    "G22",
    "G23",
    "G24",
]

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


@dataclass(frozen=True)
class G07(Problem):
    """g07: a quadratic in 10 dimensions with 3 linear and 5 nonlinear inequalities,
    ``x₁² + x₂² + x₁x₂ − 14x₁ − 16x₂ + (x₃ − 10)² + 4(x₄ − 5)² + (x₅ − 3)² + 2(x₆ − 1)² + 5x₇²
    + 7(x₈ − 11)² + 2(x₉ − 10)² + (x₁₀ − 7)² + 45``.

    Bounds [−10, 10]¹⁰; minimum 24.30620906818 at the report's x*, with six constraints active.
    The problem is convex, so the minimum is global; the report's x* exceeds g₁ by 6e-14, from
    rounding.

    The report's eqs. 16-17 (p. 5), after Hock and Schittkowski (1981).
    """

    _type: ClassVar[str] = "g07"


@dataclass(frozen=True)
class G08(Problem):
    """g08: ``−sin³(2πx₁) sin(2πx₂) / (x₁³(x₁ + x₂))`` in 2 dimensions, with 2 nonlinear
    inequalities: a maximization, negated.

    Bounds [0, 10]²; minimum −0.0958250414180359 at (1.22797135260752599, 4.24537336612274885).
    The value is 0/0, and so the fitness invalid, at x₁ = 0.

    The report's eqs. 18-19 (p. 5), after Koziel and Michalewicz (1999).
    """

    _type: ClassVar[str] = "g08"


@dataclass(frozen=True)
class G09(Problem):
    """g09: ``(x₁ − 10)² + 5(x₂ − 12)² + x₃⁴ + 3(x₄ − 11)² + 10x₅⁶ + 7x₆² + x₇⁴ − 4x₆x₇ − 10x₆
    − 8x₇``, a polynomial in 7 dimensions with 4 nonlinear inequalities.

    Bounds [−10, 10]⁷; minimum 680.630057374402 at the report's x*, with g₁ and g₄ active.

    The report's eqs. 20-21 (pp. 5-6), after Hock and Schittkowski (1981).
    """

    _type: ClassVar[str] = "g09"


@dataclass(frozen=True)
class G10(Problem):
    """g10: ``x₁ + x₂ + x₃``, linear in 8 dimensions, with 3 linear and 3 bilinear inequalities.

    Bounds x₁ in [100, 10000], x₂, x₃ in [1000, 10000], x₄…x₈ in [10, 1000]; minimum
    7049.24802052867 at the report's x*, where all six constraints are active.

    The report's eq. 22 (p. 6), after Hock and Schittkowski (1981).
    """

    _type: ClassVar[str] = "g10"


@dataclass(frozen=True)
class G11(_WithTolerance):
    """g11: ``x₁² + (x₂ − 1)²`` in 2 dimensions, subject to ``x₂ = x₁²``.

    Bounds [−1, 1]²; minimum ``3/4 − δ`` at (±√(1/2 − δ), 1/2), derived from the definition:
    0.7499 for the report's δ, and 3/4 at (±1/√2, 1/2) without the tolerance. ``tolerance`` is δ,
    :data:`EQUALITY_TOLERANCE` if None.

    The report's eqs. 23-24 (p. 6), after Koziel and Michalewicz (1999).
    """

    tolerance: float | None = None
    _type: ClassVar[str] = "g11"


@dataclass(frozen=True)
class G12(Problem):
    """g12: ``−(100 − (x₁ − 5)² − (x₂ − 5)² − (x₃ − 5)²)/100`` in 3 dimensions, feasible inside
    any of 9³ = 729 disjoint spheres of radius 0.25 centered on (p, q, r), p, q, r in 1…9: a
    maximization, negated.

    Bounds [0, 10]³; minimum −1 at (5, 5, 5). The one constraint is the squared distance to the
    nearest center, less 0.0625.

    The report's eq. 25 (p. 6), after Koziel and Michalewicz (1999).
    """

    _type: ClassVar[str] = "g12"


@dataclass(frozen=True)
class G13(_WithTolerance):
    """g13: ``exp(x₁x₂x₃x₄x₅)`` in 5 dimensions, with 3 nonlinear equalities.

    Bounds x₁, x₂ in [−2.3, 2.3], x₃…x₅ in [−3.2, 3.2]; best known 0.053941514041898, which
    meets the equalities within the tolerance only, for the report's δ only (``optimum`` is None
    for another). ``tolerance`` is δ, :data:`EQUALITY_TOLERANCE` if None.

    The report's eqs. 26-27 (pp. 6-7), after Hock and Schittkowski (1981).
    """

    tolerance: float | None = None
    _type: ClassVar[str] = "g13"


@dataclass(frozen=True)
class G14(_WithTolerance):
    """g14: ``Σ xᵢ (cᵢ + ln(xᵢ / Σⱼ xⱼ))`` in 10 dimensions, with 3 linear equalities.

    Bounds (0, 10]¹⁰, closed at 2.2e-308, the smallest positive normal number (the fitness is
    invalid at 0); best known −47.7648884594915, for the report's δ only (``optimum`` is None for
    another). ``tolerance`` is δ, :data:`EQUALITY_TOLERANCE` if None.

    The report's eqs. 28-29 (p. 7), after Himmelblau, D. M. (1972). Applied Nonlinear
    Programming. McGraw-Hill.
    """

    tolerance: float | None = None
    _type: ClassVar[str] = "g14"


@dataclass(frozen=True)
class G15(_WithTolerance):
    """g15: ``1000 − x₁² − 2x₂² − x₃² − x₁x₂ − x₁x₃``, a quadratic in 3 dimensions, subject to
    ``x₁² + x₂² + x₃² = 25`` and ``8x₁ + 14x₂ + 7x₃ = 56``.

    Bounds [0, 10]³; best known 961.715022289961, for the report's δ only (``optimum`` is None for
    another). ``tolerance`` is δ, :data:`EQUALITY_TOLERANCE` if None.

    The report's eqs. 30-31 (p. 7), after Himmelblau (1972).
    """

    tolerance: float | None = None
    _type: ClassVar[str] = "g15"


@dataclass(frozen=True)
class G16(Problem):
    """g16: a nonlinear function of 5 variables through a chain of 17 intermediate quantities,
    with 38 inequalities, most of them limits on those quantities.

    Bounds x₁ in [704.4148, 906.3855], x₂ in [68.6, 288.88], x₃ in [0, 134.75], x₄ in
    [193, 287.0966], x₅ in [25, 84.1988]; best known −1.90515525853479, with five constraints
    active.

    The report's eqs. 32-34 (pp. 7-10), after Himmelblau (1972).
    """

    _type: ClassVar[str] = "g16"


@dataclass(frozen=True)
class G17(_WithTolerance):
    """g17: ``f₁(x₁) + f₂(x₂)``, piecewise linear in 6 dimensions, with 4 nonlinear equalities:
    f₁ is 30x₁ below 300 and 31x₁ from there, f₂ is 28x₂ below 100, 29x₂ below 200 and 30x₂ from
    there.

    Bounds x₁ in [0, 400], x₂ in [0, 1000], x₃, x₄ in [340, 420], x₅ in [−1000, 1000], x₆ in
    [0, 0.5236]; best known 8853.5338748065, reached by the report's x* with x₁ lowered to
    201.78446249355, for the report's δ only (``optimum`` is None for another). The report's x*
    itself evaluates to 8853.53401643571; the report prints 8853.53967480648, the value of its
    organizers' code, which evaluates f₁ and f₂ at the right-hand sides of h₁ and h₂.
    ``tolerance`` is δ, :data:`EQUALITY_TOLERANCE` if None.

    The report's eqs. 35-36 (p. 10), after Himmelblau (1972).
    """

    tolerance: float | None = None
    _type: ClassVar[str] = "g17"


@dataclass(frozen=True)
class G18(Problem):
    """g18: ``−0.5(x₁x₄ − x₂x₃ + x₃x₉ − x₅x₉ + x₅x₈ − x₆x₇)``, a quadratic in 9 dimensions with 13
    nonlinear inequalities: a maximization, negated.

    Bounds x₁…x₈ in [−10, 10], x₉ in [0, 20]; best known −0.866025403784439 (−√3/2 to its
    digits), with six constraints active.

    The report's eqs. 37-38 (p. 11), after Himmelblau (1972).
    """

    _type: ClassVar[str] = "g18"


@dataclass(frozen=True)
class G19(Problem):
    """g19: ``Σⱼ Σᵢ cᵢⱼ x₁₀₊ᵢ x₁₀₊ⱼ + 2 Σⱼ dⱼ x₁₀₊ⱼ³ − Σᵢ bᵢ xᵢ``, a cubic in 15 dimensions with 5
    nonlinear inequalities ``−2 Σᵢ cᵢⱼ x₁₀₊ᵢ − 3dⱼ x₁₀₊ⱼ² − eⱼ + Σᵢ aᵢⱼ xᵢ <= 0``, with the data of
    the report's table 1.

    Bounds [0, 10]¹⁵; best known 32.6555929502463, where all five constraints are active (the
    report's table 3 counts none).

    The report's eqs. 39-40 and table 1 (pp. 11-12), after Himmelblau (1972).
    """

    _type: ClassVar[str] = "g19"


@dataclass(frozen=True)
class G20(_WithTolerance):
    """g20: ``Σ aᵢxᵢ``, linear in 24 dimensions, with 6 nonlinear inequalities, 12 nonlinear and 2
    linear equalities, with the data of the report's table 2, and no feasible solution.

    Bounds [0, 10]²⁴. The report finds no feasible solution, and there is none (derived for
    genoxide, not in the report): the inequalities hold only where 12 of the variables are 0, and
    the equalities then ask for Σ xᵢ >= 1.26 where h₁₃ asks for 1. The best known value is the
    report's 0.2049794002 (its table 4), at an infeasible x*, for the report's δ only (``optimum``
    is None for another). The equalities are undefined, and the fitness invalid, where x₁…x₁₂ or
    x₁₃…x₂₄ are all 0. ``tolerance`` is δ, :data:`EQUALITY_TOLERANCE` if None.

    The report's eqs. 41-42 and table 2 (pp. 12-13), after Himmelblau (1972).
    """

    tolerance: float | None = None
    _type: ClassVar[str] = "g20"


@dataclass(frozen=True)
class G21(_WithTolerance):
    """g21: ``x₁``, linear in 7 dimensions, with 1 nonlinear inequality and 5 nonlinear
    equalities.

    Bounds x₁ in [0, 1000], x₂, x₃ in [0, 40], x₄ in [100, 300], x₅ in [6.3, 6.7], x₆ in
    [5.9, 6.4], x₇ in [4.5, 6.25]; best known 193.724510070035, for the report's δ only
    (``optimum`` is None for another). ``tolerance`` is δ, :data:`EQUALITY_TOLERANCE` if None.

    The report's eqs. 43-44 (p. 13), after Epperly, T. Global optimization test problems with
    solutions (the report's reference 6).
    """

    tolerance: float | None = None
    _type: ClassVar[str] = "g21"


@dataclass(frozen=True)
class G22(_WithTolerance):
    """g22: ``x₁``, linear in 22 dimensions, with 1 nonlinear inequality, 8 linear and 11
    nonlinear equalities.

    Bounds x₁ in [0, 20000], x₂…x₄ in [0, 10⁶], x₅…x₇ in [0, 4·10⁷], x₈ in [100, 299.99], x₉ in
    [100, 399.99], x₁₀ in [100.01, 300], x₁₁ in [100, 400], x₁₂ in [100, 600], x₁₃…x₁₅ in
    [0, 500], x₁₆ in [0.01, 300], x₁₇ in [0.01, 400], x₁₈…x₂₂ in [−4.7, 6.25]; best known
    236.430975504001, for the report's δ only (``optimum`` is None for another). ``tolerance`` is
    δ, :data:`EQUALITY_TOLERANCE` if None.

    The report's eqs. 45-46 (pp. 13-14), after Epperly (the report's reference 6).
    """

    tolerance: float | None = None
    _type: ClassVar[str] = "g22"


@dataclass(frozen=True)
class G23(_WithTolerance):
    """g23: ``−9x₅ − 15x₈ + 6x₁ + 16x₂ + 10(x₆ + x₇)``, linear in 9 dimensions, with 2 nonlinear
    inequalities, 3 linear equalities and 1 nonlinear equality.

    Bounds x₁, x₂, x₆ in [0, 300], x₃, x₅, x₇ in [0, 100], x₄, x₈ in [0, 200], x₉ in
    [0.01, 0.03]; best known −400.055099999999584, for the report's δ only (``optimum`` is None
    for another). The report prints x* with a comma missing: x₈ = 200 and
    x₉ = 0.0100000100000100008. ``tolerance`` is δ, :data:`EQUALITY_TOLERANCE` if None.

    The report's eqs. 47-48 (pp. 14-15), after Xia, Q. Global optimization test problems (the
    report's reference 10).
    """

    tolerance: float | None = None
    _type: ClassVar[str] = "g23"


@dataclass(frozen=True)
class G24(Problem):
    """g24: ``−x₁ − x₂`` in 2 dimensions, subject to ``x₂ <= 2x₁⁴ − 8x₁³ + 8x₁² + 2`` and
    ``x₂ <= 4x₁⁴ − 32x₁³ + 88x₁² − 96x₁ + 36``.

    Bounds x₁ in [0, 3], x₂ in [0, 4]; minimum −5.50801327159536 at (2.32952019747762,
    3.17849307411774), where both constraints are active (the report prints x* with a comma
    missing). The feasible region's two parts meet at (1, 0).

    The report's eqs. 49-50 (p. 15), after Floudas, C. A. et al. (1999). Handbook of Test Problems
    in Local and Global Optimization. Kluwer.
    """

    _type: ClassVar[str] = "g24"
