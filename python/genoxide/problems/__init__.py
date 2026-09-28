"""Test problems from the literature, evaluated in Rust.

Each problem describes itself (its genome, objective, known optimum and reference) and is a
fitness function::

    import genoxide as gx

    problem = gx.problems.Rastrigin(10)
    de = gx.De(problem.genome, objective=problem.objective, seed=1)
    result = de.run(problem, target=problem.optimum.value + 1e-8, evaluations=200_000)
    print(result.best_fitness, result.evaluations)

``run`` evaluates a problem in Rust, with no Python call per genome: ``parallel=True`` uses every
core, and a seed gives the same result as the same Rust program. Calling ``problem(x)`` or
``problem.evaluate(genomes)`` runs the same Rust code.

The multi-objective problems, such as :class:`Zdt1`, :class:`Dtlz2`, :class:`Wfg4` and the
constrained :class:`Bnh`, derive from :class:`MultiProblem` and run with the multi-objective
algorithms. They give their ``objectives`` and, where it's known, their
``optimal_front(points)`` for :mod:`genoxide.indicators`::

    problem = gx.problems.Bnh()
    nsga2 = gx.Nsga2(
        problem.genome,
        objectives=problem.objectives,
        population_size=100,
        crossover=gx.SimulatedBinaryCrossover(20),
        mutation=gx.PolynomialMutation(20, rate=0.5),
        seed=1,
    )
    result = nsga2.run(problem, generations=100)
    front = problem.optimal_front(500)
    print(gx.indicators.igd_plus(result.front_objectives, front))

A constrained problem gives ``(objectives, violation)``, 0 when feasible, and its
``constraints(x)`` as ``g(x) <= 0``.

Two submodules hold constrained single-objective problems, whose fitness is ``(score,
violation)``: :mod:`genoxide.problems.cec2006` (g01 to g18 of the CEC 2006 competition) and
:mod:`genoxide.problems.engineering` (engineering design problems such as the welded beam and the
pressure vessel)::

    problem = gx.problems.engineering.PressureVessel()
    de = gx.De(problem.genome, objective=problem.objective, seed=1)
    result = de.run(problem, evaluations=20_000)
    print(result.best_fitness, result.violation, problem.design(result.best_genome))

All problems here are minimized, on :class:`genoxide.Real` genomes except the gear train's
:class:`genoxide.Integer` and :class:`Zdt5`'s :class:`genoxide.Binary`. Each class's docstring
gives
the function, its bounds, its optimum or front and its source. Many originals are books, reports
or proceedings that aren't online, and some functions have no known origin: their definitions
are taken from later papers that restate them, named in the docstrings, and are still to be
checked against the originals (https://github.com/tachsin/genoxide/issues/168), among them:

- Yao, X., Liu, Y. and Lin, G. (1999). Evolutionary programming made faster. IEEE Transactions on
  Evolutionary Computation 3(2): 82-102. doi:10.1109/4235.771163
- Laguna, M. and Martí, R. (2005). Experimental testing of advanced scatter search designs for
  global optimization of multimodal functions. Journal of Global Optimization 33(2): 235-255.
  doi:10.1007/s10898-004-1936-z
- Molga, M. and Smutnicki, C. (2005). Test functions for optimization needs.
- Jamil, M. and Yang, X.-S. (2013). A literature survey of benchmark functions for global
  optimisation problems. International Journal of Mathematical Modelling and Numerical
  Optimisation 4(2): 150-194. arXiv:1308.4008
- Deb, K., Pratap, A., Agarwal, S. and Meyarivan, T. (2002). A fast and elitist multiobjective
  genetic algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation 6(2): 182-197.
  doi:10.1109/4235.996017

The functions are evaluated in Rust with genoxide's portable math, so their values are the same to
the bit on every platform.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from typing import Any, ClassVar

import numpy as np

from .. import Binary, Integer, Real, _genoxide, _whole

__all__ = [
    "Problem",
    "MultiProblem",
    "Optimum",
    # single objective
    "Sphere",
    "AxisParallelEllipsoid",
    "Schwefel1_2",
    "Rastrigin",
    "Rosenbrock",
    "Ackley",
    "Griewank",
    "Schwefel2_26",
    "Levy",
    "Zakharov",
    "StyblinskiTang",
    "Michalewicz",
    "Himmelblau",
    "Branin",
    "GoldsteinPrice",
    "SixHumpCamel",
    "Hartmann3",
    "Hartmann6",
    "Shekel5",
    "Shekel7",
    "Shekel10",
    "Easom",
    "Eggholder",
    "SchafferF6",
    # multi-objective
    "Zdt1",
    "Zdt2",
    "Zdt3",
    "Zdt4",
    "Zdt5",
    "Zdt6",
    "Schaffer1",
    "Schaffer2",
    "FonsecaFleming",
    "Kursawe",
    "Poloni",
    "Viennet1",
    "Viennet2",
    "Viennet3",
    "Bnh",
    "Srn",
    "Tnk",
    "Osy",
    "Constr",
    "Dtlz1",
    "Dtlz2",
    "Dtlz3",
    "Dtlz4",
    "Dtlz5",
    "Dtlz6",
    "Dtlz7",
    "Wfg1",
    "Wfg2",
    "Wfg3",
    "Wfg4",
    "Wfg5",
    "Wfg6",
    "Wfg7",
    "Wfg8",
    "Wfg9",
]

@dataclass(frozen=True, eq=False)
class Optimum:
    """The optimum of a problem."""

    value: float
    """The optimal score."""
    solutions: np.ndarray
    """Genomes with this score, a row each: all of them when there are few (Himmelblau's four,
    Branin's three), one otherwise."""
    proven: bool
    """Whether the value is the global optimum (analytic or proven), rather than the best
    known."""


class _Described:
    """What every problem has: a description in JSON, from which Rust gives the rest."""

    _type: ClassVar[str]

    def _describe(self) -> dict[str, Any]:
        return {"type": self._type}

    def _json(self) -> str:
        return json.dumps(self._describe())

    @cached_property
    def _info(self) -> dict[str, Any]:
        return _genoxide.problem_info(self._json())

    @property
    def name(self) -> str:
        """The name, e.g. "Rastrigin" or "ZDT1"."""
        return self._info["name"]

    @property
    def dimensions(self) -> int:
        """The number of genes."""
        return len(self._info["bounds"])

    @property
    def genome(self) -> Real | Integer | Binary:
        """The search space, e.g. ``Real((-5.12, 5.12), length=10)``; ``Integer`` bounds for a
        problem of whole numbers, such as :class:`genoxide.problems.engineering.GearTrain`, and
        a ``Binary`` length for one of bit strings, such as :class:`Zdt5`."""
        if self._info["genome"] == "binary":
            return Binary(len(self._info["bounds"]))
        kind = Integer if self._info["genome"] == "integer" else Real
        bounds = [tuple(pair) for pair in self._info["bounds"]]
        if all(pair == bounds[0] for pair in bounds):
            return kind(bounds[0], length=len(bounds))
        return kind(bounds)

    @property
    def constraint_count(self) -> int:
        """The number of constraints: 0 for an unconstrained problem."""
        return self._info["constraints"]

    @property
    def reference(self) -> str:
        """The paper, book or report that defines the problem."""
        return self._info["reference"]

    @property
    def reference_url(self) -> str | None:
        """Its DOI or URL, if any."""
        return self._info["reference_url"]

    def constraints(self, genome: Any) -> np.ndarray:
        """The constraint values of ``genome``, a 1-D array, in the paper's order: the
        inequalities as ``g(x) <= 0``, then the equalities ``h(x) = 0``. Empty for an
        unconstrained problem.

        Raises
        ------
        ValueError
            If ``genome`` isn't a 1-D array of numbers with a value per dimension.
        """
        genome = np.ascontiguousarray(genome, dtype=np.float64)
        if genome.ndim != 1:
            raise ValueError(f"a genome is a 1-D array, not an array of shape {genome.shape}")
        return _genoxide.constraints(self._json(), genome)

    def _evaluate(self, genomes: Any) -> Any:
        genomes = np.ascontiguousarray(genomes, dtype=np.float64)
        if genomes.ndim != 2:
            raise ValueError(
                f"evaluate takes a 2-D array, a genome per row, not an array of shape "
                f"{genomes.shape}"
            )
        return _genoxide.evaluate(self._json(), genomes)

    def _one(self, genome: Any) -> Any:
        genome = np.asarray(genome, dtype=np.float64)
        if genome.ndim != 1:
            raise ValueError(f"a genome is a 1-D array, not an array of shape {genome.shape}")
        return self._evaluate(genome[np.newaxis, :])


class Problem(_Described):
    """A single-objective test problem: a fitness function with its genome, objective, optimum
    and reference.

    The single-objective problem classes derive from it; ``isinstance(x, Problem)`` tells such a
    problem from a Python fitness function.
    """

    @property
    def objective(self) -> str:
        """Whether the score is minimized or maximized: "minimize" for every problem here."""
        return self._info["objectives"][0]

    @property
    def optimum(self) -> Optimum | None:
        """The global optimum, or None if it isn't known for this size."""
        optimum = self._info["optimum"]
        if optimum is None:
            return None
        return Optimum(optimum["value"], optimum["solutions"], optimum["proven"])

    def evaluate(self, genomes: Any) -> Any:
        """The scores of ``genomes``, a 2-D array with a genome per row, as a 1-D array; for a
        constrained problem, a tuple of it and an array of constraint violations.

        Raises
        ------
        ValueError
            If ``genomes`` isn't a 2-D array of numbers with a column per dimension.
        """
        return self._evaluate(genomes)

    def __call__(self, genome: Any) -> Any:
        """The score of ``genome``, a 1-D array; for a constrained problem, a tuple of it and the
        constraint violation.

        Raises
        ------
        ValueError
            If ``genome`` isn't a 1-D array of numbers with a value per dimension.
        """
        result = self._one(genome)
        if isinstance(result, tuple):
            return float(result[0][0]), float(result[1][0])
        return float(result[0])


class MultiProblem(_Described):
    """A multi-objective test problem, all objectives minimized: a fitness function with its
    genome, objectives, optimal front where it's known, and reference.

    The multi-objective problem classes derive from it; ``isinstance(x, MultiProblem)`` tells
    such a problem from a Python fitness function. ``run`` of :class:`genoxide.Nsga2`,
    :class:`genoxide.Nsga3`, :class:`genoxide.Spea2`, :class:`genoxide.Moead` and
    :class:`genoxide.SmsEmoa` evaluates it in Rust.
    """

    @property
    def objectives(self) -> list[str]:
        """"minimize" for each objective, for the ``objectives`` of an algorithm."""
        return list(self._info["objectives"])

    @property
    def ideal_point(self) -> np.ndarray | None:
        """The best value of each objective on the optimal front, or None if unknown."""
        point = self._info["ideal_point"]
        return None if point is None else np.array(point)

    @property
    def nadir_point(self) -> np.ndarray | None:
        """The worst value of each objective on the optimal front, or None if unknown."""
        point = self._info["nadir_point"]
        return None if point is None else np.array(point)

    def optimal_front(self, points: int) -> np.ndarray | None:
        """At least ``points`` points of the optimal front, a row each (exactly ``points`` for 2
        objectives), for :mod:`genoxide.indicators`; None if the front isn't known.
        """
        points = _whole("points", points)
        return _genoxide.optimal_front(self._json(), points)

    def evaluate(self, genomes: Any) -> Any:
        """The objective values of ``genomes``, a 2-D array with a genome per row, as a 2-D array
        with a row per genome; for a constrained problem, a tuple of it and an array of
        constraint violations.

        Raises
        ------
        ValueError
            If ``genomes`` isn't a 2-D array of numbers with a column per dimension.
        """
        return self._evaluate(genomes)

    def __call__(self, genome: Any) -> Any:
        """The objective values of ``genome``, a 1-D array, as a 1-D array; for a constrained
        problem, a tuple of it and the constraint violation.

        Raises
        ------
        ValueError
            If ``genome`` isn't a 1-D array of numbers with a value per dimension.
        """
        result = self._one(genome)
        if isinstance(result, tuple):
            return result[0][0], float(result[1][0])
        return result[0]


class _Scalable(Problem):
    """A problem in any number of dimensions, from ``_minimum``."""

    dimensions: int
    _minimum: ClassVar[int] = 1

    def _describe(self) -> dict[str, Any]:
        name = f"{type(self).__name__}.dimensions"
        dimensions = _whole(name, self.dimensions, minimum=self._minimum)
        return {"type": self._type, "dimensions": dimensions}


@dataclass(frozen=True)
class Sphere(_Scalable):
    """The sphere, ``Σ xᵢ²``, De Jong's F1: the simplest unimodal function.

    Bounds [-100, 100]ⁿ; minimum 0 at the origin. ``dimensions`` is at least 1.

    De Jong, K. A. (1975). An Analysis of the Behavior of a Class of Genetic Adaptive Systems. PhD
    thesis, University of Michigan. Definition, bounds and dimensions as restated in Yao, Liu and
    Lin (1999, f1); De Jong's F1 is the same function in 3 dimensions on [-5.12, 5.12], as
    restated by Pohlheim (GEATbx) and Laguna and Martí (2005), and Schwefel (1977, problem 1.1)
    states it in any dimension, unbounded. Not yet checked against De Jong's thesis (#168).
    """

    dimensions: int = 30
    _type: ClassVar[str] = "sphere"


@dataclass(frozen=True)
class AxisParallelEllipsoid(_Scalable):
    """The axis-parallel hyper-ellipsoid, ``Σ i xᵢ²`` (i from 1): a sphere stretched along each
    axis. Not the rotated hyper-ellipsoid, and not :class:`Schwefel1_2`.

    Bounds [-5.12, 5.12]ⁿ; minimum 0 at the origin. ``dimensions`` is at least 1.

    Its origin is unknown. The earliest source found is Pohlheim's GEATbx documentation (function
    1a, "the weighted sphere model"), which Molga and Smutnicki (2005, section 2.2) restate with
    the same bounds; it isn't among Schwefel's (1977) problems. Not yet checked against an
    original (#168).
    """

    dimensions: int = 30
    _type: ClassVar[str] = "axis_parallel_ellipsoid"


@dataclass(frozen=True)
class Schwefel1_2(_Scalable):
    """Schwefel's problem 1.2, ``Σᵢ (Σⱼ≤ᵢ xⱼ)²``: unimodal, with strongly interacting genes.

    Bounds [-100, 100]ⁿ; minimum 0 at the origin. ``dimensions`` is at least 1.

    Schwefel, H.-P. (1981). Numerical Optimization of Computer Models. Wiley, problem 1.2, the
    translation of Numerische Optimierung von Computer-Modellen (1977, Birkhäuser, p. 319), which
    states it in any dimension and unbounded, with its minimum 0 at the origin. The bounds are
    Yao, Liu and Lin's (1999, f3).
    """

    dimensions: int = 30
    _type: ClassVar[str] = "schwefel_1_2"


@dataclass(frozen=True)
class Rastrigin(_Scalable):
    """Rastrigin's function, ``10n + Σ (xᵢ² − 10 cos 2πxᵢ)``: a sphere plus a cosine, with a
    local minimum near every integer point.

    Bounds [-5.12, 5.12]ⁿ; minimum 0 at the origin. ``dimensions`` is at least 1.

    Rastrigin, L. A. (1974). Systems of Extremal Control. Nauka, Moscow, whose function is a
    related one in two dimensions with other constants, as Törn and Žilinskas (1989) restate it.
    The n-dimensional form is credited to Rudolph (1990), and was spread by Hoffmeister and Bäck
    (1991) and Mühlenbein, H., Schomisch, M. and Born, J. (1991). The parallel genetic algorithm
    as function optimizer. Parallel Computing 17(6-7): 619-632, whose F6 this is. Definition and
    bounds as in Yao, Liu and Lin (1999, f9); not yet checked against Rastrigin's book (#168).
    """

    dimensions: int = 30
    _type: ClassVar[str] = "rastrigin"


@dataclass(frozen=True)
class Rosenbrock(_Scalable):
    """Rosenbrock's function, chained: ``Σᵢ₌₁ⁿ⁻¹ [100 (xᵢ₊₁ − xᵢ²)² + (xᵢ − 1)²]``, a narrow
    curved valley.

    Bounds [-30, 30]ⁿ; minimum 0 at (1, …, 1). ``dimensions`` is at least 2.

    Rosenbrock, H. H. (1960). An automatic method for finding the greatest or least value of a
    function. The Computer Journal 3(3): 175-184, in two dimensions, with no bounds. This is the
    chained n-dimensional form and the bounds of Yao, Liu and Lin (1999, f5), not the pairwise
    "extended" form.
    """

    dimensions: int = 30
    _type: ClassVar[str] = "rosenbrock"
    _minimum: ClassVar[int] = 2


@dataclass(frozen=True)
class Ackley(_Scalable):
    """Ackley's function, ``−20 exp(−0.2 √(Σ xᵢ² / n)) − exp(Σ cos(2πxᵢ) / n) + 20 + e``: a
    nearly flat outer region around a deep hole, covered in local minima.

    Bounds [-32, 32]ⁿ; minimum 0 at the origin. ``dimensions`` is at least 1.

    Ackley, D. H. (1987). A Connectionist Machine for Genetic Hillclimbing. Kluwer (two
    dimensions); generalized by Bäck, T. (1996). Evolutionary Algorithms in Theory and Practice.
    Oxford University Press. Definition and bounds as restated in Yao, Liu and Lin (1999, f10);
    not yet checked against the originals (#168).
    """

    dimensions: int = 30
    _type: ClassVar[str] = "ackley"


@dataclass(frozen=True)
class Griewank(_Scalable):
    """Griewank's function, ``1 + Σ xᵢ² / 4000 − Π cos(xᵢ / √i)`` (i from 1).

    Bounds [-600, 600]ⁿ; minimum 0 at the origin. ``dimensions`` is at least 1.

    Griewank, A. O. (1981). Generalized descent for global optimization. Journal of Optimization
    Theory and Applications 34(1): 11-39, whose function, as Bosse and Bücker (2024) restate it, is
    two-dimensional with the divisor 200: 1 + (x₁² + x₂²) / 200 − cos x₁ cos(x₂ / √2). The
    n-dimensional form with the divisor 4000 and the bounds are those of Mühlenbein, Schomisch and
    Born (1991, F8) and Yao, Liu and Lin (1999, f11). Not yet checked against the original (#168).
    """

    dimensions: int = 30
    _type: ClassVar[str] = "griewank"


@dataclass(frozen=True)
class Schwefel2_26(_Scalable):
    """Schwefel's problem 2.26, ``−Σ xᵢ sin √|xᵢ|``: the best local minima are far apart.

    Bounds [-500, 500]ⁿ; minimum −418.9828872724337 n at xᵢ = 420.96874635998205, derived from the
    formula. ``dimensions`` is at least 1. This is the form without an offset.

    Schwefel, H.-P. (1981). Numerical Optimization of Computer Models. Wiley, problem 2.26, the
    translation of Numerische Optimierung von Computer-Modellen (1977, Birkhäuser, p. 335), where
    it has one variable, −x₁ sin √|x₁|, no bounds and so no finite minimum (its problem 2.44 bounds
    it to [-300, 300]). The sum over n variables on [-500, 500] is Mühlenbein, Schomisch and Born's
    (1991, F7), restated in Yao, Liu and Lin (1999, f8).
    """

    dimensions: int = 30
    _type: ClassVar[str] = "schwefel_2_26"


@dataclass(frozen=True)
class Levy(_Scalable):
    """Levy's function: with ``wᵢ = 1 + (xᵢ − 1) / 4``,
    ``sin²(πw₁) + Σᵢ₌₁ⁿ⁻¹ (wᵢ − 1)² [1 + 10 sin²(πwᵢ + 1)] + (wₙ − 1)² [1 + sin²(2πwₙ)]``.

    Bounds [-10, 10]ⁿ; minimum 0 at (1, …, 1). ``dimensions`` is at least 1.

    Usually credited to Levy, A. V. and Montalvo, A. (1985). The tunneling algorithm for the
    global minimization of functions. SIAM Journal on Scientific and Statistical Computing 6(1):
    15-29, but several functions carry Levy's name. This is the form of Surjanovic and Bingham's
    Virtual Library of Simulation Experiments, with sin²(πwᵢ + 1) and bounds [-10, 10]; Laguna
    and Martí (2005, function 38) print xₙ instead of wₙ in the last sine (this uses wₙ), and the
    Levy-Montalvo functions as restated in Yao, Liu and Lin (1999, f12 and f13) have sin²(πyᵢ₊₁)
    instead, of which πwᵢ + 1 may be a misreading. The minimum is 0 at (1, …, 1) in every form.
    Not yet checked against the original (#168).
    """

    dimensions: int = 30
    _type: ClassVar[str] = "levy"


@dataclass(frozen=True)
class Zakharov(_Scalable):
    """Zakharov's function, ``Σ xᵢ² + (Σ 0.5 i xᵢ)² + (Σ 0.5 i xᵢ)⁴`` (i from 1).

    Bounds [-5, 10]ⁿ; minimum 0 at the origin. ``dimensions`` is at least 1.

    Its origin is unknown: definition and bounds as restated in Laguna and Martí (2005, function
    12); not yet checked against an original (#168).
    """

    dimensions: int = 30
    _type: ClassVar[str] = "zakharov"


@dataclass(frozen=True)
class StyblinskiTang(_Scalable):
    """The Styblinski-Tang function, ``½ Σ (xᵢ⁴ − 16xᵢ² + 5xᵢ)``: separable.

    Bounds [-5, 5]ⁿ; minimum −39.16616570377141 n at xᵢ = −2.903534027771177, derived from the
    formula. ``dimensions`` is at least 1.

    Styblinski, M. A. and Tang, T.-S. (1990). Experiments in nonconvex optimization: stochastic
    approximation with function smoothing and simulated annealing. Neural Networks 3(4): 467-483.
    Definition and bounds as restated in Jamil and Yang (2013, function 144); not yet checked
    against the original (#168).
    """

    dimensions: int = 30
    _type: ClassVar[str] = "styblinski_tang"


@dataclass(frozen=True)
class Michalewicz(_Scalable):
    """Michalewicz's function, ``−Σ sin(xᵢ) sin²ᵐ(i xᵢ² / π)`` with m = 10 (i from 1): steep
    narrow valleys on flat plateaus.

    Bounds [0, π]ⁿ. ``dimensions`` is at least 1. The minimum depends on n: the function is
    separable, so ``optimum`` minimizes each term on its own, which gives −1.8013034 for n = 2,
    −4.6876582 for n = 5 and −9.6601517 for n = 10.

    Michalewicz, Z. (1992). Genetic Algorithms + Data Structures = Evolution Programs. Springer.
    Definition, m and bounds as restated in Molga and Smutnicki (2005, section 2.11); not yet
    checked against the original (#168).
    """

    dimensions: int = 10
    _type: ClassVar[str] = "michalewicz"


@dataclass(frozen=True)
class Himmelblau(Problem):
    """Himmelblau's function, ``(x₁² + x₂ − 11)² + (x₁ + x₂² − 7)²``: four global minima.

    Bounds [-5, 5]²; minimum 0 at (3, 2) and three other points, the solutions of x₁² + x₂ = 11
    and x₁ + x₂² = 7, computed by Newton's method.

    Himmelblau, D. M. (1972). Applied Nonlinear Programming. McGraw-Hill. Definition and bounds as
    restated in Jamil and Yang (2013, function 65); not yet checked against the original (#168).
    """

    _type: ClassVar[str] = "himmelblau"


@dataclass(frozen=True)
class Branin(Problem):
    """Branin's function (RCOS),
    ``(x₂ − 5.1 x₁² / (4π²) + 5 x₁ / π − 6)² + 10 (1 − 1 / (8π)) cos x₁ + 10``: three global
    minima.

    Bounds x₁ ∈ [-5, 10], x₂ ∈ [0, 15]; minimum 5 / (4π) at (−π, 12.275), (π, 2.275) and
    (3π, 2.475), derived from the formula.

    Branin, F. H. (1972). Widely convergent method for finding multiple solutions of simultaneous
    nonlinear equations. IBM Journal of Research and Development 16(5): 504-522. Definition and
    bounds as restated in Yao, Liu and Lin (1999, f17); not yet checked against the original
    (#168).
    """

    _type: ClassVar[str] = "branin"


@dataclass(frozen=True)
class GoldsteinPrice(Problem):
    """The Goldstein-Price function.

    Bounds [-2, 2]²; minimum 3 at (0, −1); local minima (1.2, 0.8) with 840, (1.8, 0.2) with 84
    and (−0.6, −0.4) with 30. The minimum is global: with s = x₁ + x₂ the first factor is
    1 + (s + 1)² (3s² − 14s + 19) >= 1, and with t = 2x₁ − 3x₂ the second is
    30 + t² (3t² − 16t + 18) >= 3, both met only at (0, −1).

    Goldstein, A. A. and Price, J. F. (1971). On descent from local minima. Mathematics of
    Computation 25(115): 569-574. The paper gives no bounds: these are Dixon and Szegö's (1978),
    as restated in Yao, Liu and Lin (1999, f18).
    """

    _type: ClassVar[str] = "goldstein_price"


@dataclass(frozen=True)
class SixHumpCamel(Problem):
    """The six-hump camel-back function, ``(4 − 2.1x₁² + x₁⁴ / 3) x₁² + x₁x₂ + (−4 + 4x₂²) x₂²``:
    two global minima among six.

    Bounds [-5, 5]²; minimum −1.0316284534898774 at ±(0.08984201310031806, −0.7126564030207396),
    computed by Newton's method. It's the global minimum: of the 15 real stationary points, this
    pair is the lowest, the next −0.2154638 at ±(1.70361, −0.79608).

    Dixon, L. C. W. and Szegö, G. P. (eds.) (1978). Towards Global Optimisation 2. North-Holland.
    Definition and bounds as restated in Yao, Liu and Lin (1999, f16); not yet checked against the
    original (#168).
    """

    _type: ClassVar[str] = "six_hump_camel"


@dataclass(frozen=True)
class Hartmann3(Problem):
    """Hartmann's function in 3 dimensions, ``−Σᵢ₌₁⁴ cᵢ exp(−Σⱼ aᵢⱼ (xⱼ − pᵢⱼ)²)``: four
    Gaussian wells of different widths and depths.

    c = (1, 1.2, 3, 3.2); the rows of a are (3, 10, 30), (0.1, 10, 35), (3, 10, 30) and
    (0.1, 10, 35), and those of p (0.3689, 0.1170, 0.2673), (0.4699, 0.4387, 0.7470),
    (0.1091, 0.8732, 0.5547) and (0.03815, 0.5743, 0.8828).

    Bounds [0, 1]³; minimum −3.862782147820755 at (0.11461433858967196, 0.5556488499718569,
    0.8525469535208658), where the gradient is 0, computed by Newton's method; −3.86278 is the
    value later papers quote from Dixon and Szegö. The best of the local minima that searches
    from random points find, but not proven global (``optimum.proven`` is False).

    The form is Hartman's: Hartman, J. K. (1972). Some Experiments in Global Optimization. Report
    NPS-55HH72051A, Naval Postgraduate School, published in Naval Research Logistics Quarterly
    20(3): 569-576 (1973), with random constants. These constants are those of Dixon, L. C. W.
    and Szegö, G. P. (1978). The global optimisation problem: an introduction. In Towards Global
    Optimisation 2, North-Holland: 1-15, as Yao, Liu and Lin (1999, table XII) reprint them.
    """

    _type: ClassVar[str] = "hartmann3"


@dataclass(frozen=True)
class Hartmann6(Problem):
    """Hartmann's function in 6 dimensions, ``−Σᵢ₌₁⁴ cᵢ exp(−Σⱼ aᵢⱼ (xⱼ − pᵢⱼ)²)``: four
    Gaussian wells of different widths and depths, in two basins of nearly the same depth.

    c = (1, 1.2, 3, 3.2); the rows of a are (10, 3, 17, 3.5, 1.7, 8), (0.05, 10, 17, 0.1, 8, 14),
    (3, 3.5, 1.7, 10, 17, 8) and (17, 8, 0.05, 10, 0.1, 14), and those of p
    (0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886),
    (0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991),
    (0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650) and
    (0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381).

    Bounds [0, 1]⁶; minimum −3.3223680114155147 at (0.20168951100670543, 0.15001069182345797,
    0.476873974221897, 0.2753324304940561, 0.31165161660011326, 0.6573005340656204), computed by
    Newton's method; −3.32237 is the value later papers quote from Dixon and Szegö. The other
    local minimum, −3.2031619, draws a third of local searches. Not proven global
    (``optimum.proven`` is False).

    The form is Hartman's (1972, 1973), the constants Dixon and Szegö's (1978), as Yao, Liu and
    Lin (1999, table XIII) reprint them but for p₃₂, which they print as 0.1415.
    """

    _type: ClassVar[str] = "hartmann6"


@dataclass(frozen=True)
class Shekel5(Problem):
    """Shekel's function with m = 5 wells, in 4 dimensions (SQRIN5).

    Minimum −10.153199679058227 at (4.000037152819676, 4.00013327659156, 4.000037152819676,
    4.00013327659156); later papers quote −10.1532 from Dixon and Szegö.

    The function is ``−Σᵢ₌₁ᵐ 1 / ((x − aᵢ)ᵀ(x − aᵢ) + cᵢ)``, a well at each of the first m of
    the points a = (4, 4, 4, 4), (1, 1, 1, 1), (8, 8, 8, 8), (6, 6, 6, 6), (3, 7, 3, 7),
    (2, 9, 2, 9), (5, 5, 3, 3), (8, 1, 8, 1), (6, 2, 6, 2), (7, 3.6, 7, 3.6), with
    c = (0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5). Bounds [0, 10]⁴. The minimum is near
    (4, 4, 4, 4) but not at it; computed by Newton's method, and not proven global
    (``optimum.proven`` is False).

    Shekel, J. (1971). Test functions for multimodal search techniques. Proceedings of the 5th
    Annual Princeton Conference on Information Sciences and Systems, and Dixon, L. C. W. and
    Szegö, G. P. (1978). The global optimisation problem: an introduction. In Towards Global
    Optimisation 2, North-Holland: 1-15. Neither is online: the constants as Yao, Liu and Lin
    (1999, table XIV) reprint them.
    """

    _type: ClassVar[str] = "shekel5"


@dataclass(frozen=True)
class Shekel7(Problem):
    """Shekel's function with m = 7 wells, in 4 dimensions (SQRIN7).

    Minimum −10.40294056681866 at (4.000572916185823, 4.000689366185305, 3.9994897088591506,
    3.9996061588586316); later papers quote −10.4029 from Dixon and Szegö.

    The function is ``−Σᵢ₌₁ᵐ 1 / ((x − aᵢ)ᵀ(x − aᵢ) + cᵢ)``, a well at each of the first m of
    the points a = (4, 4, 4, 4), (1, 1, 1, 1), (8, 8, 8, 8), (6, 6, 6, 6), (3, 7, 3, 7),
    (2, 9, 2, 9), (5, 5, 3, 3), (8, 1, 8, 1), (6, 2, 6, 2), (7, 3.6, 7, 3.6), with
    c = (0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5). Bounds [0, 10]⁴. The minimum is near
    (4, 4, 4, 4) but not at it; computed by Newton's method, and not proven global
    (``optimum.proven`` is False).

    Shekel, J. (1971). Test functions for multimodal search techniques. Proceedings of the 5th
    Annual Princeton Conference on Information Sciences and Systems, and Dixon, L. C. W. and
    Szegö, G. P. (1978). The global optimisation problem: an introduction. In Towards Global
    Optimisation 2, North-Holland: 1-15. Neither is online: the constants as Yao, Liu and Lin
    (1999, table XIV) reprint them.
    """

    _type: ClassVar[str] = "shekel7"


@dataclass(frozen=True)
class Shekel10(Problem):
    """Shekel's function with m = 10 wells, in 4 dimensions (SQRIN10).

    Minimum −10.536409816692043 at (4.000746531592046, 4.000592934138532, 3.9996633980403224,
    3.9995098005868077); later papers quote −10.5364 from Dixon and Szegö.

    The function is ``−Σᵢ₌₁ᵐ 1 / ((x − aᵢ)ᵀ(x − aᵢ) + cᵢ)``, a well at each of the first m of
    the points a = (4, 4, 4, 4), (1, 1, 1, 1), (8, 8, 8, 8), (6, 6, 6, 6), (3, 7, 3, 7),
    (2, 9, 2, 9), (5, 5, 3, 3), (8, 1, 8, 1), (6, 2, 6, 2), (7, 3.6, 7, 3.6), with
    c = (0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5). Bounds [0, 10]⁴. The minimum is near
    (4, 4, 4, 4) but not at it; computed by Newton's method, and not proven global
    (``optimum.proven`` is False).

    Shekel, J. (1971). Test functions for multimodal search techniques. Proceedings of the 5th
    Annual Princeton Conference on Information Sciences and Systems, and Dixon, L. C. W. and
    Szegö, G. P. (1978). The global optimisation problem: an introduction. In Towards Global
    Optimisation 2, North-Holland: 1-15. Neither is online: the constants as Yao, Liu and Lin
    (1999, table XIV) reprint them.
    """

    _type: ClassVar[str] = "shekel10"


@dataclass(frozen=True)
class Easom(Problem):
    """Easom's function, ``−cos x₁ cos x₂ exp(−((x₁ − π)² + (x₂ − π)²))``: a single narrow well
    in a flat plane.

    Bounds [-100, 100]²; minimum −1 at (π, π), the global minimum: both factors are at most 1 in
    absolute value, and the exponential is 1 only at (π, π). Farther than 4.8 from it, the value
    is within 1e-10 of 0.

    Easom, E. E. (1990). A Survey of Global Optimization Techniques. M.Eng. thesis, University of
    Louisville. The thesis couldn't be read: definition and bounds as restated in Jamil and Yang
    (2013, function 50); not yet checked against the original (#168).
    """

    _type: ClassVar[str] = "easom"


@dataclass(frozen=True)
class Eggholder(Problem):
    """The eggholder function,
    ``−(x₂ + 47) sin √|x₂ + x₁ / 2 + 47| − x₁ sin √|x₁ − (x₂ + 47)|``: deep local minima all
    over, the deepest at the edge of the box.

    Bounds [-512, 512]²; minimum −959.6406627208508 at (512, 404.2318051137578), on the bound,
    computed by Newton's method; the next, −956.9182316, inside the box at (482.35331, 432.87900).
    Not proven global (``optimum.proven`` is False).

    Whitley, D., Mathias, K., Rana, S. and Dzubera, J. (1996). Evaluating evolutionary algorithms.
    Artificial Intelligence 85(1-2): 245-276, section 4.2, where it is F101, on [−512, 511] (where
    the minimum is −956.9182). The name and the bounds [−512, 512] are Mishra's (2006, MPRA paper
    2718).
    """

    _type: ClassVar[str] = "eggholder"


@dataclass(frozen=True)
class SchafferF6(Problem):
    """Schaffer's F6, ``0.5 + (sin² √(x₁² + x₂²) − 0.5) / (1 + 0.001 (x₁² + x₂²))²``: rings of
    local minima around the global one.

    Bounds [-100, 100]²; minimum 0 at the origin, the global minimum: the numerator is at least
    −0.5 and the denominator at least 1. The first ring of local minima, at a distance of
    3.1384848 from the origin, has 0.0097159.

    Schaffer, J. D., Caruana, R. A., Eshelman, L. J. and Das, R. (1989). A study of control
    parameters affecting online performance of genetic algorithms for function optimization.
    Proceedings of the Third International Conference on Genetic Algorithms: 51-60. Definition
    and bounds as Whitley, Mathias, Rana and Dzubera (1996, table 1, F9) and the CEC 2005 report
    restate it; not yet checked against the original (#168).
    """

    _type: ClassVar[str] = "schaffer_f6"


# ---- multi-objective problems -------------------------------------------------------------------


class _Sized(MultiProblem):
    """A multi-objective problem in any number of variables, from ``_minimum``."""

    variables: int
    _minimum: ClassVar[int] = 2

    def _describe(self) -> dict[str, Any]:
        name = f"{type(self).__name__}.variables"
        variables = _whole(name, self.variables, minimum=self._minimum)
        return {"type": self._type, "variables": variables}


@dataclass(frozen=True)
class Zdt1(_Sized):
    """ZDT1: a convex front, ``f₂ = 1 − √f₁`` for f₁ in [0, 1].

    ``f₁ = x₁``, ``g = 1 + 9 Σᵢ₌₂ⁿ xᵢ / (n − 1)``, ``f₂ = g (1 − √(f₁ / g))``, on [0, 1]ⁿ;
    ``variables`` is at least 2.

    Zitzler, E., Deb, K. and Thiele, L. (2000). Comparison of multiobjective evolutionary
    algorithms: empirical results. Evolutionary Computation 8(2): 173-195.
    """

    variables: int = 30
    _type: ClassVar[str] = "zdt1"


@dataclass(frozen=True)
class Zdt2(_Sized):
    """ZDT2: a concave front, ``f₂ = 1 − f₁²`` for f₁ in [0, 1]. As :class:`Zdt1`, with
    ``f₂ = g (1 − (f₁ / g)²)``.

    Zitzler, E., Deb, K. and Thiele, L. (2000). Comparison of multiobjective evolutionary
    algorithms: empirical results. Evolutionary Computation 8(2): 173-195.
    """

    variables: int = 30
    _type: ClassVar[str] = "zdt2"


@dataclass(frozen=True)
class Zdt3(_Sized):
    """ZDT3: a front of five disconnected pieces. As :class:`Zdt1`, with
    ``f₂ = g (1 − √(f₁ / g) − (f₁ / g) sin(10π f₁))``.

    Zitzler, E., Deb, K. and Thiele, L. (2000). Comparison of multiobjective evolutionary
    algorithms: empirical results. Evolutionary Computation 8(2): 173-195.
    """

    variables: int = 30
    _type: ClassVar[str] = "zdt3"


@dataclass(frozen=True)
class Zdt4(_Sized):
    """ZDT4: the convex front of ZDT1 behind 21⁹ local fronts: ``x₁`` in [0, 1] and the other
    variables in [−5, 5], with ``g = 1 + 10 (n − 1) + Σᵢ₌₂ⁿ (xᵢ² − 10 cos 4πxᵢ)``.

    Zitzler, E., Deb, K. and Thiele, L. (2000). Comparison of multiobjective evolutionary
    algorithms: empirical results. Evolutionary Computation 8(2): 173-195.
    """

    variables: int = 10
    _type: ClassVar[str] = "zdt4"


@dataclass(frozen=True)
class Zdt6(_Sized):
    """ZDT6: a concave front, ``f₂ = 1 − f₁²`` for f₁ from 0.2807753188 to 1, with solutions
    dense near its upper end: ``f₁ = 1 − exp(−4x₁) sin⁶(6πx₁)``,
    ``g = 1 + 9 (Σᵢ₌₂ⁿ xᵢ / (n − 1))^0.25``.

    Zitzler, E., Deb, K. and Thiele, L. (2000). Comparison of multiobjective evolutionary
    algorithms: empirical results. Evolutionary Computation 8(2): 173-195.
    """

    variables: int = 10
    _type: ClassVar[str] = "zdt6"


@dataclass(frozen=True)
class Zdt5(MultiProblem):
    """ZDT5: a deceptive problem on bit strings, whose front is 31 points, ``f₂ = 10 / f₁`` for
    f₁ from 1 to 31.

    The genome is a :class:`genoxide.Binary` of 80 bits: x₁ of 30 bits, then x₂ … x₁₁ of 5 bits
    each. ``f₁ = 1 + u(x₁)``, u counting the ones; ``g = Σᵢ₌₂¹¹ v(u(xᵢ))``, with
    ``v(u) = 2 + u`` for u < 5 and ``v(5) = 1``; ``f₂ = g / f₁``. The optimal solutions have
    x₂ … x₁₁ all ones, g = 10. Each 5-bit substring is deceptive, drawn to all zeros (v = 2): the
    best deceptive front has g = 11. ``optimal_front(31)`` gives the 31 points; fewer are spread
    evenly over them, and more repeat some.

    ``Zdt5(first_bits=30, substrings=10)``: other sizes have ``first_bits`` bits in x₁ and
    ``substrings`` substrings of 5 bits, at least 1 each, and the front ``f₂ = substrings / f₁``
    for f₁ from 1 to ``first_bits + 1``. The problem takes genomes of 0s and 1s, or booleans.

    Zitzler, E., Deb, K. and Thiele, L. (2000). Comparison of multiobjective evolutionary
    algorithms: empirical results. Evolutionary Computation 8(2): 173-195, eq. 11, p. 178.
    """

    first_bits: int = 30
    substrings: int = 10
    _type: ClassVar[str] = "zdt5"

    def _describe(self) -> dict[str, Any]:
        first_bits = _whole("Zdt5.first_bits", self.first_bits, minimum=1)
        substrings = _whole("Zdt5.substrings", self.substrings, minimum=1)
        return {"type": self._type, "first_bits": first_bits, "substrings": substrings}


@dataclass(frozen=True, init=False)
class _Dtlz(MultiProblem):
    """DTLZ with ``objectives`` objectives, 2 to 6, and ``variables`` variables, at least
    ``objectives``; None is the standard ``objectives + k − 1``. The field ``objective_count``
    keeps the number, as the ``objectives`` property lists the objectives."""

    objective_count: int
    variables: int | None

    def __init__(self, objectives: int = 3, variables: int | None = None) -> None:
        object.__setattr__(self, "objective_count", objectives)
        object.__setattr__(self, "variables", variables)

    def _describe(self) -> dict[str, Any]:
        name = type(self).__name__
        objectives = _whole(f"{name}.objectives", self.objective_count, minimum=2)
        if objectives > 6:
            raise ValueError(f"{name}.objectives is at most 6, not {objectives}")
        variables = self.variables
        if variables is not None:
            variables = _whole(f"{name}.variables", variables, minimum=objectives)
        return {"type": self._type, "objectives": objectives, "variables": variables}


@dataclass(frozen=True, init=False)
class Dtlz1(_Dtlz):
    """DTLZ1: a linear front, the objectives summing to 1/2, behind 11ᵏ − 1 local fronts; k = 5
    by default. ``Dtlz1(objectives=3, variables=None)``: 2 to 6 objectives, and at least as many
    variables, None for ``objectives + 4``.

    Deb, K., Thiele, L., Laumanns, M. and Zitzler, E. (2002). Scalable multi-objective
    optimization test problems. Proceedings of the 2002 Congress on Evolutionary Computation:
    825-830.
    """

    _type: ClassVar[str] = "dtlz1"


@dataclass(frozen=True, init=False)
class Dtlz2(_Dtlz):
    """DTLZ2: a spherical front, the squared objectives summing to 1.
    ``Dtlz2(objectives=3, variables=None)``: 2 to 6 objectives, and at least as many variables,
    None for ``objectives + 9``.

    Deb, K., Thiele, L., Laumanns, M. and Zitzler, E. (2002). Scalable multi-objective
    optimization test problems. Proceedings of the 2002 Congress on Evolutionary Computation:
    825-830.
    """

    _type: ClassVar[str] = "dtlz2"


@dataclass(frozen=True, init=False)
class Dtlz3(_Dtlz):
    """DTLZ3: the spherical front of DTLZ2 behind 3ᵏ − 1 local fronts.
    ``Dtlz3(objectives=3, variables=None)``: 2 to 6 objectives, and at least as many variables,
    None for ``objectives + 9``.

    Deb, K., Thiele, L., Laumanns, M. and Zitzler, E. (2002). Scalable multi-objective
    optimization test problems. Proceedings of the 2002 Congress on Evolutionary Computation:
    825-830.
    """

    _type: ClassVar[str] = "dtlz3"


@dataclass(frozen=True, init=False)
class Dtlz4(_Dtlz):
    """DTLZ4: the spherical front of DTLZ2, with solutions biased towards some objectives.
    ``Dtlz4(objectives=3, variables=None)``: 2 to 6 objectives, and at least as many variables,
    None for ``objectives + 9``.

    Deb, K., Thiele, L., Laumanns, M. and Zitzler, E. (2002). Scalable multi-objective
    optimization test problems. Proceedings of the 2002 Congress on Evolutionary Computation:
    825-830.
    """

    _type: ClassVar[str] = "dtlz4"


@dataclass(frozen=True, init=False)
class Dtlz5(_Dtlz):
    """DTLZ5 (the technical report's numbering; not in the 2002 paper): DTLZ2 with its angles
    mapped so that the front is a curve, for 2 and 3 objectives.
    ``Dtlz5(objectives=3, variables=None)``: 2 to 6 objectives, and at least as many variables,
    None for ``objectives + 9``.

    ``g = Σ (xᵢ − 0.5)²`` over the last k variables, ``θ₁ = x₁π/2``,
    ``θᵢ = π (1 + 2g xᵢ) / (4 (1 + g))`` for the others, and the objectives of DTLZ2 at these
    angles, on a sphere of radius ``1 + g``. The report's eq. 25 writes ``cos(θᵢπ/2)`` for
    ``cos θᵢ`` and leaves θ₁ undefined; its eqs. 8 and 10 give this reading. The optimal
    solutions have the last k variables at 0.5. The front is DTLZ2's quarter circle for 2
    objectives, and the curve ``f₁ = f₂ = cos θ₁ / √2``, ``f₃ = sin θ₁`` for 3. For 4 or more
    it isn't a curve (Huband et al., 2006) and isn't known: ``optimal_front`` and
    ``nadir_point`` are None, and ``ideal_point`` the origin.

    Deb, K., Thiele, L., Laumanns, M. and Zitzler, E. (2001). Scalable Test Problems for
    Evolutionary Multi-Objective Optimization. TIK-Report 112, ETH Zürich, eq. 25, p. 20.
    """

    _type: ClassVar[str] = "dtlz5"


@dataclass(frozen=True, init=False)
class Dtlz6(_Dtlz):
    """DTLZ6 (the technical report's numbering; the 2002 paper's DTLZ5): :class:`Dtlz5` with
    ``g = Σ xᵢ^0.1`` over the last k variables, which makes the same front harder to reach.
    ``Dtlz6(objectives=3, variables=None)``: 2 to 6 objectives, and at least as many variables,
    None for ``objectives + 9``.

    The optimal solutions have the last k variables at 0. The front and the ideal and nadir
    points are :class:`Dtlz5`'s.

    Deb, K., Thiele, L., Laumanns, M. and Zitzler, E. (2001). Scalable Test Problems for
    Evolutionary Multi-Objective Optimization. TIK-Report 112, ETH Zürich, eq. 26, p. 21.
    """

    _type: ClassVar[str] = "dtlz6"


@dataclass(frozen=True, init=False)
class Dtlz7(_Dtlz):
    """DTLZ7 (the technical report's numbering; the 2002 paper's DTLZ6): a front of 2^(M−1)
    disconnected regions. ``Dtlz7(objectives=3, variables=None)``: 2 to 6 objectives, and at
    least as many variables, None for ``objectives + 19``.

    ``fᵢ = xᵢ`` for the first M − 1, ``g = 1 + 9 Σ xᵢ / k`` over the last k variables,
    ``f_M = (1 + g) (M − Σ fᵢ (1 + sin 3πfᵢ) / (1 + g))``. The optimal solutions have the last k
    variables at 0: the front is ``f_M = 2M − Σ fᵢ (1 + sin 3πfᵢ)`` with each fᵢ in
    [0, 0.2514118360889171] or (0.6316265307000612, 0.8594008566447239], ranges derived from
    the definition. ``optimal_front`` is a grid over them.

    Deb, K., Thiele, L., Laumanns, M. and Zitzler, E. (2001). Scalable Test Problems for
    Evolutionary Multi-Objective Optimization. TIK-Report 112, ETH Zürich, eq. 27, p. 22.
    """

    _type: ClassVar[str] = "dtlz7"


@dataclass(frozen=True, init=False)
class _Wfg(MultiProblem):
    """WFG with ``objectives`` objectives, 2 to 6, ``position`` position parameters k (None for
    the recommended 4 with 2 objectives and 2 (objectives − 1) with more; a positive multiple of
    objectives − 1) and ``distance`` distance parameters l (20 by default): k + l variables, the
    i-th in [0, 2i]. The field ``objective_count`` keeps the number, as the ``objectives``
    property lists the objectives."""

    objective_count: int
    position: int | None
    distance: int
    _even: ClassVar[bool] = False

    def __init__(
        self, objectives: int = 3, position: int | None = None, distance: int = 20
    ) -> None:
        object.__setattr__(self, "objective_count", objectives)
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "distance", distance)

    def _describe(self) -> dict[str, Any]:
        name = type(self).__name__
        objectives = _whole(f"{name}.objectives", self.objective_count, minimum=2)
        if objectives > 6:
            raise ValueError(f"{name}.objectives is at most 6, not {objectives}")
        position = self.position
        if position is not None:
            position = _whole(f"{name}.position", position, minimum=1, maximum=2**24)
            if position % (objectives - 1):
                raise ValueError(
                    f"{name}.position is a multiple of {objectives - 1} (objectives − 1), not "
                    f"{position}"
                )
        distance = _whole(f"{name}.distance", self.distance, minimum=1, maximum=2**24)
        if self._even and distance % 2:
            raise ValueError(f"{name}.distance is even, not {distance}")
        return {
            "type": self._type,
            "objectives": objectives,
            "position": position,
            "distance": distance,
        }


@dataclass(frozen=True, init=False)
class Wfg1(_Wfg):
    """WFG1: a front of convex and mixed convex/concave parts, behind a flat region and a strong
    polynomial bias. ``Wfg1(objectives=3, position=None, distance=20)``.

    The distance parameters are shifted (``s_linear(y, 0.35)``) and given a flat region
    (``b_flat(y, 0.8, 0.75, 0.85)``), every parameter is biased (``b_poly(y, 0.02)``) and each
    group reduced by a sum weighted by 2i. h₁…h_{M−1} are convex and h_M mixed,
    ``1 − x₁ − cos(10πx₁ + π/2) / 10π``. The optimal solutions have the distance parameters at
    0.35 × 2i. In floating point, zᵢ / 2i is never exactly 0.35 for some i (3, 6, 12, 24, 48,
    …), and the bias turns that last bit into a distance of about 0.48 in the parameter: with
    such indices among the distance parameters, as in the default sizes, no genome reaches the
    front (the distance stays above about 0.069 for k = 4, l = 20), as with the authors'
    toolkit.

    Huband, S., Hingston, P., Barone, L. and While, L. (2006). A review of multiobjective test
    problems and a scalable test problem toolkit. IEEE Transactions on Evolutionary Computation
    10(5): 477-506: its table XIV, with tables X and XI; values checked against the authors'
    C++ toolkit, version 2006.03.28.
    """

    _type: ClassVar[str] = "wfg1"


@dataclass(frozen=True, init=False)
class Wfg2(_Wfg):
    """WFG2: a convex front in six disconnected regions, with a non-separable reduction.
    ``Wfg2(objectives=3, position=None, distance=20)``, ``distance`` even.

    The distance parameters are shifted (``s_linear(y, 0.35)``) and reduced in pairs
    (``r_nonsep``). h₁…h_{M−1} are convex and h_M disconnected, ``1 − x₁ cos²(5πx₁)``. The
    optimal solutions have the distance parameters at 0.35 × 2i and x₁ where h_M is below all
    its values at smaller x₁: [0, 0.0416], (0.1297, 0.2096], (0.3549, 0.4050], (0.5641, 0.6034],
    (0.7691, 0.8025] and (0.9724, 1]. For 3 objectives or more, ``optimal_front`` samples evenly
    spread directions and leaves out those between the regions.

    Huband, S., Hingston, P., Barone, L. and While, L. (2006). A review of multiobjective test
    problems and a scalable test problem toolkit. IEEE Transactions on Evolutionary Computation
    10(5): 477-506: its table XIV, with tables X and XI; values checked against the authors'
    C++ toolkit, version 2006.03.28.
    """

    _type: ClassVar[str] = "wfg2"
    _even: ClassVar[bool] = True


@dataclass(frozen=True, init=False)
class Wfg3(_Wfg):
    """WFG3: meant to have a degenerate linear front, a line for any number of objectives.
    ``Wfg3(objectives=3, position=None, distance=20)``, ``distance`` even.

    WFG2's transitions with the linear shape, and x₂…x_{M−1} fixed at 0.5 at the optimal
    distance, 0.35 × 2i: those solutions make a line, which the paper takes for the whole front.
    For 3 objectives or more, solutions off the optimal distance are optimal too (Ishibuchi,
    Masuda and Nojima (2016), IEEE Transactions on Evolutionary Computation 20(5): 807-813, not
    yet checked against the letter, #168): with 3 objectives, e.g. (3, 1, 1). The front,
    ideal and nadir points are then None; with 2 objectives, the front is the segment from
    (0, 4) to (2, 0).

    Huband, S., Hingston, P., Barone, L. and While, L. (2006). A review of multiobjective test
    problems and a scalable test problem toolkit. IEEE Transactions on Evolutionary Computation
    10(5): 477-506: its table XIV, with tables X and XI; values checked against the authors'
    C++ toolkit, version 2006.03.28.
    """

    _type: ClassVar[str] = "wfg3"
    _even: ClassVar[bool] = True


@dataclass(frozen=True, init=False)
class Wfg4(_Wfg):
    """WFG4: a concave front, ``Σ (fₘ / 2m)² = 1``, behind many local fronts: every parameter
    is shifted by the multimodal ``s_multi(y, 30, 10, 0.35)``.
    ``Wfg4(objectives=3, position=None, distance=20)``. The optimal solutions have the distance
    parameters at 0.35 × 2i.

    Huband, S., Hingston, P., Barone, L. and While, L. (2006). A review of multiobjective test
    problems and a scalable test problem toolkit. IEEE Transactions on Evolutionary Computation
    10(5): 477-506: its table XIV, with tables X and XI; values checked against the authors'
    C++ toolkit, version 2006.03.28.
    """

    _type: ClassVar[str] = "wfg4"


@dataclass(frozen=True, init=False)
class Wfg5(_Wfg):
    """WFG5: a concave front, ``Σ (fₘ / 2m)² = 1``, with deceptive parameters: every parameter
    is shifted by ``s_decept(y, 0.35, 0.001, 0.05)``.
    ``Wfg5(objectives=3, position=None, distance=20)``. The optimal solutions have the distance
    parameters at 0.35 × 2i.

    Huband, S., Hingston, P., Barone, L. and While, L. (2006). A review of multiobjective test
    problems and a scalable test problem toolkit. IEEE Transactions on Evolutionary Computation
    10(5): 477-506: its table XIV, with tables X and XI; values checked against the authors'
    C++ toolkit, version 2006.03.28.
    """

    _type: ClassVar[str] = "wfg5"


@dataclass(frozen=True, init=False)
class Wfg6(_Wfg):
    """WFG6: a concave front, ``Σ (fₘ / 2m)² = 1``, non-separable: each group is reduced by
    ``r_nonsep``, the distance parameters all together.
    ``Wfg6(objectives=3, position=None, distance=20)``. The optimal solutions have the distance
    parameters at 0.35 × 2i.

    Huband, S., Hingston, P., Barone, L. and While, L. (2006). A review of multiobjective test
    problems and a scalable test problem toolkit. IEEE Transactions on Evolutionary Computation
    10(5): 477-506: its table XIV, with tables X and XI; values checked against the authors'
    C++ toolkit, version 2006.03.28.
    """

    _type: ClassVar[str] = "wfg6"


@dataclass(frozen=True, init=False)
class Wfg7(_Wfg):
    """WFG7: a concave front, ``Σ (fₘ / 2m)² = 1``, each position parameter biased by the mean
    of the parameters after it (``b_param``). ``Wfg7(objectives=3, position=None, distance=20)``.
    The optimal solutions have the distance parameters at 0.35 × 2i.

    Huband, S., Hingston, P., Barone, L. and While, L. (2006). A review of multiobjective test
    problems and a scalable test problem toolkit. IEEE Transactions on Evolutionary Computation
    10(5): 477-506: its table XIV, with tables X and XI; values checked against the authors'
    C++ toolkit, version 2006.03.28.
    """

    _type: ClassVar[str] = "wfg7"


@dataclass(frozen=True, init=False)
class Wfg8(_Wfg):
    """WFG8: a concave front, ``Σ (fₘ / 2m)² = 1``, each distance parameter biased by the mean
    of the parameters before it (``b_param``). ``Wfg8(objectives=3, position=None,
    distance=20)``. The optimal distance parameters depend on the position:
    ``zᵢ = 2i × 0.35^(1 / (0.02 + 49.98 v(u)))`` with u the mean of y₁…yᵢ₋₁ (yⱼ = zⱼ / 2j),
    from z_{k+1} on.

    Huband, S., Hingston, P., Barone, L. and While, L. (2006). A review of multiobjective test
    problems and a scalable test problem toolkit. IEEE Transactions on Evolutionary Computation
    10(5): 477-506: its table XIV, with tables X and XI; values checked against the authors'
    C++ toolkit, version 2006.03.28.
    """

    _type: ClassVar[str] = "wfg8"


@dataclass(frozen=True, init=False)
class Wfg9(_Wfg):
    """WFG9: a concave front, ``Σ (fₘ / 2m)² = 1``, multimodal, deceptive and non-separable:
    each parameter biased by the mean of those after it, the position parameters deceptive and
    the distance parameters multimodal, then reduced as in :class:`Wfg6`.
    ``Wfg9(objectives=3, position=None, distance=20)``. The optimal distance parameters:
    zₙ = 0.35 × 2n, then ``zᵢ = 2i × 0.35^(1 / (0.02 + 1.96 u))`` with u the mean of
    yᵢ₊₁…yₙ, from z_{n−1} back.

    Huband, S., Hingston, P., Barone, L. and While, L. (2006). A review of multiobjective test
    problems and a scalable test problem toolkit. IEEE Transactions on Evolutionary Computation
    10(5): 477-506: its table XIV, with tables X and XI; values checked against the authors'
    C++ toolkit, version 2006.03.28.
    """

    _type: ClassVar[str] = "wfg9"



@dataclass(frozen=True)
class Schaffer1(MultiProblem):
    """Schaffer's first problem (SCH1): ``f₁ = x²``, ``f₂ = (x − 2)²``, on one variable.

    Bounds [−1000, 1000]. The optimal solutions are x in [0, 2], and the front is
    ``f₂ = (√f₁ − 2)²`` for f₁ in [0, 4].

    Schaffer, J. D. (1985). Multiple objective optimization with vector evaluated genetic
    algorithms. Proceedings of the First International Conference on Genetic Algorithms: 93-100.
    Definition and bounds as restated in Deb, Pratap, Agarwal and Meyarivan (2002, NSGA-II, table
    I); not yet checked against the original (#168). Other papers use other bounds.
    """

    _type: ClassVar[str] = "schaffer1"


@dataclass(frozen=True)
class Schaffer2(MultiProblem):
    """Schaffer's second problem (SCH2), on one variable: ``f₁ = −x`` for x ≤ 1, ``x − 2`` for
    1 < x ≤ 3, ``4 − x`` for 3 < x ≤ 4 and ``x − 4`` for x > 4; ``f₂ = (x − 5)²``.

    Bounds [−5, 10]. The optimal solutions are x in [1, 2) and [4, 5], and the front is in two
    pieces: ``f₂ = (f₁ − 3)²`` for f₁ in [−1, 0), and ``f₂ = (f₁ − 1)²`` for f₁ in [0, 1].

    Schaffer, J. D. (1985). Multiple objective optimization with vector evaluated genetic
    algorithms. Proceedings of the First International Conference on Genetic Algorithms: 93-100.
    Definition and bounds as restated in Van Veldhuizen, D. A. (1999). Multiobjective Evolutionary
    Algorithms: Classifications, Analyses, and New Innovations. PhD thesis AFIT/DS/ENG/99-01, Air
    Force Institute of Technology, table B.1, after Srinivas and Deb (1994); not yet checked against
    the original (#168).
    """

    _type: ClassVar[str] = "schaffer2"


@dataclass(frozen=True)
class FonsecaFleming(_Sized):
    """Fonseca and Fleming's problem (FON): ``f₁ = 1 − exp(−Σ (xᵢ − 1/√n)²)``,
    ``f₂ = 1 − exp(−Σ (xᵢ + 1/√n)²)``, in n variables, at least 1.

    Bounds [−4, 4]ⁿ. The optimal solutions have all variables equal, in [−1/√n, 1/√n], and the
    front, the same for every n, runs from (0, 1 − e⁻⁴) to (1 − e⁻⁴, 0).

    Fonseca, C. M. and Fleming, P. J. (1995). An overview of evolutionary algorithms in
    multiobjective optimization. Evolutionary Computation 3(1): 1-16. Definition and bounds as
    restated in Deb, Thiele, Laumanns and Zitzler (2001, TIK-Report 112, eq. 1) and, for 3
    variables, in Deb, Pratap, Agarwal and Meyarivan (2002, NSGA-II, table I); not yet checked
    against the original (#168).
    """

    variables: int = 3
    _type: ClassVar[str] = "fonseca_fleming"
    _minimum: ClassVar[int] = 1


@dataclass(frozen=True)
class Kursawe(_Sized):
    """Kursawe's problem (KUR): ``f₁ = Σᵢ₌₁ⁿ⁻¹ −10 exp(−0.2 √(xᵢ² + xᵢ₊₁²))``,
    ``f₂ = Σᵢ₌₁ⁿ (|xᵢ|^0.8 + 5 sin(xᵢ³))``, in n variables, at least 2.

    Bounds [−5, 5]ⁿ. The front is disconnected: for 3 variables, the point (−20, 0) at x = 0 and
    three curves. It isn't known in closed form: ``optimal_front`` is None. Deb et al. (2002) and
    Van Veldhuizen (1999) describe three regions, and plot the point apart from them.

    Its ends are known (derived from the definition): f₁ is smallest, −10(n − 1), only at x = 0,
    where f₂ = 0. f₂ is a sum of one term per variable, |x|^0.8 + 5 sin(x³), whose minimum over
    [−5, 5] is h* = −3.8757622790462816, only at x* = −1.1527408475499261 (a root of its
    derivative, polished to 50 digits: its next best local minimum is 0.09 higher). So f₂ is
    smallest, n h*, only at x = (x*, …, x*), where f₁ = −10(n − 1) exp(−0.2 √2 |x*|).
    ``ideal_point`` is (−10(n − 1), n h*) and ``nadir_point`` (−10(n − 1) exp(−0.2 √2 |x*|), 0):
    for 3 variables, (−20, −11.627286837138845) and (−14.435463549038639, 0). The non-dominated
    points of a 401 × 401 × 401 grid over the box, with x*, reach both and stay between them.

    Kursawe, F. (1991). A variant of evolution strategies for vector optimization. Parallel
    Problem Solving from Nature, LNCS 496: 193-197. The original (p. 196) prints f₁ summed to n
    and f₂ = Σ (|xᵢ|^0.8 + 5 sin(xᵢ)³), with no bounds or number of variables; its figure 2 looks
    like sin(xᵢ)³ with 2 variables. The definition here, with sin(xᵢ³), 3 variables and bounds
    [−5, 5], is Deb, Pratap, Agarwal and Meyarivan's (2002, NSGA-II, table I), the form the
    literature uses; Van Veldhuizen (1999, PhD thesis, table B.1) keeps sin(xᵢ)³ and sums f₁ to
    n − 1.
    """

    variables: int = 3
    _type: ClassVar[str] = "kursawe"


@dataclass(frozen=True)
class Poloni(MultiProblem):
    """Poloni's problem (POL): ``f₁ = 1 + (A₁ − B₁)² + (A₂ − B₂)²``,
    ``f₂ = (x₁ + 3)² + (x₂ + 1)²``, with ``A₁ = 0.5 sin 1 − 2 cos 1 + sin 2 − 1.5 cos 2``,
    ``A₂ = 1.5 sin 1 − cos 1 + 2 sin 2 − 0.5 cos 2``, ``B₁ = 0.5 sin x₁ − 2 cos x₁ + sin x₂ −
    1.5 cos x₂`` and ``B₂ = 1.5 sin x₁ − cos x₁ + 2 sin x₂ − 0.5 cos x₂``.

    Bounds [−π, π]². The front is disconnected and not known in closed form: ``optimal_front``
    is None.

    Its ends are known (derived from the definition): f₂ is 0 only at (−3, −1), where
    f₁ = 16.772337779156782, and f₁ is 1 where B = A, which in the box is at (1, 2), where
    f₂ = 25, and at (2.0228, 0.7307), where f₂ = 28.2237: (1, 2) dominates it. ``ideal_point`` is
    (1, 0) and ``nadir_point`` (16.772337779156782, 25). Newton's method from each of the 2,696
    points of a 2,001 × 2,001 grid over the box with f₁ < 1.01 finds only these two solutions of
    B = A, and the non-dominated points of a grid over the box, with the two ends, stay between
    the two points.

    Poloni, C., Giurgevich, A., Onesti, L. and Pediroda, V. (2000). Hybridization of a
    multi-objective genetic algorithm, a neural network and a classical optimizer for a complex
    design problem in fluid dynamics. Computer Methods in Applied Mechanics and Engineering
    186(2-4): 403-420. It first appeared in Poloni et al. (1996, ECCOMAS '96, Wiley: 258-264) and
    Poloni (1997, in Genetic Algorithms in Engineering and Computer Science, Wiley: 397-414),
    which, as Van Veldhuizen (1999, PhD thesis, table B.1) notes, print it mistyped. Definition
    and bounds as restated in Deb, Pratap, Agarwal and Meyarivan (2002, NSGA-II, table I),
    minimized; Van Veldhuizen restates it as the maximization of the negated objectives, and
    Rigoni and Poles (2005, Dagstuhl Seminar Proceedings 04461) minimize it with the same
    constants. Not yet checked against the originals (#168).
    """

    _type: ClassVar[str] = "poloni"


@dataclass(frozen=True)
class Viennet1(MultiProblem):
    """Viennet's first problem (VNT1), with three objectives: ``f₁ = x₁² + (x₂ − 1)²``,
    ``f₂ = x₁² + (x₂ + 1)² + 1``, ``f₃ = (x₁ − 1)² + x₂² + 2``.

    Bounds [−2, 2]². The optimal solutions are the triangle with corners (0, 1), (0, −1) and
    (1, 0), the minima of the three objectives, and the front its image.

    Viennet, R., Fonteix, C. and Marc, I. (1996). Multicriteria optimization using a genetic
    algorithm for determining a Pareto set. International Journal of Systems Science 27(2): 255-260.
    Definition and bounds as restated in Van Veldhuizen, D. A. (1999). Multiobjective Evolutionary
    Algorithms: Classifications, Analyses, and New Innovations. PhD thesis AFIT/DS/ENG/99-01, Air
    Force Institute of Technology, table B.1; not yet checked against the original (#168).
    """

    _type: ClassVar[str] = "viennet1"


@dataclass(frozen=True)
class Viennet2(MultiProblem):
    """Viennet's second problem (VNT2), with three objectives:
    ``f₁ = (x₁ − 2)²/2 + (x₂ + 1)²/13 + 3``, ``f₂ = (x₁ + x₂ − 3)²/36 + (−x₁ + x₂ + 2)²/8 − 17``,
    ``f₃ = (x₁ + 2x₂ − 1)²/175 + (2x₂ − x₁)²/17 − 13``.

    Bounds [−4, 4]². The front is not known in closed form: ``optimal_front`` is None.

    Its ideal and nadir points are known (derived from the definition): the objectives are convex
    quadratics, so the optimal solutions are the minima of the weighted sums w₁f₁ + w₂f₂ + w₃f₃ with
    w ≥ 0, each the solution of a 2 × 2 linear system: a curved triangle whose corners are the
    objectives' minima, (2, −1), (2.5, 0.5) and (0.5, 0.25). ``ideal_point`` is (3, −17, −13), and
    ``nadir_point`` (883/208, −2109/128, −35858/2975) ≈ (4.2452, −16.4766, −12.0531): f₁ and f₂ are
    worst at f₃'s minimum, and f₃ at f₁'s. The minima of 80,601 weighted sums, Das and Dennis's
    weights with 400 divisions, and the non-dominated points of a grid over the box stay between the
    two points.

    Viennet, R., Fonteix, C. and Marc, I. (1996). Multicriteria optimization using a genetic
    algorithm for determining a Pareto set. International Journal of Systems Science 27(2): 255-260.
    Definition and bounds as restated in Van Veldhuizen (1999, PhD thesis, table B.1); not yet
    checked against the original (#168).
    """

    _type: ClassVar[str] = "viennet2"


@dataclass(frozen=True)
class Viennet3(MultiProblem):
    """Viennet's third problem (VNT3), with three objectives:
    ``f₁ = 0.5 (x₁² + x₂²) + sin(x₁² + x₂²)``,
    ``f₂ = (3x₁ − 2x₂ + 4)²/8 + (x₁ − x₂ + 1)²/27 + 15``,
    ``f₃ = 1 / (x₁² + x₂² + 1) − 1.1 exp(−(x₁² + x₂²))``.

    Bounds [−3, 3]². The front is not known in closed form: ``optimal_front`` is None.

    Its ideal and nadir points are known (derived from the definition, and checked numerically): f₁
    and f₃ depend only on t = x₁² + x₂², so an optimal solution has the least f₂ on its circle, and
    the front is the image of one curve in t, two pieces of which are optimal: t from 0 to about
    1.5, and from 4π/3, where f₁ has a local minimum, to about 17.16. ``ideal_point`` is (0, 15,
    −0.1): f₁ and f₃ are smallest at the origin, and f₂ at (−2, −1). ``nadir_point`` is (7π/3 +
    √3/2, 460/27, 1/(1 + 4π/3) − 1.1 exp(−4π/3)) ≈ (8.1964, 17.0370, 0.1760): f₁ is worst at its
    local maximum t = 14π/3, on the second piece; f₂ at the origin, where f₁ is 0; and f₃ at the
    second piece's start, since it falls from there on and stays below 0.155 on the first. Checked
    against 18,001 values of t, each with the least f₂ on its circle, and against the non-dominated
    points of a grid over the box.

    Viennet, R., Fonteix, C. and Marc, I. (1996). Multicriteria optimization using a genetic
    algorithm for determining a Pareto set. International Journal of Systems Science 27(2): 255-260.
    Definition and bounds as restated in Van Veldhuizen (1999, PhD thesis, table B.1); not yet
    checked against the original (#168).
    """

    _type: ClassVar[str] = "viennet3"


@dataclass(frozen=True)
class Bnh(MultiProblem):
    """Binh and Korn's problem (BNH): ``f₁ = 4x₁² + 4x₂²``, ``f₂ = (x₁ − 5)² + (x₂ − 5)²``,
    subject to ``(x₁ − 5)² + x₂² ≤ 25`` and ``(x₁ − 8)² + (x₂ + 3)² ≥ 7.7``.

    Bounds [−15, 30]². The optimal solutions, derived from the definition, are x₁ = x₂ = t for t
    in [0, 5], and the front ``f = (8t², 2(t − 5)²)`` runs from (0, 50) to (200, 0). Later
    papers use the bounds x₁ in [0, 5] and x₂ in [0, 3], which cut the front at x₂ = 3.

    Binh, T. T. and Korn, U. (1997). MOBES: a multiobjective evolution strategy for constrained
    optimization problems. Proceedings of the Third International Conference on Genetic Algorithms
    (Mendel 97), Brno: 176-182. Definition and bounds from its section 5.2, in the authors' version
    of the paper.
    """

    _type: ClassVar[str] = "bnh"


@dataclass(frozen=True)
class Srn(MultiProblem):
    """Srinivas and Deb's problem (SRN): ``f₁ = (x₁ − 2)² + (x₂ − 1)² + 2``,
    ``f₂ = 9x₁ − (x₂ − 1)²``, subject to ``x₁² + x₂² ≤ 225`` and ``x₁ − 3x₂ ≤ −10``.

    Bounds [−20, 20]². The front, derived from the definition, has three pieces: the second
    constraint's boundary x₁ = 3x₂ − 10 for x₂ from 3.7 to 2.5, the line x₁ = −2.5 for x₂ from
    2.5 to about 14.79, and the first constraint's circle to about (−4.841, 14.197). The usually
    quoted x₁ = −2.5 is only part of it.

    Srinivas, N. and Deb, K. (1994). Multiobjective optimization using nondominated sorting in
    genetic algorithms. Evolutionary Computation 2(3): 221-248. Definition and bounds as
    restated in Deb, Pratap, Agarwal and Meyarivan (2002, NSGA-II, table V); not yet checked
    against the original (#168).
    """

    _type: ClassVar[str] = "srn"


@dataclass(frozen=True)
class Tnk(MultiProblem):
    """Tanaka's problem (TNK): ``f₁ = x₁``, ``f₂ = x₂``, subject to
    ``x₁² + x₂² − 1 − 0.1 cos(16 arctan(x₁/x₂)) ≥ 0`` and ``(x₁ − 0.5)² + (x₂ − 0.5)² ≤ 0.5``.

    Bounds [0, π]². The angle arctan(x₁/x₂) is ``atan2(x₁, x₂)``, π/2 at x₂ = 0. The front lies
    on the first constraint's boundary, in disconnected pieces; ``optimal_front`` samples it.

    Tanaka, M., Watanabe, H., Furukawa, Y. and Tanino, T. (1995). GA-based decision support system
    for multicriteria optimization. Proceedings of the IEEE International Conference on Systems,
    Man and Cybernetics 2: 1556-1561. Definition and bounds as restated in Deb, Pratap, Agarwal
    and Meyarivan (2002, NSGA-II, table V); not yet checked against the original (#168).
    """

    _type: ClassVar[str] = "tnk"


@dataclass(frozen=True)
class Osy(MultiProblem):
    """Osyczka and Kundu's problem (OSY), in six variables:
    ``f₁ = −[25 (x₁ − 2)² + (x₂ − 2)² + (x₃ − 1)² + (x₄ − 4)² + (x₅ − 1)²]``, ``f₂ = Σ xᵢ²``,
    subject to ``x₁ + x₂ ≥ 2``, ``x₁ + x₂ ≤ 6``, ``x₂ − x₁ ≤ 2``, ``x₁ − 3x₂ ≤ 2``,
    ``(x₃ − 3)² + x₄ ≤ 4`` and ``(x₅ − 3)² + x₆ ≥ 4``.

    Bounds x₁, x₂, x₆ in [0, 10], x₃, x₅ in [1, 5], x₄ in [0, 6]. The front, derived from the
    definition, has x₄ = x₆ = 0 and five pieces, from (−274, 76) to (−42, 4).

    Osyczka, A. and Kundu, S. (1995). A new method to solve generalized multicriteria optimization
    problems using the simple genetic algorithm. Structural Optimization 10(2): 94-99. Definition
    and bounds as restated in Deb, K., Pratap, A. and Meyarivan, T. (2001). Constrained test
    problems for multi-objective evolutionary optimization. Evolutionary Multi-Criterion
    Optimization (EMO 2001), LNCS 1993: 284-298, eq. 3, whose table 1 lists the same five pieces;
    not yet checked against the original (#168).
    """

    _type: ClassVar[str] = "osy"


@dataclass(frozen=True)
class Constr(MultiProblem):
    """Deb's CONSTR: ``f₁ = x₁``, ``f₂ = (1 + x₂)/x₁``, subject to ``x₂ + 9x₁ ≥ 6`` and
    ``−x₂ + 9x₁ ≥ 1``.

    Bounds x₁ in [0.1, 1], x₂ in [0, 5]. The front, derived from the definition: x₂ = 6 − 9x₁ for
    x₁ in [7/18, 2/3], then x₂ = 0 for x₁ in [2/3, 1], from (7/18, 9) to (1, 1).

    Deb, K., Pratap, A., Agarwal, S. and Meyarivan, T. (2002). A fast and elitist multiobjective
    genetic algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation 6(2): 182-197,
    table V.
    """

    _type: ClassVar[str] = "constr"


from . import cec2006, engineering  # noqa: E402
