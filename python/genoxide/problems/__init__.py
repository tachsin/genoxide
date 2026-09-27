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

The multi-objective problems, such as :class:`Zdt1`, :class:`Dtlz2` and the constrained
:class:`Bnh`, derive from :class:`MultiProblem` and run with the multi-objective algorithms. They
give their ``objectives`` and, where it's known, their ``optimal_front(points)`` for
:mod:`genoxide.indicators`::

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
violation)``: :mod:`genoxide.problems.cec2006` (g01 to g06 of the CEC 2006 competition) and
:mod:`genoxide.problems.engineering` (engineering design problems such as the welded beam and the
pressure vessel)::

    problem = gx.problems.engineering.PressureVessel()
    de = gx.De(problem.genome, objective=problem.objective, seed=1)
    result = de.run(problem, evaluations=20_000)
    print(result.best_fitness, result.violation, problem.design(result.best_genome))

All problems here are minimized, on :class:`genoxide.Real` genomes except the gear train's
:class:`genoxide.Integer`. Each class's docstring gives
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

The functions use the platform's trigonometric and exponential functions, so their values can
differ in the last bit between platforms.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from typing import Any, ClassVar

import numpy as np

from .. import Integer, Real, _genoxide, _whole

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
    # multi-objective
    "Zdt1",
    "Zdt2",
    "Zdt3",
    "Zdt4",
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
    def genome(self) -> Real | Integer:
        """The search space, e.g. ``Real((-5.12, 5.12), length=10)``; ``Integer`` bounds for a
        problem of whole numbers, such as :class:`genoxide.problems.engineering.GearTrain`."""
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
    Lin (1999, f1); not yet checked against the original (#168).
    """

    dimensions: int = 30
    _type: ClassVar[str] = "sphere"


@dataclass(frozen=True)
class AxisParallelEllipsoid(_Scalable):
    """The axis-parallel hyper-ellipsoid, ``Σ i xᵢ²`` (i from 1): a sphere stretched along each
    axis. Not the rotated hyper-ellipsoid, and not :class:`Schwefel1_2`.

    Bounds [-5.12, 5.12]ⁿ; minimum 0 at the origin. ``dimensions`` is at least 1.

    Its origin is unknown: definition and bounds as restated in Molga and Smutnicki (2005,
    section 2.2); not yet checked against an original (#168).
    """

    dimensions: int = 30
    _type: ClassVar[str] = "axis_parallel_ellipsoid"


@dataclass(frozen=True)
class Schwefel1_2(_Scalable):
    """Schwefel's problem 1.2, ``Σᵢ (Σⱼ≤ᵢ xⱼ)²``: unimodal, with strongly interacting genes.

    Bounds [-100, 100]ⁿ; minimum 0 at the origin. ``dimensions`` is at least 1.

    Schwefel, H.-P. (1981). Numerical Optimization of Computer Models. Wiley, problem 1.2.
    Definition and bounds as restated in Yao, Liu and Lin (1999, f3); not yet checked against the
    original (#168).
    """

    dimensions: int = 30
    _type: ClassVar[str] = "schwefel_1_2"


@dataclass(frozen=True)
class Rastrigin(_Scalable):
    """Rastrigin's function, ``10n + Σ (xᵢ² − 10 cos 2πxᵢ)``: a sphere plus a cosine, with a
    local minimum near every integer point.

    Bounds [-5.12, 5.12]ⁿ; minimum 0 at the origin. ``dimensions`` is at least 1.

    Rastrigin, L. A. (1974). Systems of Extremal Control. Nauka, Moscow (two dimensions);
    generalized to n dimensions by Mühlenbein, H., Schomisch, M. and Born, J. (1991). The parallel
    genetic algorithm as function optimizer. Parallel Computing 17(6-7): 619-632. Definition and
    bounds as restated in Yao, Liu and Lin (1999, f9); not yet checked against the original
    (#168).
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
    Theory and Applications 34(1): 11-39. The divisor 4000 and the bounds as restated in Yao, Liu
    and Lin (1999, f11); not yet checked against the original (#168).
    """

    dimensions: int = 30
    _type: ClassVar[str] = "griewank"


@dataclass(frozen=True)
class Schwefel2_26(_Scalable):
    """Schwefel's problem 2.26, ``−Σ xᵢ sin √|xᵢ|``: the best local minima are far apart.

    Bounds [-500, 500]ⁿ; minimum −418.9828872724337 n at xᵢ = 420.96874635998205, derived from the
    formula. ``dimensions`` is at least 1. This is the form without an offset.

    Schwefel, H.-P. (1981). Numerical Optimization of Computer Models. Wiley, problem 2.26.
    Definition and bounds as restated in Yao, Liu and Lin (1999, f8); not yet checked against the
    original (#168).
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
    15-29. Definition and bounds as restated in Laguna and Martí (2005, function 38), who print xₙ
    instead of wₙ in the last sine; this uses wₙ. Not yet checked against the original (#168).
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
    and (−0.6, −0.4) with 30.

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
    computed by Newton's method.

    Dixon, L. C. W. and Szegö, G. P. (eds.) (1978). Towards Global Optimisation 2. North-Holland.
    Definition and bounds as restated in Yao, Liu and Lin (1999, f16); not yet checked against the
    original (#168).
    """

    _type: ClassVar[str] = "six_hump_camel"


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

    Kursawe, F. (1991). A variant of evolution strategies for vector optimization. Parallel
    Problem Solving from Nature, LNCS 496: 193-197. Definition and bounds as restated in Deb,
    Pratap, Agarwal and Meyarivan (2002, NSGA-II, table I), with sin(xᵢ³); Van Veldhuizen (1999,
    PhD thesis, table B.1) prints sin(xᵢ)³ and notes that the original is misprinted. Not yet
    checked against the original (#168).
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

    Poloni, C., Giurgevich, A., Onesti, L. and Pediroda, V. (2000). Hybridization of a
    multi-objective genetic algorithm, a neural network and a classical optimizer for a complex
    design problem in fluid dynamics. Computer Methods in Applied Mechanics and Engineering
    186(2-4): 403-420. Definition and bounds as restated in Deb, Pratap, Agarwal and Meyarivan
    (2002, NSGA-II, table I); not yet checked against the original (#168).
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
