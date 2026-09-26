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

All problems here are minimized, on :class:`genoxide.Real` genomes. Each class's docstring gives
the function, its bounds, its optimum and its source. Most originals are books or reports that
aren't online, and some functions have no known origin: their definitions are taken from later
papers that restate them, and are still to be checked against the originals
(https://github.com/tachsin/genoxide/issues/168):

- Yao, X., Liu, Y. and Lin, G. (1999). Evolutionary programming made faster. IEEE Transactions on
  Evolutionary Computation 3(2): 82-102. doi:10.1109/4235.771163
- Laguna, M. and Martí, R. (2005). Experimental testing of advanced scatter search designs for
  global optimization of multimodal functions. Journal of Global Optimization 33(2): 235-255.
  doi:10.1007/s10898-004-1936-z
- Molga, M. and Smutnicki, C. (2005). Test functions for optimization needs.
- Jamil, M. and Yang, X.-S. (2013). A literature survey of benchmark functions for global
  optimisation problems. International Journal of Mathematical Modelling and Numerical
  Optimisation 4(2): 150-194. arXiv:1308.4008

The functions use the platform's trigonometric and exponential functions, so their values can
differ in the last bit between platforms.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from typing import Any, ClassVar

import numpy as np

from . import Real, _genoxide, _whole

__all__ = [
    "Problem",
    "Optimum",
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


class Problem:
    """A test problem: a fitness function with its genome, objective, optimum and reference.

    The problem classes derive from it; ``isinstance(x, Problem)`` tells a problem from a Python
    fitness function.
    """

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
        """The name, e.g. "Rastrigin"."""
        return self._info["name"]

    @property
    def dimensions(self) -> int:
        """The number of genes."""
        return len(self._info["bounds"])

    @property
    def genome(self) -> Real:
        """The search space, e.g. ``Real((-5.12, 5.12), length=10)``."""
        bounds = [tuple(pair) for pair in self._info["bounds"]]
        if all(pair == bounds[0] for pair in bounds):
            return Real(bounds[0], length=len(bounds))
        return Real(bounds)

    @property
    def objective(self) -> str:
        """Whether the score is minimized or maximized: "minimize" for every problem here."""
        return self._info["objective"]

    @property
    def optimum(self) -> Optimum | None:
        """The global optimum, or None if it isn't known for this size."""
        optimum = self._info["optimum"]
        if optimum is None:
            return None
        return Optimum(optimum["value"], optimum["solutions"], optimum["proven"])

    @property
    def reference(self) -> str:
        """The paper, book or report that defines the problem."""
        return self._info["reference"]

    @property
    def reference_url(self) -> str | None:
        """Its DOI or URL, if any."""
        return self._info["reference_url"]

    def evaluate(self, genomes: Any) -> np.ndarray:
        """The scores of ``genomes``, a 2-D array with a genome per row, as a 1-D array.

        Raises
        ------
        ValueError
            If ``genomes`` isn't a 2-D array of numbers with a column per dimension.
        """
        genomes = np.ascontiguousarray(genomes, dtype=np.float64)
        if genomes.ndim != 2:
            raise ValueError(
                f"evaluate takes a 2-D array, a genome per row, not an array of shape "
                f"{genomes.shape}"
            )
        return _genoxide.evaluate(self._json(), genomes)

    def __call__(self, genome: Any) -> float:
        """The score of ``genome``, a 1-D array.

        Raises
        ------
        ValueError
            If ``genome`` isn't a 1-D array of numbers with a value per dimension.
        """
        genome = np.asarray(genome, dtype=np.float64)
        if genome.ndim != 1:
            raise ValueError(f"a genome is a 1-D array, not an array of shape {genome.shape}")
        return float(self.evaluate(genome[np.newaxis, :])[0])


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
