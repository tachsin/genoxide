"""Evolutionary computation in Rust, for Python.

Genetic algorithms, local search, differential evolution, CMA-ES, particle swarm optimization,
and NSGA-II, NSGA-III, SPEA2, MOEA/D and SMS-EMOA for several objectives, from
`genoxide <https://github.com/tachsin/genoxide>`_, with Python fitness functions::

    import genoxide as gx

    # OneMax: the genome with the most ones
    ga = gx.Ga(
        gx.Binary(100),
        population_size=100,
        select=gx.Tournament(3),
        crossover=gx.UniformCrossover(),
        mutation=gx.BitFlip(rate=0.01),
        seed=42,
    )
    result = ga.run(lambda bits: bits.sum(), target=100, generations=1_000)
    print(result.best_fitness, result.generations)

A fitness function takes a genome as a numpy array (``bool`` for :class:`Binary`, ``float64`` for
:class:`Real`, ``int64`` for :class:`Integer` and :class:`Permutation`) and returns a number,
``None`` for an invalid solution, or a tuple ``(score, constraint_violation)``. With
``batch=True``, it takes a whole generation as a 2-D array, a genome per row, and returns an array
of scores: at most one call per generation, for vectorized numpy code.

:mod:`genoxide.problems` has test problems from the literature, which ``run`` evaluates in Rust,
and :mod:`genoxide.indicators` the quality indicators of multi-objective fronts.

A run stops at the first of its stop conditions: ``generations``, ``evaluations``, ``target``,
``time`` (seconds) and ``stagnation`` (generations without improvement), or when its
``on_generation`` callback returns False.

Settings are checked before a run: a count, a size or an integer bound is a whole number (an
``int`` or a numpy integer, not a ``bool`` or a ``float``), and a real setting is a finite number.
A wrong one is a ``ValueError`` that names it.
"""

from __future__ import annotations

import json
import math
import numbers
import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Union

import numpy as np

from . import _genoxide

__version__: str = _genoxide.__version__

__all__ = [
    # genomes
    "Binary",
    "Integer",
    "Real",
    "Permutation",
    # selection
    "Tournament",
    "Rank",
    "Roulette",
    "StochasticUniversalSampling",
    "Truncation",
    "RandomSelection",
    # crossover
    "UniformCrossover",
    "PointCrossover",
    "NoCrossover",
    "SimulatedBinaryCrossover",
    "BlendCrossover",
    "ArithmeticCrossover",
    "OrderCrossover",
    "PartiallyMappedCrossover",
    "CycleCrossover",
    "EdgeRecombinationCrossover",
    # mutation
    "BitFlip",
    "UniformMutation",
    "GaussianMutation",
    "PolynomialMutation",
    "SwapMutation",
    "InversionMutation",
    "InsertionMutation",
    "ScrambleMutation",
    # schemes and acceptance
    "Generational",
    "SteadyState",
    "MuPlusLambda",
    "MuCommaLambda",
    "Improving",
    "NotWorse",
    "Annealing",
    "Tabu",
    # decompositions of MOEA/D
    "Tchebycheff",
    "Pbi",
    # algorithms
    "Ga",
    "De",
    "Cmaes",
    "Pso",
    "LocalSearch",
    "Nsga2",
    "Nsga3",
    "Spea2",
    "Moead",
    "SmsEmoa",
    "das_dennis",
    # results and progress
    "Result",
    "MultiResult",
    "Progress",
    "MultiProgress",
    # submodules
    "problems",
    "indicators",
]

ObjectiveName = Literal["maximize", "minimize"]
Bounds = Union[tuple[float, float], Sequence[tuple[float, float]]]


def _whole(name: str, value: Any, *, minimum: int | None = 0, plural: bool = False) -> int:
    """The setting ``name`` as an ``int``: an ``int`` or a numpy integer, not a ``bool`` (an
    ``int`` to Python) nor a ``float``, even a whole one; at least ``minimum`` unless it's None."""
    wrong = f"{name} {'are whole numbers' if plural else 'is a whole number'}, not {value!r}"
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(wrong)
    try:
        number = operator.index(value)
    except TypeError:
        raise ValueError(wrong) from None
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} {'are' if plural else 'is'} at least {minimum}, not {number}")
    return number


def _number(name: str, value: Any, *, plural: bool = False) -> float:
    """The setting ``name`` as a ``float``: a finite real number, not a ``bool``."""
    wrong = f"{name} {'are finite numbers' if plural else 'is a finite number'}, not {value!r}"
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(wrong)
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(wrong)
    return number


def _optional_whole(name: str, value: Any) -> int | None:
    """``_whole``, or None for a setting left to its default."""
    return None if value is None else _whole(name, value)


def _optional_number(name: str, value: Any) -> float | None:
    """``_number``, or None for a setting left to its default."""
    return None if value is None else _number(name, value)


def _flag(name: str, value: Any) -> bool | None:
    """A yes-or-no setting, or None for its default."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(f"{name} is True or False, not {value!r}")


def _bounds(
    name: str, bounds: Any, length: Any, cast: Callable[[str, Any], Any]
) -> list[list[Any]]:
    """One ``[low, high]`` per gene, from one pair for every gene (with ``length``) or a pair per
    gene. ``cast`` checks each bound, with the setting's name."""
    if length is not None:
        length = _whole(f"{name}.length", length)
    pairs = np.asarray(bounds, dtype=object)
    bound = f"{name}.bounds"
    if pairs.ndim == 1 and len(pairs) == 2:
        if length is None:
            raise ValueError("one pair of bounds for every gene needs the genome's length")
        low, high = cast(bound, pairs[0]), cast(bound, pairs[1])
        return [[low, high] for _ in range(length)]
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("bounds are a pair (low, high), or a pair per gene")
    if length is not None and length != len(pairs):
        raise ValueError(f"length is {length}, but there are bounds for {len(pairs)} genes")
    return [[cast(bound, low), cast(bound, high)] for low, high in pairs]


def _integer_bound(name: str, value: Any) -> int:
    return _whole(name, value, minimum=None, plural=True)


def _real_bound(name: str, value: Any) -> float:
    return _number(name, value, plural=True)


def _rate_or_count(name: str, rate: float | None, count: int | None) -> dict[str, Any]:
    if (rate is None) == (count is None):
        raise ValueError(f"{name} needs either rate (per gene) or count (genes)")
    return {
        "rate": _optional_number(f"{name}.rate", rate),
        "count": _optional_whole(f"{name}.count", count),
    }


# --- genomes -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Binary:
    """Bit strings of ``length`` bits: numpy ``bool`` arrays.

    ``length`` is 1 to 2^24.
    """

    length: int

    def _describe(self) -> dict[str, Any]:
        return {"type": "binary", "length": _whole("Binary.length", self.length)}


@dataclass(frozen=True)
class Integer:
    """Whole numbers between bounds, inclusive: numpy ``int64`` arrays.

    ``bounds`` is one pair ``(low, high)`` for every gene, with ``length``, or a pair per gene.
    Bounds are whole numbers that fit in an ``int64``, with ``low <= high``; both are included.
    There is at least 1 gene.
    """

    bounds: Bounds
    length: int | None = None

    def _describe(self) -> dict[str, Any]:
        return {
            "type": "integer",
            "bounds": _bounds("Integer", self.bounds, self.length, _integer_bound),
        }


@dataclass(frozen=True)
class Real:
    """Real numbers between bounds: numpy ``float64`` arrays.

    ``bounds`` is one pair ``(low, high)`` for every gene, with ``length``, or a pair per gene.
    Bounds are finite, with ``low <= high`` and a finite width ``high - low``. There is at least 1
    gene.
    """

    bounds: Bounds
    length: int | None = None

    def _describe(self) -> dict[str, Any]:
        return {"type": "real", "bounds": _bounds("Real", self.bounds, self.length, _real_bound)}


@dataclass(frozen=True)
class Permutation:
    """Orderings of ``0 .. length - 1``: numpy ``int64`` arrays.

    ``length`` is 1 to 2^24.
    """

    length: int

    def _describe(self) -> dict[str, Any]:
        return {"type": "permutation", "length": _whole("Permutation.length", self.length)}


Genome = Union[Binary, Integer, Real, Permutation]

# --- selection -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Tournament:
    """The best of ``size`` random individuals. ``size`` is 1 to 2^24."""

    size: int

    def _describe(self) -> dict[str, Any]:
        return {"type": "tournament", "size": _whole("Tournament.size", self.size)}


@dataclass(frozen=True)
class Rank:
    """Linear ranking: the best individual is expected to be selected ``pressure`` times as often
    as the average one. ``pressure`` is 1 (uniform) to 2; 1.5 by default."""

    pressure: float = 1.5

    def _describe(self) -> dict[str, Any]:
        return {"type": "rank", "pressure": _number("Rank.pressure", self.pressure)}


@dataclass(frozen=True)
class Roulette:
    """Fitness-proportional selection."""

    def _describe(self) -> dict[str, Any]:
        return {"type": "roulette"}


@dataclass(frozen=True)
class StochasticUniversalSampling:
    """Fitness-proportional selection with evenly spaced pointers."""

    def _describe(self) -> dict[str, Any]:
        return {"type": "stochastic_universal_sampling"}


@dataclass(frozen=True)
class Truncation:
    """Uniformly among the best ``fraction`` of the population. ``fraction`` is greater than 0
    and at most 1."""

    fraction: float

    def _describe(self) -> dict[str, Any]:
        return {"type": "truncation", "fraction": _number("Truncation.fraction", self.fraction)}


@dataclass(frozen=True)
class RandomSelection:
    """Uniformly at random."""

    def _describe(self) -> dict[str, Any]:
        return {"type": "random"}


Select = Union[Tournament, Rank, Roulette, StochasticUniversalSampling, Truncation, RandomSelection]

# --- crossover -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class UniformCrossover:
    """Swaps each gene with probability 1/2. Binary, integer and real genomes."""

    def _describe(self) -> dict[str, Any]:
        return {"type": "uniform"}


@dataclass(frozen=True)
class PointCrossover:
    """Swaps the segments between ``points`` random cut points. Binary, integer and real
    genomes. ``points`` is at least 1; 1 by default."""

    points: int = 1

    def _describe(self) -> dict[str, Any]:
        return {"type": "point", "points": _whole("PointCrossover.points", self.points)}


@dataclass(frozen=True)
class NoCrossover:
    """Leaves the parents as they are: mutation only."""

    def _describe(self) -> dict[str, Any]:
        return {"type": "none"}


@dataclass(frozen=True)
class SimulatedBinaryCrossover:
    """SBX with distribution index ``eta``: higher keeps children closer to their parents. Real
    genomes. ``eta`` is 0 or more; 15 by default."""

    eta: float = 15.0

    def _describe(self) -> dict[str, Any]:
        eta = _number("SimulatedBinaryCrossover.eta", self.eta)
        return {"type": "simulated_binary", "eta": eta}


@dataclass(frozen=True)
class BlendCrossover:
    """BLX-alpha: children uniformly in the parents' interval, widened by ``alpha`` on each side.
    Real genomes. ``alpha`` is 0 or more; 0.5 by default."""

    alpha: float = 0.5

    def _describe(self) -> dict[str, Any]:
        return {"type": "blend", "alpha": _number("BlendCrossover.alpha", self.alpha)}


@dataclass(frozen=True)
class ArithmeticCrossover:
    """Random weighted averages of the parents. Real genomes."""

    def _describe(self) -> dict[str, Any]:
        return {"type": "arithmetic"}


@dataclass(frozen=True)
class OrderCrossover:
    """OX: a segment of one parent, the rest in the other's order. Permutations."""

    def _describe(self) -> dict[str, Any]:
        return {"type": "order"}


@dataclass(frozen=True)
class PartiallyMappedCrossover:
    """PMX. Permutations."""

    def _describe(self) -> dict[str, Any]:
        return {"type": "partially_mapped"}


@dataclass(frozen=True)
class CycleCrossover:
    """CX: every position keeps a value of one of the parents. Permutations."""

    def _describe(self) -> dict[str, Any]:
        return {"type": "cycle"}


@dataclass(frozen=True)
class EdgeRecombinationCrossover:
    """ERX: keeps the parents' adjacencies, for routing problems. Permutations."""

    def _describe(self) -> dict[str, Any]:
        return {"type": "edge_recombination"}


Crossover = Union[
    UniformCrossover,
    PointCrossover,
    NoCrossover,
    SimulatedBinaryCrossover,
    BlendCrossover,
    ArithmeticCrossover,
    OrderCrossover,
    PartiallyMappedCrossover,
    CycleCrossover,
    EdgeRecombinationCrossover,
]

# --- mutation ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BitFlip:
    """Flips each bit with probability ``rate``, or exactly ``count`` bits. Binary genomes.

    Give either ``rate``, greater than 0 and at most 1, or ``count``, at least 1.
    """

    rate: float | None = None
    count: int | None = None

    def _describe(self) -> dict[str, Any]:
        return {"type": "bit_flip", **_rate_or_count("BitFlip", self.rate, self.count)}


@dataclass(frozen=True)
class UniformMutation:
    """Redraws each gene uniformly within its bounds with probability ``rate``, or exactly
    ``count`` genes. Integer and real genomes.

    Give either ``rate``, greater than 0 and at most 1, or ``count``, at least 1.
    """

    rate: float | None = None
    count: int | None = None

    def _describe(self) -> dict[str, Any]:
        return {"type": "uniform", **_rate_or_count("UniformMutation", self.rate, self.count)}


@dataclass(frozen=True)
class GaussianMutation:
    """Adds normal noise with standard deviation ``sigma`` (a fraction of each gene's range) to
    each gene with probability ``rate``, or to exactly ``count`` genes. Real genomes.

    ``sigma`` is greater than 0 and finite. Give either ``rate``, greater than 0 and at most 1, or
    ``count``, at least 1.
    """

    sigma: float
    rate: float | None = None
    count: int | None = None

    def _describe(self) -> dict[str, Any]:
        return {
            "type": "gaussian",
            "sigma": _number("GaussianMutation.sigma", self.sigma),
            **_rate_or_count("GaussianMutation", self.rate, self.count),
        }


@dataclass(frozen=True)
class PolynomialMutation:
    """Deb's polynomial mutation with distribution index ``eta``, of each gene with probability
    ``rate`` (usually 1 / length), or of exactly ``count`` genes. Real genomes.

    ``eta`` is 0 or more; 20 by default. Give either ``rate``, greater than 0 and at most 1, or
    ``count``, at least 1.
    """

    eta: float = 20.0
    rate: float | None = None
    count: int | None = None

    def _describe(self) -> dict[str, Any]:
        return {
            "type": "polynomial",
            "eta": _number("PolynomialMutation.eta", self.eta),
            **_rate_or_count("PolynomialMutation", self.rate, self.count),
        }


@dataclass(frozen=True)
class SwapMutation:
    """Swaps ``count`` pairs of positions. Permutations. ``count`` is at least 1; 1 by
    default."""

    count: int = 1

    def _describe(self) -> dict[str, Any]:
        return {"type": "swap", "count": _whole("SwapMutation.count", self.count)}


@dataclass(frozen=True)
class InversionMutation:
    """Reverses a random segment. Permutations."""

    def _describe(self) -> dict[str, Any]:
        return {"type": "inversion"}


@dataclass(frozen=True)
class InsertionMutation:
    """Moves a value to another position. Permutations."""

    def _describe(self) -> dict[str, Any]:
        return {"type": "insertion"}


@dataclass(frozen=True)
class ScrambleMutation:
    """Shuffles a random segment. Permutations."""

    def _describe(self) -> dict[str, Any]:
        return {"type": "scramble"}


Mutation = Union[
    BitFlip,
    UniformMutation,
    GaussianMutation,
    PolynomialMutation,
    SwapMutation,
    InversionMutation,
    InsertionMutation,
    ScrambleMutation,
]

# --- schemes of the genetic algorithm ------------------------------------------------------------


@dataclass(frozen=True)
class Generational:
    """Children replace the population, except its ``elitism`` best individuals (the default,
    with 1). ``elitism`` is 0 or more and less than the population size."""

    elitism: int = 1

    def _describe(self) -> dict[str, Any]:
        return {"type": "generational", "elitism": _whole("Generational.elitism", self.elitism)}


@dataclass(frozen=True)
class SteadyState:
    """Each generation, ``replacements`` children replace the worst individuals. ``replacements``
    is 1 to the population size."""

    replacements: int

    def _describe(self) -> dict[str, Any]:
        return {
            "type": "steady_state",
            "replacements": _whole("SteadyState.replacements", self.replacements),
        }


@dataclass(frozen=True)
class MuPlusLambda:
    """(mu + lambda): ``offspring`` children, and the best of parents and children survive.
    ``offspring`` is 1 to 2^24."""

    offspring: int

    def _describe(self) -> dict[str, Any]:
        offspring = _whole("MuPlusLambda.offspring", self.offspring)
        return {"type": "mu_plus_lambda", "lambda": offspring}


@dataclass(frozen=True)
class MuCommaLambda:
    """(mu, lambda): ``offspring`` children, of which the best survive. ``offspring`` is the
    population size to 2^24."""

    offspring: int

    def _describe(self) -> dict[str, Any]:
        offspring = _whole("MuCommaLambda.offspring", self.offspring)
        return {"type": "mu_comma_lambda", "lambda": offspring}


Scheme = Union[Generational, SteadyState, MuPlusLambda, MuCommaLambda]

# --- acceptance of local search ------------------------------------------------------------------


@dataclass(frozen=True)
class Improving:
    """Moves to strictly better neighbors only: hill climbing that stops at the first local
    optimum."""

    def _describe(self) -> dict[str, Any]:
        return {"type": "improving"}


@dataclass(frozen=True)
class NotWorse:
    """Moves to better or equal neighbors (the default): hill climbing that drifts across
    plateaus."""

    def _describe(self) -> dict[str, Any]:
        return {"type": "not_worse"}


@dataclass(frozen=True)
class Annealing:
    """Simulated annealing: also moves to a neighbor worse by d with probability exp(-d / T).
    The temperature T starts at ``initial_temperature`` and is multiplied by ``cooling`` (e.g.
    0.999) after every step. ``initial_temperature`` is greater than 0 and finite; ``cooling`` is
    greater than 0 and at most 1."""

    initial_temperature: float
    cooling: float

    def _describe(self) -> dict[str, Any]:
        temperature = _number("Annealing.initial_temperature", self.initial_temperature)
        return {
            "type": "annealing",
            "initial_temperature": temperature,
            "cooling": _number("Annealing.cooling", self.cooling),
        }


@dataclass(frozen=True)
class Tabu:
    """Tabu search: moves to the best neighbor that isn't one of the last ``tenure`` solutions,
    even if it's worse. Use it with several neighbors per step. ``tenure`` is at least 1."""

    tenure: int

    def _describe(self) -> dict[str, Any]:
        return {"type": "tabu", "tenure": _whole("Tabu.tenure", self.tenure)}


Acceptance = Union[Improving, NotWorse, Annealing, Tabu]

# --- decompositions of MOEA/D --------------------------------------------------------------------


@dataclass(frozen=True)
class Tchebycheff:
    """The weighted Tchebycheff distance to the ideal point, ``max_j w_j |f_j - z_j|`` (the
    default), for any front shape."""

    def _describe(self) -> dict[str, Any]:
        return {"type": "tchebycheff"}


@dataclass(frozen=True)
class Pbi:
    """Penalty-based boundary intersection: the distance along the weight vector plus ``theta``
    times the distance from it. Spreads fronts of 3 or more objectives evenly. ``theta`` is 0 or
    more; 5 by default."""

    theta: float = 5.0

    def _describe(self) -> dict[str, Any]:
        return {"type": "pbi", "theta": _number("Pbi.theta", self.theta)}


Decomposition = Union[Tchebycheff, Pbi]

# --- results -------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Result:
    """The result of a single-objective run."""

    best_genome: np.ndarray
    """The best genome found."""
    best_fitness: float | None
    """Its score, or None if no valid solution was found."""
    violation: float
    """Its constraint violation: 0 for a feasible solution, and NaN if no valid solution was
    found."""
    generations: int
    """The generations completed after the initial population."""
    evaluations: int
    """The fitness evaluations. A child identical to a parent inherits its fitness without one."""
    seconds: float
    """The duration of the run, in seconds."""
    stop_reason: str
    """What stopped the run:

    - "target", "generations", "evaluations", "time" or "stagnation": that stop condition;
    - "aborted": ``on_generation`` returned False;
    - "stalled": nothing new to evaluate for 10,000 generations in a row (e.g. every child was a
      copy of a parent), while only ``target`` or ``evaluations`` could stop the run;
    - "other": a reason that the stop conditions of the package don't produce.
    """


@dataclass(frozen=True, eq=False)
class MultiResult:
    """The result of a multi-objective run: its final non-dominated front, without copies."""

    front_genomes: np.ndarray
    """The genomes of the front, a row each."""
    front_objectives: np.ndarray
    """Their objective values, a row each."""
    front_violations: np.ndarray
    """Their constraint violations: 0 for feasible solutions, and NaN for invalid ones."""
    generations: int
    """The generations completed after the initial population."""
    evaluations: int
    """The fitness evaluations. A child identical to a parent inherits its objective values
    without one."""
    seconds: float
    """The duration of the run, in seconds."""
    stop_reason: str
    """What stopped the run:

    - "generations", "evaluations", "time" or "stagnation": that stop condition;
    - "aborted": ``on_generation`` returned False;
    - "stalled": nothing new to evaluate for 10,000 generations in a row (e.g. every child was a
      copy of a parent), while only ``evaluations`` could stop the run;
    - "other": a reason that the stop conditions of the package don't produce.
    """


@dataclass(frozen=True)
class Progress:
    """A single-objective run after a generation, for ``on_generation``."""

    generation: int
    """The generations completed: 0 after the initial population."""
    evaluations: int
    """The fitness evaluations so far."""
    seconds: float
    """The time since the run started."""
    best_fitness: float | None
    """The best score so far, or None if no valid solution was found yet."""


@dataclass(frozen=True)
class MultiProgress:
    """A multi-objective run after a generation, for ``on_generation``."""

    generation: int
    """The generations completed: 0 after the initial population."""
    evaluations: int
    """The fitness evaluations so far."""
    seconds: float
    """The time since the run started."""
    front_size: int
    """The number of non-dominated individuals in the population, copies included."""


# --- running -------------------------------------------------------------------------------------


def _stop(
    generations: int | None,
    evaluations: int | None,
    target: float | None,
    time: float | None,
    stagnation: int | None,
) -> dict[str, Any]:
    stop = {
        "generations": _optional_whole("generations", generations),
        "evaluations": _optional_whole("evaluations", evaluations),
        "target": _optional_number("target", target),
        "seconds": _seconds(time),
        "stagnation": _optional_whole("stagnation", stagnation),
    }
    if all(value is None for value in stop.values()):
        raise ValueError(
            "a run needs a stop condition: generations, evaluations, target, time or stagnation"
        )
    return stop


def _seconds(time: Any) -> float | None:
    """The time limit in seconds: None for none, with ``time`` None or infinite."""
    if time is None:
        return None
    if isinstance(time, (bool, np.bool_)) or not isinstance(time, numbers.Real):
        raise ValueError(f"time is a number of seconds, not {time!r}")
    seconds = float(time)
    if seconds == math.inf:
        return None
    if not seconds >= 0:
        raise ValueError(f"time is a number of seconds, at least 0, not {time!r}")
    return seconds


def _check_callable(function: Any, name: str = "the fitness function") -> None:
    if not callable(function):
        raise TypeError(f"{name} isn't callable: {function!r}")


def _on_generation(
    callback: Callable[[Any], Any] | None, progress: type[Progress] | type[MultiProgress]
) -> Callable[[int, int, float, Any], bool] | None:
    """The callback, called with the generation, the evaluations, the seconds and the best
    fitness or the size of the front, as a ``progress``; False from it stops the run."""
    if callback is None:
        return None
    _check_callable(callback, "on_generation")

    def call(generation: int, evaluations: int, seconds: float, value: Any) -> bool:
        go_on = callback(progress(generation, evaluations, seconds, value))
        return go_on is not False and go_on is not np.False_

    return call


def _json_number(value: Any) -> Any:
    """numpy numbers in the settings, as Python numbers."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{value!r} isn't a number, text or None")


def _batch_scores(function: Callable[[np.ndarray], Any]) -> Callable[[np.ndarray], Any]:
    """A batch function returning float64 arrays: scores, and constraint violations or None."""

    def evaluate(genomes: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        result = function(genomes)
        # (scores, violations): two arrays or columns, not a pair of scores
        if isinstance(result, tuple) and len(result) == 2 and np.ndim(result[0]) in (1, 2):
            scores, violations = result
            return (
                np.asarray(scores, dtype=np.float64).reshape(-1),
                np.asarray(violations, dtype=np.float64).reshape(-1),
            )
        return np.asarray(result, dtype=np.float64).reshape(-1), None

    return evaluate


def _batch_objectives(function: Callable[[np.ndarray], Any]) -> Callable[[np.ndarray], Any]:
    """A batch function returning float64 arrays: a row of objective values per genome, and
    constraint violations or None."""

    def evaluate(genomes: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        result = function(genomes)
        # (objectives, violations): a matrix and an array
        if isinstance(result, tuple) and len(result) == 2 and np.ndim(result[0]) == 2:
            objectives, violations = result
            return (
                _objective_rows(objectives),
                np.asarray(violations, dtype=np.float64).reshape(-1),
            )
        return _objective_rows(result), None

    return evaluate


def _objective_rows(objectives: Any) -> np.ndarray:
    """The objective values of a batch, a row per genome."""
    rows = np.asarray(objectives, dtype=np.float64)
    if rows.ndim != 2:
        raise ValueError(
            "a multi-objective batch fitness function returns a 2-D array, a row of objective "
            f"values per genome, not an array of shape {rows.shape}"
        )
    return rows


class _Algorithm:
    """An algorithm with its settings. ``run`` builds it anew, so runs with a seed repeat."""

    _genome: Genome

    def _describe(self) -> dict[str, Any]:
        raise NotImplementedError

    def _objectives(self) -> list[str]:
        raise NotImplementedError

    def _run(
        self,
        fitness: Callable[[np.ndarray], Any],
        stop: dict[str, Any],
        batch: bool,
        parallel: bool,
        on_generation: Callable[[int, int, float, Any], bool] | None,
        problem: str | None = None,
    ) -> dict[str, Any]:
        run = {
            "genome": self._genome._describe(),
            "algorithm": self._describe(),
            "objectives": self._objectives(),
            "stop": stop,
        }
        # NaN and infinity aren't JSON: the settings are finite, or an error names them
        description = json.dumps(run, default=_json_number, allow_nan=False)
        return _genoxide.run(
            description, fitness, bool(batch), bool(parallel), on_generation, problem
        )


class _SingleObjective(_Algorithm):
    _objective: ObjectiveName

    def _objectives(self) -> list[str]:
        if self._objective not in ("maximize", "minimize"):
            raise ValueError(f'objective is "maximize" or "minimize", not {self._objective!r}')
        return [self._objective]

    def run(
        self,
        fitness: Callable[[np.ndarray], Any],
        *,
        generations: int | None = None,
        evaluations: int | None = None,
        target: float | None = None,
        time: float | None = None,
        stagnation: int | None = None,
        batch: bool = False,
        parallel: bool = False,
        on_generation: Callable[[Progress], bool | None] | None = None,
    ) -> Result:
        """Runs the algorithm until the first stop condition.

        Each call starts a new run from the settings. The settings are checked here, not by the
        constructor. With a ``seed``, the same call repeats the run exactly: one genome at a time,
        in batches or in parallel.

        Parameters
        ----------
        fitness : callable or problems.Problem
            Takes a genome as a 1-D numpy array and returns a number, None or NaN (an invalid
            solution), or a tuple ``(score, constraint_violation)``. The violation is 0 for a
            feasible solution and positive for an infeasible one. With ``batch=True``, it takes a
            generation as a 2-D array, a genome per row, and returns an array of scores (NaN for
            an invalid solution), or a tuple of an array of scores and an array of constraint
            violations; a column of shape ``(n, 1)`` does for an array. The function must be
            deterministic. A problem of :mod:`genoxide.problems` is evaluated in Rust, with no
            Python call: ``batch`` doesn't apply, and the genome must be a :class:`Real` with a
            gene per dimension of the problem.
        generations : int, optional
            Stops after this many generations, 0 or more. 0 evaluates only the initial
            population.
        evaluations : int, optional
            Stops after the generation that reaches this many fitness evaluations, 0 or more.
        target : float, optional
            Stops when the best score is at least as good: at least ``target`` when maximizing,
            at most when minimizing. A finite number.
        time : float, optional
            Stops when the run has taken this many seconds, 0 or more, checked after every
            generation. ``math.inf`` is no limit.
        stagnation : int, optional
            Stops after this many generations without a better best score, at least 1.
        batch : bool, default False
            Calls ``fitness`` once per generation with a 2-D array, and not for a generation of
            copies of their parents.
        parallel : bool, default False
            Calls a non-batch ``fitness`` from several threads at once. It pays off when the
            function releases the GIL (numpy on large arrays, I/O), or on free-threaded Python,
            and for a problem of :mod:`genoxide.problems`, which runs without the GIL.
        on_generation : callable, optional
            Called with a :class:`Progress` after every generation, the initial population
            (generation 0) included, on the thread that called ``run``. If it returns False, the
            run stops with the stop reason "aborted".

        At least one of ``generations``, ``evaluations``, ``target``, ``time`` and
        ``stagnation`` is needed. None is no condition.

        Returns
        -------
        Result
            The best solution found, and what the run took.

        Raises
        ------
        ValueError
            Without a stop condition; for a wrong setting of the run, the algorithm, its genome
            or its operators, or an operator that doesn't fit the genome (the message names the
            setting); for a problem whose genome isn't a :class:`Real` of its dimensions; and for
            a wrong fitness result: a negative constraint violation, or a batch result with a
            length other than the number of genomes.
        TypeError
            If ``fitness`` or ``on_generation`` isn't callable, or ``fitness`` returns something
            that isn't a number. Another error converting a result, e.g. an ``OverflowError`` for
            an int too large for a float, is raised as it is.
        Exception
            An exception raised by ``fitness`` or ``on_generation`` stops the run, and ``run``
            raises it. So does ``KeyboardInterrupt`` on Ctrl+C.
        """
        _check_callable(fitness)
        stop = _stop(generations, evaluations, target, time, stagnation)
        callback = _on_generation(on_generation, Progress)
        if isinstance(fitness, problems.Problem):
            description = fitness._json()
            return Result(**self._run(fitness, stop, False, parallel, callback, description))
        function = _batch_scores(fitness) if batch else fitness
        return Result(**self._run(function, stop, batch, parallel, callback))


class Ga(_SingleObjective):
    """A genetic algorithm. Any genome.

    Each generation, ``select`` picks parents, ``crossover`` combines pairs of them with
    probability ``crossover_rate`` and ``mutation`` changes each child with probability
    ``mutation_rate``. The ``scheme`` decides who survives: by default the children replace the
    population, except its best individual.

    Parameters
    ----------
    genome : Binary, Integer, Real or Permutation
        The search space.
    population_size : int
        The number of individuals, 1 to 2^24.
    select : Tournament, Rank, Roulette, StochasticUniversalSampling, Truncation or RandomSelection
        How parents are picked.
    crossover : a crossover
        How pairs of parents are combined. It must fit the genome.
    mutation : a mutation
        How children are changed. It must fit the genome.
    crossover_rate : float, default 0.9
        The probability that a pair of parents is combined, 0 to 1.
    mutation_rate : float, default 1
        The probability that a child is mutated, 0 to 1. 0 needs a ``crossover_rate`` above 0
        and a crossover other than :class:`NoCrossover`: otherwise every child is a copy.
    scheme : Generational, SteadyState, MuPlusLambda or MuCommaLambda, default Generational(1)
        Who survives each generation.
    objective : {"maximize", "minimize"}, default "maximize"
        Whether higher or lower scores are better.
    seed : int, optional
        The seed of the random numbers, 0 to 2^64 - 1. None is a random seed. The same seed
        repeats the run.
    """

    def __init__(
        self,
        genome: Genome,
        *,
        population_size: int,
        select: Select,
        crossover: Crossover,
        mutation: Mutation,
        crossover_rate: float | None = None,
        mutation_rate: float | None = None,
        scheme: Scheme | None = None,
        objective: ObjectiveName = "maximize",
        seed: int | None = None,
    ) -> None:
        self._genome = genome
        self._objective = objective
        self.population_size = population_size
        self.select = select
        self.crossover = crossover
        self.mutation = mutation
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.scheme = scheme
        self.seed = seed

    def _describe(self) -> dict[str, Any]:
        return {
            "type": "ga",
            "population_size": _whole("population_size", self.population_size),
            "seed": _optional_whole("seed", self.seed),
            "select": self.select._describe(),
            "crossover": self.crossover._describe(),
            "mutate": self.mutation._describe(),
            "crossover_rate": _optional_number("crossover_rate", self.crossover_rate),
            "mutation_rate": _optional_number("mutation_rate", self.mutation_rate),
            "scheme": None if self.scheme is None else self.scheme._describe(),
        }


class De(_SingleObjective):
    """Differential evolution. Real genomes.

    By default SHADE's published settings (Tanabe and Fukunaga, "Success-History Based
    Parameter Adaptation for Differential Evolution", IEEE CEC 2013): DE/current-to-pbest/1 with
    a random p per trial between 2 / population and 0.2 and an archive of the population's size,
    SHADE's adaptation of F and CR with a memory of 100, and a population of 100. genoxide adds
    restarts, which aren't part of SHADE: every individual but the best is replaced when the
    scores converge (within 1e-8, relative to the best) or the best doesn't improve for 200
    generations.

    With ``l_shade``, L-SHADE (Tanabe and Fukunaga, 2014) for a budget of that many evaluations:
    current-to-pbest/1 with p 0.11 and an archive of 2.6 times the population, SHADE's
    adaptation with a memory of 6, no restarts, and a population that shrinks linearly from its
    initial size to 4 over the budget. Stop the run at the same number of evaluations.

    Parameters
    ----------
    genome : Real
        The search space.
    population_size : int, optional
        The number of individuals, 4 to 2^24. 100 by default. With ``l_shade``, the initial size,
        18 times the number of genes (at least 4) by default.
    l_shade : int, optional
        L-SHADE's budget of evaluations, at least 1. None is SHADE.
    objective : {"maximize", "minimize"}, default "maximize"
        Whether higher or lower scores are better.
    seed : int, optional
        The seed of the random numbers, 0 to 2^64 - 1. None is a random seed. The same seed
        repeats the run.
    """

    def __init__(
        self,
        genome: Real,
        *,
        population_size: int | None = None,
        l_shade: int | None = None,
        objective: ObjectiveName = "maximize",
        seed: int | None = None,
    ) -> None:
        self._genome = genome
        self._objective = objective
        self.population_size = population_size
        self.l_shade = l_shade
        self.seed = seed

    def _describe(self) -> dict[str, Any]:
        return {
            "type": "de",
            "population_size": _optional_whole("population_size", self.population_size),
            "seed": _optional_whole("seed", self.seed),
            "l_shade": _optional_whole("l_shade", self.l_shade),
        }


class Cmaes(_SingleObjective):
    """CMA-ES, the covariance matrix adaptation evolution strategy. Real genomes.

    It adapts a full covariance matrix, from a random initial mean. At least one gene needs
    ``low < high``; genes with ``low == high`` are fixed.

    Parameters
    ----------
    genome : Real
        The search space.
    population_size : int, optional
        The samples per generation, 2 to 2^24. ``4 + floor(3 ln n)`` by default, for the ``n``
        genes with ``low < high``: 10 for 10 genes. With restarts, the size of the first run.
    restarts : {"never", "ipop", "bipop"}, default "never"
        What happens when a run converges. "never" goes on sampling around the same point.
        "ipop" restarts from a random point with a doubled population, up to 1024 times the
        initial one. "bipop" alternates such large populations with small ones of random size
        and step size. Restarts suit multimodal functions.
    initial_step : float, default 0.3
        The initial step size as a fraction of each gene's range, greater than 0 and at most 1.
    objective : {"maximize", "minimize"}, default "maximize"
        Whether higher or lower scores are better.
    seed : int, optional
        The seed of the random numbers, 0 to 2^64 - 1. None is a random seed. The same seed
        repeats the run.
    """

    def __init__(
        self,
        genome: Real,
        *,
        population_size: int | None = None,
        restarts: Literal["never", "ipop", "bipop"] | None = None,
        initial_step: float | None = None,
        objective: ObjectiveName = "maximize",
        seed: int | None = None,
    ) -> None:
        self._genome = genome
        self._objective = objective
        self.population_size = population_size
        self.restarts = restarts
        self.initial_step = initial_step
        self.seed = seed

    def _describe(self) -> dict[str, Any]:
        if self.restarts not in (None, "never", "ipop", "bipop"):
            raise ValueError(f'restarts is "never", "ipop" or "bipop", not {self.restarts!r}')
        return {
            "type": "cmaes",
            "population_size": _optional_whole("population_size", self.population_size),
            "seed": _optional_whole("seed", self.seed),
            "restarts": self.restarts,
            "initial_step": _optional_number("initial_step", self.initial_step),
        }


class Pso(_SingleObjective):
    """Particle swarm optimization. Real genomes.

    The velocities keep 0.7298 of themselves, are pulled toward the personal and the neighborhood
    best with 1.49618 each, and are at most each gene's range.

    Parameters
    ----------
    genome : Real
        The search space.
    population_size : int
        The number of particles, 2 to 2^24. 20 to 50 is common.
    ring : int, optional
        Each particle follows the best of itself and its ``ring`` neighbors on each side of a
        ring, at least 1, instead of the whole swarm's best. Good positions spread slowly, which
        suits multimodal functions. None is the whole swarm.
    objective : {"maximize", "minimize"}, default "maximize"
        Whether higher or lower scores are better.
    seed : int, optional
        The seed of the random numbers, 0 to 2^64 - 1. None is a random seed. The same seed
        repeats the run.
    """

    def __init__(
        self,
        genome: Real,
        *,
        population_size: int,
        ring: int | None = None,
        objective: ObjectiveName = "maximize",
        seed: int | None = None,
    ) -> None:
        self._genome = genome
        self._objective = objective
        self.population_size = population_size
        self.ring = ring
        self.seed = seed

    def _describe(self) -> dict[str, Any]:
        return {
            "type": "pso",
            "population_size": _whole("population_size", self.population_size),
            "seed": _optional_whole("seed", self.seed),
            "ring": _optional_whole("ring", self.ring),
        }


class LocalSearch(_SingleObjective):
    """Local search from one random solution. Any genome.

    Each step evaluates ``neighbors`` neighbors made by ``neighbor``, and ``acceptance`` decides
    whether to move to the best of them: hill climbing across plateaus by default, simulated
    annealing or tabu search. With ``restart``, iterated local search.

    Parameters
    ----------
    genome : Binary, Integer, Real or Permutation
        The search space.
    neighbor : a mutation
        Makes a neighbor from the current solution. It must fit the genome.
    neighbors : int, default 1
        The neighbors evaluated per step, 1 to 2^24.
    acceptance : Improving, NotWorse, Annealing or Tabu, default NotWorse()
        When to move to the best neighbor.
    restart : tuple of (int, int), optional
        ``(patience, kicks)``: after ``patience`` steps (at least 1) without a new best
        solution, the search restarts from the best solution changed by ``kicks`` neighbor
        moves (1 to 2^24). None is no restarts.
    objective : {"maximize", "minimize"}, default "maximize"
        Whether higher or lower scores are better.
    seed : int, optional
        The seed of the random numbers, 0 to 2^64 - 1. None is a random seed. The same seed
        repeats the run.
    """

    def __init__(
        self,
        genome: Genome,
        *,
        neighbor: Mutation,
        neighbors: int | None = None,
        acceptance: Acceptance | None = None,
        restart: tuple[int, int] | None = None,
        objective: ObjectiveName = "maximize",
        seed: int | None = None,
    ) -> None:
        self._genome = genome
        self._objective = objective
        self.neighbor = neighbor
        self.neighbors = neighbors
        self.acceptance = acceptance
        self.restart = restart
        self.seed = seed

    def _describe(self) -> dict[str, Any]:
        restart = None
        if self.restart is not None:
            try:
                patience, kicks = self.restart
            except (TypeError, ValueError):
                raise ValueError(
                    f"restart is a pair (patience, kicks), not {self.restart!r}"
                ) from None
            restart = [_whole("restart patience", patience), _whole("restart kicks", kicks)]
        return {
            "type": "local_search",
            "seed": _optional_whole("seed", self.seed),
            "neighbor": self.neighbor._describe(),
            "neighbors": _optional_whole("neighbors", self.neighbors),
            "acceptance": None if self.acceptance is None else self.acceptance._describe(),
            "restart": restart,
        }


def das_dennis(objectives: int, divisions: int) -> np.ndarray:
    """Points evenly spread on the unit simplex (Das and Dennis, 1998), a point per row: every
    point with ``objectives`` coordinates (2 to 6) that are multiples of ``1 / divisions`` and sum
    to 1, in lexicographic order. They serve as the reference directions of :class:`Nsga3` and the
    weights of :class:`Moead`. There are ``(divisions + objectives - 1)! / (divisions!
    (objectives - 1)!)`` of them: 91 for 3 objectives and 12 divisions, and none for 0 divisions.
    """
    return _genoxide.das_dennis(_whole("objectives", objectives), _whole("divisions", divisions))


def _rows(name: str, rows: Any) -> list[list[float]]:
    """Reference directions or weight vectors: a row each, with a value per objective."""
    array = np.asarray(rows, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} is a 2-D array, a row each with a value per objective")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} are finite numbers, not {array[~np.isfinite(array)][0]}")
    return array.tolist()


class _MultiObjective(_Algorithm):
    """A multi-objective algorithm, whose children are made by ``crossover`` and ``mutation``."""

    objectives: list[ObjectiveName]
    crossover: Crossover
    mutation: Mutation
    crossover_rate: float | None
    mutation_rate: float | None
    seed: int | None

    def _objectives(self) -> list[str]:
        for objective in self.objectives:
            if objective not in ("maximize", "minimize"):
                raise ValueError(f'an objective is "maximize" or "minimize", not {objective!r}')
        return list(self.objectives)

    def _variation(self) -> dict[str, Any]:
        return {
            "crossover": self.crossover._describe(),
            "mutate": self.mutation._describe(),
            "crossover_rate": _optional_number("crossover_rate", self.crossover_rate),
            "mutation_rate": _optional_number("mutation_rate", self.mutation_rate),
            "eliminate_duplicates": _flag(
                "eliminate_duplicates", getattr(self, "eliminate_duplicates", None)
            ),
        }

    def run(
        self,
        fitness: Callable[[np.ndarray], Any],
        *,
        generations: int | None = None,
        evaluations: int | None = None,
        time: float | None = None,
        stagnation: int | None = None,
        batch: bool = False,
        parallel: bool = False,
        on_generation: Callable[[MultiProgress], bool | None] | None = None,
    ) -> MultiResult:
        """Runs the algorithm until the first stop condition.

        Each call starts a new run from the settings. The settings are checked here, not by the
        constructor. With a ``seed``, the same call repeats the run exactly: one genome at a time,
        in batches or in parallel.

        Parameters
        ----------
        fitness : callable
            Takes a genome as a 1-D numpy array and returns a sequence of objective values, one
            per objective, None or NaN values (an invalid solution), or a tuple
            ``(objective_values, constraint_violation)``. The violation is 0 for a feasible
            solution and positive for an infeasible one. With ``batch=True``, it takes a
            generation as a 2-D array, a genome per row, and returns a 2-D array with a row of
            objective values per genome, or a tuple of it and an array of constraint violations.
            The function must be deterministic.
        generations : int, optional
            Stops after this many generations, 0 or more. 0 evaluates only the initial
            population.
        evaluations : int, optional
            Stops after the generation that reaches this many fitness evaluations, 0 or more.
        time : float, optional
            Stops when the run has taken this many seconds, 0 or more, checked after every
            generation. ``math.inf`` is no limit.
        stagnation : int, optional
            Stops after this many generations in which the front gained no solution that
            no earlier front member dominated or equaled, at least 1.
        batch : bool, default False
            Calls ``fitness`` once per generation with a 2-D array, and not for a generation of
            copies of their parents.
        parallel : bool, default False
            Calls a non-batch ``fitness`` from several threads at once. It pays off when the
            function releases the GIL (numpy on large arrays, I/O), or on free-threaded Python.
        on_generation : callable, optional
            Called with a :class:`MultiProgress` after every generation, the initial population
            (generation 0) included, on the thread that called ``run``. If it returns False, the
            run stops with the stop reason "aborted".

        At least one of ``generations``, ``evaluations``, ``time`` and ``stagnation`` is
        needed. None is no condition.

        Returns
        -------
        MultiResult
            The final non-dominated front, and what the run took.

        Raises
        ------
        ValueError
            Without a stop condition; for a wrong setting of the run, the algorithm, its genome
            or its operators, an operator that doesn't fit the genome, or other than 2 to 6
            objectives (the message names the setting); for a problem of
            :mod:`genoxide.problems`, which has one objective; and for a wrong fitness result: the
            wrong number of objective values, a negative constraint violation, a batch result
            that isn't a 2-D array, or one with a number of rows other than the number of genomes.
        TypeError
            If ``fitness`` or ``on_generation`` isn't callable, or ``fitness`` returns something
            that isn't a sequence of numbers. Another error converting a result, e.g. an
            ``OverflowError`` for an int too large for a float, is raised as it is.
        Exception
            An exception raised by ``fitness`` or ``on_generation`` stops the run, and ``run``
            raises it. So does ``KeyboardInterrupt`` on Ctrl+C.
        """
        _check_callable(fitness)
        if isinstance(fitness, problems.Problem):
            raise ValueError(
                f"{type(fitness).__name__} has one objective: use a single-objective algorithm"
            )
        stop = _stop(generations, evaluations, None, time, stagnation)
        function = _batch_objectives(fitness) if batch else fitness
        callback = _on_generation(on_generation, MultiProgress)
        return MultiResult(**self._run(function, stop, batch, parallel, callback))


class Nsga2(_MultiObjective):
    """NSGA-II, for 2 to 6 objectives: non-dominated sorting and crowding distance.

    ``objectives`` says, for each objective, whether to "maximize" or "minimize" it. The fitness
    function returns a sequence of objective values, None (an invalid solution) or
    ``(objective_values, constraint_violation)``; with ``batch=True``, a 2-D array with a row of
    objective values per genome, or a tuple of it and an array of constraint violations. The same
    goes for every multi-objective algorithm.

    Pairs of parents are recombined with probability ``crossover_rate`` (default 0.9), and each
    child is mutated with probability ``mutation_rate`` (default 1). With ``eliminate_duplicates``
    (the default, as in pymoo), a child that equals a member of the population or an earlier child
    is dropped and another bred instead, which keeps the population and its front free of copies;
    :class:`Nsga3`, :class:`Spea2` and :class:`SmsEmoa` have it too.

    Parameters
    ----------
    genome : Binary, Integer, Real or Permutation
        The search space.
    objectives : sequence of {"maximize", "minimize"}
        Whether to maximize or minimize each objective: 2 to 6 of them.
    population_size : int
        The number of individuals, 2 to 2^24.
    crossover : a crossover
        How pairs of parents are combined. It must fit the genome.
    mutation : a mutation
        How children are changed. It must fit the genome.
    crossover_rate : float, default 0.9
        The probability that a pair of parents is combined, 0 to 1.
    mutation_rate : float, default 1
        The probability that a child is mutated, 0 to 1. 0 needs a ``crossover_rate`` above 0
        and a crossover other than :class:`NoCrossover`: otherwise every child is a copy.
    eliminate_duplicates : bool, default True
        Whether a child equal to a member of the population or an earlier child is bred again.
    seed : int, optional
        The seed of the random numbers, 0 to 2^64 - 1. None is a random seed. The same seed
        repeats the run.
    """

    def __init__(
        self,
        genome: Genome,
        *,
        objectives: Sequence[ObjectiveName],
        population_size: int,
        crossover: Crossover,
        mutation: Mutation,
        crossover_rate: float | None = None,
        mutation_rate: float | None = None,
        eliminate_duplicates: bool | None = None,
        seed: int | None = None,
    ) -> None:
        self._genome = genome
        self.objectives = list(objectives)
        self.population_size = population_size
        self.crossover = crossover
        self.mutation = mutation
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.eliminate_duplicates = eliminate_duplicates
        self.seed = seed

    def _describe(self) -> dict[str, Any]:
        return {
            "type": "nsga2",
            "population_size": _whole("population_size", self.population_size),
            "seed": _optional_whole("seed", self.seed),
            "variation": self._variation(),
        }


class Nsga3(_MultiObjective):
    """NSGA-III (Deb and Jain, 2014), for 2 to 6 objectives: NSGA-II for many objectives, which
    spreads the front along reference directions instead of by crowding distance.

    ``reference_directions`` is a 2-D array with a direction per row and a value per objective,
    non-negative and not all 0, usually :func:`das_dennis` points, e.g. ``das_dennis(3, 12)``.
    ``population_size`` is their number by default; smaller populations can't fill every
    direction. Pairs of parents are recombined with probability ``crossover_rate`` and each child
    is mutated with probability ``mutation_rate``, both 1 by default, as in Deb and Jain. For real
    genomes, SBX with eta 30 is the usual crossover.

    The fitness function is as for :class:`Nsga2`.

    Parameters
    ----------
    genome : Binary, Integer, Real or Permutation
        The search space.
    objectives : sequence of {"maximize", "minimize"}
        Whether to maximize or minimize each objective: 2 to 6 of them.
    reference_directions : 2-D array_like of float
        At least 1 direction, a row each with a value per objective: finite, non-negative and
        not all 0.
    crossover : a crossover
        How pairs of parents are combined. It must fit the genome.
    mutation : a mutation
        How children are changed. It must fit the genome.
    population_size : int, optional
        The number of individuals, 2 to 2^24. The number of reference directions by default.
    crossover_rate : float, default 1
        The probability that a pair of parents is combined, 0 to 1.
    mutation_rate : float, default 1
        The probability that a child is mutated, 0 to 1. 0 needs a ``crossover_rate`` above 0
        and a crossover other than :class:`NoCrossover`: otherwise every child is a copy.
    eliminate_duplicates : bool, default True
        Whether a child equal to a member of the population or an earlier child is bred again.
    seed : int, optional
        The seed of the random numbers, 0 to 2^64 - 1. None is a random seed. The same seed
        repeats the run.
    """

    def __init__(
        self,
        genome: Genome,
        *,
        objectives: Sequence[ObjectiveName],
        reference_directions: np.ndarray | Sequence[Sequence[float]],
        crossover: Crossover,
        mutation: Mutation,
        population_size: int | None = None,
        crossover_rate: float | None = None,
        mutation_rate: float | None = None,
        eliminate_duplicates: bool | None = None,
        seed: int | None = None,
    ) -> None:
        self._genome = genome
        self.objectives = list(objectives)
        self.reference_directions = reference_directions
        self.population_size = population_size
        self.crossover = crossover
        self.mutation = mutation
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.eliminate_duplicates = eliminate_duplicates
        self.seed = seed

    def _describe(self) -> dict[str, Any]:
        return {
            "type": "nsga3",
            "reference_directions": _rows("reference_directions", self.reference_directions),
            "population_size": _optional_whole("population_size", self.population_size),
            "seed": _optional_whole("seed", self.seed),
            "variation": self._variation(),
        }


class Spea2(_MultiObjective):
    """SPEA2 (Zitzler, Laumanns and Thiele, 2001), for 2 to 6 objectives: an archive of
    ``population_size`` solutions, the non-dominated ones first. When they don't fit, the one
    nearest to another is removed until they do, which keeps the extremes of the front and spreads
    it evenly: O(N³) per generation at worst, fine for populations of a few hundred.

    Pairs of parents are recombined with probability ``crossover_rate`` (default 0.9), and each
    child is mutated with probability ``mutation_rate`` (default 1). The fitness function is as
    for :class:`Nsga2`.

    Parameters
    ----------
    genome : Binary, Integer, Real or Permutation
        The search space.
    objectives : sequence of {"maximize", "minimize"}
        Whether to maximize or minimize each objective: 2 to 6 of them.
    population_size : int
        The size of the archive and the number of children per generation, 2 to 2^24.
    crossover : a crossover
        How pairs of parents are combined. It must fit the genome.
    mutation : a mutation
        How children are changed. It must fit the genome.
    crossover_rate : float, default 0.9
        The probability that a pair of parents is combined, 0 to 1.
    mutation_rate : float, default 1
        The probability that a child is mutated, 0 to 1. 0 needs a ``crossover_rate`` above 0
        and a crossover other than :class:`NoCrossover`: otherwise every child is a copy.
    eliminate_duplicates : bool, default True
        Whether a child equal to a member of the population or an earlier child is bred again.
    seed : int, optional
        The seed of the random numbers, 0 to 2^64 - 1. None is a random seed. The same seed
        repeats the run.
    """

    def __init__(
        self,
        genome: Genome,
        *,
        objectives: Sequence[ObjectiveName],
        population_size: int,
        crossover: Crossover,
        mutation: Mutation,
        crossover_rate: float | None = None,
        mutation_rate: float | None = None,
        eliminate_duplicates: bool | None = None,
        seed: int | None = None,
    ) -> None:
        self._genome = genome
        self.objectives = list(objectives)
        self.population_size = population_size
        self.crossover = crossover
        self.mutation = mutation
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.eliminate_duplicates = eliminate_duplicates
        self.seed = seed

    def _describe(self) -> dict[str, Any]:
        return {
            "type": "spea2",
            "population_size": _whole("population_size", self.population_size),
            "seed": _optional_whole("seed", self.seed),
            "variation": self._variation(),
        }


class Moead(_MultiObjective):
    """MOEA/D (Zhang and Li, 2007), for 2 to 6 objectives: a single-objective subproblem per
    weight vector, which shares good solutions with the subproblems of its nearest weight vectors.

    ``weights`` is a 2-D array with a weight vector per row and a value per objective,
    non-negative and not all 0, usually :func:`das_dennis` points; the population size is their
    number, at least 2. ``decomposition`` turns the objectives into one value per subproblem:
    :class:`Tchebycheff` (the default) or :class:`Pbi`, which spreads fronts of 3 or more
    objectives well. Each generation, every subproblem gets a child, one of the crossover's two.

    For real genomes, SBX with eta 20 is the usual crossover. The fitness function is as for
    :class:`Nsga2`. MOEA/D has no ``eliminate_duplicates``: it replaces its neighbors one child at
    a time.

    Parameters
    ----------
    genome : Binary, Integer, Real or Permutation
        The search space.
    objectives : sequence of {"maximize", "minimize"}
        Whether to maximize or minimize each objective: 2 to 6 of them.
    weights : 2-D array_like of float
        At least 2 weight vectors, a row each with a value per objective: finite, non-negative
        and not all 0. Their number is the population size.
    crossover : a crossover
        How pairs of parents are combined. It must fit the genome.
    mutation : a mutation
        How children are changed. It must fit the genome.
    neighbors : int, default 20
        The size of each neighborhood, itself included, at least 2. More than the number of
        weight vectors is all of them.
    neighbor_mating : float, default 0.9
        The probability that the parents come from the neighborhood rather than the whole
        population, 0 to 1.
    max_replacements : int, default 2
        The most solutions a child replaces, at least 1. The neighborhood size or more removes
        the limit.
    decomposition : Tchebycheff or Pbi, default Tchebycheff()
        How the objectives become one value per subproblem.
    crossover_rate : float, default 1
        The probability that a pair of parents is combined, 0 to 1.
    mutation_rate : float, default 1
        The probability that a child is mutated, 0 to 1. 0 needs a ``crossover_rate`` above 0
        and a crossover other than :class:`NoCrossover`: otherwise every child is a copy.
    seed : int, optional
        The seed of the random numbers, 0 to 2^64 - 1. None is a random seed. The same seed
        repeats the run.
    """

    def __init__(
        self,
        genome: Genome,
        *,
        objectives: Sequence[ObjectiveName],
        weights: np.ndarray | Sequence[Sequence[float]],
        crossover: Crossover,
        mutation: Mutation,
        neighbors: int | None = None,
        neighbor_mating: float | None = None,
        max_replacements: int | None = None,
        decomposition: Decomposition | None = None,
        crossover_rate: float | None = None,
        mutation_rate: float | None = None,
        seed: int | None = None,
    ) -> None:
        self._genome = genome
        self.objectives = list(objectives)
        self.weights = weights
        self.crossover = crossover
        self.mutation = mutation
        self.neighbors = neighbors
        self.neighbor_mating = neighbor_mating
        self.max_replacements = max_replacements
        self.decomposition = decomposition
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.seed = seed

    def _describe(self) -> dict[str, Any]:
        return {
            "type": "moead",
            "weights": _rows("weights", self.weights),
            "neighbors": _optional_whole("neighbors", self.neighbors),
            "neighbor_mating": _optional_number("neighbor_mating", self.neighbor_mating),
            "max_replacements": _optional_whole("max_replacements", self.max_replacements),
            "decomposition": None if self.decomposition is None else self.decomposition._describe(),
            "seed": _optional_whole("seed", self.seed),
            "variation": self._variation(),
        }


class SmsEmoa(_MultiObjective):
    """SMS-EMOA (Beume, Naujoks and Emmerich, 2007), for 2 to 6 objectives: survival by
    hypervolume contribution, for fronts that are well spread and converged, at a higher cost per
    generation than NSGA-II: O(N log N) per removal for 2 objectives, O(N²) for 3, O(N³) for 4 and
    O(N⁴) for 5, where :class:`Nsga3` or :class:`Moead` are better choices.

    ``offspring``: the children per generation, ``population_size`` by default; 1 is the original
    steady-state algorithm. Pairs of parents are recombined with probability ``crossover_rate``
    (default 0.9), and each child is mutated with probability ``mutation_rate`` (default 1). The
    fitness function is as for :class:`Nsga2`.

    Parameters
    ----------
    genome : Binary, Integer, Real or Permutation
        The search space.
    objectives : sequence of {"maximize", "minimize"}
        Whether to maximize or minimize each objective: 2 to 6 of them.
    population_size : int
        The number of individuals, 2 to 2^24.
    crossover : a crossover
        How pairs of parents are combined. It must fit the genome.
    mutation : a mutation
        How children are changed. It must fit the genome.
    offspring : int, optional
        The children per generation, 1 to 2^24. ``population_size`` by default.
    crossover_rate : float, default 0.9
        The probability that a pair of parents is combined, 0 to 1.
    mutation_rate : float, default 1
        The probability that a child is mutated, 0 to 1. 0 needs a ``crossover_rate`` above 0
        and a crossover other than :class:`NoCrossover`: otherwise every child is a copy.
    eliminate_duplicates : bool, default True
        Whether a child equal to a member of the population or an earlier child is bred again.
    seed : int, optional
        The seed of the random numbers, 0 to 2^64 - 1. None is a random seed. The same seed
        repeats the run.
    """

    def __init__(
        self,
        genome: Genome,
        *,
        objectives: Sequence[ObjectiveName],
        population_size: int,
        crossover: Crossover,
        mutation: Mutation,
        offspring: int | None = None,
        crossover_rate: float | None = None,
        mutation_rate: float | None = None,
        eliminate_duplicates: bool | None = None,
        seed: int | None = None,
    ) -> None:
        self._genome = genome
        self.objectives = list(objectives)
        self.population_size = population_size
        self.crossover = crossover
        self.mutation = mutation
        self.offspring = offspring
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.eliminate_duplicates = eliminate_duplicates
        self.seed = seed

    def _describe(self) -> dict[str, Any]:
        return {
            "type": "sms_emoa",
            "population_size": _whole("population_size", self.population_size),
            "offspring": _optional_whole("offspring", self.offspring),
            "seed": _optional_whole("seed", self.seed),
            "variation": self._variation(),
        }


# the submodules use the classes above
from . import indicators, problems  # noqa: E402
