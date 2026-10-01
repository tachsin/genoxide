"""Evolutionary computation in Rust, for Python.

Genetic algorithms, local search, the Nelder-Mead simplex method, differential evolution,
evolution strategies, CMA-ES, OpenAI's evolution strategy, NEAT, particle swarm optimization, the
island model, and NSGA-II, NSGA-III, SPEA2, MOEA/D and SMS-EMOA for several objectives, from
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
:class:`Real` and :class:`AdaptiveReal`, ``int64`` for :class:`Integer` and :class:`Permutation`)
and returns a number,
``None`` for an invalid solution, or a tuple ``(score, constraint_violation)``. With
``batch=True``, it takes a whole generation as a 2-D array, a genome per row, and returns an array
of scores: at most one call per generation, for vectorized numpy code.

:mod:`genoxide.problems` has test problems from the literature, single- and multi-objective,
which ``run`` evaluates in Rust, and :mod:`genoxide.indicators` the quality indicators of
multi-objective fronts. :mod:`genoxide.nn` has neural networks whose weights a genome holds, and
:mod:`genoxide.problems.control` pole-balancing tasks for them (neuroevolution); :class:`Neat`
evolves networks' structure too, as :mod:`genoxide.neat`'s networks. :mod:`genoxide.gp` has
genetic programming: trees of formulas and Boolean functions, with symbolic regression and the
Boolean problems evaluated in Rust.
:mod:`genoxide.math` has genoxide's portable math functions, the same to the bit on every
platform.

A run stops at the first of its stop conditions: ``generations``, ``evaluations``, ``target``,
``time`` (seconds) and ``stagnation`` (generations without improvement), or when its
``on_generation`` callback returns False.

A single-objective run's ``control`` callback gets the running algorithm once per generation, to
change its settings (parameter control, e.g. an annealed mutation step) or to re-evaluate it after
the fitness function changed: see :class:`Running`.

``run(..., checkpoint="run.ckpt", checkpoint_every=100)`` saves the run as it goes, and
``run(..., resume="run.ckpt")`` continues it later, with the results of an uninterrupted run.

Settings are checked before a run: a count, a size or an integer bound is a whole number (an
``int`` or a numpy integer, not a ``bool`` or a ``float``), and a real setting is a finite number.
A wrong one is a ``ValueError`` that names it.
"""

from __future__ import annotations

import json
import math as _math
import numbers
import operator
import os
from collections.abc import Callable, Sequence
from dataclasses import FrozenInstanceError, dataclass
from functools import cached_property
from typing import Any, Literal, Union, cast

import numpy as np

from . import _genoxide
from . import neat

__version__: str = _genoxide.__version__

__all__ = [
    # genomes
    "Binary",
    "Integer",
    "Real",
    "Permutation",
    "AdaptiveReal",
    # selection
    "Tournament",
    "Rank",
    "Roulette",
    "StochasticUniversalSampling",
    "Truncation",
    "RandomSelection",
    "LexicographicTournament",
    "DoubleTournament",
    "Tarpeian",
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
    "SelfAdaptiveMutation",
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
    "Es",
    "Cmaes",
    "OpenEs",
    "Neat",
    "Adam",
    "Sgd",
    "Pso",
    "LocalSearch",
    "NelderMead",
    "Islands",
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
    "NeatResult",
    "NeatProgress",
    # parameter control
    "Running",
    "RunningGa",
    "RunningDe",
    "RunningEs",
    "RunningCmaes",
    "RunningOpenEs",
    "RunningNeat",
    "RunningPso",
    "RunningLocalSearch",
    "RunningNelderMead",
    "RunningIslands",
    # submodules
    "problems",
    "indicators",
    "nn",
    "neat",
    "math",
    "gp",
]

ObjectiveName = Literal["maximize", "minimize"]
# De's settings, see De
DeStrategy = Union[Literal["rand1", "best1"], dict[str, float]]
DeControl = dict[str, float]
DeRestarts = Union[Literal["never"], dict[str, float]]
Bounds = Union[tuple[float, float], Sequence[tuple[float, float]]]


def _whole(
    name: str,
    value: Any,
    *,
    minimum: int | None = 0,
    maximum: int = 2**64 - 1,
    plural: bool = False,
) -> int:
    """The setting ``name`` as an ``int``: an ``int`` or a numpy integer, not a ``bool`` (an
    ``int`` to Python) nor a ``float``, even a whole one; at least ``minimum`` unless it's None,
    and at most ``maximum``, the largest the Rust side can read."""
    wrong = f"{name} {'are whole numbers' if plural else 'is a whole number'}, not {value!r}"
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(wrong)
    try:
        number = operator.index(value)
    except TypeError:
        raise ValueError(wrong) from None
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} {'are' if plural else 'is'} at least {minimum}, not {number}")
    if number > maximum:
        raise ValueError(f"{name} {'are' if plural else 'is'} at most {maximum}, not {number}")
    return number


# what each setting object is, for the error when it's something else
_GENOME = "a genome such as gx.Binary(8) or gx.Real((0, 1), length=5)"
_REAL = "a gx.Real, e.g. gx.Real((0, 1), length=5)"
_SELECT = "a selection such as gx.Tournament(3)"
_CROSSOVER = "a crossover such as gx.UniformCrossover()"
_MUTATION = "a mutation such as gx.BitFlip(rate=0.01)"
_SCHEME = "a scheme such as gx.Generational(elitism=1)"
_NEIGHBOR = "a mutation such as gx.SwapMutation()"
_ACCEPTANCE = "an acceptance such as gx.NotWorse()"
_DECOMPOSITION = "gx.Tchebycheff() or gx.Pbi(theta)"


def _describe_setting(name: str, value: Any, what: str) -> dict[str, Any]:
    """The description of an operator, genome or other setting object, or a ValueError that names
    the setting: for a class passed without calling it, and for anything else, such as a string."""
    if isinstance(value, type) and hasattr(value, "_describe"):
        raise ValueError(
            f"{name} is {what}: pass {value.__name__}(...), an instance, not the class "
            f"{value.__name__}"
        )
    if isinstance(value, type) or not callable(getattr(value, "_describe", None)):
        raise ValueError(f"{name} is {what}, not {value!r}")
    return cast(dict[str, Any], value._describe())


def _number(name: str, value: Any, *, plural: bool = False) -> float:
    """The setting ``name`` as a ``float``: a finite real number, not a ``bool``."""
    wrong = f"{name} {'are finite numbers' if plural else 'is a finite number'}, not {value!r}"
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(wrong)
    number = float(value)
    if not _math.isfinite(number):
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
        # at most 2^24 genes, checked before the list is made
        length = _whole(f"{name}.length", length, maximum=2**24)
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
    # the genes are int64
    return _whole(name, value, minimum=-(2**63), maximum=2**63 - 1, plural=True)


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

    def random_genome(self, seed: int) -> np.ndarray:
        """A random genome, uniform within the bounds: the one that genoxide's Rust
        ``real.random_genome(&mut StreamRng::seed_from_u64(seed))`` gives, so that a Python program
        can start from the same point as a Rust one, e.g. ``OpenEs(initial_mean=...)``.

        ``seed`` is 0 to 2^64 - 1.
        """
        bounds = [tuple(pair) for pair in self._describe()["bounds"]]
        return _genoxide.random_real(bounds, _whole("seed", seed))


@dataclass(frozen=True)
class Permutation:
    """Orderings of ``0 .. length - 1``: numpy ``int64`` arrays.

    ``length`` is 1 to 2^24.
    """

    length: int

    def _describe(self) -> dict[str, Any]:
        return {"type": "permutation", "length": _whole("Permutation.length", self.length)}


@dataclass(frozen=True)
class AdaptiveReal:
    """Real numbers between bounds, each genome with a mutation step size that evolves with it:
    numpy ``float64`` arrays of the genes, as :class:`Real`'s, without the step size.

    ``real`` gives the bounds. :class:`SelfAdaptiveMutation` changes the step size, a fraction
    of each gene's range starting at ``initial_step`` (greater than 0 and finite, e.g. 0.3 to start
    broad), and then the genes by normal steps of it: steps that lead to good genomes survive with
    them, so the search tunes its own step size. With :class:`NoCrossover` and
    :class:`MuCommaLambda`, a :class:`Ga` is then a (mu, lambda) evolution strategy; :class:`Es`
    is one, with a step size per gene and recombination. Crossovers: :class:`UniformCrossover`,
    :class:`PointCrossover` and :class:`NoCrossover`.
    """

    real: Real
    initial_step: float

    def _describe(self) -> dict[str, Any]:
        if not isinstance(self.real, Real):
            raise ValueError(f"AdaptiveReal.real is {_REAL}, not {self.real!r}")
        return {
            "type": "adaptive_real",
            "bounds": self.real._describe()["bounds"],
            "initial_step": _number("AdaptiveReal.initial_step", self.initial_step),
        }


# trees of genetic programming are a genome of `gp`, which imports this module
Genome = Union[Binary, Integer, Real, Permutation, AdaptiveReal, "gp.Gp"]

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


@dataclass(frozen=True)
class LexicographicTournament:
    """Lexicographic parsimony pressure (Luke and Panait 2002): tournaments of ``size``
    individuals (1 to 2^24) in which, of equal fitness, the smaller genome wins: a tree with fewer
    nodes (:mod:`genoxide.gp`), else the shorter genome. Strong where fitness values repeat
    (Boolean problems, counts), little with continuous fitness, for which ``bucket_ratio``
    (greater than 0 and at most 1, Luke and Panait's 1/2) makes near values equal: the worst
    ``bucket_ratio`` of the population in the lowest bucket, the worst ``bucket_ratio`` of the
    rest in the next, and so on, and tournaments compare the buckets, then sizes."""

    size: int
    bucket_ratio: float | None = None

    def _describe(self) -> dict[str, Any]:
        ratio = None
        if self.bucket_ratio is not None:
            ratio = _number("LexicographicTournament.bucket_ratio", self.bucket_ratio)
        return {
            "type": "lexicographic_tournament",
            "size": _whole("LexicographicTournament.size", self.size),
            "bucket_ratio": ratio,
        }


@dataclass(frozen=True)
class DoubleTournament:
    """Double tournament (Luke and Panait 2002), against bloat: a size tournament between the
    winners of two fitness tournaments of ``fitness_size`` individuals (1 to 2^24), in which the
    smaller of the two wins with probability ``parsimony / 2``, ``parsimony`` from 1 (no pressure)
    to 2 (always the smaller). With ``size_first``, the fitness tournament's contestants are the
    winners of size tournaments instead. Size is a tree's number of nodes (:mod:`genoxide.gp`),
    or a genome's length. Fitness tournaments of 7 and a parsimony of 1.4 were the best in Luke
    and Panait's comparison (2006)."""

    fitness_size: int
    parsimony: float
    size_first: bool = False

    def _describe(self) -> dict[str, Any]:
        return {
            "type": "double_tournament",
            "fitness_size": _whole("DoubleTournament.fitness_size", self.fitness_size),
            "parsimony": _number("DoubleTournament.parsimony", self.parsimony),
            "size_first": bool(_flag("DoubleTournament.size_first", self.size_first)),
        }


@dataclass(frozen=True)
class Tarpeian:
    """Tarpeian bloat control (Poli 2003): ``select``, any selection but another Tarpeian one,
    with each genome larger than the population's mean size counted as invalid with probability
    ``rate`` (greater than 0 and at most 1), drawn anew at each selection. Size is a tree's
    number of nodes (:mod:`genoxide.gp`), or a genome's length."""

    select: Select
    rate: float

    def _describe(self) -> dict[str, Any]:
        return {
            "type": "tarpeian",
            "select": _describe_setting("Tarpeian.select", self.select, _SELECT),
            "rate": _number("Tarpeian.rate", self.rate),
        }


Select = Union[
    Tournament,
    Rank,
    Roulette,
    StochasticUniversalSampling,
    Truncation,
    RandomSelection,
    LexicographicTournament,
    DoubleTournament,
    Tarpeian,
]

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
    "gp.SubtreeCrossover",
    "gp.OnePointCrossover",
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


@dataclass(frozen=True)
class SelfAdaptiveMutation:
    """Self-adaptive Gaussian mutation, for :class:`AdaptiveReal` genomes: first the genome's step
    size changes log-normally, ``step * exp(learning_rate * N(0, 1))``, then every gene moves by a
    normal step with that standard deviation times its range, mirrored at the bounds.

    ``learning_rate`` is how fast the step size changes, greater than 0 and finite; None is
    ``1 / sqrt(n)`` for the ``n`` genes that can take more than one value. ``min_step`` is the
    smallest step size, greater than 0 and finite; None is 1e-12. The step size is at most 10.
    """

    learning_rate: float | None = None
    min_step: float | None = None

    def _describe(self) -> dict[str, Any]:
        return {
            "type": "self_adaptive",
            "learning_rate": _optional_number(
                "SelfAdaptiveMutation.learning_rate", self.learning_rate
            ),
            "min_step": _optional_number("SelfAdaptiveMutation.min_step", self.min_step),
        }


Mutation = Union[
    BitFlip,
    UniformMutation,
    GaussianMutation,
    PolynomialMutation,
    SwapMutation,
    InversionMutation,
    InsertionMutation,
    ScrambleMutation,
    SelfAdaptiveMutation,
    "gp.SubtreeMutation",
    "gp.PointMutation",
    "gp.HoistMutation",
    "gp.ShrinkMutation",
    "gp.ConstantMutation",
    "gp.Mutations",
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

    best_genome: Any
    """The best genome found: a 1-D numpy array, or a :class:`genoxide.gp.Tree` for a
    :class:`genoxide.gp.Gp` genome."""
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
    - "converged": the algorithm converged with nothing more to do, e.g. a :class:`NelderMead`
      whose simplex shrank within its tolerance, with no restart left;
    - "stalled": nothing new to evaluate for 10,000 generations in a row (e.g. every child was a
      copy of a parent), while only ``target`` or ``evaluations`` could stop the run;
    - "other": a reason that the stop conditions of the package don't produce.
    """


@dataclass(frozen=True, eq=False)
class MultiResult:
    """The result of a multi-objective run: its final non-dominated front, each genome once (the
    first of its copies), as in the last generation's ``MultiProgress``. Different genomes with the
    same objective values each have a row."""

    front_genomes: Any
    """The genomes of the front, a row each; a tuple of :class:`genoxide.gp.Tree` objects for a
    :class:`genoxide.gp.Gp` genome."""
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


class _Arrays:
    """A population's arrays, already made: what a copied or unpickled progress object reads, in
    place of the run's copy of the population."""

    __slots__ = ("_genomes", "_values", "_violations")

    def __init__(self, genomes: Any, values: np.ndarray, violations: np.ndarray) -> None:
        self._genomes = genomes
        self._values = values
        self._violations = violations

    def genomes(self) -> Any:
        return self._genomes

    def values(self) -> np.ndarray:
        return self._values

    def violations(self) -> np.ndarray:
        return self._violations


class _ReadOnly:
    """A progress object: read-only, with the population's arrays made when first read and then
    kept (``cached_property`` writes them to the instance's ``__dict__`` directly)."""

    _repr_fields: tuple[str, ...] = ()

    def __setattr__(self, name: str, value: Any) -> None:
        raise FrozenInstanceError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> None:
        raise FrozenInstanceError(f"cannot delete field {name!r}")

    def __repr__(self) -> str:
        fields = ", ".join(f"{name}={getattr(self, name)!r}" for name in self._repr_fields)
        return f"{type(self).__name__}({fields})"


class Progress(_ReadOnly):
    """A single-objective run after a generation, for ``on_generation`` and ``control``.

    It's read-only. ``population``, ``scores`` and ``violations`` are made when first read, from a
    copy of the population that the run takes after the generation, and then kept: a callback that
    doesn't read them doesn't pay for their arrays. A progress object kept after its callback
    returned stays valid, and can be copied and pickled. It isn't a dataclass:
    ``dataclasses.fields``, ``asdict`` and ``replace`` don't apply to it.
    """

    __match_args__ = (
        "generation",
        "evaluations",
        "seconds",
        "best_fitness",
        "best_genome",
        "population",
        "scores",
        "violations",
    )
    _repr_fields = ("generation", "evaluations", "seconds", "best_fitness")

    generation: int
    """The generations completed: 0 after the initial population."""
    evaluations: int
    """The fitness evaluations so far."""
    seconds: float
    """The time since the run started."""
    best_fitness: float | None
    """The best score so far, or None if no valid solution was found yet."""
    best_genome: Any
    """The best genome so far: a 1-D numpy array, or a :class:`genoxide.gp.Tree`."""
    _population: _genoxide.Snapshot | _Arrays

    def __init__(
        self,
        generation: int,
        evaluations: int,
        seconds: float,
        best_fitness: float | None,
        best_genome: Any,
        population: Any,
    ) -> None:
        # `population`: the run's copy of the population, or its arrays (`_Arrays`)
        self.__dict__.update(
            generation=generation,
            evaluations=evaluations,
            seconds=seconds,
            best_fitness=best_fitness,
            best_genome=best_genome,
            _population=population,
        )

    @cached_property
    def population(self) -> Any:
        """The population after the generation, a genome per row; a tuple of
        :class:`genoxide.gp.Tree` objects for a :class:`genoxide.gp.Gp` genome."""
        # a population has genomes: only a front's arrays have none
        return self._population.genomes()

    @cached_property
    def scores(self) -> np.ndarray:
        """The population's scores: NaN for an invalid solution."""
        return self._population.values()

    @cached_property
    def violations(self) -> np.ndarray:
        """The population's constraint violations: 0 for a feasible solution, NaN for an invalid
        one."""
        return self._population.violations()

    def __reduce__(self) -> tuple[Any, ...]:
        arrays = _Arrays(self.population, self.scores, self.violations)
        return (
            Progress,
            (
                self.generation,
                self.evaluations,
                self.seconds,
                self.best_fitness,
                self.best_genome,
                arrays,
            ),
        )


class MultiProgress(_ReadOnly):
    """A multi-objective run after a generation, for ``on_generation``.

    It's read-only. ``population``, ``objectives``, ``violations``, ``front_objectives`` and
    ``front_violations`` are made when first read, as :class:`Progress`'s population, and then
    kept. It isn't a dataclass either.
    """

    __match_args__ = (
        "generation",
        "evaluations",
        "seconds",
        "front_size",
        "population",
        "objectives",
        "violations",
        "front_objectives",
        "front_violations",
    )
    _repr_fields = ("generation", "evaluations", "seconds", "front_size")

    generation: int
    """The generations completed: 0 after the initial population."""
    evaluations: int
    """The fitness evaluations so far."""
    seconds: float
    """The time since the run started."""
    front_size: int
    """The number of non-dominated individuals in the population, each genome once."""
    _population: _genoxide.Snapshot | _Arrays
    _front: _genoxide.Snapshot | _Arrays

    def __init__(
        self,
        generation: int,
        evaluations: int,
        seconds: float,
        front_size: int,
        population: Any,
        front: Any,
    ) -> None:
        # `population` and `front`: the run's copies of them, or their arrays (`_Arrays`)
        self.__dict__.update(
            generation=generation,
            evaluations=evaluations,
            seconds=seconds,
            front_size=front_size,
            _population=population,
            _front=front,
        )

    @cached_property
    def population(self) -> Any:
        """The population after the generation, a genome per row; a tuple of
        :class:`genoxide.gp.Tree` objects for a :class:`genoxide.gp.Gp` genome."""
        # a population has genomes: only a front's arrays have none
        return self._population.genomes()

    @cached_property
    def objectives(self) -> np.ndarray:
        """The population's objective values, a row each: NaN for an invalid solution."""
        return self._population.values()

    @cached_property
    def violations(self) -> np.ndarray:
        """The population's constraint violations: 0 for a feasible solution, NaN for an invalid
        one."""
        return self._population.violations()

    @cached_property
    def front_objectives(self) -> np.ndarray:
        """The objective values of the population's non-dominated individuals, each genome once, a
        row each: the ``front_objectives`` of ``MultiResult`` after the last generation."""
        return self._front.values()

    @cached_property
    def front_violations(self) -> np.ndarray:
        """Their constraint violations."""
        return self._front.violations()

    def __reduce__(self) -> tuple[Any, ...]:
        population = _Arrays(self.population, self.objectives, self.violations)
        front = _Arrays(None, self.front_objectives, self.front_violations)
        return (
            MultiProgress,
            (self.generation, self.evaluations, self.seconds, self.front_size, population, front),
        )


@dataclass(frozen=True, eq=False)
class NeatResult:
    """The result of a :class:`Neat` run: :class:`Result`'s fields, with the best network."""

    best_genome: neat.Network
    """The best network found."""
    best_fitness: float | None
    """Its score, or None if no valid solution was found."""
    violation: float
    """Its constraint violation: 0 for a feasible solution, and NaN if no valid solution was
    found."""
    generations: int
    """The generations completed after the initial population."""
    evaluations: int
    """The fitness evaluations. A network copied unchanged keeps its fitness without one."""
    seconds: float
    """The duration of the run, in seconds."""
    stop_reason: str
    """What stopped the run, as :attr:`Result.stop_reason`."""


class NeatProgress(_ReadOnly):
    """A :class:`Neat` run after a generation, for ``on_generation`` and ``control``: as
    :class:`Progress`, with networks for genomes. ``population`` is a tuple of the generation's
    :class:`genoxide.neat.Network` objects, made when first read."""

    __match_args__ = (
        "generation",
        "evaluations",
        "seconds",
        "best_fitness",
        "best_genome",
        "population",
        "scores",
        "violations",
    )
    _repr_fields = ("generation", "evaluations", "seconds", "best_fitness")

    generation: int
    """The generations completed: 0 after the initial population."""
    evaluations: int
    """The fitness evaluations so far."""
    seconds: float
    """The time since the run started."""
    best_fitness: float | None
    """The best score so far, or None if no valid solution was found yet."""
    best_genome: neat.Network
    """The best network so far."""
    _population: _genoxide.Snapshot | _Arrays

    def __init__(
        self,
        generation: int,
        evaluations: int,
        seconds: float,
        best_fitness: float | None,
        best_genome: neat.Network,
        population: Any,
    ) -> None:
        # `population`: the run's copy of the population, or its arrays (`_Arrays`)
        self.__dict__.update(
            generation=generation,
            evaluations=evaluations,
            seconds=seconds,
            best_fitness=best_fitness,
            best_genome=best_genome,
            _population=population,
        )

    @cached_property
    def population(self) -> tuple[neat.Network, ...]:
        """The networks after the generation, in the population's order."""
        return cast(tuple[neat.Network, ...], self._population.genomes())

    @cached_property
    def scores(self) -> np.ndarray:
        """The networks' scores: NaN for an invalid solution."""
        return self._population.values()

    @cached_property
    def violations(self) -> np.ndarray:
        """The networks' constraint violations: 0 for a feasible solution, NaN for an invalid
        one."""
        return self._population.violations()

    def __reduce__(self) -> tuple[Any, ...]:
        arrays = _Arrays(self.population, self.scores, self.violations)
        return (
            NeatProgress,
            (
                self.generation,
                self.evaluations,
                self.seconds,
                self.best_fitness,
                self.best_genome,
                arrays,
            ),
        )


# --- parameter control ---------------------------------------------------------------------------


class Running:
    """The algorithm of a running single-objective run, for its ``control`` callback.

    ``run(..., control=callback)`` calls ``callback(algorithm, progress)`` once per generation,
    the initial population (generation 0) included, with this handle to the running algorithm and
    a :class:`Progress`, after ``on_generation``. It's for parameter control, e.g. a mutation step
    annealed over the run, and for re-evaluation after the fitness function changed. A change
    applies from the next generation. The callback is also called after the last generation, so
    the settings it makes there are never used; it isn't called again after a re-evaluation's
    evaluations, which complete no generation. What it returns is ignored: return False from
    ``on_generation`` to stop a run.

    Each algorithm has a class of its own, with its settings as properties: :class:`RunningGa`,
    :class:`RunningDe`, :class:`RunningEs`, :class:`RunningCmaes`, :class:`RunningOpenEs`,
    :class:`RunningNeat`, :class:`RunningPso`, :class:`RunningLocalSearch`,
    :class:`RunningNelderMead` and :class:`RunningIslands`. A new value is checked as in the
    algorithm's constructor: a wrong one raises a ``ValueError`` and changes nothing. The handle
    works only during the callback; afterwards it raises a ``RuntimeError``.

    A control that changes nothing leaves the run as it is: with a seed, the same result as
    without the control.
    """

    __slots__ = ("_native",)

    def __init__(self, native: Any, algorithm: _Single) -> None:
        self._native = native

    def reevaluate(self) -> None:
        """Scores again what the algorithm keeps, for a fitness function that changed during the
        run: adaptive penalty weights, a retrained surrogate model, a moving optimum.

        Instead of the next generation's children, the population is evaluated again (a particle
        swarm's positions and personal bests, a local search's current and best solution, a
        Nelder-Mead simplex, whose iteration under way is dropped), without breeding and without
        a new generation: ``on_generation`` is then called again with the same generation number,
        and ``control`` isn't. The best solution is then the best of the new values, as old and
        new values aren't comparable. The evaluations count, and no random numbers are drawn, so
        a seeded run that re-evaluates at the same generations repeats.
        """
        self._native.reevaluate()

    def _get(self, name: str) -> Any:
        return json.loads(self._native.get(name))

    def _set(self, name: str, value: Any) -> None:
        self._native.set(name, json.dumps(value, default=_json_number, allow_nan=False))


class RunningGa(Running):
    """A running :class:`Ga`, for ``control``: its rates and operators.

    ``select``, ``crossover`` and ``mutation`` take a new operator of any kind that fits the
    genome, e.g. a :class:`GaussianMutation` with a smaller ``sigma``, or a
    :class:`PolynomialMutation` instead of it. ``mutation_rate`` 0 needs a crossover that
    recombines, with ``crossover_rate`` above 0, as in :class:`Ga`.
    """

    __slots__ = ("_select", "_crossover", "_mutation")

    def __init__(self, native: Any, algorithm: _Single) -> None:
        super().__init__(native, algorithm)
        assert isinstance(algorithm, Ga)
        self._select = algorithm.select
        self._crossover = algorithm.crossover
        self._mutation = algorithm.mutation

    @property
    def crossover_rate(self) -> float:
        """The probability that a pair of parents is combined, 0 to 1."""
        return float(self._get("crossover_rate"))

    @crossover_rate.setter
    def crossover_rate(self, rate: float) -> None:
        self._set("crossover_rate", _number("crossover_rate", rate))

    @property
    def mutation_rate(self) -> float:
        """The probability that a child is mutated, 0 to 1."""
        return float(self._get("mutation_rate"))

    @mutation_rate.setter
    def mutation_rate(self, rate: float) -> None:
        self._set("mutation_rate", _number("mutation_rate", rate))

    @property
    def select(self) -> Select:
        """How parents are picked."""
        return self._select

    @select.setter
    def select(self, select: Select) -> None:
        self._set("select", _describe_setting("select", select, _SELECT))
        self._select = select

    @property
    def crossover(self) -> Crossover:
        """How pairs of parents are combined. It must fit the genome."""
        return self._crossover

    @crossover.setter
    def crossover(self, crossover: Crossover) -> None:
        self._set("crossover", _describe_setting("crossover", crossover, _CROSSOVER))
        self._crossover = crossover

    @property
    def mutation(self) -> Mutation:
        """How children are changed. It must fit the genome."""
        return self._mutation

    @mutation.setter
    def mutation(self, mutation: Mutation) -> None:
        self._set("mutation", _describe_setting("mutation", mutation, _MUTATION))
        self._mutation = mutation


class RunningDe(Running):
    """A running :class:`De`, for ``control``: its ``strategy`` and its ``control`` of F and CR,
    in the forms of :class:`De`'s settings.

    Reading one gives the setting in use, the default included, e.g. ``{"memory": 100}``.
    JADE's and SHADE's adaptation goes on across a change of JADE's ``c``, or to a SHADE control
    with the same memory size; a switch to JADE or SHADE from another control, or to another
    memory size, starts it over. A strategy with a smaller archive drops random members of it.
    With ``l_shade``, the population goes on shrinking over the budget of evaluations.
    """

    __slots__ = ()

    @property
    def strategy(self) -> DeStrategy:
        """How each mutant vector is built: ``"rand1"``, ``"best1"``, ``{"p": ..., "archive":
        ...}`` or ``{"max_p": ..., "archive": ...}``."""
        strategy: DeStrategy = self._get("strategy")
        return strategy

    @strategy.setter
    def strategy(self, strategy: DeStrategy) -> None:
        self._set("strategy", _de_running("strategy", strategy, _DE_STRATEGIES, _DE_STRATEGY))

    @property
    def control(self) -> DeControl:
        """Where F and CR come from: ``{"f": ..., "cr": ...}``, ``{"min_f": ..., "max_f": ...,
        "cr": ...}``, ``{"c": ...}`` or ``{"memory": ...}``."""
        control: DeControl = self._get("control")
        return control

    @control.setter
    def control(self, control: DeControl) -> None:
        self._set("control", _de_running("control", control, _DE_CONTROLS, _DE_CONTROL))


class RunningCmaes(Running):
    """A running :class:`Cmaes`, for ``control``: CMA-ES adapts its own settings, and only
    re-evaluates, keeping its distribution. The convergence criteria that compare values across
    generations start over."""

    __slots__ = ()


class RunningEs(Running):
    """A running :class:`Es`, for ``control``: an evolution strategy adapts its own step sizes,
    and only re-evaluates, keeping its parents' step sizes."""

    __slots__ = ()


class RunningOpenEs(Running):
    """A running :class:`OpenEs`, for ``control``: its sigma and learning rate, e.g. both decayed
    over the run. Sigma applies from the next samples, the learning rate from the next step of the
    mean; the optimizer keeps its memory. Re-evaluation scores the last samples again without
    moving the mean."""

    __slots__ = ()

    @property
    def sigma(self) -> float:
        """The perturbations' standard deviation, a fraction of each gene's range, greater than
        0."""
        return float(self._get("sigma"))

    @sigma.setter
    def sigma(self, sigma: float) -> None:
        self._set("sigma", _number("sigma", sigma))

    @property
    def learning_rate(self) -> float:
        """The optimizer's learning rate, a fraction of each gene's range, greater than 0."""
        return float(self._get("learning_rate"))

    @learning_rate.setter
    def learning_rate(self, learning_rate: float) -> None:
        self._set("learning_rate", _number("learning_rate", learning_rate))


class RunningNeat(Running):
    """A running :class:`Neat`, for ``control``: its species and the innovation numbers it gave,
    to read. NEAT has no setting to change during a run. Re-evaluation scores the population
    again, keeping the species but forgetting their best fitness."""

    __slots__ = ()

    @property
    def species(self) -> tuple[neat.Species, ...]:
        """The species of the current generation, by id: :class:`genoxide.neat.Species`."""
        return tuple(
            neat.Species(
                id=int(species["id"]),
                members=tuple(int(member) for member in species["members"]),
                best_fitness=species["best_fitness"],
                improved=int(species["improved"]),
                created=int(species["created"]),
                representative=_genoxide.neat_network(json.dumps(species["representative"])),
            )
            for species in self._get("species")
        )

    @property
    def innovations(self) -> int:
        """The innovation numbers given so far: the distinct connections the run has made."""
        return int(self._get("innovations"))

    @property
    def seed(self) -> int:
        """The seed of the run: a random one when ``seed`` was None."""
        return int(self._get("seed"))


class RunningPso(Running):
    """A running :class:`Pso`, for ``control``: its inertia and accelerations, e.g. an inertia
    falling from 0.9 to 0.4 over the run (Shi and Eberhart, 1998), or accelerations from a large
    cognitive and a small social one to the reverse (Ratnaweera, Halgamuge and Watson, 2004).
    They apply from the next move of the swarm."""

    __slots__ = ()

    @property
    def inertia(self) -> float:
        """The share of its velocity that a particle keeps, 0 or more: 0.7298 by default."""
        return float(self._get("inertia"))

    @inertia.setter
    def inertia(self, inertia: float) -> None:
        self._set("inertia", _number("inertia", inertia))

    @property
    def acceleration(self) -> tuple[float, float]:
        """``(cognitive, social)``: how strongly a particle is pulled toward its personal best
        and toward its neighborhood's best, each 0 or more: 1.49618 each by default."""
        cognitive, social = self._get("acceleration")
        return float(cognitive), float(social)

    @acceleration.setter
    def acceleration(self, acceleration: tuple[float, float]) -> None:
        try:
            cognitive, social = acceleration
        except (TypeError, ValueError):
            raise ValueError(
                f"acceleration is a pair (cognitive, social), not {acceleration!r}"
            ) from None
        pair = [_number("acceleration", cognitive), _number("acceleration", social)]
        self._set("acceleration", pair)


class RunningLocalSearch(Running):
    """A running :class:`LocalSearch`, for ``control``: its neighbor operator and the neighbors
    per step, e.g. smaller moves as the search settles. They apply from the next step, to the
    kicks of a restart too."""

    __slots__ = ("_neighbor",)

    def __init__(self, native: Any, algorithm: _Single) -> None:
        super().__init__(native, algorithm)
        assert isinstance(algorithm, LocalSearch)
        self._neighbor = algorithm.neighbor

    @property
    def neighbor(self) -> Mutation:
        """Makes a neighbor from the current solution. It must fit the genome."""
        return self._neighbor

    @neighbor.setter
    def neighbor(self, neighbor: Mutation) -> None:
        self._set("neighbor", _describe_setting("neighbor", neighbor, _NEIGHBOR))
        self._neighbor = neighbor

    @property
    def neighbors(self) -> int:
        """The neighbors evaluated per step, 1 to 2^24."""
        return int(self._get("neighbors"))

    @neighbors.setter
    def neighbors(self, neighbors: int) -> None:
        self._set("neighbors", _whole("neighbors", neighbors))


class RunningNelderMead(Running):
    """A running :class:`NelderMead`, for ``control``: the state of its simplex, read-only. The
    method's steps follow from its simplex, so it has no settings to change; ``reevaluate()``
    scores the simplex again, and the next iteration starts from it, reordered."""

    __slots__ = ()

    @property
    def converged(self) -> bool:
        """Whether the current run has converged: its simplex is within the tolerance. With
        restarts left, the next generation starts a new run from a random point."""
        return bool(self._get("converged"))

    @property
    def size(self) -> float:
        """The size of the simplex: the largest difference between a vertex and the best vertex
        in any searched gene, as a fraction of the gene's range. The run has converged once it's
        within the tolerance."""
        return float(self._get("size"))

    @property
    def iterations(self) -> int:
        """The completed iterations, of every run: those that replaced the worst vertex, and the
        shrinks. An iteration takes one to three generations."""
        return int(self._get("iterations"))

    @property
    def restart_count(self) -> int:
        """The restarts so far."""
        return int(self._get("restart_count"))


class _Island:
    """The native handle of one island of a running :class:`Islands`: its settings, named
    ``index/setting`` for the islands' handle."""

    __slots__ = ("_native", "_index")

    def __init__(self, native: Any, index: int) -> None:
        self._native = native
        self._index = index

    def get(self, name: str) -> str:
        result: str = self._native.get(f"{self._index}/{name}")
        return result

    def set(self, name: str, value: str) -> None:
        self._native.set(f"{self._index}/{name}", value)

    def reevaluate(self) -> None:
        raise ValueError(
            "the islands are re-evaluated together: call reevaluate() on the RunningIslands"
        )


class RunningIslands(Running):
    """A running :class:`Islands`, for ``control``: ``islands`` has a handle per island, a
    :class:`RunningGa` or :class:`RunningDe`, to change that island's settings, e.g. a mutation
    step per island. ``reevaluate()`` re-evaluates every island; an island's handle can't
    re-evaluate it alone."""

    __slots__ = ("_islands",)

    def __init__(self, native: Any, algorithm: _Single) -> None:
        super().__init__(native, algorithm)
        assert isinstance(algorithm, Islands)
        self._islands = tuple(
            island._running(_Island(native, index), island)
            for index, island in enumerate(algorithm.islands)
        )

    @property
    def islands(self) -> tuple[Running, ...]:
        """A handle per island, in the order of :class:`Islands`'s ``islands``."""
        return self._islands


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
    if seconds == _math.inf:
        return None
    if not seconds >= 0:
        raise ValueError(f"time is a number of seconds, at least 0, not {time!r}")
    return seconds


def _check_callable(function: Any, name: str = "the fitness function") -> None:
    if not callable(function):
        raise TypeError(f"{name} isn't callable: {function!r}")


def _on_generation(
    callback: Callable[[Any], Any] | None,
    progress: type[Progress] | type[MultiProgress] | type[NeatProgress],
) -> Callable[..., bool] | None:
    """The callback, called with the generation, the evaluations, the seconds, the best fitness
    or the size of the front, and the run's copies of the population (and of the front), as a
    ``progress``; False from it stops the run."""
    if callback is None:
        return None
    _check_callable(callback, "on_generation")

    def call(*state: Any) -> bool:
        go_on = callback(progress(*state))
        return go_on is not False and go_on is not np.False_

    return call


def _control(
    control: Callable[[Any, Any], Any] | None,
    algorithm: _Single,
    progress: type[Progress] | type[NeatProgress] = Progress,
) -> Callable[..., None] | None:
    """The control callback, called with the native handle of the running algorithm and the
    arguments of ``on_generation``, as ``control(running, progress)``, with the same
    :class:`Running` every generation."""
    if control is None:
        return None
    _check_callable(control, "control")
    running: Running | None = None

    def call(native: Any, *state: Any) -> None:
        nonlocal running
        if running is None:
            running = algorithm._running(native, algorithm)
        control(running, progress(*state))

    return call


def _path(name: str, path: Any) -> str | None:
    """A file's path, from a ``str`` or an ``os.PathLike``, or None."""
    if path is None:
        return None
    try:
        path = os.fspath(path)
    except TypeError:
        raise ValueError(f"{name} is a path, a str or a pathlib.Path, not {path!r}") from None
    if not isinstance(path, str):
        raise ValueError(f"{name} is a path, a str or a pathlib.Path, not {path!r}")
    return path


def _checkpoints(checkpoint: Any, checkpoint_every: Any, resume: Any) -> dict[str, Any]:
    """The checkpoint settings of a run, for the native ``run``."""
    path = _path("checkpoint", checkpoint)
    if (path is None) != (checkpoint_every is None):
        raise ValueError("checkpoint and checkpoint_every go together")
    every = None if checkpoint_every is None else _whole(
        "checkpoint_every", checkpoint_every, minimum=1
    )
    return {"checkpoint": path, "checkpoint_every": every, "resume": _path("resume", resume)}


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


def _batch_objectives(
    function: Callable[[np.ndarray], Any], objectives: int
) -> Callable[[np.ndarray], Any]:
    """A batch function returning float64 arrays: a row of objective values per genome, and
    constraint violations or None."""

    def evaluate(genomes: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        result = function(genomes)
        # (objectives, violations): the objective values, in rows or columns, and an array
        if isinstance(result, tuple) and len(result) == 2 and np.ndim(result[0]) == 2:
            values, violations = result
            return (
                _objective_rows(values, len(genomes), objectives),
                np.asarray(violations, dtype=np.float64).reshape(-1),
            )
        return _objective_rows(result, len(genomes), objectives), None

    return evaluate


def _objective_rows(values: Any, genomes: int, objectives: int) -> np.ndarray:
    """The objective values of a batch, a row per genome: from a 2-D array or a list of rows,
    or from a tuple of one 1-D array per objective (``return f1, f2``), its columns."""
    if isinstance(values, tuple) and len(values) == objectives and all(
        np.ndim(column) == 1 for column in values
    ):
        return np.column_stack([np.asarray(column, dtype=np.float64) for column in values])
    rows = np.asarray(values, dtype=np.float64)
    if rows.ndim != 2:
        raise ValueError(
            "a multi-objective batch fitness function returns a 2-D array, a row of objective "
            f"values per genome, not an array of shape {rows.shape}"
        )
    if rows.shape == (objectives, genomes) and genomes != objectives:
        raise ValueError(
            f"the batch fitness function returned {objectives} rows of {genomes} values: return a "
            "row of objective values per genome, e.g. np.column_stack([f1, f2]), or a tuple of "
            "one array per objective, (f1, f2)"
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
        on_generation: Callable[..., bool] | None,
        problem: str | None = None,
        control: Callable[..., None] | None = None,
        checkpoints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        run = {
            "genome": _describe_setting("genome", self._genome, _GENOME),
            "algorithm": self._describe(),
            "objectives": self._objectives(),
            "stop": stop,
        }
        # NaN and infinity aren't JSON: the settings are finite, or an error names them
        description = json.dumps(run, default=_json_number, allow_nan=False)
        return _genoxide.run(
            description,
            fitness,
            bool(batch),
            bool(parallel),
            on_generation,
            problem,
            control,
            **(checkpoints or {}),
        )


class _Single(_Algorithm):
    """A single-objective algorithm, without its `run`: the genome it gives a fitness function
    decides what `run` takes and returns."""

    _objective: ObjectiveName
    # the handle that its control gets
    _running: type[Running]

    def _objectives(self) -> list[str]:
        if self._objective not in ("maximize", "minimize"):
            raise ValueError(f'objective is "maximize" or "minimize", not {self._objective!r}')
        return [self._objective]


class _SingleObjective(_Single):
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
        control: Callable[[Any, Progress], Any] | None = None,
        checkpoint: str | os.PathLike[str] | None = None,
        checkpoint_every: int | None = None,
        resume: str | os.PathLike[str] | None = None,
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
            Python call: ``batch`` doesn't apply, the genome must be the problem's
            (``problem.genome``: a :class:`Real`, or an :class:`Integer` for
            :class:`~genoxide.problems.engineering.GearTrain`), and the objective "minimize". So
            is a :class:`~genoxide.problems.control.Balance`, with a :class:`Real` genome of a
            gene per weight of its network, and the objective "maximize". For a
            :class:`genoxide.gp.Gp` genome, the function takes a :class:`genoxide.gp.Tree`; a
            :class:`genoxide.gp.regression.Regression`, a regression problem or a Boolean problem
            of :mod:`genoxide.gp` is evaluated in Rust, with the Gp built from its primitives and
            the objective "minimize".
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
            function releases the GIL (numpy on large arrays, I/O), or on free-threaded Python
            (3.14t), and for a problem of :mod:`genoxide.problems`, which runs without the GIL.
        on_generation : callable, optional
            Called with a :class:`Progress` after every generation, the initial population
            (generation 0) included, on the thread that called ``run``. If it returns False, the
            run stops with the stop reason "aborted".
        control : callable, optional
            Called as ``control(algorithm, progress)`` once per generation, after
            ``on_generation``, on the same thread, with the running algorithm (a
            :class:`RunningGa`, :class:`RunningDe`, :class:`RunningEs`, :class:`RunningCmaes`,
            :class:`RunningOpenEs`, :class:`RunningPso`, :class:`RunningLocalSearch`,
            :class:`RunningNelderMead` or :class:`RunningIslands`) and a
            :class:`Progress`: to change the algorithm's
            settings for the next generation, or to re-evaluate it after the fitness function
            changed. See :class:`Running`.
        checkpoint : str or os.PathLike, optional
            Saves the run to this file every ``checkpoint_every`` generations and when it stops,
            to resume it later with ``resume``. The file is replaced atomically: a crash while
            saving keeps the previous checkpoint. None saves nothing.
        checkpoint_every : int, optional
            The generations between checkpoints, at least 1, with ``checkpoint``. A checkpoint
            is saved when the generation is a multiple of it, so a resumed run keeps the
            schedule.
        resume : str or os.PathLike, optional
            Continues the run saved in this checkpoint, with the results it would have had
            without the interruption. The algorithm, its genome and the objectives must be the
            ones that saved it; the fitness function, the stop conditions, ``batch``,
            ``parallel``, the callbacks and ``checkpoint`` can change, e.g. to run longer.
            ``generations`` and ``evaluations`` count from the start of the first run, ``time``
            from the start of this one. The file must come from the same version of genoxide.
            Load only checkpoints you trust, like the program that saved them: the checksum
            detects accidental damage, not tampering, and a crafted checkpoint can make a run
            loop or fail, though never break memory safety.

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
            setting); for a problem whose genome isn't a :class:`Real` of its dimensions, or a
            multi-objective problem; for a checkpoint to resume from that isn't one, is damaged,
            comes from another version of genoxide or was saved with other settings; and for
            a wrong fitness result: a negative constraint violation, or a batch result with a
            length other than the number of genomes.
        OSError
            If the checkpoint to resume from can't be read, e.g. a ``FileNotFoundError``, or a
            checkpoint can't be saved.
        TypeError
            If ``fitness``, ``on_generation`` or ``control`` isn't callable, or ``fitness``
            returns something that isn't a number. Another error converting a result, e.g. an
            ``OverflowError`` for an int too large for a float, is raised as it is.
        Exception
            An exception raised by ``fitness``, ``on_generation`` or ``control`` stops the run,
            and ``run`` raises it, e.g. the ``ValueError`` of a wrong setting in ``control``. So
            does ``KeyboardInterrupt`` on Ctrl+C: with ``parallel``, once the calls of
            ``fitness`` under way return.
        """
        _check_callable(fitness)
        if isinstance(fitness, problems.MultiProblem):
            raise ValueError(
                f"{type(fitness).__name__} has {len(fitness.objectives)} objectives: use a "
                "multi-objective algorithm"
            )
        stop = _stop(generations, evaluations, target, time, stagnation)
        callback = _on_generation(on_generation, Progress)
        controls = _control(control, self)
        saving = _checkpoints(checkpoint, checkpoint_every, resume)
        if gp._is_tree_fitness(fitness):
            # a fitness of trees, evaluated in Rust
            return Result(
                **self._run(fitness, stop, False, parallel, callback, None, controls, saving)
            )
        if isinstance(fitness, (problems.Problem, problems.control.Balance)):
            description = fitness._json()
            return Result(
                **self._run(
                    fitness, stop, False, parallel, callback, description, controls, saving
                )
            )
        function = _batch_scores(fitness) if batch else fitness
        return Result(
            **self._run(function, stop, batch, parallel, callback, None, controls, saving)
        )


class Ga(_SingleObjective):
    """A genetic algorithm. Any genome.

    Each generation, ``select`` picks parents, ``crossover`` combines pairs of them with
    probability ``crossover_rate`` and ``mutation`` changes each child with probability
    ``mutation_rate``. The ``scheme`` decides who survives: by default the children replace the
    population, except its best individual.

    Parameters
    ----------
    genome : Binary, Integer, Real, Permutation, AdaptiveReal or gp.Gp
        The search space.
    population_size : int
        The number of individuals, 1 to 2^24.
    select : a selection, e.g. Tournament, Rank or DoubleTournament
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
    parallel_breeding : bool, default False
        Whether each pair of parents is crossed over and mutated on all cores, each pair with
        random numbers of its own: a seed gives other results than without it, but the same on
        any number of cores. It pays off with thousands of children per generation and a fast
        or batch fitness function.
    objective : {"maximize", "minimize"}, default "maximize"
        Whether higher or lower scores are better.
    seed : int, optional
        The seed of the random numbers, 0 to 2^64 - 1. None is a random seed. The same seed
        repeats the run.
    initial_genomes : sequence of gp.Tree, optional
        For a :class:`genoxide.gp.Gp` genome, trees of its set to start from, at most
        ``population_size``, e.g. ``gp.ramped_half_and_half(population_size, seed)``; random
        trees fill the rest.
    """

    _running = RunningGa

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
        parallel_breeding: bool | None = None,
        objective: ObjectiveName = "maximize",
        seed: int | None = None,
        initial_genomes: Sequence[gp.Tree] | None = None,
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
        self.parallel_breeding = parallel_breeding
        self.seed = seed
        self.initial_genomes = initial_genomes

    def _describe(self) -> dict[str, Any]:
        return {
            "type": "ga",
            "population_size": _whole("population_size", self.population_size),
            "seed": _optional_whole("seed", self.seed),
            "select": _describe_setting("select", self.select, _SELECT),
            "crossover": _describe_setting("crossover", self.crossover, _CROSSOVER),
            "mutate": _describe_setting("mutation", self.mutation, _MUTATION),
            "crossover_rate": _optional_number("crossover_rate", self.crossover_rate),
            "mutation_rate": _optional_number("mutation_rate", self.mutation_rate),
            "scheme": (
                None if self.scheme is None else _describe_setting("scheme", self.scheme, _SCHEME)
            ),
            "parallel_breeding": _flag("parallel_breeding", self.parallel_breeding),
            "initial_genomes": gp._initial_genomes(self._genome, self.initial_genomes),
        }


class De(_SingleObjective):
    """Differential evolution. Real genomes.

    By default SHADE's published settings (Tanabe and Fukunaga, "Success-History Based
    Parameter Adaptation for Differential Evolution", IEEE CEC 2013): DE/current-to-pbest/1 with
    a random p per trial between 2 / population and 0.2 and an archive of the population's size,
    SHADE's adaptation of F and CR with a memory of 100, and a population of 100. genoxide adds
    restarts, which aren't part of SHADE: every individual but the best is replaced when the
    population has converged, each gene's values within 1e-12 of its range of each other and the
    scores within 1e-12 of the best, or when the best doesn't improve for 200 generations.

    With ``l_shade``, L-SHADE (Tanabe and Fukunaga, 2014) for a budget of that many evaluations:
    current-to-pbest/1 with p 0.11 and an archive of 2.6 times the population, SHADE's
    adaptation with a memory of 6, no restarts, and a population that shrinks linearly from its
    initial size to 4 over the budget. Stop the run at the same number of evaluations.
    ``strategy``, ``control`` and ``restarts`` replace L-SHADE's.

    Parameters
    ----------
    genome : Real
        The search space.
    population_size : int, optional
        The number of individuals, 4 to 2^24. 100 by default. With ``l_shade``, the initial size,
        18 times the number of genes (at least 4) by default.
    l_shade : int, optional
        L-SHADE's budget of evaluations, at least 1. None is SHADE.
    strategy : str or dict, optional
        How each mutant vector is built, for the individual x, from random other individuals r1,
        r2 and r3 and the scale factor F:

        - ``"rand1"``: ``r1 + F (r2 - r3)``, robust, explores well;
        - ``"best1"``: ``best + F (r1 - r2)``, greedy: use it with a dither ``control``, or the
          population can collapse onto one point;
        - ``{"p": 0.1, "archive": 1.0}``: current-to-pbest/1 (JADE),
          ``x + F (pbest - x) + F (r1 - r2)``, with ``pbest`` one of the best ``p`` fraction of
          the population (0 < p <= 1) and ``r2`` also from an archive of replaced individuals,
          ``archive`` times the population's size (0 or more, 0 for none);
        - ``{"max_p": 0.2, "archive": 1.0}``: the same with a random p for every trial, between
          2 / population and ``max_p`` (0 < max_p <= 1), as in SHADE.

        None is SHADE's ``{"max_p": 0.2, "archive": 1.0}``.
    control : dict, optional
        Where F and CR (the probability that a gene comes from the mutant) come from:

        - ``{"f": 0.5, "cr": 0.9}``: fixed, 0 < f <= 2 and 0 <= cr <= 1;
        - ``{"min_f": 0.5, "max_f": 1.0, "cr": 0.9}``: dither, a random F between ``min_f`` and
          ``max_f`` for every trial (0 < min_f <= max_f <= 2), and a fixed CR;
        - ``{"c": 0.1}``: JADE's adaptation of their means, at the rate ``c`` (0 < c <= 1);
        - ``{"memory": 100}``: SHADE's adaptation, with a memory of that many (F, CR) pairs, 1 to
          2^24.

        None is SHADE's ``{"memory": 100}``.
    restarts : str or dict, optional
        When the population starts over from new random individuals, all but the best:

        - ``"never"``;
        - ``{"tolerance": 1e-12, "patience": 200}``: when the population has converged, each
          gene's values within ``tolerance`` times its range of each other and the scores
          within ``tolerance`` of each other relative to the best (0 or more), or when the best
          hasn't improved for ``patience`` generations (at least 1).

        None is ``{"tolerance": 1e-12, "patience": 200}``, or ``"never"`` with ``l_shade``.
    parallel_breeding : bool, default False
        Whether the trials are built on all cores, each with random numbers of its own: a seed
        gives other results than without it, but the same on any number of cores. It pays off
        with hundreds of individuals or genes and a fast or batch fitness function.
    objective : {"maximize", "minimize"}, default "maximize"
        Whether higher or lower scores are better.
    seed : int, optional
        The seed of the random numbers, 0 to 2^64 - 1. None is a random seed. The same seed
        repeats the run.
    """

    _running = RunningDe

    def __init__(
        self,
        genome: Real,
        *,
        population_size: int | None = None,
        l_shade: int | None = None,
        strategy: DeStrategy | None = None,
        control: DeControl | None = None,
        restarts: DeRestarts | None = None,
        parallel_breeding: bool | None = None,
        objective: ObjectiveName = "maximize",
        seed: int | None = None,
    ) -> None:
        self._genome = genome
        self._objective = objective
        self.population_size = population_size
        self.l_shade = l_shade
        self.strategy = strategy
        self.control = control
        self.restarts = restarts
        self.parallel_breeding = parallel_breeding
        self.seed = seed

    def _describe(self) -> dict[str, Any]:
        return {
            "type": "de",
            "population_size": _optional_whole("population_size", self.population_size),
            "seed": _optional_whole("seed", self.seed),
            "l_shade": _optional_whole("l_shade", self.l_shade),
            "strategy": _de_setting("strategy", self.strategy, _DE_STRATEGIES, _DE_STRATEGY),
            "control": _de_setting("control", self.control, _DE_CONTROLS, _DE_CONTROL),
            "restarts": _de_setting("restarts", self.restarts, _DE_RESTARTS, _DE_RESTART),
            "parallel_breeding": _flag("parallel_breeding", self.parallel_breeding),
        }


# the forms of De's settings: the names without settings, and the keys of each dict with a
# function that reads one value
_Read = Callable[[str, Any], Any]
_DE_STRATEGIES: tuple[tuple[str, ...], tuple[dict[str, _Read], ...]] = (
    ("rand1", "best1"),
    ({"p": _number, "archive": _number}, {"max_p": _number, "archive": _number}),
)
_DE_STRATEGY = (
    '"rand1", "best1", {"p": ..., "archive": ...} or {"max_p": ..., "archive": ...}'
)
_DE_CONTROLS: tuple[tuple[str, ...], tuple[dict[str, _Read], ...]] = (
    (),
    (
        {"f": _number, "cr": _number},
        {"min_f": _number, "max_f": _number, "cr": _number},
        {"c": _number},
        {"memory": _whole},
    ),
)
_DE_CONTROL = (
    '{"f": ..., "cr": ...}, {"min_f": ..., "max_f": ..., "cr": ...}, {"c": ...} or '
    '{"memory": ...}'
)
_DE_RESTARTS: tuple[tuple[str, ...], tuple[dict[str, _Read], ...]] = (
    ("never",),
    ({"tolerance": _number, "patience": _whole},),
)
_DE_RESTART = '"never" or {"tolerance": ..., "patience": ...}'


def _de_setting(
    name: str,
    value: Any,
    forms: tuple[tuple[str, ...], tuple[dict[str, _Read], ...]],
    what: str,
) -> str | dict[str, Any] | None:
    """A setting of De: None for its default, one of the names, or a dict with exactly the keys
    of one of the forms, each value read as ``name.key``; else a ValueError that names it."""
    names, dicts = forms
    if value is None:
        return None
    if isinstance(value, str) and value in names:
        return value
    if isinstance(value, dict):
        for keys in dicts:
            if set(value) == set(keys):
                return {key: read(f"{name}.{key}", value[key]) for key, read in keys.items()}
    raise ValueError(f"{name} is {what}, not {value!r}")


def _de_running(
    name: str,
    value: Any,
    forms: tuple[tuple[str, ...], tuple[dict[str, _Read], ...]],
    what: str,
) -> str | dict[str, Any]:
    """A new setting of a running De: as in its constructor, but not None."""
    setting = None if value is None else _de_setting(name, value, forms, what)
    if setting is None:
        raise ValueError(f"{name} is {what}, not None")
    return setting


class Es(_SingleObjective):
    """A (mu/rho +, lambda) evolution strategy with self-adapted step sizes. Real genomes.

    Each generation makes ``offspring`` (lambda) offspring from the ``parents`` (mu): each
    recombines ``rho`` random parents, then its step sizes change log-normally (Schwefel's
    self-adaptation) and each gene moves by a normal step of its step size, a fraction of the
    gene's range, mirrored at the bounds. Good step sizes survive with the good solutions they
    made, so the search tunes them: for smooth problems that need precise answers. For rotated,
    badly conditioned or multimodal problems, :class:`Cmaes` is stronger.

    Parameters
    ----------
    genome : Real
        The search space. At least one gene needs ``low < high``.
    parents : int
        The parents mu, 1 to 2^24.
    offspring : int
        The offspring per generation lambda, 1 to 2^24, and at least ``parents`` with comma
        selection; e.g. 5 to 7 times ``parents``.
    recombination : {"intermediate", "dominant"}, default "intermediate"
        "intermediate" gives an offspring the mean of its parents' genes and the geometric mean
        of their step sizes; "dominant" gives it each gene, with its step size, from a random
        one of its parents.
    rho : int, optional
        The parents of each offspring, 1 to ``parents``; all of them by default. 1 is no
        recombination: a mutated copy of one random parent.
    selection : {"comma", "plus"}, default "comma"
        "comma" (mu, lambda): the best offspring become the parents, which never survive; it
        forgets misadapted step sizes, and suits self-adaptation best. "plus" (mu + lambda): the
        best of parents and offspring survive, offspring first on ties.
    step_sizes : {"per_gene", "one"}, default "per_gene"
        A step size per gene, which learns the scaling of each gene, or one for all genes.
    initial_step : float, default 0.3
        The initial step size as a fraction of each gene's range, greater than 0 and at most 10.
    parallel_breeding : bool, default False
        Whether the offspring are made on all cores, each with random numbers of its own: a
        seed gives other results than without it, but the same on any number of cores. It pays
        off with many genes or offspring and a fast or batch fitness function.
    objective : {"maximize", "minimize"}, default "maximize"
        Whether higher or lower scores are better.
    seed : int, optional
        The seed of the random numbers, 0 to 2^64 - 1. None is a random seed. The same seed
        repeats the run.
    """

    _running = RunningEs

    def __init__(
        self,
        genome: Real,
        *,
        parents: int,
        offspring: int,
        recombination: Literal["intermediate", "dominant"] | None = None,
        rho: int | None = None,
        selection: Literal["comma", "plus"] | None = None,
        step_sizes: Literal["per_gene", "one"] | None = None,
        initial_step: float | None = None,
        parallel_breeding: bool | None = None,
        objective: ObjectiveName = "maximize",
        seed: int | None = None,
    ) -> None:
        self._genome = genome
        self._objective = objective
        self.parents = parents
        self.offspring = offspring
        self.recombination = recombination
        self.rho = rho
        self.selection = selection
        self.step_sizes = step_sizes
        self.initial_step = initial_step
        self.parallel_breeding = parallel_breeding
        self.seed = seed

    def _describe(self) -> dict[str, Any]:
        if self.recombination not in (None, "intermediate", "dominant"):
            raise ValueError(
                f'recombination is "intermediate" or "dominant", not {self.recombination!r}'
            )
        if self.selection not in (None, "comma", "plus"):
            raise ValueError(f'selection is "comma" or "plus", not {self.selection!r}')
        if self.step_sizes not in (None, "per_gene", "one"):
            raise ValueError(f'step_sizes is "per_gene" or "one", not {self.step_sizes!r}')
        return {
            "type": "es",
            "parents": _whole("parents", self.parents),
            "offspring": _whole("offspring", self.offspring),
            "recombination": self.recombination,
            "rho": _optional_whole("rho", self.rho),
            "selection": self.selection,
            "step_sizes": self.step_sizes,
            "initial_step": _optional_number("initial_step", self.initial_step),
            "parallel_breeding": _flag("parallel_breeding", self.parallel_breeding),
            "seed": _optional_whole("seed", self.seed),
        }


class Cmaes(_SingleObjective):
    """CMA-ES, the covariance matrix adaptation evolution strategy. Real genomes.

    It adapts a full or a diagonal covariance matrix, from a random initial mean. At least one
    gene needs ``low < high``; genes with ``low == high`` are fixed.

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
    covariance : {"full", "diagonal"}, default "full"
        "full" learns the correlations between genes; each sample costs O(n^2) and each
        eigendecomposition O(n^3), for n genes: best up to a few hundred genes. "diagonal" is
        sep-CMA-ES (Ros and Hansen, 2008): only each gene's variance is learned, with larger
        learning rates, and each sample costs O(n). It suits separable problems and hundreds to
        thousands of genes, but can't learn correlations between genes.
    min_step : float, default 0
        A lower bound on the step size, as a fraction of each gene's range, 0 to
        ``initial_step``: the search goes on exploring around its mean instead of converging,
        for a fitness that stops pointing at the goal near its best, such as a control task
        scored on short episodes but solved by long ones (Igel 2003). The convergence on the step
        size, and a restart through it, can't happen with a bound.
    objective : {"maximize", "minimize"}, default "maximize"
        Whether higher or lower scores are better.
    seed : int, optional
        The seed of the random numbers, 0 to 2^64 - 1. None is a random seed. The same seed
        repeats the run.
    """

    _running = RunningCmaes

    def __init__(
        self,
        genome: Real,
        *,
        population_size: int | None = None,
        restarts: Literal["never", "ipop", "bipop"] | None = None,
        initial_step: float | None = None,
        covariance: Literal["full", "diagonal"] | None = None,
        min_step: float | None = None,
        objective: ObjectiveName = "maximize",
        seed: int | None = None,
    ) -> None:
        self._genome = genome
        self._objective = objective
        self.population_size = population_size
        self.restarts = restarts
        self.initial_step = initial_step
        self.covariance = covariance
        self.min_step = min_step
        self.seed = seed

    def _describe(self) -> dict[str, Any]:
        if self.restarts not in (None, "never", "ipop", "bipop"):
            raise ValueError(f'restarts is "never", "ipop" or "bipop", not {self.restarts!r}')
        if self.covariance not in (None, "full", "diagonal"):
            raise ValueError(f'covariance is "full" or "diagonal", not {self.covariance!r}')
        return {
            "type": "cmaes",
            "population_size": _optional_whole("population_size", self.population_size),
            "seed": _optional_whole("seed", self.seed),
            "restarts": self.restarts,
            "initial_step": _optional_number("initial_step", self.initial_step),
            "covariance": self.covariance,
            "min_step": _optional_number("min_step", self.min_step),
        }


@dataclass(frozen=True)
class Adam:
    """Adam (Kingma and Ba, 2015), the optimizer of :class:`OpenEs` by default: steps of about
    ``learning_rate`` per gene whatever the scale of the gradient, from averages of the gradient
    and of its square, corrected for their start at 0.

    ``learning_rate`` is a fraction of each gene's range, greater than 0; ``beta1`` and ``beta2``
    the decays of the averages, in [0, 1): Kingma and Ba's 0.9 and 0.999 by default.
    """

    learning_rate: float
    beta1: float = 0.9
    beta2: float = 0.999

    def _describe(self) -> dict[str, Any]:
        return {
            "type": "adam",
            "learning_rate": _number("Adam.learning_rate", self.learning_rate),
            "beta1": _number("Adam.beta1", self.beta1),
            "beta2": _number("Adam.beta2", self.beta2),
        }


@dataclass(frozen=True)
class Sgd:
    """Gradient ascent with momentum, an optimizer of :class:`OpenEs`: the velocity
    ``v = momentum v + (1 - momentum) g``, then the mean moves by ``learning_rate v``.

    ``learning_rate`` is a fraction of each gene's range, greater than 0; ``momentum`` how much
    of the last direction is kept, in [0, 1): 0.9 is common, 0 for none.
    """

    learning_rate: float
    momentum: float

    def _describe(self) -> dict[str, Any]:
        return {
            "type": "sgd",
            "learning_rate": _number("Sgd.learning_rate", self.learning_rate),
            "momentum": _number("Sgd.momentum", self.momentum),
        }


Optimizer = Union[Adam, Sgd]
_OPTIMIZER = "gx.Adam(learning_rate) or gx.Sgd(learning_rate, momentum)"


class OpenEs(_SingleObjective):
    """OpenAI's evolution strategy (Salimans, Ho, Chen, Sidor and Sutskever, 2017). Real genomes.

    The baseline of neuroevolution at scale: a mean moves along an estimate of the gradient of the
    expected fitness of the samples ``mean + sigma eps``, ``eps`` standard normal, gene by gene in
    units of each gene's range. Each generation draws ``population_size / 2`` perturbations and
    asks for ``mean + sigma eps`` and ``mean - sigma eps`` (mirrored sampling), clamped to the
    bounds; replaces their scores by their ranks, spread evenly over [-0.5, 0.5]; and lets the
    ``optimizer`` move the mean along the rank-weighted sum of the perturbations. Its cost per
    sample is linear in the number of genes, without the covariance matrix of :class:`Cmaes`, so it
    scales to tens of thousands of genes, such as a network's weights (:mod:`genoxide.nn`). The
    population is the last samples, followed by the mean if it is evaluated.

    Parameters
    ----------
    genome : Real
        The search space.
    population_size : int
        The samples per generation, even (mirrored pairs), 2 to 2^24: tens to thousands.
    sigma : float, default 0.02
        The perturbations' standard deviation, a fraction of each gene's range, greater than 0.
    optimizer : Adam or Sgd, default Adam(0.01)
        How the mean follows the gradient estimate. Adam steps about its learning rate per gene
        even near the optimum: lower it for precise answers.
    weight_decay : float, default 0
        The pull of the mean toward 0 (L2 regularization, as Salimans et al. used on networks'
        weights): the gradient loses ``weight_decay`` times the mean, in units of each gene's
        range. 0 or more.
    evaluate_mean : bool, default False
        Whether the mean is evaluated each generation too, after the samples: one more evaluation
        per generation, counted, and a candidate for the best solution. On many problems it is
        better than all the samples.
    initial_mean : array_like, optional
        The initial mean, a gene per gene of ``genome``, within its bounds. A random point within
        the bounds by default; small weights, e.g. from ``Real((-0.25, 0.25),
        length=n).random_genome(seed)``, suit networks.
    parallel_breeding : bool, default False
        Whether the samples are drawn on all cores, each pair with random numbers of its own: a
        seed gives other results than without it, but the same on any number of cores. It pays
        off with thousands of genes and a fast fitness function.
    objective : {"maximize", "minimize"}, default "maximize"
        Whether higher or lower scores are better.
    seed : int, optional
        The seed of the random numbers, 0 to 2^64 - 1. None is a random seed. The same seed
        repeats the run.
    """

    _running = RunningOpenEs

    def __init__(
        self,
        genome: Real,
        *,
        population_size: int,
        sigma: float | None = None,
        optimizer: Optimizer | None = None,
        weight_decay: float | None = None,
        evaluate_mean: bool | None = None,
        initial_mean: Sequence[float] | np.ndarray | None = None,
        parallel_breeding: bool | None = None,
        objective: ObjectiveName = "maximize",
        seed: int | None = None,
    ) -> None:
        self._genome = genome
        self._objective = objective
        self.population_size = population_size
        self.sigma = sigma
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.evaluate_mean = evaluate_mean
        self.initial_mean = initial_mean
        self.parallel_breeding = parallel_breeding
        self.seed = seed

    def _describe(self) -> dict[str, Any]:
        optimizer = None
        if self.optimizer is not None:
            if not isinstance(self.optimizer, (Adam, Sgd)):
                raise ValueError(f"optimizer is {_OPTIMIZER}, not {self.optimizer!r}")
            optimizer = self.optimizer._describe()
        initial_mean = None
        if self.initial_mean is not None:
            mean = np.asarray(self.initial_mean, dtype=object)
            if mean.ndim != 1:
                raise ValueError(
                    f"initial_mean is a 1-D array, a value per gene, not of shape {mean.shape}"
                )
            initial_mean = [_number("initial_mean", gene, plural=True) for gene in mean]
        return {
            "type": "open_es",
            "population_size": _whole("population_size", self.population_size),
            "sigma": _optional_number("sigma", self.sigma),
            "optimizer": optimizer,
            "weight_decay": _optional_number("weight_decay", self.weight_decay),
            "evaluate_mean": _flag("evaluate_mean", self.evaluate_mean),
            "initial_mean": initial_mean,
            "parallel_breeding": _flag("parallel_breeding", self.parallel_breeding),
            "seed": _optional_whole("seed", self.seed),
        }


@dataclass(frozen=True)
class _Networks:
    """NEAT's genome: networks of ``inputs`` inputs, a bias and ``outputs`` outputs."""

    inputs: Any
    outputs: Any

    def _describe(self) -> dict[str, Any]:
        return {
            "type": "network",
            "inputs": _whole("inputs", self.inputs, minimum=1, maximum=2**24),
            "outputs": _whole("outputs", self.outputs, minimum=1, maximum=2**24),
        }


def _numbers(name: str, value: Any, form: str, wholes: Sequence[bool]) -> list[Any] | None:
    """A setting of several numbers, e.g. ``compatibility=(1.0, 1.0, 0.4, 3.0)``, as a list:
    whole numbers where ``wholes`` says so, real ones elsewhere; None for its default."""
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, (Sequence, np.ndarray)):
        raise ValueError(f"{name} is {form}, not {value!r}")
    if len(value) != len(wholes):
        raise ValueError(f"{name} is {form}, {len(wholes)} numbers, not {value!r}")
    return [
        _whole(name, item, plural=True) if whole else _number(name, item, plural=True)
        for item, whole in zip(value, wholes)
    ]


_NEAT_ACTIVATIONS = ("identity", "tanh", "sigmoid", "relu", "steep_sigmoid")


class Neat(_Single):
    """NEAT, NeuroEvolution of Augmenting Topologies (Stanley and Miikkulainen, 2002): networks
    whose structure evolves with their weights, from minimal networks up.

    Each genome is a :class:`genoxide.neat.Network`: node genes and connection genes, each
    connection with an innovation number that records its history, so crossover aligns the genes
    of networks of different shapes. Mutations perturb weights, add connections, and add nodes
    that split a connection. Speciation groups similar networks by the compatibility distance
    ``c1 E / N + c2 D / N + c3 W`` (excess and disjoint genes, mean weight difference), and
    explicit fitness sharing gives each species offspring in proportion to its members' mean
    fitness, which protects new structure while its weights are tuned. The settings default to
    the paper's.

    The fitness function takes a network and returns a score, as for the other algorithms;
    evaluate it with ``network.feed_forward()`` (or ``network.recurrent()`` with
    ``feed_forward=False``), whose ``activate`` runs in Rust. Both evaluators, and their
    ``policy(scale=2, offset=-1)`` (NEAT's sigmoid outputs, in (0, 1), as forces in (-1, 1)),
    drive the tasks of :mod:`genoxide.problems.control` in Rust, without the GIL::

        import genoxide as gx

        CASES = [((0.0, 0.0), 0.0), ((0.0, 1.0), 1.0), ((1.0, 0.0), 1.0), ((1.0, 1.0), 0.0)]

        def xor(network):
            evaluator = network.feed_forward()
            error = sum(abs(evaluator.activate(x)[0] - y) for x, y in CASES)
            return (4.0 - error) ** 2

        neat = gx.Neat(2, 1, sharing="raw", seed=1)
        result = neat.run(xor, target=15.0, generations=500)
        print(result.best_genome.hidden(), result.best_fitness)

    Parameters
    ----------
    inputs, outputs : int
        The networks' inputs (a bias input is added) and outputs, each 1 to 2^24.
    population_size : int, default 150
        The number of networks, 1 to 2^24.
    compatibility : (float, float, float, float), default (1.0, 1.0, 0.4, 3.0)
        The compatibility coefficients ``c1`` (excess genes), ``c2`` (disjoint genes) and ``c3``
        (the mean weight difference), each at least 0, and the threshold, greater than 0, of the
        distance under which a network joins a species. The paper used ``c3 = 3`` and a threshold
        of 4 with a population of 1000.
    weight_mutation : (float, float), default (0.8, 0.1)
        The probability that a child's weights are mutated, and that each of them is then
        replaced by a new random weight rather than perturbed, both in [0, 1].
    weight_deviations : (float, float), default (1.0, 1.0)
        The standard deviations of a weight's normal perturbation and of a new weight, both
        greater than 0.
    structural_mutation : (float, float), default (0.03, 0.05)
        The probabilities that a child gets a new node and a new connection, in [0, 1]. The paper
        used 0.3 for new connections in its population of 1000.
    reproduction : (float, float, float), default (0.25, 0.001, 0.75)
        The fraction of offspring made by mutation alone, the probability that a crossover takes
        its second parent from another species, and that a gene disabled in either parent is
        disabled in the child, each in [0, 1].
    selection : (int, float), default (5, 0.2)
        The size a species must exceed for its champion to be copied unchanged, and the fraction
        of each species, its best, that breeds, in (0, 1].
    stagnation : int, default 15
        The generations without improvement after which a species stops reproducing, at least 1.
        The species with the best network never stops.
    activation : str, default "steep_sigmoid"
        The activation of the outputs and the hidden nodes, as in :mod:`genoxide.nn`: "identity",
        "tanh", "sigmoid", "relu" or "steep_sigmoid", the paper's ``1 / (1 + exp(-4.9 x))``.
    feed_forward : bool, default True
        Whether the networks stay feed-forward: a new connection never closes a cycle, so every
        network has a ``feed_forward()`` evaluator. False allows recurrent connections and loops,
        for ``recurrent()``.
    initial : {"fully_connected", "unconnected"}, default "fully_connected"
        The initial networks: every input and the bias connected to every output with random
        weights, as in the paper, or no connections.
    sharing : {"normalized", "raw"}, default "normalized"
        How fitness is shared within species. "normalized": each network's score normalized in
        its generation, 0 for the worst and 1 for the best in the objective's direction (0 for
        an invalid or infeasible one), then divided by its species' size, for any objective and
        any scores. "raw": the paper's, the score divided by the species' size, which needs the
        objective "maximize" and valid, non-negative scores.
    objective : {"maximize", "minimize"}, default "maximize"
        Whether higher or lower scores are better.
    seed : int, optional
        The seed of the random numbers, 0 to 2^64 - 1. None is a random seed. The same seed
        repeats the run.
    """

    _running = RunningNeat

    def __init__(
        self,
        inputs: int,
        outputs: int,
        *,
        population_size: int | None = None,
        compatibility: tuple[float, float, float, float] | None = None,
        weight_mutation: tuple[float, float] | None = None,
        weight_deviations: tuple[float, float] | None = None,
        structural_mutation: tuple[float, float] | None = None,
        reproduction: tuple[float, float, float] | None = None,
        selection: tuple[int, float] | None = None,
        stagnation: int | None = None,
        activation: nn.Activation | None = None,
        feed_forward: bool | None = None,
        initial: Literal["fully_connected", "unconnected"] | None = None,
        sharing: Literal["normalized", "raw"] | None = None,
        objective: ObjectiveName = "maximize",
        seed: int | None = None,
    ) -> None:
        self.inputs = inputs
        self.outputs = outputs
        self._objective = objective
        self.population_size = population_size
        self.compatibility = compatibility
        self.weight_mutation = weight_mutation
        self.weight_deviations = weight_deviations
        self.structural_mutation = structural_mutation
        self.reproduction = reproduction
        self.selection = selection
        self.stagnation = stagnation
        self.activation = activation
        self.feed_forward = feed_forward
        self.initial = initial
        self.sharing = sharing
        self.seed = seed

    @property
    def _genome(self) -> _Networks:  # type: ignore[override]
        return _Networks(self.inputs, self.outputs)

    def _describe(self) -> dict[str, Any]:
        if self.activation is not None and self.activation not in _NEAT_ACTIVATIONS:
            names = ", ".join(f'"{name}"' for name in _NEAT_ACTIVATIONS)
            raise ValueError(f"activation is one of {names}, not {self.activation!r}")
        if self.initial not in (None, "fully_connected", "unconnected"):
            raise ValueError(
                f'initial is "fully_connected" or "unconnected", not {self.initial!r}'
            )
        if self.sharing not in (None, "normalized", "raw"):
            raise ValueError(f'sharing is "normalized" or "raw", not {self.sharing!r}')
        return {
            "type": "neat",
            "population_size": _optional_whole("population_size", self.population_size),
            "compatibility": _numbers(
                "compatibility",
                self.compatibility,
                "(c1, c2, c3, threshold)",
                [False, False, False, False],
            ),
            "weight_mutation": _numbers(
                "weight_mutation", self.weight_mutation, "(rate, replace)", [False, False]
            ),
            "weight_deviations": _numbers(
                "weight_deviations", self.weight_deviations, "(perturbation, new)", [False, False]
            ),
            "structural_mutation": _numbers(
                "structural_mutation",
                self.structural_mutation,
                "(add_node, add_connection)",
                [False, False],
            ),
            "reproduction": _numbers(
                "reproduction",
                self.reproduction,
                "(mutation_only, interspecies, disable)",
                [False, False, False],
            ),
            "selection": _numbers(
                "selection", self.selection, "(elitism_size, survival)", [True, False]
            ),
            "stagnation": _optional_whole("stagnation", self.stagnation),
            "activation": self.activation,
            "feed_forward": _flag("feed_forward", self.feed_forward),
            "initial": self.initial,
            "sharing": self.sharing,
            "seed": _optional_whole("seed", self.seed),
        }

    def run(
        self,
        fitness: Callable[[neat.Network], Any],
        *,
        generations: int | None = None,
        evaluations: int | None = None,
        target: float | None = None,
        time: float | None = None,
        stagnation: int | None = None,
        batch: bool = False,
        parallel: bool = False,
        on_generation: Callable[[NeatProgress], bool | None] | None = None,
        control: Callable[[RunningNeat, NeatProgress], Any] | None = None,
        checkpoint: str | os.PathLike[str] | None = None,
        checkpoint_every: int | None = None,
        resume: str | os.PathLike[str] | None = None,
    ) -> NeatResult:
        """Runs NEAT until the first stop condition, as :meth:`Ga.run` runs a GA.

        ``fitness`` takes a :class:`genoxide.neat.Network` and returns a number, None or NaN (an
        invalid solution), or a tuple ``(score, constraint_violation)``; with ``batch=True``, it
        takes a tuple of the generation's networks and returns an array of scores. The test
        problems of :mod:`genoxide.problems` and ``Balance`` take weights, not networks: write
        the fitness with ``network.feed_forward()`` and a task's ``run``, which runs in Rust.

        ``on_generation`` gets a :class:`NeatProgress`, and ``control`` a :class:`RunningNeat`
        (the species) and a :class:`NeatProgress`. The other parameters, the errors and the
        checkpoints are :meth:`Ga.run`'s.

        Returns
        -------
        NeatResult
            The best network found, and what the run took.
        """
        _check_callable(fitness)
        if isinstance(fitness, (problems.Problem, problems.MultiProblem, problems.control.Balance)):
            raise ValueError(
                f"Neat's fitness function takes a gx.neat.Network, but {type(fitness).__name__} "
                "evaluates an array of numbers: evaluate the network with feed_forward() instead"
            )
        stop = _stop(generations, evaluations, target, time, stagnation)
        callback = _on_generation(on_generation, NeatProgress)
        controls = _control(control, self, NeatProgress)
        saving = _checkpoints(checkpoint, checkpoint_every, resume)
        function = _batch_scores(cast(Callable[[np.ndarray], Any], fitness)) if batch else fitness
        return NeatResult(
            **self._run(
                cast(Callable[[np.ndarray], Any], function),
                stop,
                batch,
                parallel,
                callback,
                None,
                controls,
                saving,
            )
        )


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

    _running = RunningPso

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

    _running = RunningLocalSearch

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
            "neighbor": _describe_setting("neighbor", self.neighbor, _NEIGHBOR),
            "neighbors": _optional_whole("neighbors", self.neighbors),
            "acceptance": (
                None
                if self.acceptance is None
                else _describe_setting("acceptance", self.acceptance, _ACCEPTANCE)
            ),
            "restart": restart,
        }


class NelderMead(_SingleObjective):
    """The Nelder-Mead simplex method. Real genomes.

    A local method without derivatives, for low dimensions (up to about 10 genes, more with the
    adaptive coefficients) and for functions that are non-smooth or noisy in their last digits. It
    keeps a simplex of ``n + 1`` points for the ``n`` genes with ``low < high`` (the others stay
    fixed). Each iteration replaces the worst point by its reflection through the centroid of the
    others, by an expansion further out, or by a contraction nearer; when none of them is better,
    the simplex shrinks towards its best point. The steps, their conditions and the order of tied
    points are those of Lagarias, Reeds, Wright and Wright (1998), and the coefficients by default
    Gao and Han's (2012), which adapt to the number of genes and keep the simplex from flattening
    from about 5 genes up. Points outside the bounds are moved to the nearest point inside, gene
    by gene. A run has converged when every vertex is within ``tolerance`` of the best one, in
    every gene, as a fraction of the gene's range: without restarts, ``run`` then stops with the
    stop reason "converged"; with them, the search starts again from a random point, and the
    result is the best of all the runs. A generation is one round of evaluations: a new simplex,
    one trial point, or the points of a shrink. With ``speculative``, the reflection, the
    expansion and both contractions are evaluated in one round, for a slow fitness function
    evaluated in parallel. The method only compares scores, so invalid and constrained solutions
    rank as everywhere in the package.

    Rosenbrock's valley from the classic start ``(-1.2, 1)``, to convergence::

        import genoxide as gx

        def rosenbrock(x):
            return 100 * (x[1] - x[0] * x[0]) ** 2 + (1 - x[0]) ** 2

        nelder_mead = gx.NelderMead(
            gx.Real((-5, 5), length=2), initial_genome=[-1.2, 1], objective="minimize"
        )
        result = nelder_mead.run(rosenbrock, evaluations=1_000)
        print(result.stop_reason, result.best_genome)  # converged [1. 1.]

    A run still needs a stop condition, here ``evaluations``, which it may not reach.

    Parameters
    ----------
    genome : Real
        The search space. At least one gene needs ``low < high``.
    coefficients : "adaptive", "standard" or tuple of 4 floats, default "adaptive"
        How far the simplex reflects, expands, contracts and shrinks. "adaptive" is Gao and
        Han's for ``n`` searched genes: reflection 1, expansion ``1 + 2/n``, contraction
        ``0.75 - 1/(2n)`` and shrink ``1 - 1/n`` (the standard ones for 1 or 2 genes).
        "standard" is Nelder and Mead's (1965): 1, 2, 1/2 and 1/2. A tuple
        ``(reflection, expansion, contraction, shrink)`` sets them, with ``reflection > 0``,
        ``expansion > 1`` and greater than ``reflection``, and ``contraction`` and ``shrink``
        between 0 and 1 (exclusive).
    initial_step : float, default 0.1
        The size of the first simplex of every run, as a fraction of each gene's range, greater
        than 0 and at most 1: each other vertex is the start moved up by it in one gene, or down
        if up leaves the bounds.
    tolerance : float, default 1e-10
        The simplex size at which a run has converged, as a fraction of each gene's range,
        greater than 0 and smaller than ``initial_step``.
    restarts : int, optional
        Random restarts after a run converges, at least 1, each from a new random point in the
        bounds: for multimodal functions, or to make sure of a minimum. None is no restarts.
    speculative : bool, default False
        Whether the reflection, the expansion and both contractions of an iteration are
        evaluated in one round: the same simplexes in fewer generations, for 4 evaluations per
        iteration. It pays off with a slow fitness function evaluated in parallel
        (``parallel=True`` or ``batch=True``).
    initial_genome : array-like of float, optional
        The point to start from, a number per gene within the bounds, e.g. a known good solution
        or the best of a global method. None is a random point. Restarts start from random
        points.
    objective : {"maximize", "minimize"}, default "maximize"
        Whether higher or lower scores are better.
    seed : int, optional
        The seed of the random numbers (the random start and the restarts), 0 to 2^64 - 1. None
        is a random seed. The same seed repeats the run.

    References: Nelder, J. A. and Mead, R. (1965). A simplex method for function minimization.
    *The Computer Journal* 7(4): 308-313. Lagarias, J. C., Reeds, J. A., Wright, M. H. and
    Wright, P. E. (1998). Convergence properties of the Nelder-Mead simplex method in low
    dimensions. *SIAM Journal on Optimization* 9(1): 112-147. Gao, F. and Han, L. (2012).
    Implementing the Nelder-Mead simplex algorithm with adaptive parameters. *Computational
    Optimization and Applications* 51(1): 259-277.
    """

    _running = RunningNelderMead

    def __init__(
        self,
        genome: Real,
        *,
        coefficients: Literal["adaptive", "standard"] | tuple[float, float, float, float] = (
            "adaptive"
        ),
        initial_step: float = 0.1,
        tolerance: float = 1e-10,
        restarts: int | None = None,
        speculative: bool = False,
        initial_genome: Sequence[float] | np.ndarray | None = None,
        objective: ObjectiveName = "maximize",
        seed: int | None = None,
    ) -> None:
        self._genome = genome
        self._objective = objective
        self.coefficients = coefficients
        self.initial_step = initial_step
        self.tolerance = tolerance
        self.restarts = restarts
        self.speculative = speculative
        self.initial_genome = initial_genome
        self.seed = seed

    def _describe(self) -> dict[str, Any]:
        wrong = (
            'coefficients is "adaptive", "standard" or a tuple (reflection, expansion, '
            f"contraction, shrink), not {self.coefficients!r}"
        )
        coefficients: str | dict[str, float]
        if isinstance(self.coefficients, str):
            if self.coefficients not in ("adaptive", "standard"):
                raise ValueError(wrong)
            coefficients = self.coefficients
        else:
            try:
                values = tuple(self.coefficients)
            except TypeError:
                raise ValueError(wrong) from None
            if len(values) != 4:
                raise ValueError(wrong)
            names = ("reflection", "expansion", "contraction", "shrink")
            coefficients = {
                name: _number(f"coefficients.{name}", value) for name, value in zip(names, values)
            }
        restarts = None
        if self.restarts is not None:
            restarts = _whole("restarts", self.restarts, minimum=None)
            if restarts < 1:
                raise ValueError(f"restarts is at least 1, or None for no restarts, not {restarts}")
        initial_genome = None
        if self.initial_genome is not None:
            genes = np.asarray(self.initial_genome, dtype=object)
            if genes.ndim != 1:
                raise ValueError(
                    "initial_genome is a sequence of numbers, one per gene, not "
                    f"{self.initial_genome!r}"
                )
            initial_genome = [_number("initial_genome", gene, plural=True) for gene in genes]
        return {
            "type": "nelder_mead",
            "coefficients": coefficients,
            "initial_step": _number("initial_step", self.initial_step),
            "tolerance": _number("tolerance", self.tolerance),
            "restarts": restarts,
            "speculative": _flag("speculative", self.speculative),
            "initial_genome": initial_genome,
            "seed": _optional_whole("seed", self.seed),
        }


class Islands(_SingleObjective):
    """The island model: several :class:`Ga` or several :class:`De`, the islands, that evolve
    apart and, every ``interval`` generations, send copies of their ``migrants`` best individuals
    to other islands, where they replace the worst. Isolation keeps the islands diverse, and
    migration spreads what they find: often better than one large population on multimodal
    problems.

    The islands share their genome and objective, and should have different seeds; their other
    settings can differ, e.g. a mutation step per island. Each generation, every island makes a
    generation, and ``run`` evaluates the candidates of all islands together (in parallel with
    ``parallel=True``, or in one batch). Breeding and migration are sequential, so a seeded run is
    the same on any number of threads. A progress's population is the islands' populations one
    after another. Each island counts its own evaluations: give an ``l_shade`` island its share
    of the budget.

    Parameters
    ----------
    islands : sequence of Ga, or of De
        At least 2 islands, all :class:`Ga` or all :class:`De`, with the same genome and
        objective.
    topology : {"ring", "fully_connected", "random", "isolated"}, default "ring"
        Where the migrants go: "ring", each island to the next one and the last to the first,
        which spreads good solutions slowly and keeps the islands diverse; "fully_connected",
        each island to every other one, the fastest; "random", each island to another one
        chosen at random at every migration; "isolated", no migration: the islands evolve apart
        for the whole run, e.g. with settings of their own to compare or hedge between them.
    interval : int, default 10
        The generations between migrations, at least 1.
    migrants : int, default 2
        The individuals each island sends at a migration, copies of its best, at least 1; a few
        percent of an island's population is common.
    seed : int, optional
        The seed of the random topology's choices, 0 to 2^64 - 1. None is a random seed.
    """

    _running = RunningIslands

    def __init__(
        self,
        islands: Sequence[Ga] | Sequence[De],
        *,
        topology: Literal["ring", "fully_connected", "random", "isolated"] | None = None,
        interval: int | None = None,
        migrants: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.islands: list[Ga | De] = list(islands)
        self.topology = topology
        self.interval = interval
        self.migrants = migrants
        self.seed = seed

    @property
    def _genome(self) -> Genome:  # type: ignore[override]
        return self._first()._genome

    @property
    def _objective(self) -> ObjectiveName:  # type: ignore[override]
        return self._first()._objective

    def _first(self) -> Ga | De:
        if len(self.islands) < 2:
            raise ValueError(
                f"invalid setting `islands`: at least 2 islands, got {len(self.islands)}"
            )
        first = self.islands[0]
        kind = type(first)
        for index, island in enumerate(self.islands):
            if type(island) not in (Ga, De):
                raise ValueError(f"islands are Ga or De, not {island!r} (island {index})")
            if type(island) is not kind:
                raise ValueError("the islands are all Ga or all De")
        return first

    def _describe(self) -> dict[str, Any]:
        first = self._first()
        genome = _describe_setting("genome", first._genome, _GENOME)
        for index, island in enumerate(self.islands):
            if _describe_setting("genome", island._genome, _GENOME) != genome:
                raise ValueError(
                    "invalid setting `islands`: the islands must share a genome, so migrants "
                    f"fit: island {index} has another one than island 0"
                )
            if island._objective != first._objective:
                raise ValueError(
                    "invalid setting `islands`: the islands must share an objective: island "
                    f"{index} has another one than island 0"
                )
        topologies = (None, "ring", "fully_connected", "random", "isolated")
        if self.topology not in topologies:
            raise ValueError(
                'topology is "ring", "fully_connected", "random" or "isolated", not '
                f"{self.topology!r}"
            )
        return {
            "type": "islands",
            "islands": [island._describe() for island in self.islands],
            "topology": self.topology,
            "interval": _optional_whole("interval", self.interval),
            "migrants": _optional_whole("migrants", self.migrants),
            "seed": _optional_whole("seed", self.seed),
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
    return cast(list[list[float]], array.tolist())


def _objective_list(objectives: Sequence[ObjectiveName]) -> list[ObjectiveName]:
    """A copy of the objectives as a list; a string as it is, for ``run`` to name it."""
    if isinstance(objectives, str):
        return cast(list[ObjectiveName], objectives)
    return list(objectives)


class _MultiObjective(_Algorithm):
    """A multi-objective algorithm, whose children are made by ``crossover`` and ``mutation``."""

    objectives: list[ObjectiveName]
    crossover: Crossover
    mutation: Mutation
    crossover_rate: float | None
    mutation_rate: float | None
    seed: int | None

    def _objectives(self) -> list[str]:
        if isinstance(self.objectives, str):
            raise ValueError(
                'objectives is a list, one "maximize" or "minimize" per objective, e.g. '
                f'["minimize", "minimize"], not the string {self.objectives!r}'
            )
        for objective in self.objectives:
            if objective not in ("maximize", "minimize"):
                raise ValueError(f'an objective is "maximize" or "minimize", not {objective!r}')
        return list(self.objectives)

    def _variation(self) -> dict[str, Any]:
        return {
            "crossover": _describe_setting("crossover", self.crossover, _CROSSOVER),
            "mutate": _describe_setting("mutation", self.mutation, _MUTATION),
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
        checkpoint: str | os.PathLike[str] | None = None,
        checkpoint_every: int | None = None,
        resume: str | os.PathLike[str] | None = None,
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
            generation as a 2-D array, a genome per row, and returns the objective values: a 2-D
            array or a list with a row per genome, or a tuple of one 1-D array per objective
            (``return f1, f2``); or a tuple of those and an array of constraint violations.
            The function must be deterministic. A multi-objective problem of
            :mod:`genoxide.problems` is evaluated in Rust, with no Python call: ``batch`` doesn't
            apply, and the genome must be the problem's (``problem.genome``: a :class:`Real`
            with a gene per variable, or a :class:`Binary` for :class:`~genoxide.problems.Zdt5`),
            and the objectives the problem's, all "minimize". For a :class:`genoxide.gp.Gp`
            genome, the function takes a :class:`genoxide.gp.Tree`; a
            :class:`genoxide.gp.WithSize` (a fitness of :mod:`genoxide.gp` and the tree's size)
            is evaluated in Rust, with 2 objectives, both "minimize".
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
            the previous generation's front didn't dominate or equal, at least 1. With a front
            of more trade-offs than the population holds, survival drops some and brings them
            back later, which counts as a gain, and the run may never stagnate: add
            ``generations`` or ``evaluations``.
        batch : bool, default False
            Calls ``fitness`` once per generation with a 2-D array, and not for a generation of
            copies of their parents.
        parallel : bool, default False
            Calls a non-batch ``fitness`` from several threads at once. It pays off when the
            function releases the GIL (numpy on large arrays, I/O), or on free-threaded Python
            (3.14t).
        on_generation : callable, optional
            Called with a :class:`MultiProgress` after every generation, the initial population
            (generation 0) included, on the thread that called ``run``. If it returns False, the
            run stops with the stop reason "aborted".
        checkpoint : str or os.PathLike, optional
            Saves the run to this file every ``checkpoint_every`` generations and when it stops,
            to resume it later with ``resume``. The file is replaced atomically: a crash while
            saving keeps the previous checkpoint. None saves nothing.
        checkpoint_every : int, optional
            The generations between checkpoints, at least 1, with ``checkpoint``. A checkpoint
            is saved when the generation is a multiple of it, so a resumed run keeps the
            schedule.
        resume : str or os.PathLike, optional
            Continues the run saved in this checkpoint, with the results it would have had
            without the interruption. The algorithm, its genome and the objectives must be the
            ones that saved it; the fitness function, the stop conditions, ``batch``,
            ``parallel``, the callbacks and ``checkpoint`` can change, e.g. to run longer.
            ``generations`` and ``evaluations`` count from the start of the first run, ``time``
            from the start of this one. The file must come from the same version of genoxide.
            Load only checkpoints you trust, like the program that saved them: the checksum
            detects accidental damage, not tampering, and a crafted checkpoint can make a run
            loop or fail, though never break memory safety.

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
            objectives (the message names the setting); for a single-objective problem of
            :mod:`genoxide.problems`, or a multi-objective one whose objectives, number of genes
            or genome don't match the algorithm's; and for a wrong fitness result: the
            wrong number of objective values, a negative constraint violation, a batch result
            that isn't a 2-D array or a tuple of an array per objective, or one with a number of
            rows other than the number of genomes; and for a checkpoint to resume from that isn't
            one, is damaged, comes from another version of genoxide or was saved with other
            settings.
        OSError
            If the checkpoint to resume from can't be read, e.g. a ``FileNotFoundError``, or a
            checkpoint can't be saved.
        TypeError
            If ``fitness`` or ``on_generation`` isn't callable, or ``fitness`` returns something
            that isn't a sequence of numbers. Another error converting a result, e.g. an
            ``OverflowError`` for an int too large for a float, is raised as it is.
        Exception
            An exception raised by ``fitness`` or ``on_generation`` stops the run, and ``run``
            raises it. So does ``KeyboardInterrupt`` on Ctrl+C: with ``parallel``, once the calls
            of ``fitness`` under way return.
        """
        _check_callable(fitness)
        if isinstance(fitness, (problems.Problem, problems.control.Balance)):
            raise ValueError(
                f"{type(fitness).__name__} has one objective: use a single-objective algorithm"
            )
        stop = _stop(generations, evaluations, None, time, stagnation)
        callback = _on_generation(on_generation, MultiProgress)
        saving = _checkpoints(checkpoint, checkpoint_every, resume)
        if gp._is_tree_fitness(fitness):
            # a fitness of trees with their size, evaluated in Rust
            return MultiResult(
                **self._run(fitness, stop, False, parallel, callback, None, None, saving)
            )
        if isinstance(fitness, problems.MultiProblem):
            description = fitness._json()
            return MultiResult(
                **self._run(fitness, stop, False, parallel, callback, description, None, saving)
            )
        function = _batch_objectives(fitness, len(self._objectives())) if batch else fitness
        return MultiResult(
            **self._run(function, stop, batch, parallel, callback, None, None, saving)
        )


class Nsga2(_MultiObjective):
    """NSGA-II, for 2 to 6 objectives: non-dominated sorting and crowding distance.

    ``objectives`` says, for each objective, whether to "maximize" or "minimize" it. The fitness
    function returns a sequence of objective values, None (an invalid solution) or
    ``(objective_values, constraint_violation)``; with ``batch=True``, a 2-D array with a row of
    objective values per genome, or a tuple of it and an array of constraint violations. The same
    goes for every multi-objective algorithm.

    Pairs of parents are recombined with probability ``crossover_rate`` (default 0.9), and each
    child is mutated with probability ``mutation_rate`` (default 1). With ``eliminate_duplicates``
    (the default), a child that equals a member of the population or an earlier child is dropped
    and another bred instead, which keeps the population and its front free of copies;
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
    initial_genomes : sequence of gp.Tree, optional
        For a :class:`genoxide.gp.Gp` genome (with 2 objectives, e.g. the error and the size,
        :class:`genoxide.gp.WithSize`), trees of its set to start from, at most
        ``population_size``; random trees fill the rest.
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
        initial_genomes: Sequence[gp.Tree] | None = None,
    ) -> None:
        self._genome = genome
        self.objectives = _objective_list(objectives)
        self.population_size = population_size
        self.crossover = crossover
        self.mutation = mutation
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.eliminate_duplicates = eliminate_duplicates
        self.seed = seed
        self.initial_genomes = initial_genomes

    def _describe(self) -> dict[str, Any]:
        return {
            "type": "nsga2",
            "population_size": _whole("population_size", self.population_size),
            "seed": _optional_whole("seed", self.seed),
            "variation": self._variation(),
            "initial_genomes": gp._initial_genomes(self._genome, self.initial_genomes),
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
        self.objectives = _objective_list(objectives)
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
        self.objectives = _objective_list(objectives)
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
        self.objectives = _objective_list(objectives)
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
            "decomposition": (
                None
                if self.decomposition is None
                else _describe_setting("decomposition", self.decomposition, _DECOMPOSITION)
            ),
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
        self.objectives = _objective_list(objectives)
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
from . import indicators, math, nn, problems  # noqa: E402
from . import gp  # noqa: E402
