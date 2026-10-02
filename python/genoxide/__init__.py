"""Evolutionary computation in Rust, for Python.

Genetic algorithms, local search, the Nelder-Mead simplex method, L-BFGS-B and first-order
gradient methods (gradient descent, momentum, Nesterov, Adam and AdamW), Bayesian optimization
(with the Gaussian processes of :mod:`genoxide.model.gp`), differential evolution,
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
    "Bo",
    "ProbabilityOfImprovement",
    "UpperConfidenceBound",
    "FirstOrder",
    "Lbfgsb",
    "Mma",
    "Continuation",
    "Islands",
    "Nsga2",
    "Nsga3",
    "Spea2",
    "Moead",
    "SmsEmoa",
    "das_dennis",
    # results and progress
    "Result",
    "Stage",
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
    "RunningBo",
    "RunningFirstOrder",
    "RunningLbfgsb",
    "RunningMma",
    "RunningIslands",
    # submodules
    "problems",
    "indicators",
    "nn",
    "neat",
    "math",
    "gp",
    "model",
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


@dataclass(frozen=True)
class Stage:
    """What a finished stage of a :class:`Continuation` did."""

    index: int
    """The stage's index, from 0."""
    generations: int
    """The generations of the wrapped method in the stage, its iterations; the re-evaluation that
    starts a stage isn't one."""
    evaluations: int
    """The evaluations of the stage, the re-evaluation that starts it included."""
    best_fitness: float | None
    """The best score at the end of the stage, by the stage's fitness function; None if it has
    no valid solution."""
    end: str
    """Why the stage ended: "finished", the method converged; or "generations", the stage's
    budget ran out."""


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
      whose simplex shrank within its tolerance, an :class:`Lbfgsb` or :class:`FirstOrder` at a
      minimum, with no restart
      left, a :class:`Cmaes` with ``restarts="stop"`` whose run met a stop criterion, or a
      :class:`Continuation` after its last stage;
    - "stalled": nothing new to evaluate for 10,000 generations in a row (e.g. every child was a
      copy of a parent), while only ``target`` or ``evaluations`` could stop the run;
    - "other": a reason that the stop conditions of the package don't produce.
    """
    stages: tuple[Stage, ...] = ()
    """For a :class:`Continuation`, what each finished stage did, in order; empty otherwise."""


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
    :class:`RunningNelderMead`, :class:`RunningBo`, :class:`RunningLbfgsb`,
    :class:`RunningFirstOrder`, :class:`RunningMma` and :class:`RunningIslands`. A new value is
    checked as in the algorithm's constructor: a wrong one raises a ``ValueError`` and changes
    nothing. The handle works only during the callback; afterwards it raises a ``RuntimeError``.

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
        Nelder-Mead simplex, whose iteration under way is dropped, a first-order method's point
        and its gradient, with the velocity or Adam's averages kept), without breeding and without
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
        in any searched gene, as a fraction of the gene's initial step (1 or less for the first
        simplex). The run has converged once it's within the tolerance."""
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


class RunningBo(Running):
    """A running :class:`Bo`, for ``control``: its acquisition function, which can change (e.g.
    UCB's ``beta`` on a schedule), and the model that chose the last point, with the
    acquisition's values, e.g. for a plot. ``reevaluate()`` asks every evaluated point again."""

    __slots__ = ()

    @property
    def acquisition(self) -> BoAcquisition:
        """The acquisition function: "log-ei", "ei", a :class:`ProbabilityOfImprovement` or an
        :class:`UpperConfidenceBound`. A new one applies from the next point."""
        return _acquisition_of(self._get("acquisition"))

    @acquisition.setter
    def acquisition(self, value: BoAcquisition) -> None:
        self._set("acquisition", _acquisition(value))

    @property
    def initial_points(self) -> int:
        """The points of the initial design."""
        return int(self._get("initial_points"))

    @property
    def batch(self) -> int:
        """The points of each generation after the initial design, at least 1. A new value
        applies from the next generation."""
        return int(self._get("batch"))

    @batch.setter
    def batch(self, value: int) -> None:
        self._set("batch", _whole("batch", value, minimum=None))

    @property
    def fantasy(self) -> BoFantasy:
        """What the points of a batch are taken to be worth while the next ones are chosen:
        "believer", "liar-min", "liar-mean" or "liar-max". A new value applies from the next
        generation."""
        return cast(BoFantasy, self._get("fantasy"))

    @fantasy.setter
    def fantasy(self, value: BoFantasy) -> None:
        self._set("fantasy", _fantasy(value))

    @property
    def constraints(self) -> int:
        """The number of inequality constraints whose values the fitness function gives: 0
        without constraints."""
        return int(self._get("constraints") or 0)

    @property
    def model(self) -> _surrogate.GaussianProcess | None:
        """The Gaussian process that chose the last point, fitted to the evaluations before it:
        a model of the values the search minimizes (the scores, negated when maximizing, through
        ``output``). None in generation 0, and after resuming from a checkpoint until the next
        point."""
        native = self._native.model()
        return None if native is None else _surrogate.GaussianProcess(native)

    def acquisition_at(self, points: Any) -> np.ndarray:
        """The acquisition function under :attr:`model` at ``points``, a point per row (a 1-D
        array is one point), as the search maximizes it: the log expected improvement, the
        expected improvement, the logarithm of the probability of improvement or the negated
        lower confidence bound, in the model's standardized units. Raises a ``ValueError``
        without a model."""
        rows = np.ascontiguousarray(points, dtype=np.float64)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        return np.asarray(self._native.acquisition(rows))

    def probability_of_feasibility_at(self, points: Any) -> np.ndarray:
        """The probability that ``points`` are feasible under the constraints' models, a point
        per row (a 1-D array is one point): the product of the probabilities that each
        constraint's value is at most 0, the models taken as independent. 1 without constraints.
        Raises a ``ValueError`` without a model."""
        rows = np.ascontiguousarray(points, dtype=np.float64)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        return np.asarray(self._native.feasibility(rows))


class RunningLbfgsb(Running):
    """A running :class:`Lbfgsb`, for ``control``: its ``memory``, which can change, and its
    state, read-only. ``reevaluate()`` scores the current point again and drops the correction
    pairs, which describe the old function; the search goes on from the point along steepest
    descent."""

    __slots__ = ()

    @property
    def memory(self) -> int:
        """The correction pairs the limited-memory matrix keeps at most, at least 1. A new value
        applies from the next iteration and keeps the newest pairs that fit."""
        return int(self._get("memory"))

    @memory.setter
    def memory(self, value: int) -> None:
        self._set("memory", _whole("memory", value, minimum=None))

    @property
    def pairs(self) -> int:
        """The correction pairs stored now, at most ``memory``."""
        return int(self._get("pairs"))

    @property
    def converged(self) -> str | None:
        """Why the current run has converged: "projected_gradient", "relative_decrease",
        "line_search" (no step lowers the function) or "not_finite" (the function or its
        gradient isn't finite at the start); None while it's still searching."""
        value = self._get("converged")
        return None if value is None else str(value)

    @property
    def projected_gradient(self) -> float:
        """The largest component of the projected gradient at the current point: 0 at a minimum
        in the bounds. NaN before the first evaluation."""
        value = self._get("projected_gradient")
        return _math.nan if value is None else float(value)

    @property
    def iterations(self) -> int:
        """The completed iterations, of every run: the steps taken."""
        return int(self._get("iterations"))

    @property
    def gradients(self) -> str:
        """Where the gradients come from in this run: "supplied", "forward" or "central", the
        ``gradients`` setting resolved ("auto" becomes "supplied" with a gradient, "forward"
        without)."""
        return str(self._get("gradients"))

    @property
    def gradient_evaluations(self) -> int:
        """The gradients computed: by the fitness function, or each from a stencil of finite
        differences."""
        return int(self._get("gradient_evaluations"))

    @property
    def stencil_evaluations(self) -> int:
        """The evaluations of finite-difference points, part of the run's evaluations: the cost
        of the gradients, 0 when they're supplied."""
        return int(self._get("stencil_evaluations"))

    @property
    def skipped_pairs(self) -> int:
        """The correction pairs skipped for too little curvature."""
        return int(self._get("skipped_pairs"))

    @property
    def memory_resets(self) -> int:
        """The times the pairs were dropped to search along steepest descent again."""
        return int(self._get("memory_resets"))

    @property
    def restart_count(self) -> int:
        """The restarts so far."""
        return int(self._get("restart_count"))


class RunningMma(Running):
    """A running :class:`Mma`, for ``control``: its state, read-only. Its steps follow from its
    approximations, so it has no settings to change; ``reevaluate()`` scores the current point
    again, for a fitness function that changed, and the next iteration starts from it with the
    asymptotes kept."""

    __slots__ = ()

    @property
    def converged(self) -> str | None:
        """Why the run has converged: "kkt" (the KKT residual is within the tolerance) or "step"
        (the last step is); None while it goes on."""
        converged = self._get("converged")
        return None if converged is None else str(converged)

    @property
    def iterations(self) -> int:
        """The completed iterations: points accepted after the initial one."""
        return int(self._get("iterations"))

    @property
    def inner_iterations(self) -> int:
        """GCMMA's inner iterations: points rejected because an approximation wasn't
        conservative there. 0 with ``method="mma"``."""
        return int(self._get("inner_iterations"))

    @property
    def multipliers(self) -> list[float]:
        """The constraints' multipliers from the last subproblem, at least 0: at a solution, the
        Lagrange multipliers of the problem minimized (of the negated score when maximizing). A
        multiplier at ``constraint_cost`` or above means its constraint couldn't be met."""
        return [float(value) for value in self._get("multipliers")]

    @property
    def kkt_residual(self) -> float:
        """The KKT residual at the current point with those multipliers (NaN before the first
        iteration)."""
        residual = self._get("kkt_residual")
        return _math.nan if residual is None else float(residual)


class RunningFirstOrder(Running):
    """A running :class:`FirstOrder` method, for ``control``: its learning rate and schedule
    multiplier, e.g. lowered over the run (a learning-rate schedule), and its state, read-only.
    A change applies from the next step; the velocity and Adam's averages are kept.
    ``reevaluate()`` scores the point again, with its gradient, and keeps them too."""

    __slots__ = ()

    @property
    def learning_rate(self) -> float:
        """The learning rate alpha, greater than 0."""
        return float(self._get("learning_rate"))

    @learning_rate.setter
    def learning_rate(self, learning_rate: float) -> None:
        self._set("learning_rate", _number("learning_rate", learning_rate))

    @property
    def multiplier(self) -> float:
        """The schedule multiplier eta of Loshchilov and Hutter, greater than 0, 1 unless
        changed: it multiplies every step, AdamW's weight decay included, where the learning rate
        doesn't change the decay."""
        return float(self._get("multiplier"))

    @multiplier.setter
    def multiplier(self, multiplier: float) -> None:
        self._set("multiplier", _number("multiplier", multiplier))

    @property
    def converged(self) -> str | None:
        """Whether the current run has converged, and why: "gradient" (the projected gradient
        within its tolerance), "step" (the last step within its tolerance) or "invalid" (an
        invalid point with no valid one to step back to); None if not. With restarts left, the
        next generation starts a new run from a random point."""
        converged = self._get("converged")
        return None if converged is None else str(converged)

    @property
    def gradient_norm(self) -> float | None:
        """The largest component of the projected gradient at the point (0 for a gene at a bound
        it points out of): what the gradient tolerance measures. None before the first
        evaluation."""
        norm = self._get("gradient_norm")
        return None if norm is None else float(norm)

    @property
    def gradient(self) -> np.ndarray:
        """The gradient of the last evaluation, a copy: at the point, or at an invalid point
        the method steps back from."""
        return np.asarray(self._get("gradient"), dtype=np.float64)

    @property
    def iterations(self) -> int:
        """The steps of every run; a step back from an invalid point isn't one."""
        return int(self._get("iterations"))

    @property
    def steps(self) -> int:
        """The steps t of the current run, the exponent of Adam's bias corrections."""
        return int(self._get("steps"))

    @property
    def restart_count(self) -> int:
        """The restarts so far."""
        return int(self._get("restart_count"))

    @property
    def gradients(self) -> str:
        """Where the gradients come from in this run: "supplied", "forward" or "central"."""
        return str(self._get("gradients"))


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


def _gradient_array(gradient: Callable[[np.ndarray], Any]) -> Callable[[np.ndarray], Any]:
    """A gradient function returning a float64 array, a value per gene."""

    def evaluate(genome: np.ndarray) -> np.ndarray:
        return np.asarray(gradient(genome), dtype=np.float64).reshape(-1)

    return evaluate


def _with_gradient(function: Callable[[np.ndarray], Any]) -> Callable[[np.ndarray], Any]:
    """A function returning ``(value, gradient)``, the gradient a float64 array."""

    def evaluate(genome: np.ndarray) -> tuple[Any, np.ndarray]:
        result = function(genome)
        if not (isinstance(result, tuple) and len(result) == 2):
            raise TypeError(
                "with gradient=True, the fitness function returns a tuple (value, gradient), "
                f"not {type(result).__name__}"
            )
        value, gradient = result
        return value, np.asarray(gradient, dtype=np.float64).reshape(-1)

    return evaluate


def _with_constraints(
    function: Callable[[np.ndarray], Any], constraints: int
) -> Callable[[np.ndarray], Any]:
    """A function returning ``(value, gradient, g, jacobian)``: the gradient and the
    ``constraints`` values ``g`` float64 arrays, the Jacobian a 2-D one, a row per constraint."""

    def evaluate(genome: np.ndarray) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
        result = function(genome)
        if not (isinstance(result, tuple) and len(result) == 4):
            raise TypeError(
                "with gradient=True and constraints, the fitness function returns a tuple "
                "(value, gradient, constraint values, jacobian), not "
                + type(result).__name__
                + (f" of length {len(result)}" if isinstance(result, tuple) else "")
            )
        value, gradient, values, jacobian = result
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.shape != (constraints,):
            raise ValueError(
                f"the fitness function returns {values.size} constraint values, for "
                f"{constraints} constraints"
            )
        return (
            value,
            np.asarray(gradient, dtype=np.float64).reshape(-1),
            values,
            np.asarray(jacobian, dtype=np.float64).reshape(constraints, -1),
        )

    return evaluate


def _batch_gradient_rows(gradient: Callable[[np.ndarray], Any]) -> Callable[[np.ndarray], Any]:
    """A batch gradient function returning a float64 array, a row per genome."""

    def evaluate(genomes: np.ndarray) -> np.ndarray:
        return np.asarray(gradient(genomes), dtype=np.float64).reshape(len(genomes), -1)

    return evaluate


def _batch_with_gradients(function: Callable[[np.ndarray], Any]) -> Callable[[np.ndarray], Any]:
    """A batch function returning ``(values, gradients)``: float64 arrays, a value and a row of
    the gradient per genome."""

    def evaluate(genomes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        result = function(genomes)
        if not (isinstance(result, tuple) and len(result) == 2):
            raise TypeError(
                "with gradient=True and batch=True, the fitness function returns a tuple "
                f"(values, gradients), not {type(result).__name__}"
            )
        values, gradients = result
        return (
            np.asarray(values, dtype=np.float64).reshape(-1),
            np.asarray(gradients, dtype=np.float64).reshape(len(genomes), -1),
        )

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
        gradient: Callable[[np.ndarray], Any] | None = None,
        combined_gradient: bool = False,
        constraints: int = 0,
    ) -> dict[str, Any]:
        run = {
            "genome": _describe_setting("genome", self._genome, _GENOME),
            "algorithm": self._describe(),
            "objectives": self._objectives(),
            "stop": stop,
        }
        # NaN and infinity aren't JSON: the settings are finite, or an error names them
        description = json.dumps(run, default=_json_number, allow_nan=False)
        result = _genoxide.run(
            description,
            fitness,
            bool(batch),
            bool(parallel),
            on_generation,
            problem,
            control,
            **(checkpoints or {}),
            gradient=gradient,
            combined_gradient=combined_gradient,
            constraints=constraints,
            **self._stage_callbacks(),
        )
        if "stages" in result:
            result["stages"] = tuple(Stage(**stage) for stage in result["stages"])
        return result

    def _stage_callbacks(self) -> dict[str, Any]:
        """A continuation's callbacks, for ``_genoxide.run``: none for other algorithms."""
        return {}


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
            :class:`RunningNelderMead`, :class:`RunningBo`, :class:`RunningLbfgsb`,
            :class:`RunningFirstOrder`, :class:`RunningMma` or :class:`RunningIslands`) and a
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
    restarts : {"never", "ipop", "bipop", "stop"}, default "never"
        What happens when a run converges. "never" goes on sampling around the same point.
        "ipop" restarts from a random point with a doubled population, up to 1024 times the
        initial one. "bipop" alternates such large populations with small ones of random size
        and step size. Restarts suit multimodal functions. "stop" ends the run once it meets one
        of Hansen's stop criteria, as his reference code does: ``run`` returns with the stop
        reason "converged", unless a stop condition is met in the same generation; up to there,
        the run is the one of "never". It saves most of a budget on smooth problems without
        constraints. On flat or quantized fitness (plateaus, e.g. Easom's function) and with
        constraints, the criteria can fire while the best would still improve: use "never"
        there, or a ``target``.
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
        restarts: Literal["never", "ipop", "bipop", "stop"] | None = None,
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
        if self.restarts not in (None, "never", "ipop", "bipop", "stop"):
            raise ValueError(
                f'restarts is "never", "ipop", "bipop" or "stop", not {self.restarts!r}'
            )
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
    Gao and Han's (2012), which adapt to the number of genes: about as fast as the standard ones
    up to 8 genes, faster from about 9, and far more reliable from about 20. A trial point outside
    the bounds is mirrored back in at the bound it crossed, gene by gene, so the simplex doesn't
    flatten against a bound; a minimum on a bound takes more evaluations than one inside. A run
    has converged when every vertex is within ``tolerance`` of the best one, in every gene, as a
    fraction of the initial step: without restarts, ``run`` then stops with the
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
        ``0.75 - 1/(2n)`` and shrink ``1 - 1/n``, the standard ones for 2 genes. The paper
        defines them for 2 genes or more; for 1 gene, the standard ones.
        "standard" is Nelder and Mead's (1965): 1, 2, 1/2 and 1/2. A tuple
        ``(reflection, expansion, contraction, shrink)`` sets them, with ``reflection > 0``,
        ``expansion > 1`` and greater than ``reflection``, and ``contraction`` and ``shrink``
        between 0 and 1 (exclusive).
    initial_step : float, optional
        The size of the first simplex of every run, as a fraction of each gene's range, greater
        than 0 and at most 1: each other vertex is the start moved up by it in one gene, or down
        if up leaves the bounds. None is 0.1, unless ``initial_step_absolute`` is given.
    initial_step_absolute : float, optional
        The size of the first simplex as a distance, the same in every gene, positive and finite,
        instead of ``initial_step``: for a wide box around an unbounded problem (e.g. ±1e10),
        where a fraction of the range would be far too large.
    tolerance : float, default 1e-9
        The simplex size at which a run has converged, as a fraction of each gene's initial step,
        greater than 0 and smaller than 1: 1e-10 of the range with the default step, and the same
        test in a wide box as in a narrow one.
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
        initial_step: float | None = None,
        initial_step_absolute: float | None = None,
        tolerance: float = 1e-9,
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
        self.initial_step_absolute = initial_step_absolute
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
        if self.initial_step is not None and self.initial_step_absolute is not None:
            raise ValueError("give initial_step or initial_step_absolute, not both")
        return {
            "type": "nelder_mead",
            "coefficients": coefficients,
            "initial_step": _optional_number("initial_step", self.initial_step),
            "initial_step_absolute": _optional_number(
                "initial_step_absolute", self.initial_step_absolute
            ),
            "tolerance": _number("tolerance", self.tolerance),
            "restarts": restarts,
            "speculative": _flag("speculative", self.speculative),
            "initial_genome": initial_genome,
            "seed": _optional_whole("seed", self.seed),
        }


@dataclass(frozen=True)
class ProbabilityOfImprovement:
    """Bayesian optimization's probability of improving on the best value by more than ``xi``
    (Kushner 1964), maximized through its logarithm, which has the same maximizer. ``xi`` is 0 or
    more, in the units the model fits (the scores, or their logarithm with ``output="log"``):
    larger explores more."""

    xi: float

    def _describe(self) -> dict[str, Any]:
        return {"type": "probability_of_improvement", "xi": _number("xi", self.xi)}


@dataclass(frozen=True)
class UpperConfidenceBound:
    """Bayesian optimization's confidence bound ``mean - sqrt(beta) sd``, minimized (``mean +
    sqrt(beta) sd`` maximized when maximizing): Srinivas, Krause, Kakade and Seeger (2010).
    ``beta`` is 0 or more: 0 exploits the model's mean alone, larger explores more."""

    beta: float

    def _describe(self) -> dict[str, Any]:
        return {"type": "upper_confidence_bound", "beta": _number("beta", self.beta)}


BoAcquisition = Union[Literal["log-ei", "ei"], ProbabilityOfImprovement, UpperConfidenceBound]

# what Bayesian optimization takes a point not evaluated yet to be worth
BoFantasy = Literal["believer", "liar-min", "liar-mean", "liar-max"]


def _fantasy(fantasy: Any) -> str:
    if isinstance(fantasy, str) and fantasy in ("believer", "liar-min", "liar-mean", "liar-max"):
        return fantasy
    raise ValueError(
        f'fantasy is "believer", "liar-min", "liar-mean" or "liar-max", not {fantasy!r}'
    )


def _with_values(
    function: Callable[[np.ndarray], Any], constraints: int
) -> Callable[[np.ndarray], Any]:
    """A function returning ``(value, g)``, ``g`` the ``constraints`` values a float64 array."""

    def evaluate(genome: np.ndarray) -> tuple[Any, np.ndarray]:
        result = function(genome)
        if not (isinstance(result, tuple) and len(result) == 2):
            raise TypeError(
                "with constraints, the fitness function returns a tuple (value, constraint "
                "values), not "
                + type(result).__name__
                + (f" of length {len(result)}" if isinstance(result, tuple) else "")
            )
        value, values = result
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.shape != (constraints,):
            raise ValueError(
                f"the fitness function returns {values.size} constraint values, for "
                f"{constraints} constraints"
            )
        return value, values

    return evaluate


def _acquisition(acquisition: Any) -> dict[str, Any]:
    if isinstance(acquisition, (ProbabilityOfImprovement, UpperConfidenceBound)):
        return acquisition._describe()
    if isinstance(acquisition, str) and acquisition == "log-ei":
        return {"type": "log_expected_improvement"}
    if isinstance(acquisition, str) and acquisition == "ei":
        return {"type": "expected_improvement"}
    raise ValueError(
        'acquisition is "log-ei", "ei", ProbabilityOfImprovement(xi) or '
        f"UpperConfidenceBound(beta), not {acquisition!r}"
    )


def _acquisition_of(description: dict[str, Any]) -> BoAcquisition:
    kind = description["type"]
    if kind == "probability_of_improvement":
        return ProbabilityOfImprovement(float(description["xi"]))
    if kind == "upper_confidence_bound":
        return UpperConfidenceBound(float(description["beta"]))
    return "ei" if kind == "expected_improvement" else "log-ei"


class Bo(_SingleObjective):
    """Bayesian optimization. Real or Integer genomes.

    For expensive black-box functions, such as a simulation that runs for minutes or a physical
    experiment, where tens to a few hundred evaluations must do. A Gaussian process
    (:mod:`genoxide.model.gp`) models the function from every evaluation so far, and an
    acquisition function of its posterior picks the next points, trading the model's best guesses
    against its uncertainty. Generation 0 evaluates an initial design: the ``initial_genomes``,
    and a Latin hypercube (McKay, Beckman and Conover 1979) for the rest of ``initial_points``.
    Each later generation fits the model and evaluates ``batch`` points, by default one, the
    acquisition's maximum: from ``raw_samples`` random points, then L-BFGS-B with the
    acquisition's gradient from the best ``acquisition_starts`` of them and from the best point so
    far. Each further point of a batch is chosen after the ones before it are added to the model
    with a ``fantasy`` value (Ginsbourger, Le Riche and Carraro 2010), the hyperparameters kept;
    with ``parallel=True`` in :meth:`run`, the points of a batch are evaluated together. A point
    is never evaluated twice. The model fits the scores to minimize (negated when maximizing),
    through ``output``; an invalid score enters it at the worst value of the others, so the search
    learns to avoid where the function fails. With ``constraints=m`` in :meth:`run` (or a test
    problem of :mod:`genoxide.problems` with constraints, which gives their values), a Gaussian
    process models each constraint, and the acquisition is weighed by the probability that a
    point is feasible (Gardner et al. 2014); until a point is feasible, that probability alone is
    maximized. Without them, a constraint violation is ignored by the search (use a penalty). On
    an :class:`Integer` genome, the genes are rounded to the nearest integer inside the model's
    kernel (Garrido-Merchán and Hernández-Lobato 2020), and the acquisition is maximized on the
    lattice, by a hill climb of steps of one in one gene. The model's fit costs O(N^3) for N
    evaluations: up to a few hundred evaluations of a function far more expensive than that, in
    up to about 10 to 20 genes.

    Branin's function, whose three global minima are 0.397887, in 40 evaluations::

        import genoxide as gx

        problem = gx.problems.Branin()
        bo = gx.Bo(problem.genome, objective="minimize", seed=1)
        result = bo.run(problem, evaluations=40)
        print(result.best_fitness)  # 0.3978873...

    A constraint's values one by one: the fitness function returns ``(value, g)``, feasible
    where every value of ``g`` is at most 0::

        import numpy as np

        def disc(x):
            # the sum inside the unit disc: the minimum -1.414... at (-0.707..., -0.707...)
            return x[0] + x[1], np.array([x[0] ** 2 + x[1] ** 2 - 1.0])

        bo = gx.Bo(gx.Real((-2.0, 2.0), length=2), objective="minimize", seed=1)
        result = bo.run(disc, constraints=1, evaluations=30)
        print(result.best_fitness)  # -1.41...

    The model, its likelihood's maximization and the acquisition's are seeded and portable: a
    seed repeats the run on every platform, and gives a Rust program's results.

    Parameters
    ----------
    genome : Real or Integer
        The search space. At least one gene needs ``low < high``.
    initial_points : int, optional
        The points of the initial design, at least 1. None is 2(n + 1) for the n genes with
        ``low < high``: a small design leaves most evaluations to the model. 10n is the usual
        size for an accurate model of the whole box (Loeppky, Sacks and Welch 2009), more than
        finding a minimum needs.
    initial_genomes : 2-D array-like of float, optional
        Genomes evaluated first, in the initial design, a genome per row: at most
        ``initial_points`` of them, distinct, within the bounds.
    acquisition : "log-ei", "ei", ProbabilityOfImprovement or UpperConfidenceBound
        The acquisition function, "log-ei" by default: the logarithm of the expected
        improvement, computed so that it and its gradient stay finite where the expected
        improvement underflows (Ament et al. 2023), the same maximizer found far more reliably
        than with "ei", the expected improvement itself (Mockus 1975; Jones, Schonlau and Welch
        1998).
    kernel : "matern52" or "squared_exponential", default "matern52"
        The model's kernel, with a length scale per gene.
    noise : float or genoxide.model.gp.Learned, default 0.0
        The model's noise variance, a fraction of the values' variance: 0 interpolates the
        values, as suits a deterministic function; :class:`genoxide.model.gp.Learned` learns it,
        for a noisy one.
    output : "standardize" or "log", default "standardize"
        What the model fits: the values themselves, standardized; or ``log(v - v_best + d)`` of
        the values to minimize, ``d`` the first quartile of their distances above the best, for
        objectives that span orders of magnitude (Goldstein-Price's 3 to 10^6).
    raw_samples : int, default 1000
        The random points at which the acquisition is evaluated before its maximization, at
        least 1.
    acquisition_starts : int, default 10
        The best raw samples from which L-BFGS-B maximizes the acquisition, at least 1 and at
        most ``raw_samples``.
    hyperparameter_starts : int, default 5
        The starts of the likelihood's maximization at each fit, at least 1: the last fit's
        hyperparameters, then random ones.
    batch : int, default 1
        The points of each generation after the initial design, at least 1, chosen one after the
        other, each after the ones before it are added to the model with a ``fantasy`` value:
        with ``parallel=True``, a generation takes about as long as one evaluation. A batch needs
        more evaluations than a point at a time, and fewer generations.
    fantasy : {"believer", "liar-min", "liar-mean", "liar-max"}, default "believer"
        What a point of a batch is taken to be worth while the next ones are chosen: the model's
        mean there (the Kriging believer, which only removes the uncertainty there), or a lie,
        the lowest, mean or highest value the model fits (the constant liar: the higher, the
        farther the next points go). The constraints' models take their mean either way.
    objective : {"maximize", "minimize"}, default "maximize"
        Whether higher or lower scores are better.
    seed : int, optional
        The seed of the random numbers (the design, the starts of both maximizations), 0 to
        2^64 - 1. None is a random seed. The same seed repeats the run.

    References: Jones, D. R., Schonlau, M. and Welch, W. J. (1998). Efficient global optimization
    of expensive black-box functions. *Journal of Global Optimization* 13(4): 455-492. Rasmussen,
    C. E. and Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning.* MIT Press.
    Ament, S., Daulton, S., Eriksson, D., Balandat, M. and Bakshy, E. (2023). Unexpected
    improvements to expected improvement for Bayesian optimization. *NeurIPS 2023*. Ginsbourger,
    D., Le Riche, R. and Carraro, L. (2010). Kriging is well-suited to parallelize optimization.
    In *Computational Intelligence in Expensive Optimization Problems*, Springer: 131-162.
    Gardner, J. R., Kusner, M. J., Xu, Z., Weinberger, K. Q. and Cunningham, J. P. (2014).
    Bayesian optimization with inequality constraints. *ICML 2014*. Garrido-Merchán, E. C. and
    Hernández-Lobato, D. (2020). Dealing with categorical and integer-valued variables in
    Bayesian optimization with Gaussian processes. *Neurocomputing* 380: 20-35.
    """

    _running = RunningBo

    def __init__(
        self,
        genome: Real | Integer,
        *,
        initial_points: int | None = None,
        initial_genomes: Sequence[Sequence[float]] | np.ndarray | None = None,
        acquisition: BoAcquisition = "log-ei",
        kernel: Literal["matern52", "squared_exponential"] = "matern52",
        noise: Any = 0.0,
        output: Literal["standardize", "log"] = "standardize",
        raw_samples: int = 1000,
        acquisition_starts: int = 10,
        hyperparameter_starts: int = 5,
        batch: int = 1,
        fantasy: BoFantasy = "believer",
        objective: ObjectiveName = "maximize",
        seed: int | None = None,
    ) -> None:
        self._genome = genome
        self._objective = objective
        self.initial_points = initial_points
        self.initial_genomes = initial_genomes
        self.acquisition = acquisition
        self.kernel = kernel
        self.noise = noise
        self.output = output
        self.raw_samples = raw_samples
        self.acquisition_starts = acquisition_starts
        self.hyperparameter_starts = hyperparameter_starts
        self.batch = batch
        self.fantasy = fantasy
        self.seed = seed

    def _describe(self) -> dict[str, Any]:
        initial_genomes = None
        if self.initial_genomes is not None:
            rows = np.asarray(self.initial_genomes, dtype=object)
            if rows.ndim != 2:
                raise ValueError(
                    "initial_genomes is a genome per row, a number per gene, not "
                    f"{self.initial_genomes!r}"
                )
            initial_genomes = [
                [_number("initial_genomes", gene, plural=True) for gene in row] for row in rows
            ]
        if self.output not in ("standardize", "log"):
            raise ValueError(f'output is "standardize" or "log", not {self.output!r}')
        return {
            "type": "bo",
            "initial_points": _optional_whole("initial_points", self.initial_points),
            "initial_genomes": initial_genomes,
            "acquisition": _acquisition(self.acquisition),
            "kernel": _surrogate._kernel(self.kernel),
            "noise": _surrogate._noise(self.noise),
            "output": self.output,
            "raw_samples": _whole("raw_samples", self.raw_samples),
            "acquisition_starts": _whole("acquisition_starts", self.acquisition_starts),
            "hyperparameter_starts": _whole("hyperparameter_starts", self.hyperparameter_starts),
            "batch": _whole("batch", self.batch),
            "fantasy": _fantasy(self.fantasy),
            "seed": _optional_whole("seed", self.seed),
        }

    def run(
        self,
        fitness: Callable[[np.ndarray], Any],
        *,
        constraints: int = 0,
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
        """Runs the search until the first stop condition, as :meth:`Ga.run`, with the values of
        ``constraints`` inequality constraints if there are any.

        Parameters
        ----------
        fitness : callable or problems.Problem
            As for :meth:`Ga.run`. With ``constraints`` above 0, it returns ``(value, g)``,
            ``g`` an array of the constraints' values, feasible at 0 or below: a Gaussian process
            models each. A problem of :mod:`genoxide.problems` with constraints gives their
            values in Rust: leave ``constraints`` out.
        constraints : int, default 0
            The number of inequality constraints whose values ``fitness`` returns.

        The other parameters are those of :meth:`Ga.run`; with constraints, ``batch`` is False:
        the function is called a point at a time, in parallel with ``parallel=True``.

        Returns
        -------
        Result
            The best solution found by Deb's rules, and what the run took.

        Raises
        ------
        ValueError
            As :meth:`Ga.run`; for constraints with ``batch=True`` or a problem of
            :mod:`genoxide.problems`, or a function that returns another number of constraint
            values; and for the upper confidence bound with constraints.
        TypeError
            As :meth:`Ga.run`; and if a function with constraints doesn't return a pair.
        """
        count = _whole("constraints", constraints)
        if count == 0:
            return _SingleObjective.run(
                self,
                fitness,
                generations=generations,
                evaluations=evaluations,
                target=target,
                time=time,
                stagnation=stagnation,
                batch=batch,
                parallel=parallel,
                on_generation=on_generation,
                control=control,
                checkpoint=checkpoint,
                checkpoint_every=checkpoint_every,
                resume=resume,
            )
        if batch:
            raise ValueError("constraints need batch=False: their values come a point at a time")
        _check_callable(fitness)
        if isinstance(fitness, (problems.Problem, problems.MultiProblem, problems.control.Balance)):
            raise ValueError(
                f"{type(fitness).__name__} is evaluated in Rust, with its own constraints: leave "
                "constraints out"
            )
        stop = _stop(generations, evaluations, target, time, stagnation)
        callback = _on_generation(on_generation, Progress)
        controls = _control(control, self)
        saving = _checkpoints(checkpoint, checkpoint_every, resume)
        return Result(
            **self._run(
                _with_values(fitness, count),
                stop,
                False,
                parallel,
                callback,
                None,
                controls,
                saving,
                constraints=count,
            )
        )


class _GradientMethod(_SingleObjective):
    """A gradient-based method: its ``run`` takes the gradient of the fitness function."""

    def run(
        self,
        fitness: Callable[[np.ndarray], Any],
        *,
        gradient: Callable[[np.ndarray], Any] | bool | None = None,
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
        """Runs the method until the first stop condition, as :meth:`Ga.run`, with the
        gradient.

        Parameters
        ----------
        fitness : callable or problems.Problem
            As for :meth:`Ga.run`. A smooth problem of :mod:`genoxide.problems` gives its
            analytic gradient, computed in Rust.
        gradient : callable or True, optional
            The gradient of the score as ``fitness`` returns it, whatever the objective:
            ``gradient(x)`` returns an array of a derivative per gene; with ``batch=True``, it
            takes the 2-D array of genomes and returns a row of derivatives per genome. True
            means ``fitness`` returns both, ``(value, gradient)`` (with ``batch=True``,
            ``(values, gradients)``), computed together. None is no gradient: forward
            differences, unless ``gradients`` says otherwise. With a gradient, the function is
            called on the thread that called ``run``, ``parallel`` or not.

        The other parameters are those of :meth:`Ga.run`.

        Returns
        -------
        Result
            The best solution found, and what the run took.

        Raises
        ------
        ValueError
            As :meth:`Ga.run`; and for a gradient with a problem of :mod:`genoxide.problems`,
            which has its own, or a gradient of the wrong length.
        TypeError
            As :meth:`Ga.run`; and if ``gradient`` isn't callable, True or None, or a function
            with ``gradient=True`` doesn't return a pair.
        """
        return self._run_gradient(
            fitness,
            gradient=gradient,
            constraints=0,
            generations=generations,
            evaluations=evaluations,
            target=target,
            time=time,
            stagnation=stagnation,
            batch=batch,
            parallel=parallel,
            on_generation=on_generation,
            control=control,
            checkpoint=checkpoint,
            checkpoint_every=checkpoint_every,
            resume=resume,
        )

    def _run_gradient(
        self,
        fitness: Callable[[np.ndarray], Any],
        *,
        gradient: Callable[[np.ndarray], Any] | bool | None,
        constraints: int,
        generations: int | None,
        evaluations: int | None,
        target: float | None,
        time: float | None,
        stagnation: int | None,
        batch: bool,
        parallel: bool,
        on_generation: Callable[[Progress], bool | None] | None,
        control: Callable[[Any, Progress], Any] | None,
        checkpoint: str | os.PathLike[str] | None,
        checkpoint_every: int | None,
        resume: str | os.PathLike[str] | None,
    ) -> Result:
        """``run`` with the gradient, and for MMA the values and Jacobian of ``constraints``
        inequality constraints."""
        count = _whole("constraints", constraints)
        if count > 0:
            return self._run_constrained(
                fitness,
                count,
                gradient,
                batch,
                parallel,
                on_generation,
                control,
                lambda: _stop(generations, evaluations, target, time, stagnation),
                lambda: _checkpoints(checkpoint, checkpoint_every, resume),
            )
        if gradient is None:
            return _SingleObjective.run(
                self,
                fitness,
                generations=generations,
                evaluations=evaluations,
                target=target,
                time=time,
                stagnation=stagnation,
                batch=batch,
                parallel=parallel,
                on_generation=on_generation,
                control=control,
                checkpoint=checkpoint,
                checkpoint_every=checkpoint_every,
                resume=resume,
            )
        if gradient is not True:
            _check_callable(gradient, "the gradient")
        _check_callable(fitness)
        if isinstance(fitness, (problems.Problem, problems.MultiProblem, problems.control.Balance)):
            raise ValueError(
                f"{type(fitness).__name__} is evaluated in Rust, with its own gradient: leave "
                "gradient out"
            )
        stop = _stop(generations, evaluations, target, time, stagnation)
        callback = _on_generation(on_generation, Progress)
        controls = _control(control, self)
        saving = _checkpoints(checkpoint, checkpoint_every, resume)
        combined = gradient is True
        function: Callable[[np.ndarray], Any]
        derivative: Callable[[np.ndarray], Any] | None = None
        if batch:
            function = _batch_with_gradients(fitness) if combined else _batch_scores(fitness)
            if not combined:
                derivative = _batch_gradient_rows(gradient)  # type: ignore[arg-type]
        else:
            function = _with_gradient(fitness) if combined else fitness
            if not combined:
                derivative = _gradient_array(gradient)  # type: ignore[arg-type]
        return Result(
            **self._run(
                function,
                stop,
                batch,
                parallel,
                callback,
                None,
                controls,
                saving,
                gradient=derivative,
                combined_gradient=combined,
            )
        )


    def _run_constrained(
        self,
        fitness: Callable[[np.ndarray], Any],
        count: int,
        gradient: Callable[[np.ndarray], Any] | bool | None,
        batch: bool,
        parallel: bool,
        on_generation: Callable[[Progress], bool | None] | None,
        control: Callable[[Any, Progress], Any] | None,
        stop_conditions: Callable[[], dict[str, Any]],
        checkpoints: Callable[[], dict[str, Any]],
    ) -> Result:
        """``run`` with ``count`` inequality constraints, for MMA: the stop conditions and the
        checkpoints checked after the constraints' settings."""
        if gradient is not True:
            raise ValueError(
                "with constraints, gradient=True: the fitness function returns (value, gradient, "
                "constraint values, jacobian)"
            )
        if batch:
            raise ValueError("constraints need batch=False: MMA evaluates one point at a time")
        _check_callable(fitness)
        if isinstance(fitness, (problems.Problem, problems.MultiProblem, problems.control.Balance)):
            raise ValueError(
                f"{type(fitness).__name__} is evaluated in Rust: leave gradient and constraints out"
            )
        stop = stop_conditions()
        callback = _on_generation(on_generation, Progress)
        controls = _control(control, self)
        saving = checkpoints()
        return Result(
            **self._run(
                _with_constraints(fitness, count),
                stop,
                False,
                parallel,
                callback,
                None,
                controls,
                saving,
                combined_gradient=True,
                constraints=count,
            )
        )


class Lbfgsb(_GradientMethod):
    """L-BFGS-B, the limited-memory BFGS method with bounds. Real genomes.

    A local method for smooth functions with a gradient, from a few genes to millions. Each
    iteration models the function by its gradient and the curvature of the last ``memory`` steps
    (a limited-memory BFGS matrix in compact form), minimizes the model within the bounds of the
    genome in two stages, the generalized Cauchy point along the projected steepest descent path,
    then a step over the genes it leaves free, projected into the bounds, and searches along the
    step with the Moré-Thuente line search. A minimum on a bound is landed on exactly. Memory and
    work per iteration grow as ``memory`` times the genes: no matrix of genes by genes.

    The gradient comes from ``gradients``: by default ("auto"), the one ``run`` is given (a
    ``gradient`` function, or ``gradient=True`` with a fitness function returning ``(value,
    gradient)``), the analytic one of a problem of :mod:`genoxide.problems` (evaluated in Rust),
    or else forward differences, which cost a point per gene for each gradient, evaluated with the
    point in one generation. A run has converged when the largest component of the projected
    gradient is within ``gradient_tolerance``, when a step lowers the function by less than
    ``function_tolerance`` of its value, or when no step lowers it: ``run`` then stops with the
    stop reason "converged", unless restarts are left. :class:`RunningLbfgsb` reads the state
    (the iterations, the criterion, the gradient evaluations) in a ``control``.

    Rosenbrock's function in 100 dimensions, with its gradient computed in Rust::

        import genoxide as gx

        problem = gx.problems.Rosenbrock(100)
        lbfgsb = gx.Lbfgsb(problem.genome, initial_genome=[-1.2, 1.0] * 50, objective="minimize")
        result = lbfgsb.run(problem, evaluations=10_000)
        print(result.stop_reason, result.best_fitness)  # converged 1.08e-09

    A Python function with its gradient::

        import numpy as np

        import genoxide as gx

        def sphere(x):
            return float(x @ x), 2 * x

        lbfgsb = gx.Lbfgsb(gx.Real((-5, 5), length=10), objective="minimize", seed=1)
        result = lbfgsb.run(sphere, gradient=True, evaluations=1_000)

    Parameters
    ----------
    genome : Real
        The search space; its bounds are the box of the method. At least one gene needs
        ``low < high`` (the others stay fixed).
    memory : int, default 10
        The correction pairs the limited-memory matrix keeps, at least 1; 3 to 20 is usual.
    gradients : {"auto", "supplied", "forward", "central"}, default "auto"
        Where the gradients come from: "auto" is the fitness function's if it gives one and
        forward differences otherwise (for at most 10,000 genes that aren't fixed); "supplied"
        requires it; "forward" (a point per gene) and "central" (two points per gene, more
        accurate) use finite differences even if a gradient is given.
    difference_step : float, optional
        The relative step of finite differences, above 0: ``h = step * max(|x|, 1)``. None is
        √ε ≈ 1.5e-8 for forward differences and ε^(1/3) ≈ 6.1e-6 for central ones.
    gradient_tolerance : float, default 1e-5
        A run has converged when the projected gradient's largest component is at most this,
        0 or more. Absolute: it depends on the scale of the function and of the genes. Forward
        differences rarely meet much less than 1e-7 of the function's scale.
    function_tolerance : float, default 2.220446049250313e-9
        A run has converged when a step lowers the function by at most this times
        ``max(|f|, 1)``, 0 or more: 10⁷ times the machine epsilon, the authors' "moderate
        accuracy". Smaller for more digits; 0 stops only when a step changes nothing.
    max_line_search : int, default 20
        The most trial steps of a line search, at least 1.
    restarts : int, optional
        Random restarts after a run converges, at least 1, each from a new random point in the
        bounds. None is no restarts.
    keep_pairs : bool, default False
        In a :class:`Continuation` that keeps the state (``keep="state"``), whether the next
        stage keeps the correction pairs, which describe the old function's curvature: for
        stages whose functions differ little.
    initial_genome : array-like of float, optional
        The point to start from, a number per gene within the bounds, e.g. the best of a global
        method, to polish it. None is a random point. Restarts start from random points.
    objective : {"maximize", "minimize"}, default "maximize"
        Whether higher or lower scores are better. A maximized function's gradient is of the
        score as returned: no sign changes.
    seed : int, optional
        The seed of the random numbers (the random start and the restarts), 0 to 2^64 - 1. None
        is a random seed. The same seed repeats the run.

    References: Byrd, R. H., Lu, P., Nocedal, J. and Zhu, C. (1995). A limited memory algorithm
    for bound constrained optimization. *SIAM Journal on Scientific Computing* 16(5): 1190-1208.
    Zhu, C., Byrd, R. H., Lu, P. and Nocedal, J. (1997). Algorithm 778: L-BFGS-B. *ACM
    Transactions on Mathematical Software* 23(4): 550-560. Morales, J. L. and Nocedal, J. (2011).
    Remark on "Algorithm 778". *ACM Transactions on Mathematical Software* 38(1): 7. Moré, J. J.
    and Thuente, D. J. (1994). Line search algorithms with guaranteed sufficient decrease. *ACM
    Transactions on Mathematical Software* 20(3): 286-307.
    """

    _running = RunningLbfgsb

    def __init__(
        self,
        genome: Real,
        *,
        memory: int = 10,
        gradients: Literal["auto", "supplied", "forward", "central"] = "auto",
        difference_step: float | None = None,
        gradient_tolerance: float = 1e-5,
        function_tolerance: float = 2.220446049250313e-9,
        max_line_search: int = 20,
        restarts: int | None = None,
        keep_pairs: bool = False,
        initial_genome: Sequence[float] | np.ndarray | None = None,
        objective: ObjectiveName = "maximize",
        seed: int | None = None,
    ) -> None:
        self._genome = genome
        self._objective = objective
        self.keep_pairs = keep_pairs
        self.memory = memory
        self.gradients = gradients
        self.difference_step = difference_step
        self.gradient_tolerance = gradient_tolerance
        self.function_tolerance = function_tolerance
        self.max_line_search = max_line_search
        self.restarts = restarts
        self.initial_genome = initial_genome
        self.seed = seed

    def _describe(self) -> dict[str, Any]:
        if self.gradients not in ("auto", "supplied", "forward", "central"):
            raise ValueError(
                'gradients is "auto", "supplied", "forward" or "central", not '
                f"{self.gradients!r}"
            )
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
            "type": "lbfgsb",
            "memory": _whole("memory", self.memory, minimum=None),
            "gradients": self.gradients,
            "difference_step": _optional_number("difference_step", self.difference_step),
            "gradient_tolerance": _number("gradient_tolerance", self.gradient_tolerance),
            "function_tolerance": _number("function_tolerance", self.function_tolerance),
            "max_line_search": _whole("max_line_search", self.max_line_search, minimum=None),
            "restarts": restarts,
            "keep_pairs": _flag("keep_pairs", self.keep_pairs),
            "initial_genome": initial_genome,
            "seed": _optional_whole("seed", self.seed),
        }


_STEPS = ("gradient", "momentum", "nesterov", "adam", "adamw")
_GRADIENTS = ("auto", "supplied", "forward", "central")


class FirstOrder(_GradientMethod):
    """A first-order method: gradient descent, momentum, Nesterov, Adam or AdamW. Real genomes.

    Steps along the gradient of the score, with the step's size a setting rather than found by a
    line search: for smooth problems with many variables, up to millions, where a line search
    costs too much, such as fitting a model's parameters. Memory and work per step are linear in
    the genes. Each generation evaluates the point and its gradient: supplied by ``run``'s
    ``gradient``, or by a problem of :mod:`genoxide.problems` in Rust, one evaluation each; or by
    finite differences, ``n`` more evaluations forward (``2n`` central) in the same generation, up
    to 10,000 genes with ``gradients="auto"``. The gradient is of the score as returned: with the
    objective "maximize", the method climbs it.

    The step rules, with ``g`` the gradient (its sign turned when maximizing), ``alpha`` the
    learning rate and ``eta`` the schedule multiplier (1 unless a ``control`` changes it), per
    gene:

    - "gradient": ``x -= eta alpha g``.
    - "momentum", Polyak's heavy ball: ``v = momentum v - eta alpha g``, ``x += v``.
    - "nesterov", Nesterov's accelerated gradient in Sutskever et al.'s form: the same, with the
      gradient taken at the look-ahead point ``x + momentum v``, which is the point evaluated.
    - "adam", Kingma and Ba's Algorithm 1: averages ``m`` and ``v`` of the gradient and of its
      square with the decays ``beta1`` and ``beta2``, corrected for their start at 0 (``m_hat``,
      ``v_hat``), and ``x -= eta alpha m_hat / (sqrt(v_hat) + epsilon)``: steps of about
      ``alpha`` per gene whatever the size of the gradient.
    - "adamw", Loshchilov and Hutter's Algorithm 2: Adam with decoupled weight decay,
      ``x -= eta (alpha m_hat / (sqrt(v_hat) + epsilon) + weight_decay x)``. Adam with the decay
      in the gradient (L2 regularization) is another method: its decay shrinks where the
      gradient is large.

    Each new point is projected onto the bounds, gene by gene (a projected gradient method: a
    minimum on a bound is landed on, and kept while the gradient points out). A run has
    converged when the projected gradient's largest component is within ``gradient_tolerance``,
    or the last step moved no gene by more than ``step_tolerance`` (relative to ``max(1,
    |x|)``); without restarts, ``run`` then stops with the stop reason "converged". An invalid
    point (an invalid fitness, or a gradient that isn't finite) moves the method halfway back to
    the last valid one. A ``control`` changes the learning rate and the multiplier between steps
    (:class:`RunningFirstOrder`), for a learning-rate schedule.

    A least-squares fit, with its gradient::

        import numpy as np
        import genoxide as gx

        t = np.linspace(0, 1, 50)
        y = 2 * t - 1

        def loss(p):
            return float(np.sum((p[0] * t + p[1] - y) ** 2))

        def gradient(p):
            r = p[0] * t + p[1] - y
            return np.array([2 * np.sum(r * t), 2 * np.sum(r)])

        adam = gx.FirstOrder(
            gx.Real((-5, 5), length=2), step="adam", learning_rate=0.05, objective="minimize"
        )
        result = adam.run(loss, gradient=gradient, generations=20_000)
        print(result.stop_reason, result.best_genome)  # converged, near [2, -1]

    Parameters
    ----------
    genome : Real
        The search space; the rules are sensitive to the scale of the genes and of the score, so
        scale the genes alike.
    step : {"gradient", "momentum", "nesterov", "adam", "adamw"}, default "adam"
        The step rule.
    learning_rate : float, optional
        alpha, greater than 0, in the units of the genes: required for "gradient", "momentum"
        and "nesterov"; Kingma and Ba's 0.001 by default for "adam" and "adamw".
    momentum : float, optional
        mu, how much of the velocity each step keeps, in [0, 1): required for "momentum" and
        "nesterov", and only for them. 0.9 is common.
    beta1, beta2, epsilon : float, optional
        Adam's decays, in [0, 1), and its epsilon, greater than 0: Kingma and Ba's 0.9, 0.999 and
        1e-8 by default. Only for "adam" and "adamw".
    weight_decay : float, optional
        AdamW's decay lambda per step, 0 or more: required for "adamw", and only for it.
    gradients : {"auto", "supplied", "forward", "central"}, default "auto"
        Where the gradients come from: "auto" is the supplied gradient if there is one (``run``'s
        ``gradient``, or a problem's), forward differences otherwise; "supplied" requires one;
        "forward" and "central" are finite differences even with a supplied gradient.
    difference_step : float, optional
        The relative step of finite differences, greater than 0: about 1.5e-8 forward and 6.1e-6
        central by default. Only with "forward" or "central".
    gradient_tolerance : float, default 1e-6
        The largest component of the projected gradient at which a run has converged, 0 or
        more. Forward differences can't bring it much below about 1e-8 times the score.
    step_tolerance : float, default 1e-12
        The step at which a run has converged, 0 or more: the largest change of a gene in the
        last step, relative to ``max(1, |x|)``.
    restarts : int, optional
        Random restarts after a run converges, at least 1, each from a random point with the
        velocity and Adam's averages reset. None is no restarts.
    initial_genome : array-like of float, optional
        The point to start from, a number per gene within the bounds. None is a random point.
    objective : {"maximize", "minimize"}, default "maximize"
        Whether higher or lower scores are better.
    seed : int, optional
        The seed of the random numbers (the random start and the restarts), 0 to 2^64 - 1. None
        is a random seed. The same seed repeats the run.

    References: Polyak, B. T. (1964). Some methods of speeding up the convergence of iteration
    methods. *USSR Computational Mathematics and Mathematical Physics* 4(5): 1-17. Nesterov, Y.
    (1983). A method for solving the convex programming problem with convergence rate O(1/k^2).
    *Soviet Mathematics Doklady* 27: 372-376. Sutskever, I., Martens, J., Dahl, G. and Hinton,
    G. (2013). On the importance of initialization and momentum in deep learning. *ICML 2013*,
    eqs. 1-4. Kingma, D. P. and Ba, J. (2015). Adam: a method for stochastic optimization.
    *ICLR 2015*, arXiv:1412.6980. Loshchilov, I. and Hutter, F. (2019). Decoupled weight decay
    regularization. *ICLR 2019*, arXiv:1711.05101.
    """

    _running = RunningFirstOrder

    def __init__(
        self,
        genome: Real,
        *,
        step: Literal["gradient", "momentum", "nesterov", "adam", "adamw"] = "adam",
        learning_rate: float | None = None,
        momentum: float | None = None,
        beta1: float | None = None,
        beta2: float | None = None,
        epsilon: float | None = None,
        weight_decay: float | None = None,
        gradients: Literal["auto", "supplied", "forward", "central"] = "auto",
        difference_step: float | None = None,
        gradient_tolerance: float = 1e-6,
        step_tolerance: float = 1e-12,
        restarts: int | None = None,
        initial_genome: Sequence[float] | np.ndarray | None = None,
        objective: ObjectiveName = "maximize",
        seed: int | None = None,
    ) -> None:
        self._genome = genome
        self._objective = objective
        self.step = step
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.gradients = gradients
        self.difference_step = difference_step
        self.gradient_tolerance = gradient_tolerance
        self.step_tolerance = step_tolerance
        self.restarts = restarts
        self.initial_genome = initial_genome
        self.seed = seed

    def _step(self) -> dict[str, Any]:
        step = self.step
        if step not in _STEPS:
            raise ValueError(
                'step is "gradient", "momentum", "nesterov", "adam" or "adamw", not ' f"{step!r}"
            )
        adam = step in ("adam", "adamw")
        given = {
            "momentum": self.momentum,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "weight_decay": self.weight_decay,
        }
        allowed = {
            "gradient": (),
            "momentum": ("momentum",),
            "nesterov": ("momentum",),
            "adam": ("beta1", "beta2", "epsilon"),
            "adamw": ("beta1", "beta2", "epsilon", "weight_decay"),
        }[step]
        for name, value in given.items():
            if value is not None and name not in allowed:
                raise ValueError(f'{name} doesn\'t go with step="{step}"')
        required = [] if adam else ["learning_rate"]
        required += [name for name in ("momentum", "weight_decay") if name in allowed]
        for name in required:
            if getattr(self, name) is None:
                raise ValueError(f'step="{step}" needs {name}')
        description: dict[str, Any] = {"type": step}
        description["learning_rate"] = _optional_number("learning_rate", self.learning_rate)
        for name in allowed:
            description[name] = _optional_number(name, given[name])
        return description

    def _describe(self) -> dict[str, Any]:
        if self.gradients not in _GRADIENTS:
            raise ValueError(
                'gradients is "auto", "supplied", "forward" or "central", not '
                f"{self.gradients!r}"
            )
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
            "type": "first_order",
            "step": self._step(),
            "gradients": self.gradients,
            "difference_step": _optional_number("difference_step", self.difference_step),
            "gradient_tolerance": _number("gradient_tolerance", self.gradient_tolerance),
            "step_tolerance": _number("step_tolerance", self.step_tolerance),
            "restarts": restarts,
            "initial_genome": initial_genome,
            "seed": _optional_whole("seed", self.seed),
        }


class Mma(_GradientMethod):
    """The method of moving asymptotes (MMA), or its globally convergent form (GCMMA). Real
    genomes.

    For smooth problems with very many variables (up to millions) and few inequality constraints
    ``g(x) <= 0`` (up to a few hundred), with the gradient of the score and of every constraint.
    Each iteration replaces the objective and every constraint by a convex, separable
    approximation around the current point, ``r + Σ p/(u − x) + q/(x − l)`` with two asymptotes
    ``l < x < u`` per gene that move with the iterates (closer where a gene oscillates, farther
    where it moves steadily), and solves the approximate problem through its dual, in the
    constraints' multipliers: an iteration costs a few passes over the genes, with no matrix of
    them. ``method="gcmma"`` accepts a point only once the approximations are conservative there,
    and otherwise adds curvature and solves again (another evaluation): convergence from any
    start. Plain MMA is faster where it converges; around a minimum inside the bounds where the
    objective's gradient vanishes, it can cycle, and GCMMA converges. Each constraint is relaxed
    by an artificial variable at ``constraint_cost`` per unit, so every subproblem has a
    solution: the cost must exceed the constraints' multipliers. Equality constraints aren't
    supported. A run stops with the stop reason "converged" when the KKT residual is within
    ``kkt_tolerance`` or a step within ``step_tolerance``; one that converges to a point
    infeasible by rounding ends with a restoration step onto the feasible side of its active
    constraints. The best solution is the best evaluated by Deb's rules, as everywhere in the
    package. :class:`RunningMma` reads the state in a ``control``.

    ``run`` takes the gradient as :class:`Lbfgsb` does (a ``gradient`` function, or
    ``gradient=True`` with a fitness function returning ``(value, gradient)``), or a smooth
    problem of :mod:`genoxide.problems`, evaluated in Rust with its gradient; MMA doesn't use
    finite differences. With ``constraints=m``, ``gradient=True`` and the fitness function
    returns ``(value, gradient, g, jacobian)``: ``g`` the constraints' values (feasible at 0 or
    below) and ``jacobian`` an array of a row per constraint, its gradient.

    Minimize the sum of ``c / x`` subject to ``sum(x) <= 10``; the minimum is at
    ``x = 10 sqrt(c) / sum(sqrt(c))``::

        import numpy as np

        import genoxide as gx

        c = np.array([1.0, 4.0, 9.0, 16.0])

        def volume(x):
            return (c / x).sum(), -c / x**2, np.array([x.sum() - 10.0]), np.ones((1, 4))

        mma = gx.Mma(gx.Real((0.1, 10), length=4), initial_genome=[1.0] * 4, objective="minimize")
        result = mma.run(volume, gradient=True, constraints=1, evaluations=500)
        print(result.stop_reason, result.best_genome)  # converged [1. 2. 3. 4.]

    Parameters
    ----------
    genome : Real
        The search space. At least one gene needs ``low < high``; fixed genes stay as they are.
    method : {"mma", "gcmma"}, default "mma"
        MMA, one evaluation per iteration, or its globally convergent form.
    asymptote_initial : float, default 0.5
        The asymptotes' distance from the point in the first two iterations, as a fraction of each
        gene's range, above 0 (Svanberg's ``asyinit``). Smaller is more conservative.
    asymptote_decrease : float, default 0.7
        The factor that brings a gene's asymptotes nearer when it oscillates, above 0 and at most
        1 (``asydecr``).
    asymptote_increase : float, default 1.2
        The factor that moves them away when it moves the same way twice, at least 1
        (``asyincr``).
    move_limit : float, default 0.5
        The largest step of a gene in an iteration, as a fraction of its range, above 0
        (``move``).
    constraint_cost : float, default 1000
        The cost per unit of the artificial variable that relaxes each constraint, above 0: it
        must exceed the constraints' multipliers at the solution, which depend on the scaling of
        the score and the constraints.
    kkt_tolerance : float, default 1e-9
        The KKT residual (the root mean square of the KKT conditions' residuals, in the score's
        units) at which a run has converged, at least 0.
    step_tolerance : float, default 1e-10
        The step, as a fraction of each gene's range, at which a run has converged, at least 0.
    restoration : bool, default True
        Whether a run that converges to an infeasible point ends with the restoration step.
    parallel_sums : bool, default False
        Whether the dual's sums over the genes run on all cores, in fixed chunks: the same results
        as without it. It pays off from about 10^5 genes.
    initial_genome : array-like of float, optional
        The point to start from, a number per gene within the bounds. None is a random point.
    objective : {"maximize", "minimize"}, default "maximize"
        Whether higher or lower scores are better. The gradient is of the score as the function
        returns it.
    seed : int, optional
        The seed of the random start, 0 to 2^64 - 1. None is a random seed.

    References: Svanberg, K. (1987). The method of moving asymptotes: a new method for structural
    optimization. *International Journal for Numerical Methods in Engineering* 24(2): 359-373.
    Svanberg, K. (2002). A class of globally convergent optimization methods based on
    conservative convex separable approximations. *SIAM Journal on Optimization* 12(2): 555-573.
    Svanberg, K. (2007). *MMA and GCMMA – two methods for nonlinear optimization.* Notes, KTH.
    """

    _running = RunningMma

    def __init__(
        self,
        genome: Real,
        *,
        method: Literal["mma", "gcmma"] = "mma",
        asymptote_initial: float = 0.5,
        asymptote_decrease: float = 0.7,
        asymptote_increase: float = 1.2,
        move_limit: float = 0.5,
        constraint_cost: float = 1000.0,
        kkt_tolerance: float = 1e-9,
        step_tolerance: float = 1e-10,
        restoration: bool = True,
        parallel_sums: bool = False,
        initial_genome: Sequence[float] | np.ndarray | None = None,
        objective: ObjectiveName = "maximize",
        seed: int | None = None,
    ) -> None:
        self._genome = genome
        self._objective = objective
        self.method = method
        self.asymptote_initial = asymptote_initial
        self.asymptote_decrease = asymptote_decrease
        self.asymptote_increase = asymptote_increase
        self.move_limit = move_limit
        self.constraint_cost = constraint_cost
        self.kkt_tolerance = kkt_tolerance
        self.step_tolerance = step_tolerance
        self.restoration = restoration
        self.parallel_sums = parallel_sums
        self.initial_genome = initial_genome
        self.seed = seed

    def _describe(self) -> dict[str, Any]:
        if self.method not in ("mma", "gcmma"):
            raise ValueError(f'method is "mma" or "gcmma", not {self.method!r}')
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
            "type": "mma",
            "method": self.method,
            "asymptote_initial": _number("asymptote_initial", self.asymptote_initial),
            "asymptote_decrease": _number("asymptote_decrease", self.asymptote_decrease),
            "asymptote_increase": _number("asymptote_increase", self.asymptote_increase),
            "move_limit": _number("move_limit", self.move_limit),
            "constraint_cost": _number("constraint_cost", self.constraint_cost),
            "kkt_tolerance": _number("kkt_tolerance", self.kkt_tolerance),
            "step_tolerance": _number("step_tolerance", self.step_tolerance),
            "restoration": _flag("restoration", self.restoration),
            "parallel_sums": _flag("parallel_sums", self.parallel_sums),
            "initial_genome": initial_genome,
            "seed": _optional_whole("seed", self.seed),
        }

    def run(
        self,
        fitness: Callable[[np.ndarray], Any],
        *,
        gradient: Callable[[np.ndarray], Any] | bool | None = None,
        constraints: int = 0,
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
        """Runs MMA until it converges or the first stop condition, as :meth:`Lbfgsb.run`, with
        the constraints.

        Parameters
        ----------
        fitness : callable or problems.Problem
            As for :meth:`Lbfgsb.run`. With ``constraints`` above 0, a function that returns
            ``(value, gradient, g, jacobian)``: the gradient an array of a number per gene,
            ``g`` an array of the constraints' values (feasible at 0 or below) and ``jacobian``
            an array of a row per constraint, a number per gene in each. A NaN value is an
            invalid point, which MMA doesn't accept.
        gradient : callable or True
            As for :meth:`Lbfgsb.run`, and required: True with ``constraints`` above 0. None
            only for a problem of :mod:`genoxide.problems` with a gradient.
        constraints : int, default 0
            The number of inequality constraints whose values and Jacobian ``fitness`` returns.

        The other parameters are those of :meth:`Ga.run`; with constraints, ``batch`` is False.
        MMA evaluates one point per generation, so ``batch`` and ``parallel`` gain nothing.

        Returns
        -------
        Result
            The best solution found, and what the run took.

        Raises
        ------
        ValueError
            As :meth:`Lbfgsb.run`; and for constraints with ``gradient`` other than True or with
            ``batch=True``, or a function's constraint values of another number.
        TypeError
            As :meth:`Lbfgsb.run`; and if a function with constraints doesn't return a tuple of
            four.
        """
        return self._run_gradient(
            fitness,
            gradient=gradient,
            constraints=constraints,
            generations=generations,
            evaluations=evaluations,
            target=target,
            time=time,
            stagnation=stagnation,
            batch=batch,
            parallel=parallel,
            on_generation=on_generation,
            control=control,
            checkpoint=checkpoint,
            checkpoint_every=checkpoint_every,
            resume=resume,
        )


class Continuation(_GradientMethod):
    """A gradient method run through stages of one problem, its state kept between them. Real
    genomes.

    Some smooth problems are solved best in stages: a smooth version first, then sharper ones,
    each started from the last stage's result. Between stages a parameter of the fitness function
    changes (the sharpness of a smoothed maximum or absolute value, the steepness of a projection,
    a penalty weight), and the method goes on where it was: :class:`FirstOrder` with its velocity
    or Adam's averages and step count, :class:`Mma` with its asymptotes, :class:`Lbfgsb` along
    steepest descent again (its curvature pairs dropped, unless ``keep_pairs``).

    A stage ends when the method has converged, or after ``generations`` generations. Then
    ``on_stage`` is called with the next stage's index, to set its parameters, which the fitness
    function reads; the method's point is evaluated again on the changed function, and it goes on
    from there. The run stops with the stop reason "converged" after the last stage, not when a
    stage converges; the stop conditions of ``run`` apply to the whole run. The result's
    ``stages`` say what each stage did.

    The smoothed maximum of ``|x - a|`` over a few points, sharper in each stage::

        import numpy as np

        import genoxide as gx

        points = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        p = 2

        def smoothed(x):
            # (Σ |x − a|^2p)^(1/p): the largest squared distance, smoothed
            g = np.sum((x - points) ** 2, axis=1)
            total = np.sum(g**p)
            gradient = total ** (1 / p - 1) * (g ** (p - 1)) @ (2 * (x - points))
            return float(total ** (1 / p)), gradient

        def stage(index):
            global p
            p = [2, 4, 8, 16][index]

        adam = gx.FirstOrder(gx.Real((-2, 2), length=2), step="adam", learning_rate=0.05,
                             initial_genome=[1.5, -0.5], objective="minimize")
        continuation = gx.Continuation(adam, stages=4, on_stage=stage)
        result = continuation.run(smoothed, gradient=True, generations=10_000)
        print(result.stop_reason, [s.generations for s in result.stages])

    Parameters
    ----------
    algorithm : FirstOrder, Lbfgsb or Mma
        The method, with its settings: the first stage starts as it would.
    stages : int
        The number of stages, at least 1.
    on_stage : callable
        Called as ``on_stage(index)`` with a stage's index, from 0, to set its parameters, e.g. a
        variable the fitness function reads: when a run starts (with the stage it's in: 0, or the
        stage of the checkpoint it resumes from), and when each next stage begins, before its
        first evaluation. It must set everything from the index alone. An exception stops the run,
        and ``run`` raises it.
    generations : int, optional
        The most generations of each stage, at least 1: the method's iterations, without the
        re-evaluation that starts a stage. None runs each stage until the method converges.
    keep : {"state", "point"}, default "state"
        What the method keeps between stages: its state (Adam's averages and step count, the
        velocity of momentum, MMA's asymptotes and the iterates they're updated from), or only the
        point, the state starting again as at the start of a run.
    on_stage_finished : callable, optional
        Called as ``on_stage_finished(stage, point)`` with each finished :class:`Stage` and the
        method's last point, a 1-D array, before the next stage's ``on_stage``. An exception stops
        the run, and ``run`` raises it.

    A ``control`` gets the wrapped method's handle (:class:`RunningFirstOrder`,
    :class:`RunningLbfgsb` or :class:`RunningMma`). A checkpoint holds the stage and resumes in
    it, with the results the run would have had without the interruption: ``on_stage`` is called
    with that stage when the resumed run starts.
    """

    def __init__(
        self,
        algorithm: FirstOrder | Lbfgsb | Mma,
        *,
        stages: int,
        on_stage: Callable[[int], Any],
        generations: int | None = None,
        keep: Literal["state", "point"] = "state",
        on_stage_finished: Callable[[Stage, np.ndarray], Any] | None = None,
    ) -> None:
        self.algorithm = algorithm
        self.stages = stages
        self.on_stage = on_stage
        self.generations = generations
        self.keep = keep
        self.on_stage_finished = on_stage_finished

    @property
    def _genome(self) -> Genome:  # type: ignore[override]
        return self.algorithm._genome

    @property
    def _objective(self) -> ObjectiveName:  # type: ignore[override]
        return self.algorithm._objective

    @property
    def _running(self) -> type[Running]:  # type: ignore[override]
        return self.algorithm._running

    def _describe(self) -> dict[str, Any]:
        if not isinstance(self.algorithm, (FirstOrder, Lbfgsb, Mma)):
            raise ValueError(
                f"algorithm is a gx.FirstOrder, gx.Lbfgsb or gx.Mma, not {self.algorithm!r}"
            )
        if self.keep not in ("state", "point"):
            raise ValueError(f'keep is "state" or "point", not {self.keep!r}')
        stages = _whole("stages", self.stages, minimum=1)
        generations = None
        if self.generations is not None:
            generations = _whole("generations", self.generations, minimum=1)
        return {
            "type": "continuation",
            "algorithm": self.algorithm._describe(),
            "stages": stages,
            "generations": generations,
            "keep": self.keep,
        }

    def _stage_callbacks(self) -> dict[str, Any]:
        _check_callable(self.on_stage, "on_stage")
        callbacks: dict[str, Any] = {"on_stage": self.on_stage}
        finished = self.on_stage_finished
        if finished is not None:
            _check_callable(finished, "on_stage_finished")

            def on_stage_finished(stage: dict[str, Any], point: np.ndarray) -> None:
                finished(Stage(**stage), point)

            callbacks["on_stage_finished"] = on_stage_finished
        return callbacks

    def run(
        self,
        fitness: Callable[[np.ndarray], Any],
        *,
        gradient: Callable[[np.ndarray], Any] | bool | None = None,
        constraints: int = 0,
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
        """Runs the method through the stages, until the last has ended or the first stop
        condition, as the wrapped method's ``run``.

        Parameters
        ----------
        fitness : callable
            As for :meth:`Lbfgsb.run`: a function that reads the stage's parameters, which
            ``on_stage`` sets.
        gradient : callable or True, optional
            As for :meth:`Lbfgsb.run`.
        constraints : int, default 0
            With an :class:`Mma`, as for :meth:`Mma.run`.

        The other parameters are those of :meth:`Ga.run`. ``generations`` here is the run's,
        not a stage's.

        Returns
        -------
        Result
            The best solution of the last stage the run reached, by its function, what the run
            took, and its ``stages``.

        Raises
        ------
        ValueError
            As the wrapped method's ``run``; and for constraints without an :class:`Mma`.
        TypeError
            As the wrapped method's ``run``; and if ``on_stage`` or ``on_stage_finished`` isn't
            callable.
        """
        if constraints != 0 and not isinstance(self.algorithm, Mma):
            raise ValueError("constraints need a gx.Mma")
        return self._run_gradient(
            fitness,
            gradient=gradient,
            constraints=constraints,
            generations=generations,
            evaluations=evaluations,
            target=target,
            time=time,
            stagnation=stagnation,
            batch=batch,
            parallel=parallel,
            on_generation=on_generation,
            control=control,
            checkpoint=checkpoint,
            checkpoint_every=checkpoint_every,
            resume=resume,
        )


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
from . import model  # noqa: E402
from .model import gp as _surrogate  # noqa: E402
