"""Genetic programming: trees of genoxide's built-in primitives, evolved and evaluated in Rust.

A genetic program is a tree of primitives: functions (``add``, ``if``) whose children are their
arguments, and terminals (inputs such as ``x``) and constants at the leaves (Koza 1992). Evolved
by a :class:`genoxide.Ga` with the tree operators here, it finds formulas that fit data (symbolic
regression) or Boolean functions, as readable expressions::

    import genoxide as gx

    # Koza's quartic, x^4 + x^3 + x^2 + x, from 20 points: its function set and data
    problem = gx.gp.regression.problems.Koza1()
    gp = gx.gp.Gp(problem.primitives())
    ga = gx.Ga(
        gp,
        population_size=500,
        # Koza's even division among the depths and methods
        initial_genomes=gp.ramped_half_and_half(500, seed=2),
        select=gx.Tournament(7),
        crossover=gx.gp.SubtreeCrossover(),
        mutation=gx.gp.SubtreeMutation(),
        mutation_rate=0.1,
        objective="minimize",
        seed=2,
    )
    # the RMSE of the tree itself, evaluated in Rust, with no Python call per tree
    regression = problem.regression(linear_scaling=False)
    result = ga.run(regression, target=1e-10, generations=100)
    print(result.best_genome)  # the quartic, found in 8 generations

The pieces:

- :class:`PrimitiveSet`: the functions, terminals and constants, from
  :func:`genoxide.gp.regression.primitives` (mathematical functions and named variables, with
  :class:`Constants`) or a problem's ``primitives()``. Its ``parse`` reads a tree.
- :class:`Gp`: the genome, a set with Koza's depth limit of 17, a size limit of 1024 nodes and
  the initialization, :class:`RampedHalfAndHalf` of depths 2 to 6 by default (or :class:`Full`,
  :class:`Grow`).
- :class:`Tree`: a genome as a fitness function gets it, in Rust: ``len(tree)``, ``depth``,
  ``str(tree)`` (``add(x, mul(x, 0.5))``), and ``evaluate(x)``, its values at points, a row
  each, on numpy arrays.
- The operators: :class:`SubtreeCrossover` and :class:`OnePointCrossover`;
  :class:`SubtreeMutation`, :class:`PointMutation`, :class:`HoistMutation`,
  :class:`ShrinkMutation`, :class:`ConstantMutation` and a mix of them by weight,
  :class:`Mutations`. Against bloat (trees that grow without getting better),
  :class:`genoxide.DoubleTournament`, :class:`genoxide.LexicographicTournament` and
  :class:`genoxide.Tarpeian` select by size too.
- The fitness functions evaluated in Rust: :class:`genoxide.gp.regression.Regression` (the
  error on data, after linear scaling), its test problems in
  :mod:`genoxide.gp.regression.problems`, and the Boolean problems of
  :mod:`genoxide.gp.boolean`. :class:`WithSize` adds the size as a second objective, for
  :class:`genoxide.Nsga2`. Any Python function of a :class:`Tree` is a fitness function too.

Trees run with :class:`genoxide.Ga` (any scheme), :class:`genoxide.Islands` of them and
:class:`genoxide.Nsga2` with two objectives; ``Ga(initial_genomes=gp.ramped_half_and_half(n,
seed))`` starts from Koza's even division among the depths and methods. The primitives are
genoxide's built-in ones; primitives of your own come in a later version.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Union

from .. import _genoxide, _number, _whole

__all__ = [
    "PrimitiveSet",
    "Tree",
    "Constants",
    "Gp",
    "Full",
    "Grow",
    "RampedHalfAndHalf",
    "Init",
    "SubtreeCrossover",
    "OnePointCrossover",
    "SubtreeMutation",
    "PointMutation",
    "HoistMutation",
    "ShrinkMutation",
    "ConstantMutation",
    "Mutations",
    "TreeMutation",
    "Crossover",
    "Mutation",
    "WithSize",
    "regression",
    "boolean",
]

PrimitiveSet = _genoxide.PrimitiveSet
Tree = _genoxide.Tree
WithSize = _genoxide.WithSize


@dataclass(frozen=True)
class Constants:
    """Ephemeral random constants (Koza 1992): a constant leaf draws its value once, when it's
    made, and keeps it. Make them with :meth:`uniform`, :meth:`integers`, :meth:`choice` or
    :meth:`normal`, for :func:`genoxide.gp.regression.primitives`."""

    kind: str
    values: tuple[float, ...]

    @classmethod
    def uniform(cls, low: float, high: float) -> Constants:
        """Uniform over ``[low, high]``: finite, with ``low <= high`` and a finite width."""
        return cls("uniform", (low, high))

    @classmethod
    def integers(cls, low: int, high: int) -> Constants:
        """Whole numbers uniform over ``[low, high]``, within ±2^53, ``low <= high``."""
        return cls("integers", (low, high))

    @classmethod
    def choice(cls, values: Sequence[float]) -> Constants:
        """One of ``values``, at least one and each finite, with the same probability."""
        return cls("choice", tuple(values))

    @classmethod
    def normal(cls, mean: float, deviation: float) -> Constants:
        """Normal of ``mean`` (finite) and standard ``deviation`` (positive and finite), e.g.
        Keijzer's N(0, 5)."""
        return cls("normal", (mean, deviation))

    def _describe(self) -> dict[str, Any]:
        name = f"Constants.{self.kind}"
        if self.kind in ("uniform", "normal") and len(self.values) == 2:
            first, second = (_number(name, value, plural=True) for value in self.values)
            keys = ("low", "high") if self.kind == "uniform" else ("mean", "deviation")
            return {"type": self.kind, keys[0]: first, keys[1]: second}
        if self.kind == "integers" and len(self.values) == 2:
            low, high = (
                _whole(name, value, minimum=-(2**63), maximum=2**63 - 1, plural=True)
                for value in self.values
            )
            return {"type": "integers", "low": low, "high": high}
        if self.kind == "choice":
            return {
                "type": "choice",
                "values": [_number(name, value, plural=True) for value in self.values],
            }
        raise ValueError(
            "constants are Constants.uniform, Constants.integers, Constants.choice or "
            f"Constants.normal, not {self!r}"
        )


# --- the representation --------------------------------------------------------------------------


def _depths(name: str, depths: Any) -> list[int]:
    try:
        low, high = depths
    except (TypeError, ValueError):
        raise ValueError(f"{name}.depths is a pair (lowest, highest), not {depths!r}") from None
    return [_whole(f"{name}.depths", low, plural=True), _whole(f"{name}.depths", high, plural=True)]


@dataclass(frozen=True)
class Full:
    """Koza's full method: functions down to a depth drawn uniformly from ``depths``, ``(lowest,
    highest)`` inclusive, then terminals."""

    depths: tuple[int, int] = (2, 6)

    def _describe(self) -> dict[str, Any]:
        return {"type": "full", "depths": _depths("Full", self.depths)}


@dataclass(frozen=True)
class Grow:
    """Koza's grow method: any primitive that fits the remaining depth, drawn uniformly from
    ``depths``, ``(lowest, highest)`` inclusive: trees of every shape up to it."""

    depths: tuple[int, int] = (2, 6)

    def _describe(self) -> dict[str, Any]:
        return {"type": "grow", "depths": _depths("Grow", self.depths)}


@dataclass(frozen=True)
class RampedHalfAndHalf:
    """Koza's ramped half-and-half: :class:`Full` or :class:`Grow`, each with probability 1/2, of
    a depth drawn uniformly from ``depths``, ``(lowest, highest)`` inclusive. Koza divides the
    population evenly among the depths and the two methods: :meth:`Gp.ramped_half_and_half` does
    exactly that."""

    depths: tuple[int, int] = (2, 6)

    def _describe(self) -> dict[str, Any]:
        return {"type": "ramped_half_and_half", "depths": _depths("RampedHalfAndHalf", self.depths)}


Init = Union[Full, Grow, RampedHalfAndHalf]


@dataclass(frozen=True)
class Gp:
    """Trees of a genetic program: the genome of a run on trees, whose fitness function gets a
    :class:`Tree`.

    Every tree the run makes is of the set's types and within the limits.

    Parameters
    ----------
    primitives : PrimitiveSet
        The functions, terminals and constants, e.g. from
        :func:`genoxide.gp.regression.primitives` or a problem's ``primitives()``.
    max_depth : int, default 17
        The largest depth of a tree, the root at depth 0 (Koza's limit), at most 2^24.
    max_size : int, default 1024
        The most nodes of a tree, 1 to 2^24: a memory guard beside the depth limit.
    init : Full, Grow or RampedHalfAndHalf, default RampedHalfAndHalf((2, 6))
        How random trees are made, with the range of their depths, within ``max_depth``.
    """

    primitives: PrimitiveSet
    max_depth: int = 17
    max_size: int = 1024
    init: Init = RampedHalfAndHalf()

    def _describe(self) -> dict[str, Any]:
        if not isinstance(self.primitives, PrimitiveSet):
            raise ValueError(
                "Gp.primitives is a gx.gp.PrimitiveSet, e.g. from "
                f"gx.gp.regression.primitives(...), not {self.primitives!r}"
            )
        if not isinstance(self.init, (Full, Grow, RampedHalfAndHalf)):
            raise ValueError(
                f"Gp.init is gx.gp.RampedHalfAndHalf, Full or Grow, not {self.init!r}"
            )
        return {
            "type": "gp",
            "primitives": json.loads(self.primitives._json()),
            "max_depth": _whole("Gp.max_depth", self.max_depth),
            "max_size": _whole("Gp.max_size", self.max_size),
            "init": self.init._describe(),
        }

    def _json(self) -> str:
        return json.dumps(self._describe())

    def check(self) -> None:
        """Checks the settings: a ``ValueError`` that names a wrong one, e.g. depths of the
        initialization beyond ``max_depth``."""
        _genoxide.gp_check(self._json())

    def ramped_half_and_half(self, count: int, seed: int) -> list[Tree]:
        """``count`` trees of Koza's ramped half-and-half, from ``seed``: divided evenly among
        the depths of ``init`` and the full and grow methods, without duplicates. The trees that
        genoxide's Rust ``gp.ramped_half_and_half(count, &mut StreamRng::seed_from_u64(seed))``
        makes, for ``Ga(initial_genomes=...)``."""
        return _genoxide.gp_ramped_half_and_half(
            self._json(), _whole("count", count, maximum=2**24), _whole("seed", seed)
        )

    def random_genome(self, seed: int) -> Tree:
        """A random tree of ``init``, from ``seed``."""
        return _genoxide.gp_random_genome(self._json(), _whole("seed", seed))

    def validate(self, tree: Tree) -> None:
        """Checks that ``tree`` is a tree of the set, within the limits: a ``ValueError`` that
        says why if it isn't."""
        if not isinstance(tree, Tree):
            raise ValueError(f"a gx.gp.Tree, not {tree!r}")
        _genoxide.gp_validate(self._json(), tree)

    def parse(self, text: str) -> Tree:
        """The tree written in ``text`` as ``str(tree)`` writes it, e.g. ``add(mul(x, x), 0.5)``,
        checked against the limits: a ``ValueError`` if it isn't a tree of the set within
        them."""
        return _genoxide.gp_parse(self._json(), str(text))


# --- the operators --------------------------------------------------------------------------------


@dataclass(frozen=True)
class SubtreeCrossover:
    """Koza's subtree crossover: a subtree of each parent exchanged for a subtree of the same type
    of the other, both children within the limits. The first point is at a function node with
    probability ``internal_rate`` (0 to 1, Koza's 0.9 by default), else at a leaf."""

    internal_rate: float | None = None

    def _describe(self) -> dict[str, Any]:
        rate = None
        if self.internal_rate is not None:
            rate = _number("SubtreeCrossover.internal_rate", self.internal_rate)
        return {"type": "subtree", "internal_rate": rate}


@dataclass(frozen=True)
class OnePointCrossover:
    """Poli and Langdon's one-point crossover: subtrees exchanged at a point of the two trees'
    common region, where they have the same shape from the root; within the limits."""

    def _describe(self) -> dict[str, Any]:
        return {"type": "one_point"}


@dataclass(frozen=True)
class SubtreeMutation:
    """A node chosen uniformly gets a new subtree of its type, grown to depth at most
    ``max_depth`` (4 by default, at most 2^24) within the limits; never the same subtree."""

    max_depth: int | None = None

    def _describe(self) -> dict[str, Any]:
        depth = None
        if self.max_depth is not None:
            depth = _whole("SubtreeMutation.max_depth", self.max_depth)
        return {"type": "subtree", "max_depth": depth}


@dataclass(frozen=True)
class PointMutation:
    """Nodes replaced by other primitives of their signature (a leaf by another terminal or a
    new constant): each node with probability ``rate`` (greater than 0 and at most 1), or
    ``count`` of them (at least 1). Nodes without a replacement are never picked."""

    rate: float | None = None
    count: int | None = None

    def _describe(self) -> dict[str, Any]:
        if (self.rate is None) == (self.count is None):
            raise ValueError("PointMutation needs either rate (per node) or count (nodes)")
        return {
            "type": "point",
            "rate": None if self.rate is None else _number("PointMutation.rate", self.rate),
            "count": None if self.count is None else _whole("PointMutation.count", self.count),
        }


@dataclass(frozen=True)
class HoistMutation:
    """The tree replaced by one of its subtrees of its type (Kinnear): always smaller."""

    def _describe(self) -> dict[str, Any]:
        return {"type": "hoist"}


@dataclass(frozen=True)
class ShrinkMutation:
    """A function's subtree replaced by a leaf of its type (Angeline): always smaller."""

    def _describe(self) -> dict[str, Any]:
        return {"type": "shrink"}


@dataclass(frozen=True)
class ConstantMutation:
    """One constant moved by normal noise of standard deviation ``sigma`` (positive and finite)
    times its range, or the deviation of :meth:`Constants.normal` constants; mirrored at the
    ends of a range. :meth:`gaussian` makes one too."""

    sigma: float

    @classmethod
    def gaussian(cls, sigma: float) -> ConstantMutation:
        """Normal noise of standard deviation ``sigma``, as Rust's
        ``ConstantMutation::gaussian``."""
        return cls(sigma)

    def _describe(self) -> dict[str, Any]:
        return {"type": "constant", "sigma": _number("ConstantMutation.sigma", self.sigma)}


TreeMutation = Union[
    SubtreeMutation, PointMutation, HoistMutation, ShrinkMutation, ConstantMutation
]

_SINGLE = (SubtreeMutation, PointMutation, HoistMutation, ShrinkMutation, ConstantMutation)


@dataclass(frozen=True)
class Mutations:
    """A mix of tree mutations: each call applies one of ``mutations``, ``(weight, mutation)``
    pairs, chosen by weight, in order; if the chosen one can't change the tree (hoist of a single
    node, constant mutation without constants), another by weight among the rest. The weights are
    finite and at least 0, with a positive total; they needn't add up to 1. E.g.
    ``Mutations([(0.5, SubtreeMutation()), (0.3, PointMutation(count=1)), (0.1,
    HoistMutation()), (0.1, ShrinkMutation())])``, Rust's ``Mutations::builder().subtree(0.5)
    .point(0.3).hoist(0.1).shrink(0.1)``."""

    mutations: Sequence[tuple[float, TreeMutation]]

    def _describe(self) -> dict[str, Any]:
        described = []
        for index, pair in enumerate(self.mutations):
            try:
                weight, mutation = pair
            except (TypeError, ValueError):
                raise ValueError(
                    f"Mutations.mutations are (weight, mutation) pairs, not {pair!r}"
                ) from None
            if not isinstance(mutation, _SINGLE):
                raise ValueError(
                    "Mutations.mutations are (weight, mutation) pairs of SubtreeMutation, "
                    "PointMutation, HoistMutation, ShrinkMutation or ConstantMutation, not "
                    f"{mutation!r} (pair {index})"
                )
            described.append(
                [_number("Mutations.mutations", weight, plural=True), mutation._describe()]
            )
        return {"type": "mutations", "mutations": described}


Crossover = Union[SubtreeCrossover, OnePointCrossover]
"""A crossover of trees: :class:`SubtreeCrossover` or :class:`OnePointCrossover` (or
:class:`genoxide.NoCrossover`)."""

Mutation = Union[TreeMutation, Mutations]
"""A mutation of trees: one of them, or a mix, :class:`Mutations`."""


def _is_tree_fitness(fitness: Any) -> bool:
    """Whether ``fitness`` is evaluated in Rust: a regression, a regression or Boolean problem,
    or ``WithSize`` of one."""
    return isinstance(
        fitness,
        (
            _genoxide.Regression,
            _genoxide.RegressionProblem,
            _genoxide.BooleanProblem,
            _genoxide.WithSize,
        ),
    )


def _initial_genomes(genome: Any, trees: Any) -> list[Any] | None:
    """The description of a run's initial trees, checked to be trees of ``genome``'s set."""
    if trees is None:
        return None
    if not isinstance(genome, Gp):
        raise ValueError("initial_genomes are the trees of a gx.gp.Gp genome")
    described = []
    for index, tree in enumerate(trees):
        if not isinstance(tree, Tree):
            raise ValueError(f"initial_genomes are gx.gp.Tree objects, not {tree!r} ({index})")
        if tree.primitives != genome.primitives:
            raise ValueError(
                f"initial_genomes are trees of the Gp's primitive set: tree {index} isn't"
            )
        described.append(json.loads(tree._json()))
    return described


from . import boolean, regression  # noqa: E402
