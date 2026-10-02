"""Binary and combinatorial test problems, on :class:`genoxide.Binary` genomes, all maximized and
evaluated in Rust::

    import genoxide as gx

    problem = gx.problems.binary.Trap(10, 4)
    ga = gx.Ga(
        problem.genome,
        population_size=1000,
        select=gx.Tournament(4),
        crossover=gx.PointCrossover(2),
        mutation=gx.BitFlip(rate=1 / problem.dimensions),
        seed=1,
    )
    result = ga.run(problem, target=problem.optimum.value, generations=300)

The problems are :class:`OneMax`, :class:`LeadingOnes`, the deceptive :class:`Trap`, the royal
roads R1 and R2 (:class:`RoyalRoad`), :class:`NkLandscape` and the 0/1 :class:`Knapsack` with
Pisinger's generated instance classes (:class:`KnapsackItems` for given items). NK landscapes and
knapsack instances are drawn from a seed with genoxide's portable random numbers: the same seed
gives the same problem on every platform, and the same as in Rust. Their ``optimum`` is computed,
exactly: by dynamic programming (NK landscapes with adjacent neighborhoods, and the knapsack) or by
evaluating every bit string (small NK landscapes); None when that would take too long.

Each class's docstring gives the definition and its source, read for it:

- Droste, S., Jansen, T. and Wegener, I. (2002). On the analysis of the (1+1) evolutionary
  algorithm. Theoretical Computer Science 276(1-2): 51-81. doi:10.1016/S0304-3975(01)00182-7
- Deb, K. and Goldberg, D. E. (1993). Analyzing deception in trap functions. Foundations of
  Genetic Algorithms 2: 93-108. doi:10.1016/B978-0-08-094832-4.50012-X
- Mitchell, M., Forrest, S. and Holland, J. H. (1992). The royal road for genetic algorithms:
  fitness landscapes and GA performance. Proceedings of the First European Conference on
  Artificial Life: 245-254.
- Mitchell, M., Holland, J. H. and Forrest, S. (1994). When will a genetic algorithm outperform
  hill climbing? Advances in Neural Information Processing Systems 6: 51-58.
- Kauffman, S. A. and Weinberger, E. D. (1989). The NK model of rugged fitness landscapes and its
  application to maturation of the immune response. Journal of Theoretical Biology 141(2):
  211-245. doi:10.1016/S0022-5193(89)80019-0
- Pisinger, D. (2005). Where are the hard knapsack problems? Computers & Operations Research
  32(9): 2271-2284. doi:10.1016/j.cor.2004.03.002
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, ClassVar, Literal, Union, cast

import numpy as np

from .. import Binary, _genoxide, _number, _whole
from . import Problem

__all__ = [
    "OneMax",
    "LeadingOnes",
    "Trap",
    "RoyalRoad",
    "NkLandscape",
    "Knapsack",
    "KnapsackItems",
    "Spanner",
    "MultipleStronglyCorrelated",
    "ProfitCeiling",
    "Circle",
]

_MAX_BITS = 2**24


@dataclass(frozen=True)
class OneMax(Problem[Binary]):
    """OneMax: the number of ones, ``Σ xᵢ``, the simplest function of bit strings.

    Every string but the optimum has a better neighbor one flip away. The (1+1) evolutionary
    algorithm needs Θ(n log n) evaluations on average (Droste et al., Lemma 10).

    Maximum ``bits`` at all ones; ``bits`` is 1 to 2^24.

    Droste, S., Jansen, T. and Wegener, I. (2002). On the analysis of the (1+1) evolutionary
    algorithm. Theoretical Computer Science 276(1-2): 51-81, Definition 9. The function has no
    single origin; Ackley (1987, A Connectionist Machine for Genetic Hillclimbing, section 3.3.1)
    tests a "One Max" that is ten times the number of ones.
    """

    bits: int = 100
    _type: ClassVar[str] = "one_max"

    def _describe(self) -> dict[str, Any]:
        bits = _whole("OneMax.bits", self.bits, minimum=1, maximum=_MAX_BITS)
        return {"type": self._type, "bits": bits}


@dataclass(frozen=True)
class LeadingOnes(Problem[Binary]):
    """LeadingOnes: the number of ones before the first zero, ``Σᵢ Πⱼ≤ᵢ xⱼ``.

    Unimodal, but only the first zero can improve a string. The (1+1) evolutionary algorithm
    needs Θ(n²) evaluations on average, and at most e n² (Droste et al., Theorem 17).

    Maximum ``bits`` at all ones; ``bits`` is 1 to 2^24.

    Droste, S., Jansen, T. and Wegener, I. (2002). On the analysis of the (1+1) evolutionary
    algorithm. Theoretical Computer Science 276(1-2): 51-81, Definition 16, from Rudolph, G.
    (1997). Convergence Properties of Evolutionary Algorithms. Kovač, Hamburg, whom they credit.
    """

    bits: int = 100
    _type: ClassVar[str] = "leading_ones"

    def _describe(self) -> dict[str, Any]:
        bits = _whole("LeadingOnes.bits", self.bits, minimum=1, maximum=_MAX_BITS)
        return {"type": self._type, "bits": bits}


@dataclass(frozen=True)
class Trap(Problem[Binary]):
    """The deceptive trap: ``blocks`` consecutive blocks of ``k`` bits, each scored by Deb and
    Goldberg's trap function of its number of ones u, summed.

    ``f(u) = a (z − u) / z`` for u ≤ z, and ``b (u − z) / (k − z)`` otherwise: it falls from a at
    no ones, the deceptive attractor, to 0 at z, and rises to b > a with all ones. a, b and z are
    k − 1, k and k − 1 unless given (each on its own): a block scores k − 1 − u below k ones,
    and k with all of them, fully deceptive for k ≥ 3. Ackley's trap of n bits is
    ``Trap(1, n, a=8n, b=10n, z=3n // 4)``.

    Maximum ``blocks * b`` at all ones; ``blocks`` at least 1, ``k`` at least 2, at most 2^24
    bits; z from 1 to k − 1, and 0 ≤ a < b.

    Deb, K. and Goldberg, D. E. (1993). Analyzing deception in trap functions. Foundations of
    Genetic Algorithms 2: 93-108, equation 1, after Ackley, D. H. (1987). A Connectionist Machine
    for Genetic Hillclimbing. Kluwer, section 3.3.3.
    """

    blocks: int = 10
    k: int = 4
    a: float | None = None
    b: float | None = None
    z: int | None = None
    _type: ClassVar[str] = "trap"

    def _describe(self) -> dict[str, Any]:
        return {
            "type": self._type,
            "blocks": _whole("Trap.blocks", self.blocks, minimum=1, maximum=_MAX_BITS),
            "k": _whole("Trap.k", self.k, minimum=2, maximum=_MAX_BITS),
            "a": None if self.a is None else _number("Trap.a", self.a),
            "b": None if self.b is None else _number("Trap.b", self.b),
            "z": None if self.z is None else _whole("Trap.z", self.z, maximum=_MAX_BITS),
        }


@dataclass(frozen=True)
class RoyalRoad(Problem[Binary]):
    """The royal road functions: ``blocks`` blocks of ``block_size`` consecutive ones, each
    scoring ``block_size`` when complete (R1), and with ``hierarchical``, the complete pairs,
    quadruples and so on up to all the blocks scoring again, each its number of bits (R2).

    :meth:`r1` is R1, 8 blocks of 8 bits, maximum 64 (Mitchell, Holland and Forrest, 1994,
    Figure 1); :meth:`r2` is R2, maximum 8 · 8 + 4 · 16 + 2 · 32 + 64 = 256 (Mitchell, Forrest and
    Holland, 1992, Figure 1). ``blocks`` is a power of 2 with ``hierarchical``.

    Mitchell, M., Forrest, S. and Holland, J. H. (1992). The royal road for genetic algorithms:
    fitness landscapes and GA performance. Proceedings of the First European Conference on
    Artificial Life: 245-254; Mitchell, M., Holland, J. H. and Forrest, S. (1994). When will a
    genetic algorithm outperform hill climbing? Advances in Neural Information Processing Systems
    6: 51-58.
    """

    blocks: int = 8
    block_size: int = 8
    hierarchical: bool = False
    _type: ClassVar[str] = "royal_road"

    @classmethod
    def r1(cls) -> RoyalRoad:
        """R1: 8 blocks of 8 bits, maximum 64."""
        return cls(8, 8)

    @classmethod
    def r2(cls) -> RoyalRoad:
        """R2: 8 blocks of 8 bits and the levels above them, maximum 256."""
        return cls(8, 8, hierarchical=True)

    def _describe(self) -> dict[str, Any]:
        if not isinstance(self.hierarchical, (bool, np.bool_)):
            raise ValueError(f"RoyalRoad.hierarchical is True or False, not {self.hierarchical!r}")
        return {
            "type": self._type,
            "blocks": _whole("RoyalRoad.blocks", self.blocks, minimum=1, maximum=_MAX_BITS),
            "block_size": _whole(
                "RoyalRoad.block_size", self.block_size, minimum=1, maximum=_MAX_BITS
            ),
            "hierarchical": bool(self.hierarchical),
        }


@dataclass(frozen=True)
class NkLandscape(Problem[Binary]):
    """An NK landscape: ``n`` bits, each contributing a value drawn uniformly from (0, 1) for each
    combination of its bit and those of ``k`` others, the fitness their mean.

    ``neighborhood`` "adjacent" takes a site's flanking sites on a circle (k/2 on each side; for
    an odd k, one more after it), "random" k distinct other sites drawn for each. The neighbors
    and contributions are drawn from ``seed``: the same seed gives the same landscape on every
    platform and in Rust. :attr:`neighbors` and :attr:`contributions` give them.

    ``optimum`` is computed exactly: by dynamic programming for adjacent neighborhoods, or by
    evaluating every string, whichever is less work; None when both take more than 2^30 steps.

    ``n`` is 1 to 2^24, ``k`` below ``n``, and the tables at most 2^24 values (n 2^(k+1)).

    Kauffman, S. A. and Weinberger, E. D. (1989). The NK model of rugged fitness landscapes and
    its application to maturation of the immune response. Journal of Theoretical Biology 141(2):
    211-245.
    """

    n: int
    k: int
    neighborhood: Literal["adjacent", "random"] = "random"
    seed: int = 0
    _type: ClassVar[str] = "nk_landscape"

    def _describe(self) -> dict[str, Any]:
        if self.neighborhood not in ("adjacent", "random"):
            raise ValueError(
                f'NkLandscape.neighborhood is "adjacent" or "random", not {self.neighborhood!r}'
            )
        return {
            "type": self._type,
            "n": _whole("NkLandscape.n", self.n, minimum=1, maximum=_MAX_BITS),
            "k": _whole("NkLandscape.k", self.k, maximum=_MAX_BITS),
            "neighborhood": self.neighborhood,
            "seed": _whole("NkLandscape.seed", self.seed),
        }

    @cached_property
    def _tables(self) -> tuple[np.ndarray, np.ndarray]:
        return _genoxide.nk_tables(self._json())

    @property
    def neighbors(self) -> np.ndarray:
        """The k sites that bear on each site, a row per site, in the order of the bits of its
        table's index."""
        return self._tables[0]

    @property
    def contributions(self) -> np.ndarray:
        """The contributions of each site, a row per site with 2^(k+1) values: the site's own bit
        is bit k of the index (the highest), and its j-th neighbor's bit k − 1 − j."""
        return self._tables[1]


@dataclass(frozen=True)
class Spanner:
    """span(v, m) knapsack instances: every item a multiple (1 to ``m``) of one of ``v`` spanner
    items, drawn with weights in [1, R] and profits by ``distribution``, then scaled down to
    ⌈2p/m⌉ and ⌈2w/m⌉. The paper uses span(2, 10)."""

    v: int = 2
    m: int = 10
    distribution: Literal["uncorrelated", "weakly_correlated", "strongly_correlated"] = (
        "uncorrelated"
    )

    def _describe(self) -> dict[str, Any]:
        if self.distribution not in ("uncorrelated", "weakly_correlated", "strongly_correlated"):
            raise ValueError(
                'Spanner.distribution is "uncorrelated", "weakly_correlated" or '
                f'"strongly_correlated", not {self.distribution!r}'
            )
        return {
            "type": "spanner",
            "v": _whole("Spanner.v", self.v, minimum=1, maximum=_MAX_BITS),
            "m": _whole("Spanner.m", self.m, minimum=1, maximum=2**32),
            "distribution": self.distribution,
        }


@dataclass(frozen=True)
class MultipleStronglyCorrelated:
    """mstr(k1, k2, d) knapsack instances: weights in [1, R], profits w + k1 for the weights
    divisible by d and w + k2 for the others. The paper uses mstr(3R/10, 2R/10, 6), which None
    for ``k1`` and ``k2`` gives."""

    k1: int | None = None
    k2: int | None = None
    d: int = 6

    def _describe(self, data_range: int) -> dict[str, Any]:
        k1 = 3 * data_range // 10 if self.k1 is None else self.k1
        k2 = 2 * data_range // 10 if self.k2 is None else self.k2
        return {
            "type": "multiple_strongly_correlated",
            "k1": _whole("MultipleStronglyCorrelated.k1", k1, maximum=2**32),
            "k2": _whole("MultipleStronglyCorrelated.k2", k2, maximum=2**32),
            "d": _whole("MultipleStronglyCorrelated.d", self.d, minimum=1, maximum=2**32),
        }


@dataclass(frozen=True)
class ProfitCeiling:
    """pceil(d) knapsack instances: weights in [1, R], profits d⌈w/d⌉. The paper uses
    pceil(3)."""

    d: int = 3

    def _describe(self) -> dict[str, Any]:
        return {
            "type": "profit_ceiling",
            "d": _whole("ProfitCeiling.d", self.d, minimum=1, maximum=2**32),
        }


@dataclass(frozen=True)
class Circle:
    """circle(d) knapsack instances: weights in [1, R], profits ``d √(4R² − (w − 2R)²)`` rounded
    down, with d = ``numerator / denominator``. The paper uses circle(2/3)."""

    numerator: int = 2
    denominator: int = 3

    def _describe(self) -> dict[str, Any]:
        return {
            "type": "circle",
            "numerator": _whole("Circle.numerator", self.numerator, minimum=1, maximum=2**16),
            "denominator": _whole(
                "Circle.denominator", self.denominator, minimum=1, maximum=2**16
            ),
        }


KnapsackClassName = Literal[
    "uncorrelated",
    "weakly_correlated",
    "strongly_correlated",
    "inverse_strongly_correlated",
    "almost_strongly_correlated",
    "subset_sum",
    "uncorrelated_similar_weights",
    "spanner",
    "multiple_strongly_correlated",
    "profit_ceiling",
    "circle",
]
"""The names of Pisinger's instance classes; the last four with the paper's parameters."""

_SIMPLE = (
    "uncorrelated",
    "weakly_correlated",
    "strongly_correlated",
    "inverse_strongly_correlated",
    "almost_strongly_correlated",
    "subset_sum",
    "uncorrelated_similar_weights",
)

KnapsackClass = Union[
    KnapsackClassName, Spanner, MultipleStronglyCorrelated, ProfitCeiling, Circle
]
"""An instance class: a name, or one of the classes with parameters."""


class _KnapsackProblem(Problem[Binary]):
    """What a knapsack has: its items and capacity."""

    @property
    def weights(self) -> np.ndarray:
        """The weight of each item (int64)."""
        return np.array(self._info["weights"], dtype=np.int64)

    @property
    def profits(self) -> np.ndarray:
        """The profit of each item (int64)."""
        return np.array(self._info["profits"], dtype=np.int64)

    @property
    def capacity(self) -> int:
        """The capacity."""
        return cast(int, self._info["capacity"])


@dataclass(frozen=True)
class Knapsack(_KnapsackProblem):
    """The 0/1 knapsack problem on a generated instance of ``items`` items: the most profitable
    selection, a bit per item, whose weight fits the capacity.

    The fitness is ``(profit, violation)``: the total profit, and how far the weight exceeds the
    capacity, 0 when it fits (Deb's rules). ``instance_class`` is one of Pisinger's classes, "in
    [x, y]" drawn uniformly and R/10, R/500 rounded down:

    - "uncorrelated": weights and profits in [1, R];
    - "weakly_correlated": weights in [1, R], profits in [max(1, w − R/10), w + R/10];
    - "strongly_correlated": weights in [1, R], profits w + R/10;
    - "inverse_strongly_correlated": profits in [1, R], weights p + R/10;
    - "almost_strongly_correlated": weights in [1, R], profits in [w + R/10 − R/500,
      w + R/10 + R/500];
    - "subset_sum": weights in [1, R], profits equal to them;
    - "uncorrelated_similar_weights": weights in [100 000, 100 100], profits in [1, 1000];
    - :class:`Spanner`, :class:`MultipleStronglyCorrelated`, :class:`ProfitCeiling` and
      :class:`Circle`, or their names for the paper's parameters: span(2, 10) of uncorrelated
      items, mstr(3R/10, 2R/10, 6), pceil(3) and circle(2/3).

    ``data_range`` is R (1 to 2^32), and the capacity is ⌊h/(H + 1) Σ w⌋ for ``instance`` h of
    ``instances`` H (eq. 5): about half the total weight by default. The items are drawn from
    ``seed``, the same on every platform and in Rust. ``optimum`` is computed by dynamic
    programming, None when the items times the capacity are above 2^28.

    Pisinger, D. (2005). Where are the hard knapsack problems? Computers & Operations Research
    32(9): 2271-2284, sections 3 and 3.3.
    """

    items: int
    instance_class: KnapsackClass = "uncorrelated"
    data_range: int = 1000
    instance: int = 50
    instances: int = 100
    seed: int = 0
    _type: ClassVar[str] = "knapsack"

    def _describe(self) -> dict[str, Any]:
        data_range = _whole("Knapsack.data_range", self.data_range, minimum=1, maximum=2**32)
        kind = self.instance_class
        if isinstance(kind, str):
            if kind in _SIMPLE:
                described: dict[str, Any] = {"type": kind}
            elif kind == "spanner":
                described = Spanner()._describe()
            elif kind == "multiple_strongly_correlated":
                described = MultipleStronglyCorrelated()._describe(data_range)
            elif kind == "profit_ceiling":
                described = ProfitCeiling()._describe()
            elif kind == "circle":
                described = Circle()._describe()
            else:
                raise ValueError(f"Knapsack.instance_class: no class {kind!r}")
        elif isinstance(kind, MultipleStronglyCorrelated):
            described = kind._describe(data_range)
        elif isinstance(kind, (Spanner, ProfitCeiling, Circle)):
            described = kind._describe()
        else:
            raise ValueError(
                "Knapsack.instance_class is a class name, Spanner, MultipleStronglyCorrelated, "
                f"ProfitCeiling or Circle, not {kind!r}"
            )
        return {
            "type": self._type,
            "class": described,
            "items": _whole("Knapsack.items", self.items, minimum=1, maximum=_MAX_BITS),
            "range": data_range,
            "instance": _whole("Knapsack.instance", self.instance, minimum=1, maximum=2**32),
            "instances": _whole("Knapsack.instances", self.instances, minimum=1, maximum=2**32),
            "seed": _whole("Knapsack.seed", self.seed),
        }


class KnapsackItems(_KnapsackProblem):
    """The 0/1 knapsack problem on given items: ``weights`` and ``profits``, whole numbers of at
    least 0, one each per item, and ``capacity``; as :class:`Knapsack`, whose fitness and optimum
    it has. The totals and the capacity are at most 2^53."""

    _type: ClassVar[str] = "knapsack_items"

    def __init__(self, weights: Any, profits: Any, capacity: int) -> None:
        self._weights = tuple(_wholes("KnapsackItems.weights", weights))
        self._profits = tuple(_wholes("KnapsackItems.profits", profits))
        self._capacity = _whole("KnapsackItems.capacity", capacity, maximum=2**53)

    def __repr__(self) -> str:
        return (
            f"KnapsackItems({list(self._weights)}, {list(self._profits)}, {self._capacity})"
        )

    def _describe(self) -> dict[str, Any]:
        return {
            "type": self._type,
            "weights": list(self._weights),
            "profits": list(self._profits),
            "capacity": self._capacity,
        }


def _wholes(name: str, values: Any) -> list[int]:
    """``values`` as whole numbers of at least 0."""
    values = np.asarray(values).tolist()
    return [_whole(name, value, maximum=2**53, plural=True) for value in values]
