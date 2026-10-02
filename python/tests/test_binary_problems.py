"""gx.problems.binary: the problems of bit strings, evaluated in Rust."""

import typing

import numpy as np
import pytest

import genoxide as gx

binary = gx.problems.binary


def bits(text):
    return np.array([bit == "1" for bit in text])


def test_the_module_lists_its_problems():
    assert binary.__all__ == [
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


@pytest.mark.parametrize(
    "problem",
    [
        binary.OneMax(),
        binary.LeadingOnes(),
        binary.Trap(),
        binary.RoyalRoad(),
        binary.NkLandscape(12, 2),
        binary.Knapsack(20),
        binary.KnapsackItems([1, 2], [3, 4], 2),
    ],
)
def test_every_problem_describes_itself(problem):
    assert problem.objective == "maximize"
    assert isinstance(problem.genome, gx.Binary)
    assert problem.reference
    assert problem.reference_url is None or problem.reference_url.startswith("https://")
    optimum = problem.optimum
    assert optimum.proven
    assert optimum.solutions.dtype == np.bool_
    assert optimum.solutions.shape == (1, problem.dimensions)
    value = problem(optimum.solutions[0])
    if isinstance(value, tuple):
        assert value == (optimum.value, 0.0)
    else:
        assert value == optimum.value
    # Problem[Binary], for type checkers
    declared = next(
        typing.get_args(base)[0]
        for ancestor in type(problem).__mro__
        for base in getattr(ancestor, "__orig_bases__", ())
        if typing.get_origin(base) is gx.problems.Problem
    )
    assert declared is gx.Binary


def test_the_definitions():
    assert binary.OneMax(5)(bits("10110")) == 3.0
    assert binary.LeadingOnes(6)(bits("110111")) == 2.0
    assert binary.LeadingOnes(6)(bits("011111")) == 0.0
    # k = 5: 4 − u below 5 ones, 5 with all of them
    assert binary.Trap(3, 5)(bits("111110000000100")) == 5.0 + 4.0 + 3.0
    # a = 6, b = 10, z = 2: 6, 3, 0, 5, 10
    trap = binary.Trap(5, 4, a=6, b=10, z=2)
    assert trap(bits("0000" "1000" "1100" "1110" "1111")) == 24.0
    assert trap.optimum.value == 50.0
    first_two = bits("1" * 16 + "0" * 48)
    assert binary.RoyalRoad.r1()(first_two) == 16.0
    assert binary.RoyalRoad.r2()(first_two) == 32.0
    assert binary.RoyalRoad.r1().optimum.value == 64.0
    assert binary.RoyalRoad.r2().optimum.value == 256.0
    assert binary.RoyalRoad.r2().name == "RoyalRoadR2"
    assert binary.RoyalRoad(16, 3, hierarchical=True).optimum.value == 48.0 * 5


def test_an_nk_landscape_is_the_same_as_in_rust():
    # the values of the Rust tests, to the bit
    landscape = binary.NkLandscape(10, 3, "random", seed=42)
    assert landscape.neighbors[0].tolist() == [2, 6, 7]
    assert landscape.neighbors[9].tolist() == [4, 5, 8]
    assert landscape.contributions.shape == (10, 16)
    assert landscape.contributions[0, 0].hex() == "0x1.208b86307d686p-2"
    assert landscape.contributions[9, 15].hex() == "0x1.bfb4ddb16b0c9p-1"
    assert landscape(bits("1100101101")).hex() == "0x1.5562443ce9d63p-2"
    # the mean of the contributions
    x = bits("1100101101")
    total = 0.0
    for site in range(10):
        index = int(x[site])
        for neighbor in landscape.neighbors[site]:
            index = index << 1 | int(x[neighbor])
        total += landscape.contributions[site, index]
    assert landscape(x) == pytest.approx(total / 10, abs=1e-15)
    adjacent = binary.NkLandscape(8, 4, "adjacent")
    assert adjacent.neighbors[0].tolist() == [6, 7, 1, 2]


def test_the_optimum_of_an_nk_landscape_is_the_best_string():
    landscape = binary.NkLandscape(10, 3, "adjacent", seed=3)
    strings = np.array([[(s >> i) & 1 for i in range(10)] for s in range(1 << 10)])
    assert landscape.optimum.value == landscape.evaluate(strings).max()
    assert binary.NkLandscape(200, 4, "random").optimum is None
    assert binary.NkLandscape(200, 4, "adjacent").optimum.proven


def test_a_knapsack_is_the_same_as_in_rust():
    knapsack = binary.Knapsack(5, seed=1)
    assert knapsack.weights.tolist() == [613, 858, 688, 724, 989]
    assert knapsack.profits.tolist() == [705, 45, 998, 22, 438]
    assert knapsack.capacity == 1916
    circle = binary.Knapsack(3, "circle", seed=2)
    assert circle.weights.tolist() == [898, 956, 782]
    assert circle.profits.tolist() == [1112, 1137, 1057]
    assert circle.constraint_count == 1


def test_the_knapsack_classes():
    for kind, holds in [
        ("strongly_correlated", lambda w, p: p == w + 100),
        ("inverse_strongly_correlated", lambda w, p: w == p + 100),
        ("subset_sum", lambda w, p: p == w),
        ("profit_ceiling", lambda w, p: p % 3 == 0 and w <= p < w + 3),
        (binary.ProfitCeiling(5), lambda w, p: p % 5 == 0 and w <= p < w + 5),
        ("multiple_strongly_correlated", lambda w, p: p == w + (300 if w % 6 == 0 else 200)),
        (
            binary.MultipleStronglyCorrelated(10, 20, 4),
            lambda w, p: p == w + (10 if w % 4 == 0 else 20),
        ),
        ("uncorrelated_similar_weights", lambda w, p: 100_000 <= w <= 100_100 and 1 <= p <= 1000),
    ]:
        knapsack = binary.Knapsack(200, kind, seed=4)
        for w, p in zip(knapsack.weights.tolist(), knapsack.profits.tolist()):
            assert holds(w, p), (kind, w, p)
    spanner = binary.Knapsack(100, binary.Spanner(2, 10, "strongly_correlated"), seed=5)
    reduced = {
        (w // np.gcd(w, p), p // np.gcd(w, p))
        for w, p in zip(spanner.weights.tolist(), spanner.profits.tolist())
    }
    assert len(reduced) <= 2
    # eq. 5: instance 30 of 100
    knapsack = binary.Knapsack(50, "weakly_correlated", data_range=10_000, instance=30, seed=6)
    assert knapsack.capacity == knapsack.weights.sum() * 30 // 101
    assert knapsack.weights.max() <= 10_000


def test_the_knapsack_fitness_is_the_profit_and_the_excess_weight():
    knapsack = binary.KnapsackItems([5, 4, 3], [10, 40, 30], 7)
    assert knapsack(bits("011")) == (70.0, 0.0)
    assert knapsack(bits("111")) == (80.0, 5.0)
    scores, violations = knapsack.evaluate(np.array([[0, 1, 1], [1, 1, 1]]))
    assert scores.tolist() == [70.0, 80.0] and violations.tolist() == [0.0, 5.0]
    assert knapsack.constraints(bits("111")).tolist() == [5.0]
    assert knapsack.optimum.value == 70.0
    assert knapsack.weights.tolist() == [5, 4, 3]
    assert knapsack.capacity == 7


def test_a_native_run_equals_a_run_with_python_calls():
    for problem in (binary.Trap(5, 4), binary.Knapsack(20, seed=1), binary.NkLandscape(12, 2)):
        ga = gx.Ga(
            problem.genome,
            population_size=40,
            select=gx.Tournament(3),
            crossover=gx.UniformCrossover(),
            mutation=gx.BitFlip(rate=1 / problem.dimensions),
            seed=3,
        )
        native = ga.run(problem, generations=20)
        python = ga.run(lambda x, problem=problem: problem(x), generations=20)
        assert native.best_fitness == python.best_fitness
        assert np.array_equal(native.best_genome, python.best_genome)


def test_the_examples_methods_reach_the_optimum():
    # the (1+1) evolutionary algorithm on LeadingOnes, and random-mutation hill climbing on R1
    leading_ones = binary.LeadingOnes(30)
    search = gx.LocalSearch(leading_ones.genome, neighbor=gx.BitFlip(rate=1 / 30), seed=1)
    result = search.run(leading_ones, target=30, evaluations=100_000)
    assert result.best_fitness == 30
    r1 = binary.RoyalRoad.r1()
    search = gx.LocalSearch(r1.genome, neighbor=gx.BitFlip(count=1), seed=1)
    assert search.run(r1, target=64, evaluations=256_000).best_fitness == 64


@pytest.mark.parametrize(
    "make, message",
    [
        (lambda: binary.OneMax(0), "OneMax.bits is at least 1"),
        (lambda: binary.Trap(1, 1), "Trap.k is at least 2"),
        (lambda: binary.Trap(1, 4, z=4), "must be between 1 and k − 1"),
        (lambda: binary.Trap(1, 4, a=5), "0 ≤ a < b"),
        (lambda: binary.RoyalRoad(6, 8, hierarchical=True), "a power of 2 blocks, not 6"),
        (lambda: binary.RoyalRoad(8, 8, hierarchical=1), "True or False"),
        (lambda: binary.NkLandscape(5, 5), "must be below n = 5"),
        (lambda: binary.NkLandscape(100, 30), r"values, more than 2\^24"),
        (lambda: binary.NkLandscape(5, 2, "ring"), '"adjacent" or "random"'),
        (lambda: binary.Knapsack(0), "Knapsack.items is at least 1"),
        (lambda: binary.Knapsack(5, "nope"), "no class 'nope'"),
        (lambda: binary.Knapsack(5, instance=101), "instance"),
        (lambda: binary.Knapsack(5, binary.Spanner(0)), "Spanner.v is at least 1"),
        (lambda: binary.Knapsack(5, binary.Spanner(distribution="x")), "Spanner.distribution"),
        (lambda: binary.Knapsack(5, binary.Circle(2, 0)), "Circle.denominator is at least 1"),
        (lambda: binary.KnapsackItems([1, 2], [1], 3), "must have a profit per item"),
        (lambda: binary.KnapsackItems([1, -2], [1, 1], 3), "at least 0"),
    ],
)
def test_wrong_settings_are_errors(make, message):
    with pytest.raises(ValueError, match=message):
        make().name


def test_a_binary_problem_needs_its_genome_and_objective():
    problem = binary.OneMax(10)
    with pytest.raises(ValueError, match="OneMax maximizes its objective"):
        gx.LocalSearch(
            problem.genome, neighbor=gx.BitFlip(count=1), objective="minimize", seed=1
        ).run(problem, generations=1)
    with pytest.raises(ValueError, match="OneMax has 10 bits, but the genome has 12"):
        gx.LocalSearch(gx.Binary(12), neighbor=gx.BitFlip(count=1), seed=1).run(
            problem, generations=1
        )
    with pytest.raises(ValueError, match="OneMax needs a Binary genome"):
        gx.De(gx.Real((0, 1), length=10), seed=1).run(problem, generations=1)
    with pytest.raises(ValueError, match="OneMax takes genomes of 10 bits, not 3"):
        problem([1, 0, 1])
    with pytest.raises(ValueError, match="OneMax takes bits, 0 or 1, as genes, not 0.5"):
        problem([0.5] * 10)
    with pytest.raises(ValueError, match="one objective, and an optimum instead of a front"):
        gx._genoxide.optimal_front(problem._json(), 10)
