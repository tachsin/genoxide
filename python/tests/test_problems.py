"""gx.problems and gx.indicators: the test problems evaluated in Rust, and the indicators of
fronts."""

import dataclasses
import math

import numpy as np
import pytest

import genoxide as gx

PROBLEMS = [
    gx.problems.Sphere,
    gx.problems.AxisParallelEllipsoid,
    gx.problems.Schwefel1_2,
    gx.problems.Rastrigin,
    gx.problems.Rosenbrock,
    gx.problems.Ackley,
    gx.problems.Griewank,
    gx.problems.Schwefel2_26,
    gx.problems.Levy,
    gx.problems.Zakharov,
    gx.problems.StyblinskiTang,
    gx.problems.Michalewicz,
    gx.problems.Himmelblau,
    gx.problems.Branin,
    gx.problems.GoldsteinPrice,
    gx.problems.SixHumpCamel,
]


def test_the_classes_are_the_rust_registry():
    assert [cls().name for cls in PROBLEMS] == gx._genoxide.problem_names()
    assert sorted(cls.__name__ for cls in PROBLEMS) == sorted(
        name for name in gx.problems.__all__ if name not in ("Problem", "Optimum")
    )


@pytest.mark.parametrize("cls", PROBLEMS)
def test_every_problem_describes_itself(cls):
    problem = cls()
    assert problem.objective == "minimize"
    assert problem.reference
    assert problem.reference_url is None or problem.reference_url.startswith("https://")
    assert cls.__doc__
    genome = problem.genome
    assert isinstance(genome, gx.Real)
    bounds = np.array(genome._describe()["bounds"])
    assert bounds.shape == (problem.dimensions, 2)
    optimum = problem.optimum
    assert optimum.proven
    assert optimum.solutions.shape[1] == problem.dimensions
    for solution in optimum.solutions:
        assert np.all(bounds[:, 0] <= solution) and np.all(solution <= bounds[:, 1])
        assert problem(solution) == pytest.approx(optimum.value, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("cls", PROBLEMS)
def test_calls_and_batches_agree(cls):
    problem = cls()
    genomes = np.random.default_rng(1).uniform(-2, 2, size=(20, problem.dimensions))
    scores = problem.evaluate(genomes)
    assert scores.shape == (20,)
    assert scores.dtype == np.float64
    assert [problem(genome) for genome in genomes] == list(scores)


def test_values_at_chosen_points():
    # 1 − 10 cos 2π + 10 in the first dimension, 0 in the others
    assert gx.problems.Rastrigin(3)([1.0, 0.0, 0.0]) == pytest.approx(1.0)
    # Rosenbrock's starting point: 100 (1 − 1.44)² + (−2.2)²
    assert gx.problems.Rosenbrock(2)([-1.2, 1.0]) == pytest.approx(24.2)
    # 1 + 4 + 9
    assert gx.problems.Sphere(3)([1, -2, 3]) == 14
    # the local minima that Goldstein and Price list
    assert gx.problems.GoldsteinPrice()([1.8, 0.2]) == pytest.approx(84)
    assert gx.problems.Himmelblau()([0, 0]) == 170
    assert gx.problems.Branin().optimum.value == pytest.approx(5 / (4 * math.pi))
    assert len(gx.problems.Himmelblau().optimum.solutions) == 4
    assert gx.problems.Michalewicz(10).optimum.value == pytest.approx(-9.6601517, abs=1e-7)


def test_sizes():
    assert gx.problems.Rastrigin().dimensions == 30
    assert gx.problems.Michalewicz().dimensions == 10
    assert gx.problems.Rastrigin(dimensions=5).genome == gx.Real((-5.12, 5.12), length=5)
    assert gx.problems.Branin().genome == gx.Real([(-5.0, 10.0), (0.0, 15.0)])
    assert gx.problems.Rastrigin(4) == gx.problems.Rastrigin(4)
    assert gx.problems.Rastrigin(4) != gx.problems.Rastrigin(5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        gx.problems.Rastrigin(4).dimensions = 5


@pytest.mark.parametrize(
    "problem, message",
    [
        (gx.problems.Rosenbrock(1), "Rosenbrock.dimensions is at least 2, not 1"),
        (gx.problems.Sphere(0), "Sphere.dimensions is at least 1, not 0"),
        (gx.problems.Sphere(2.0), "Sphere.dimensions is a whole number"),
    ],
)
def test_wrong_sizes_are_errors(problem, message):
    with pytest.raises(ValueError, match=message):
        problem.genome
    with pytest.raises(ValueError, match=message):
        gx.Cmaes(gx.Real((0, 1), length=2), seed=1).run(problem, generations=1)


def test_wrong_genomes_are_errors():
    problem = gx.problems.Sphere(3)
    with pytest.raises(ValueError, match="Sphere takes genomes of 3 genes, not 2"):
        problem([1.0, 2.0])
    with pytest.raises(ValueError, match="a genome is a 1-D array"):
        problem([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="evaluate takes a 2-D array"):
        problem.evaluate([1.0, 2.0, 3.0])


def test_a_native_run_reaches_the_optimum_and_repeats():
    problem = gx.problems.Rastrigin(10)
    de = gx.De(problem.genome, objective=problem.objective, seed=1)
    target = problem.optimum.value + 1e-8
    result = de.run(problem, target=target, evaluations=200_000)
    assert result.stop_reason == "target"
    assert result.best_fitness <= target
    again = de.run(problem, target=target, evaluations=200_000)
    assert again.evaluations == result.evaluations
    assert np.array_equal(again.best_genome, result.best_genome)


@pytest.mark.parametrize("parallel", [False, True])
def test_a_native_run_equals_a_run_with_python_calls(parallel):
    problem = gx.problems.Ackley(5)
    cmaes = gx.Cmaes(problem.genome, objective="minimize", seed=3)
    native = cmaes.run(problem, generations=50, parallel=parallel)
    python = cmaes.run(lambda x: problem(x), generations=50)
    batch = cmaes.run(problem.evaluate, generations=50, batch=True)
    for other in (python, batch):
        assert other.best_fitness == native.best_fitness
        assert np.array_equal(other.best_genome, native.best_genome)
        assert other.evaluations == native.evaluations


def test_a_native_run_needs_a_matching_genome_and_one_objective():
    problem = gx.problems.Sphere(3)
    with pytest.raises(ValueError, match="Sphere has 3 dimensions, but the genome has 2 genes"):
        gx.Cmaes(gx.Real((0, 1), length=2), seed=1).run(problem, generations=1)
    with pytest.raises(ValueError, match="Sphere needs a Real genome"):
        ga = gx.Ga(
            gx.Integer((0, 5), length=3),
            population_size=10,
            select=gx.Tournament(2),
            crossover=gx.UniformCrossover(),
            mutation=gx.UniformMutation(rate=0.5),
        )
        ga.run(problem, generations=1)
    nsga2 = gx.Nsga2(
        problem.genome,
        objectives=["minimize", "minimize"],
        population_size=10,
        crossover=gx.SimulatedBinaryCrossover(15),
        mutation=gx.PolynomialMutation(20, rate=0.5),
    )
    with pytest.raises(ValueError, match="Sphere has one objective"):
        nsga2.run(problem, generations=1)


def test_a_native_run_calls_on_generation():
    problem = gx.problems.Sphere(4)
    seen = []

    def stop_at_three(progress):
        seen.append(progress.generation)
        return progress.generation < 3

    pso = gx.Pso(problem.genome, population_size=10, objective="minimize", seed=1)
    result = pso.run(problem, generations=100, on_generation=stop_at_three)
    assert result.stop_reason == "aborted"
    assert seen == [0, 1, 2, 3]


# ---- indicators ------------------------------------------------------------------------------

FRONT = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])


def test_hypervolume():
    # 3 + 2 + 1 unit squares below the reference point (4, 4)
    assert gx.indicators.hypervolume(FRONT, [4.0, 4.0]) == 6.0
    # maximizing both: the same squares, mirrored
    mirrored = -FRONT
    assert gx.indicators.hypervolume(mirrored, [-4, -4], ["maximize", "maximize"]) == 6.0
    # a point beyond the reference point adds nothing
    assert gx.indicators.hypervolume([[5.0, 0.0]], [4.0, 4.0]) == 0.0
    assert gx.indicators.hypervolume([], [4.0, 4.0]) == 0.0
    # three objectives: the unit cube below (1, 1, 1)
    assert gx.indicators.hypervolume([[0.0, 0.0, 0.0]], [1.0, 1.0, 1.0]) == 1.0


def test_distances():
    reference = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert gx.indicators.igd(reference, reference) == 0.0
    # (0, 1) covers itself, and is √2 from (1, 0)
    assert gx.indicators.igd([[0.0, 1.0]], reference) == pytest.approx(math.sqrt(2) / 2)
    assert gx.indicators.gd([[0.0, 1.0]], reference) == 0.0
    # better than the reference front: IGD sees a distance, IGD+ doesn't
    assert gx.indicators.igd([[0.0, 0.0]], reference) == 1.0
    assert gx.indicators.igd_plus([[0.0, 0.0]], reference) == 0.0
    assert gx.indicators.igd_plus([[0.0, 0.0]], reference, ["maximize", "maximize"]) == 1.0
    assert gx.indicators.igd([], reference) == math.inf
    assert math.isnan(gx.indicators.igd(reference, []))


def test_spread():
    reference = np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]])
    # evenly spaced from extreme to extreme
    assert gx.indicators.spread(reference, reference) == 0.0
    assert gx.indicators.spread(reference[:1], reference) == 1.0


def test_indicators_check_their_arguments():
    with pytest.raises(ValueError, match="the front has 3 objectives and the reference front 2"):
        gx.indicators.igd(np.zeros((2, 3)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="the indicators take 2 to 6 objectives, not 7"):
        gx.indicators.igd(np.zeros((2, 7)), np.zeros((2, 7)))
    with pytest.raises(ValueError, match="objectives has 1 entries, for 2 objectives"):
        gx.indicators.igd_plus(FRONT, FRONT, ["minimize"])
    with pytest.raises(ValueError, match='"maximize" or "minimize", not "best"'):
        gx.indicators.spread(FRONT, FRONT, ["best", "minimize"])
    with pytest.raises(ValueError, match="reference_point is a 1-D array"):
        gx.indicators.hypervolume(FRONT, [[4.0, 4.0]])
    with pytest.raises(ValueError, match="front is a 2-D array"):
        gx.indicators.gd([1.0, 2.0], FRONT)


def test_indicators_of_a_run():
    problem_front = np.column_stack([np.linspace(0, 1, 101), 1 - np.sqrt(np.linspace(0, 1, 101))])

    def zdt1(x):
        g = 1 + 9 * x[:, 1:].mean(axis=1)
        return np.column_stack([x[:, 0], g * (1 - np.sqrt(x[:, 0] / g))])

    nsga2 = gx.Nsga2(
        gx.Real((0.0, 1.0), length=10),
        objectives=["minimize", "minimize"],
        population_size=40,
        crossover=gx.SimulatedBinaryCrossover(15),
        mutation=gx.PolynomialMutation(20, rate=0.1),
        seed=1,
    )
    result = nsga2.run(zdt1, batch=True, generations=100)
    front = result.front_objectives
    assert gx.indicators.igd_plus(front, problem_front) < gx.indicators.igd_plus(
        front + 1, problem_front
    )
    assert 0 < gx.indicators.hypervolume(front, [1.1, 1.1]) < 0.8767
