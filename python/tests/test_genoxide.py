import dataclasses
import importlib.metadata
import json
import math
import pathlib
import re
import signal
import threading

import numpy as np
import pytest

import genoxide as gx


def rastrigin(x):
    return 10 * len(x) + float(np.sum(x * x - 10 * np.cos(2 * np.pi * x)))


def rastrigin_batch(x):
    return 10 * x.shape[1] + np.sum(x * x - 10 * np.cos(2 * np.pi * x), axis=1)


def onemax_ga(**settings):
    defaults = {
        "population_size": 40,
        "select": gx.Tournament(3),
        "crossover": gx.UniformCrossover(),
        "mutation": gx.BitFlip(rate=1 / 64),
        "seed": 7,
    }
    return gx.Ga(gx.Binary(64), **{**defaults, **settings})


def test_onemax_reaches_the_target():
    result = onemax_ga().run(lambda bits: bits.sum(), target=64, generations=2_000)
    assert result.stop_reason == "target"
    assert result.best_fitness == 64
    assert result.best_genome.dtype == np.bool_
    assert result.best_genome.shape == (64,)
    assert result.best_genome.all()
    assert result.violation == 0
    assert result.evaluations > 0 and result.seconds >= 0


def test_a_seed_repeats_the_run():
    first = onemax_ga().run(lambda bits: bits.sum(), generations=20)
    second = onemax_ga().run(lambda bits: bits.sum(), generations=20)
    assert first.best_fitness == second.best_fitness
    assert np.array_equal(first.best_genome, second.best_genome)
    assert first.evaluations == second.evaluations


def test_batch_and_parallel_evaluation_give_the_same_run():
    cmaes = gx.Cmaes(gx.Real((-5.12, 5.12), length=5), objective="minimize", seed=3)
    one = cmaes.run(rastrigin, evaluations=3_000)
    batch = cmaes.run(rastrigin_batch, evaluations=3_000, batch=True)
    parallel = cmaes.run(rastrigin, evaluations=3_000, parallel=True)
    for other in (batch, parallel):
        assert other.best_fitness == pytest.approx(one.best_fitness, rel=1e-12, abs=1e-12)
        assert np.allclose(other.best_genome, one.best_genome, rtol=1e-12, atol=1e-12)
        assert other.evaluations == one.evaluations


def test_the_batch_function_gets_a_generation():
    shapes = []

    def sphere(x):
        shapes.append(x.shape)
        return (x * x).sum(axis=1)

    result = gx.De(gx.Real((-1.0, 1.0), length=4), population_size=12, objective="minimize", seed=1).run(
        sphere, generations=5, batch=True
    )
    assert all(shape[1] == 4 for shape in shapes)
    assert sum(shape[0] for shape in shapes) == result.evaluations


def test_de_defaults_to_shades_population():
    # the Rust defaults: SHADE's population of 100, whatever the number of genes
    shapes = []

    def sphere(x):
        shapes.append(x.shape)
        return (x * x).sum(axis=1)

    gx.De(gx.Real((-1.0, 1.0), length=4), objective="minimize", seed=1).run(sphere, generations=2, batch=True)
    assert shapes == [(100, 4)] * 3


@pytest.mark.parametrize(
    "algorithm",
    [
        gx.De(gx.Real((-5.12, 5.12), length=5), objective="minimize", seed=1),
        gx.De(gx.Real((-5.12, 5.12), length=5), population_size=50, objective="minimize", seed=1),
        gx.De(gx.Real((-5.12, 5.12), length=5), l_shade=40_000, objective="minimize", seed=1),
        gx.Cmaes(gx.Real((-5.12, 5.12), length=5), restarts="ipop", objective="minimize", seed=1),
        gx.Pso(gx.Real((-5.12, 5.12), length=5), population_size=40, ring=2, objective="minimize", seed=1),
        gx.Ga(
            gx.Real((-5.12, 5.12), length=5),
            population_size=60,
            select=gx.Tournament(2),
            crossover=gx.SimulatedBinaryCrossover(15),
            mutation=gx.PolynomialMutation(20, rate=0.2),
            objective="minimize",
            seed=1,
        ),
    ],
)
def test_real_algorithms_minimize(algorithm):
    shift = np.linspace(-2.0, 2.0, 5)
    result = algorithm.run(lambda x: ((x - shift) ** 2).sum(axis=1), evaluations=40_000, batch=True)
    assert result.best_fitness < 1e-2
    assert result.best_genome.dtype == np.float64
    assert np.all(np.abs(result.best_genome) <= 5.12)


@pytest.mark.parametrize(
    "strategy, control, restarts",
    [
        ("rand1", {"f": 0.5, "cr": 0.9}, None),
        ("best1", {"min_f": 0.5, "max_f": 1.0, "cr": 0.9}, "never"),
        ({"p": 0.1, "archive": 1.0}, {"c": 0.1}, {"tolerance": 1e-12, "patience": 50}),
        ({"max_p": 0.2, "archive": 0.0}, {"memory": 6}, "never"),
    ],
)
def test_de_strategies_controls_and_restarts(strategy, control, restarts):
    shift = np.linspace(-2.0, 2.0, 5)
    de = gx.De(
        gx.Real((-5.12, 5.12), length=5),
        population_size=40,
        strategy=strategy,
        control=control,
        restarts=restarts,
        objective="minimize",
        seed=1,
    )
    result = de.run(lambda x: ((x - shift) ** 2).sum(axis=1), target=1e-8, evaluations=200_000, batch=True)
    assert result.best_fitness <= 1e-8


def test_de_settings_change_the_run():
    def run(**settings):
        de = gx.De(gx.Real((-5.12, 5.12), length=5), population_size=20, objective="minimize", seed=3, **settings)
        return de.run(lambda x: (x * x).sum(axis=1), generations=300, batch=True)

    default = run()
    # SHADE's defaults, spelled out
    same = run(
        strategy={"max_p": 0.2, "archive": 1.0},
        control={"memory": 100},
        restarts={"tolerance": 1e-12, "patience": 200},
    )
    assert same.best_fitness == default.best_fitness
    assert np.array_equal(same.best_genome, default.best_genome)
    for settings in [
        {"strategy": "rand1"},
        {"control": {"f": 0.5, "cr": 0.9}},
        # a restart after every generation without a better best
        {"restarts": {"tolerance": 0.0, "patience": 1}},
    ]:
        assert run(**settings).best_fitness != default.best_fitness, settings


def test_de_default_restarts_refine_large_scores():
    # #256: g04's minimum is about -30665.5; SHADE's restarts used to stop it about 1e-5 above
    problem = gx.problems.cec2006.G04()
    target = problem.optimum.value + 1e-8

    def run(restarts):
        de = gx.De(problem.genome, objective=problem.objective, restarts=restarts, seed=1)
        return de.run(problem, target=target, evaluations=500_000)

    default, never = run(None), run("never")
    assert default.best_fitness <= target
    assert default.evaluations == never.evaluations


def test_per_gene_bounds():
    bounds = [(0.0, 1.0), (10.0, 11.0), (-3.0, -2.0)]
    result = gx.Cmaes(gx.Real(bounds), seed=2).run(lambda x: -np.sum(x), generations=5)
    for gene, (low, high) in zip(result.best_genome, bounds):
        assert low <= gene <= high


def test_integer_genomes():
    target = np.array([3, -2, 7, 0, 5])
    result = gx.Ga(
        gx.Integer((-10, 10), length=5),
        population_size=50,
        select=gx.Tournament(3),
        crossover=gx.PointCrossover(2),
        mutation=gx.UniformMutation(count=1),
        objective="minimize",
        seed=4,
    ).run(lambda x: float(np.abs(x - target).sum()), target=0, generations=2_000)
    assert result.best_genome.dtype == np.int64
    assert np.array_equal(result.best_genome, target)


def queens_conflicts(order):
    n = len(order)
    rows = np.arange(n)
    diagonals = order - rows
    anti_diagonals = order + rows
    return float((n - len(np.unique(diagonals))) + (n - len(np.unique(anti_diagonals))))


def test_local_search_solves_n_queens():
    result = gx.LocalSearch(
        gx.Permutation(12),
        neighbor=gx.SwapMutation(),
        neighbors=8,
        restart=(200, 3),
        objective="minimize",
        seed=5,
    ).run(queens_conflicts, target=0, evaluations=200_000)
    assert result.best_fitness == 0
    assert result.best_genome.dtype == np.int64
    assert sorted(result.best_genome) == list(range(12))


def test_permutation_ga_with_every_operator():
    for crossover in (
        gx.OrderCrossover(),
        gx.PartiallyMappedCrossover(),
        gx.CycleCrossover(),
        gx.EdgeRecombinationCrossover(),
    ):
        for mutation in (
            gx.SwapMutation(2),
            gx.InversionMutation(),
            gx.InsertionMutation(),
            gx.ScrambleMutation(),
        ):
            result = gx.Ga(
                gx.Permutation(8),
                population_size=20,
                select=gx.Rank(),
                crossover=crossover,
                mutation=mutation,
                objective="minimize",
                seed=6,
            ).run(queens_conflicts, generations=10)
            assert sorted(result.best_genome) == list(range(8))


def test_schemes_and_acceptances():
    for scheme in (gx.Generational(2), gx.SteadyState(5), gx.MuPlusLambda(40), gx.MuCommaLambda(40)):
        result = onemax_ga(scheme=scheme).run(lambda bits: bits.sum(), generations=10)
        assert result.best_fitness > 32
    for acceptance in (gx.Improving(), gx.NotWorse(), gx.Annealing(1.0, 0.99), gx.Tabu(5)):
        result = gx.LocalSearch(
            gx.Binary(32), neighbor=gx.BitFlip(count=1), neighbors=4, acceptance=acceptance, seed=1
        ).run(lambda bits: bits.sum(), generations=50)
        assert result.best_fitness > 16


def test_every_selection():
    for select in (
        gx.Tournament(2),
        gx.Rank(1.8),
        gx.Roulette(),
        gx.StochasticUniversalSampling(),
        gx.Truncation(0.5),
        gx.RandomSelection(),
    ):
        result = onemax_ga(select=select).run(lambda bits: bits.sum() + 1.0, generations=5)
        assert result.generations == 5


def test_invalid_and_constrained_solutions():
    # x < 0.5 is infeasible by 0.5 - x; None for x > 0.9
    def fitness(x):
        if x[0] > 0.9:
            return None
        return (-x[0], max(0.0, 0.5 - x[0]))

    result = gx.Ga(
        gx.Real((0.0, 1.0), length=1),
        population_size=30,
        select=gx.Tournament(2),
        crossover=gx.BlendCrossover(),
        mutation=gx.GaussianMutation(0.1, rate=1.0),
        seed=8,
    ).run(fitness, generations=60)
    assert result.violation == 0
    assert 0.5 <= result.best_genome[0] <= 0.9
    assert result.best_fitness == pytest.approx(-0.5, abs=0.05)


def test_batch_constraint_violations():
    def fitness(x):
        return -x[:, 0], np.maximum(0.0, 0.5 - x[:, 0])

    result = gx.De(gx.Real((0.0, 1.0), length=1), population_size=20, seed=9).run(
        fitness, generations=40, batch=True
    )
    assert result.violation == 0
    assert result.best_genome[0] == pytest.approx(0.5, abs=0.01)


def zdt1(x):
    f1 = x[:, 0]
    g = 1 + 9 * x[:, 1:].mean(axis=1)
    return np.column_stack([f1, g * (1 - np.sqrt(f1 / g))])


def zdt1_one(x):
    return zdt1(x[None, :])[0]


def dtlz2(x, objectives=3):
    g = ((x[:, objectives - 1 :] - 0.5) ** 2).sum(axis=1)
    angles = x[:, : objectives - 1] * np.pi / 2
    values = np.empty((len(x), objectives))
    for i in range(objectives):
        value = 1 + g
        for j in range(objectives - 1 - i):
            value = value * np.cos(angles[:, j])
        if i > 0:
            value = value * np.sin(angles[:, objectives - 1 - i])
        values[:, i] = value
    return values


def assert_non_dominated(front):
    for a in front:
        assert not np.any(np.all(front <= a, axis=1) & np.any(front < a, axis=1))


def hypervolume(front):
    """The hypervolume of a front of 2 objectives to minimize, with the reference point
    (1.1, 1.1)."""
    front = front[np.argsort(front[:, 0])]
    widths = np.diff(np.append(front[:, 0], 1.1))
    return float(np.sum(widths * (1.1 - front[:, 1])))


MULTI_OBJECTIVE = ["nsga2", "nsga3", "spea2", "moead", "sms_emoa"]


def multi_objective(name, genome, objectives, divisions, **settings):
    """The algorithm `name`, with a population of as many Das-Dennis points."""
    settings = {"objectives": objectives, "seed": 10, **settings}
    directions = gx.das_dennis(len(objectives), divisions)
    if name == "nsga2":
        return gx.Nsga2(genome, population_size=len(directions), **settings)
    if name == "nsga3":
        return gx.Nsga3(genome, reference_directions=directions, **settings)
    if name == "spea2":
        return gx.Spea2(genome, population_size=len(directions), **settings)
    if name == "moead":
        return gx.Moead(genome, weights=directions, **settings)
    return gx.SmsEmoa(genome, population_size=len(directions), **settings)


def zdt1_algorithm(name):
    return multi_objective(
        name,
        gx.Real((0.0, 1.0), length=10),
        ["minimize", "minimize"],
        39,
        crossover=gx.SimulatedBinaryCrossover(15),
        mutation=gx.PolynomialMutation(20, rate=0.1),
    )


@pytest.mark.parametrize("name", MULTI_OBJECTIVE)
def test_multi_objective_algorithms_find_a_front(name):
    result = zdt1_algorithm(name).run(zdt1, evaluations=8_000, batch=True)
    front = result.front_objectives
    assert front.shape[1] == 2 and len(front) == len(result.front_genomes) > 5
    assert result.front_genomes.shape[1] == 10
    assert np.all(result.front_violations == 0)
    assert result.stop_reason == "evaluations"
    assert_non_dominated(front)
    # 40 points of the whole front, whose hypervolume is 0.8767
    assert hypervolume(front) > 0.85


@pytest.mark.parametrize("name", MULTI_OBJECTIVE)
def test_a_seed_repeats_a_multi_objective_run(name):
    first = zdt1_algorithm(name).run(zdt1, generations=30, batch=True)
    second = zdt1_algorithm(name).run(zdt1, generations=30, batch=True)
    assert np.array_equal(first.front_objectives, second.front_objectives)
    assert np.array_equal(first.front_genomes, second.front_genomes)
    assert first.evaluations == second.evaluations


@pytest.mark.parametrize("name", MULTI_OBJECTIVE)
def test_batch_and_per_genome_evaluation_give_the_same_front(name):
    algorithm = zdt1_algorithm(name)
    batch = algorithm.run(zdt1, evaluations=4_000, batch=True)
    one = algorithm.run(zdt1_one, evaluations=4_000)
    parallel = algorithm.run(zdt1_one, evaluations=4_000, parallel=True)
    for other in (one, parallel):
        assert np.allclose(
            np.sort(other.front_objectives, axis=0), np.sort(batch.front_objectives, axis=0)
        )
        assert other.evaluations == batch.evaluations


@pytest.mark.parametrize("name", MULTI_OBJECTIVE)
def test_three_objectives_and_tuples(name):
    result = multi_objective(
        name,
        gx.Binary(20),
        ["maximize", "maximize", "minimize"],
        4,
        crossover=gx.UniformCrossover(),
        mutation=gx.BitFlip(rate=0.05),
    ).run(lambda bits: (bits[:10].sum(), bits[10:].sum(), bits.sum()), generations=10)
    front = result.front_objectives
    assert front.shape[1] == 3
    assert result.front_genomes.dtype == np.bool_
    assert_non_dominated(front * [-1, -1, 1])


@pytest.mark.parametrize(
    "algorithm",
    [
        gx.Nsga3(
            gx.Real((0.0, 1.0), length=7),
            objectives=["minimize"] * 3,
            reference_directions=gx.das_dennis(3, 6),
            crossover=gx.SimulatedBinaryCrossover(30),
            mutation=gx.PolynomialMutation(20, rate=1 / 7),
            seed=1,
        ),
        gx.Moead(
            gx.Real((0.0, 1.0), length=7),
            objectives=["minimize"] * 3,
            weights=gx.das_dennis(3, 6),
            decomposition=gx.Pbi(5.0),
            neighbors=10,
            neighbor_mating=0.8,
            max_replacements=3,
            crossover=gx.SimulatedBinaryCrossover(20),
            mutation=gx.PolynomialMutation(20, rate=1 / 7),
            seed=1,
        ),
    ],
)
def test_many_objectives_on_dtlz2(algorithm):
    result = algorithm.run(dtlz2, generations=200, batch=True)
    # the front of DTLZ2 is on the unit sphere
    radius = np.linalg.norm(result.front_objectives, axis=1)
    assert len(radius) > 20
    assert np.all(np.abs(radius - 1) < 0.05)


def test_das_dennis():
    points = gx.das_dennis(3, 12)
    assert points.shape == (91, 3)
    assert np.allclose(points.sum(axis=1), 1)
    assert np.array_equal(gx.das_dennis(2, 2), [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    with pytest.raises(ValueError, match="2 to 6 objectives, not 7"):
        gx.das_dennis(7, 2)
    # too many points is an error, not an allocation that aborts the interpreter
    with pytest.raises(ValueError, match=r"das_dennis\(6, 200\) would have .* more than 2\^24"):
        gx.das_dennis(6, 200)
    with pytest.raises(ValueError, match=r"more than 2\^24"):
        gx.das_dennis(2, 2**24)
    with pytest.raises(ValueError, match=r"optimal_front takes at most 2\^24 points"):
        gx.problems.Dtlz2(objectives=3).optimal_front(10**11)


def test_a_generation_of_copies_makes_no_batch_call():
    rows = []

    def fitness(bits):
        rows.append(len(bits))
        return np.column_stack([bits[:, :4].sum(axis=1), bits[:, 4:].sum(axis=1)])

    # one child per generation, often a copy of a parent when copies are kept
    result = gx.SmsEmoa(
        gx.Binary(8),
        objectives=["maximize", "minimize"],
        population_size=10,
        offspring=1,
        crossover=gx.UniformCrossover(),
        mutation=gx.BitFlip(rate=0.05),
        eliminate_duplicates=False,
        seed=12,
    ).run(fitness, generations=100, batch=True)
    assert all(count > 0 for count in rows)
    assert sum(rows) == result.evaluations
    assert len(rows) < result.generations + 1


def test_duplicates_are_eliminated_by_default():
    def fitness(bits):
        return np.column_stack([bits[:, :4].sum(axis=1), bits[:, 4:].sum(axis=1)])

    def run(**settings):
        return gx.Nsga2(
            gx.Binary(12),
            objectives=["maximize", "minimize"],
            population_size=20,
            crossover=gx.UniformCrossover(),
            mutation=gx.BitFlip(rate=0.02),
            seed=3,
            **settings,
        ).run(fitness, generations=30, batch=True)

    # without copies, every child is evaluated; with them, copies of parents aren't
    assert run().evaluations == 20 * 31
    assert run(eliminate_duplicates=False).evaluations < 20 * 31
    # MOEA/D replaces its neighbors one child at a time, without duplicate elimination
    with pytest.raises(TypeError, match="eliminate_duplicates"):
        gx.Moead(
            gx.Binary(12),
            objectives=["maximize", "minimize"],
            weights=gx.das_dennis(2, 9),
            crossover=gx.UniformCrossover(),
            mutation=gx.BitFlip(rate=0.02),
            eliminate_duplicates=True,
        )


def test_multi_objective_settings_errors():
    genome = gx.Real((0.0, 1.0), length=4)
    two = ["minimize", "minimize"]
    operators = {
        "crossover": gx.SimulatedBinaryCrossover(),
        "mutation": gx.PolynomialMutation(rate=0.25),
    }
    with pytest.raises(ValueError, match="has a value per objective, 2, not 3"):
        gx.Nsga3(
            genome, objectives=two, reference_directions=gx.das_dennis(3, 4), **operators
        ).run(zdt1_one, generations=1)
    with pytest.raises(ValueError, match="2-D array"):
        gx.Moead(genome, objectives=two, weights=[0.5, 0.5], **operators).run(
            zdt1_one, generations=1
        )
    with pytest.raises(ValueError, match="weights"):
        gx.Moead(genome, objectives=two, weights=[[1.0, -1.0], [0.0, 1.0]], **operators).run(
            zdt1_one, generations=1
        )
    with pytest.raises(ValueError, match="theta"):
        gx.Moead(
            genome,
            objectives=two,
            weights=gx.das_dennis(2, 9),
            decomposition=gx.Pbi(-1.0),
            **operators,
        ).run(zdt1_one, generations=1)
    with pytest.raises(ValueError, match="2 to 6 objectives, not 7"):
        gx.Spea2(genome, objectives=["minimize"] * 7, population_size=10, **operators).run(
            zdt1_one, generations=1
        )
    with pytest.raises(ValueError, match="2 to 6 objectives, not 1"):
        gx.SmsEmoa(genome, objectives=["minimize"], population_size=10, **operators).run(
            zdt1_one, generations=1
        )
    with pytest.raises(ValueError, match="offspring"):
        gx.SmsEmoa(genome, objectives=two, population_size=10, offspring=0, **operators).run(
            zdt1_one, generations=1
        )


def test_on_generation_is_called_after_every_generation():
    main = threading.get_ident()
    progress = []
    threads = set()

    def record(state):
        progress.append(state)
        threads.add(threading.get_ident())

    result = onemax_ga().run(
        lambda bits: bits.sum(), generations=15, parallel=True, on_generation=record
    )
    assert all(isinstance(state, gx.Progress) for state in progress)
    # the initial population is generation 0
    assert [state.generation for state in progress] == list(range(16))
    # on the thread that called run, even with parallel=True
    assert threads == {main}
    evaluations = [state.evaluations for state in progress]
    assert evaluations == sorted(evaluations) and evaluations[-1] == result.evaluations
    seconds = [state.seconds for state in progress]
    assert seconds == sorted(seconds) and seconds[-1] <= result.seconds
    best = [state.best_fitness for state in progress]
    assert best == sorted(best) and best[-1] == result.best_fitness
    # the population, its scores and the best genome so far
    last = progress[-1]
    assert last.population.dtype == bool and last.population.shape[0] == len(last.scores)
    assert np.array_equal(last.scores, last.population.sum(axis=1))
    assert np.all(last.violations == 0)
    assert np.array_equal(last.best_genome, result.best_genome)
    assert all(state.scores.max() <= state.best_fitness for state in progress)
    with pytest.raises(dataclasses.FrozenInstanceError):
        progress[0].generation = 3

    progress.clear()
    result = zdt1_algorithm("spea2").run(
        zdt1, generations=10, batch=True, on_generation=progress.append
    )
    assert all(isinstance(state, gx.MultiProgress) for state in progress)
    assert [state.generation for state in progress] == list(range(11))
    assert progress[-1].evaluations == result.evaluations
    assert all(1 <= state.front_size <= 40 for state in progress)
    # the result's front is without copies
    assert progress[-1].front_size >= len(result.front_objectives)
    for state in progress:
        assert state.objectives.shape == (len(state.population), 2)
        assert np.array_equal(state.objectives, zdt1(state.population))
        assert state.front_objectives.shape == (state.front_size, 2)
        assert len(state.violations) == len(state.population)
        assert len(state.front_violations) == state.front_size
    assert {tuple(row) for row in progress[-1].front_objectives} == {
        tuple(row) for row in result.front_objectives
    }


def test_on_generation_returning_false_aborts():
    result = onemax_ga().run(
        lambda bits: bits.sum(), generations=100, on_generation=lambda state: state.generation < 5
    )
    assert result.stop_reason == "aborted"
    assert result.generations == 5
    # numpy's False too
    result = zdt1_algorithm("moead").run(
        zdt1,
        generations=100,
        batch=True,
        on_generation=lambda state: np.bool_(state.generation < 3),
    )
    assert result.stop_reason == "aborted"
    assert result.generations == 3
    assert_non_dominated(result.front_objectives)
    # anything else goes on
    result = onemax_ga().run(lambda bits: bits.sum(), generations=5, on_generation=lambda state: 0)
    assert result.stop_reason == "generations"


def test_an_exception_in_on_generation_is_raised():
    calls = []

    def callback(state):
        calls.append(state)
        if state.generation == 3:
            raise KeyError("enough")

    with pytest.raises(KeyError, match="enough"):
        onemax_ga().run(lambda bits: bits.sum(), generations=100, on_generation=callback)
    # the run stops after the exception
    assert len(calls) == 4

    def multi_callback(state):
        raise ZeroDivisionError

    with pytest.raises(ZeroDivisionError):
        zdt1_algorithm("nsga3").run(
            zdt1, generations=100, batch=True, on_generation=multi_callback
        )
    with pytest.raises(TypeError, match="on_generation"):
        onemax_ga().run(lambda bits: bits.sum(), generations=1, on_generation=42)


def test_an_exception_in_the_fitness_function_is_raised():
    calls = []

    def fitness(bits):
        calls.append(1)
        if len(calls) == 50:
            raise KeyError("boom")
        return bits.sum()

    with pytest.raises(KeyError, match="boom"):
        onemax_ga().run(fitness, generations=1_000)
    # the run stops after the exception
    assert len(calls) == 50

    def batch(bits):
        raise ZeroDivisionError

    with pytest.raises(ZeroDivisionError):
        onemax_ga().run(batch, generations=10, batch=True)


def test_an_exception_in_a_parallel_run_is_raised():
    def fitness(bits):
        raise RuntimeError("from a thread")

    with pytest.raises(RuntimeError, match="from a thread"):
        onemax_ga().run(fitness, generations=10, parallel=True)


def test_wrong_results_are_errors():
    with pytest.raises(TypeError, match="returns a number"):
        onemax_ga().run(lambda bits: "one", generations=1)
    with pytest.raises(ValueError, match="returned 3 scores for 40 genomes"):
        onemax_ga().run(lambda bits: [1.0, 2.0, 3.0], generations=1, batch=True)
    nsga2 = gx.Nsga2(
        gx.Binary(8),
        objectives=["maximize", "maximize"],
        population_size=8,
        crossover=gx.UniformCrossover(),
        mutation=gx.BitFlip(count=1),
    )
    with pytest.raises(ValueError, match="returned 3 scores, for 2 objectives"):
        nsga2.run(lambda bits: [1.0, 2.0, 3.0], generations=1)


def test_nan_is_an_invalid_solution():
    result = onemax_ga().run(lambda bits: float("nan") if bits[0] else bits.sum(), generations=30)
    assert not result.best_genome[0]


def test_settings_errors():
    with pytest.raises(ValueError, match="stop condition"):
        onemax_ga().run(lambda bits: 0.0)
    with pytest.raises(ValueError, match="doesn't work with binary genomes"):
        gx.Ga(
            gx.Binary(8),
            population_size=10,
            select=gx.Tournament(2),
            crossover=gx.OrderCrossover(),
            mutation=gx.BitFlip(count=1),
        ).run(lambda bits: 0.0, generations=1)
    with pytest.raises(TypeError, match="population_size"):
        gx.Pso(gx.Real((0.0, 1.0), length=2))
    with pytest.raises(ValueError, match="Real genome"):
        gx.Cmaes(gx.Binary(8)).run(lambda bits: 0.0, generations=1)
    with pytest.raises(ValueError, match="rate"):
        gx.BitFlip()._describe()
    with pytest.raises(ValueError, match="length"):
        gx.Real((0.0, 1.0))._describe()
    with pytest.raises(ValueError):
        onemax_ga(select=gx.Tournament(0)).run(lambda bits: 0.0, generations=1)
    with pytest.raises(ValueError, match="maximize"):
        onemax_ga(objective="max").run(lambda bits: 0.0, generations=1)
    with pytest.raises(TypeError, match="callable"):
        onemax_ga().run(42, generations=1, batch=True)


def test_numpy_numbers_in_the_settings():
    result = gx.Ga(
        gx.Binary(np.int64(16)),
        population_size=np.int64(10),
        select=gx.Tournament(np.int32(2)),
        crossover=gx.UniformCrossover(),
        mutation=gx.BitFlip(rate=np.float64(0.1)),
        seed=np.uint64(3),
    ).run(lambda bits: bits.sum(), generations=np.int64(3))
    assert result.generations == 3


def test_time_and_stagnation():
    result = onemax_ga().run(lambda bits: 1.0, stagnation=5)
    assert result.stop_reason == "stagnation"
    result = onemax_ga().run(lambda bits: bits.sum(), time=0.05)
    assert result.stop_reason == "time"


@pytest.mark.skipif(not hasattr(signal, "pthread_kill"), reason="needs pthread_kill")
def test_ctrl_c_stops_the_run():
    main = threading.get_ident()
    timer = threading.Timer(0.2, lambda: signal.pthread_kill(main, signal.SIGINT))
    timer.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            onemax_ga().run(lambda bits: 1.0, time=30)
    finally:
        timer.cancel()


def test_other_threads_run_during_a_run():
    ticks = []
    done = threading.Event()

    def tick():
        while not done.is_set():
            ticks.append(1)
            done.wait(0.001)

    thread = threading.Thread(target=tick)
    thread.start()
    try:
        gx.Cmaes(gx.Real((-1.0, 1.0), length=3), seed=1).run(rastrigin_batch, time=0.2, batch=True)
    finally:
        done.set()
        thread.join()
    assert len(ticks) > 10


def test_version():
    assert gx.__version__.count(".") == 2


# --- settings: NaN, infinity, wrong types and numbers that aren't whole --------------------------


def test_no_crossover_needs_a_mutation_rate_above_zero():
    # with neither crossover nor mutation, every child is a copy of its parent and the run stalls
    total = lambda x: float(x.sum())
    for genome, mutation, fitness in [
        (gx.Binary(8), gx.BitFlip(count=1), total),
        (gx.Integer((0, 9), length=4), gx.UniformMutation(count=1), total),
        (gx.Real((0.0, 1.0), length=4), gx.PolynomialMutation(rate=0.5), total),
        (gx.Permutation(5), gx.SwapMutation(), lambda x: float(x[0])),
    ]:
        ga = gx.Ga(
            genome,
            population_size=6,
            select=gx.Tournament(2),
            crossover=gx.NoCrossover(),
            mutation=mutation,
            mutation_rate=0.0,
            seed=1,
        )
        with pytest.raises(ValueError, match="mutation_rate"):
            ga.run(fitness, generations=5)
    nsga2 = gx.Nsga2(
        gx.Binary(8),
        objectives=["maximize"] * 2,
        population_size=6,
        crossover=gx.NoCrossover(),
        mutation=gx.BitFlip(count=1),
        mutation_rate=0.0,
    )
    with pytest.raises(ValueError, match="mutation_rate"):
        nsga2.run(lambda x: [float(x.sum()), float(x[0])], generations=5)
    # a crossover that recombines may run without mutation
    result = onemax_ga(mutation_rate=0.0).run(total, generations=5)
    assert result.generations == 5


def test_common_setting_mistakes_name_the_setting():
    zero = lambda x: 0.0
    # an operator, genome, scheme or acceptance class without ()
    with pytest.raises(ValueError, match=r"crossover is a crossover .*: pass UniformCrossover\(\.\.\.\)"):
        onemax_ga(crossover=gx.UniformCrossover).run(zero, generations=1)
    with pytest.raises(ValueError, match=r"scheme is a scheme .*: pass Generational\(\.\.\.\)"):
        onemax_ga(scheme=gx.Generational).run(zero, generations=1)
    with pytest.raises(ValueError, match=r"genome is a genome .*: pass Binary\(\.\.\.\)"):
        gx.Ga(
            gx.Binary,
            population_size=10,
            select=gx.Tournament(2),
            crossover=gx.UniformCrossover(),
            mutation=gx.BitFlip(count=1),
        ).run(zero, generations=1)
    with pytest.raises(ValueError, match=r"acceptance is an acceptance .*: pass NotWorse\(\.\.\.\)"):
        gx.LocalSearch(gx.Binary(8), neighbor=gx.BitFlip(count=1), acceptance=gx.NotWorse).run(
            zero, generations=1
        )
    # a string or a number instead of an object
    with pytest.raises(ValueError, match=r"select is a selection such as gx.Tournament\(3\), not 'tournament'"):
        onemax_ga(select="tournament").run(zero, generations=1)
    with pytest.raises(ValueError, match="genome is a genome such as .*, not 5"):
        gx.Ga(
            5,
            population_size=10,
            select=gx.Tournament(2),
            crossover=gx.UniformCrossover(),
            mutation=gx.BitFlip(count=1),
        ).run(zero, generations=1)
    # a string instead of a list of objectives
    nsga2 = gx.Nsga2(
        gx.Binary(5),
        objectives="minimize",
        population_size=10,
        crossover=gx.UniformCrossover(),
        mutation=gx.BitFlip(count=1),
    )
    with pytest.raises(ValueError, match="objectives is a list, one"):
        nsga2.run(lambda x: [0.0, 0.0], generations=1)
    # whole numbers beyond what Rust reads
    with pytest.raises(ValueError, match="seed is at most 18446744073709551615, not"):
        onemax_ga(seed=2**64).run(zero, generations=1)
    with pytest.raises(ValueError, match="population_size is at most"):
        onemax_ga(population_size=2**64).run(zero, generations=1)
    with pytest.raises(ValueError, match="at most"):
        onemax_ga(select=gx.Tournament(2**70)).run(zero, generations=1)
    with pytest.raises(ValueError, match="Integer.bounds are at most 9223372036854775807"):
        gx.Ga(
            gx.Integer((0, 2**63), length=2),
            population_size=10,
            select=gx.Tournament(2),
            crossover=gx.UniformCrossover(),
            mutation=gx.UniformMutation(count=1),
        ).run(zero, generations=1)
    # a tuple that isn't (score, violation)
    with pytest.raises(TypeError, match=r"tuple is \(score, constraint violation\), two numbers, not \(float, NoneType\)"):
        onemax_ga().run(lambda x: (1.0, None), generations=1)


def test_an_infinite_time_limit_is_no_limit():
    result = onemax_ga().run(lambda bits: 0.0, generations=3, time=math.inf)
    assert result.stop_reason == "generations"
    with pytest.raises(ValueError, match="stop condition"):
        onemax_ga().run(lambda bits: 0.0, time=math.inf)


def test_nan_and_infinite_settings_are_errors_that_name_them():
    with pytest.raises(ValueError, match="Real.bounds are finite numbers, not nan"):
        gx.Cmaes(gx.Real((0.0, math.nan), length=2)).run(lambda x: 0.0, generations=1)
    with pytest.raises(ValueError, match="Real.bounds"):
        gx.Cmaes(gx.Real([(0.0, 1.0), (-math.inf, 1.0)])).run(lambda x: 0.0, generations=1)
    with pytest.raises(ValueError, match="BitFlip.rate is a finite number"):
        onemax_ga(mutation=gx.BitFlip(rate=math.nan)).run(lambda bits: 0.0, generations=1)
    with pytest.raises(ValueError, match="mutation_rate"):
        onemax_ga(mutation_rate=math.inf).run(lambda bits: 0.0, generations=1)
    with pytest.raises(ValueError, match="target"):
        onemax_ga().run(lambda bits: 0.0, target=math.inf)
    with pytest.raises(ValueError, match="time"):
        onemax_ga().run(lambda bits: 0.0, time=math.nan)
    with pytest.raises(ValueError, match="time"):
        onemax_ga().run(lambda bits: 0.0, time=-math.inf)
    with pytest.raises(ValueError, match="Annealing.cooling"):
        gx.LocalSearch(
            gx.Binary(8), neighbor=gx.BitFlip(count=1), acceptance=gx.Annealing(1.0, math.nan)
        ).run(lambda bits: 0.0, generations=1)
    with pytest.raises(ValueError, match="reference_directions are finite numbers"):
        gx.Nsga3(
            gx.Real((0.0, 1.0), length=3),
            objectives=["minimize", "minimize"],
            reference_directions=[[0.0, 1.0], [math.nan, 0.5]],
            crossover=gx.SimulatedBinaryCrossover(),
            mutation=gx.PolynomialMutation(rate=0.3),
        ).run(lambda x: (0.0, 0.0), generations=1)


def test_a_wrong_type_is_an_error_that_names_the_setting():
    with pytest.raises(ValueError, match="generations is a whole number, not 3.0"):
        onemax_ga().run(lambda bits: 0.0, generations=3.0)
    with pytest.raises(ValueError, match="population_size is a whole number"):
        onemax_ga(population_size=10.5).run(lambda bits: 0.0, generations=1)
    with pytest.raises(ValueError, match="seed is a whole number, not True"):
        onemax_ga(seed=True).run(lambda bits: 0.0, generations=1)
    with pytest.raises(ValueError, match="Tournament.size is at least 0, not -2"):
        onemax_ga(select=gx.Tournament(-2)).run(lambda bits: 0.0, generations=1)
    with pytest.raises(ValueError, match="Rank.pressure is a finite number, not '1.5'"):
        onemax_ga(select=gx.Rank("1.5")).run(lambda bits: 0.0, generations=1)
    with pytest.raises(ValueError, match="restart kicks"):
        gx.LocalSearch(gx.Binary(8), neighbor=gx.BitFlip(count=1), restart=(5, 2.5)).run(
            lambda bits: 0.0, generations=1
        )
    with pytest.raises(ValueError, match="restarts"):
        gx.Cmaes(gx.Real((0.0, 1.0), length=2), restarts="always").run(
            lambda x: 0.0, generations=1
        )
    real = gx.Real((0.0, 1.0), length=2)
    for settings, message in [
        ({"strategy": "rand2"}, 'strategy is "rand1", "best1"'),
        ({"strategy": {"p": 0.1}}, "strategy is"),
        ({"strategy": {"p": 0.1, "archive": 1.0, "max_p": 0.2}}, "strategy is"),
        ({"strategy": {"p": "0.1", "archive": 1.0}}, "strategy.p is a finite number"),
        ({"control": "shade"}, 'control is {"f": ..., "cr": ...}'),
        ({"control": {"f": 0.5}}, "control is"),
        ({"control": {"memory": 6.0}}, "control.memory is a whole number"),
        ({"restarts": "always"}, 'restarts is "never" or'),
        ({"restarts": {"tolerance": 1e-12}}, "restarts is"),
        ({"restarts": {"tolerance": float("nan"), "patience": 200}}, "restarts.tolerance is a finite number"),
        ({"restarts": {"tolerance": 1e-12, "patience": 2.5}}, "restarts.patience is a whole number"),
    ]:
        with pytest.raises(ValueError, match=re.escape(message)):
            gx.De(real, **settings).run(lambda x: 0.0, generations=1)
    with pytest.raises(ValueError, match="eliminate_duplicates is True or False"):
        gx.Nsga2(
            gx.Binary(8),
            objectives=["maximize", "maximize"],
            population_size=8,
            crossover=gx.UniformCrossover(),
            mutation=gx.BitFlip(count=1),
            eliminate_duplicates=1,
        ).run(lambda bits: (0.0, 0.0), generations=1)
    with pytest.raises(ValueError, match="divisions is a whole number"):
        gx.das_dennis(3, 12.0)


@pytest.mark.parametrize(
    "algorithm, name",
    [
        (onemax_ga(select=gx.Tournament(0)), "`Tournament.size`"),
        (onemax_ga(select=gx.Rank(2.5)), "`Rank.pressure`"),
        (onemax_ga(crossover=gx.PointCrossover(0)), "`PointCrossover.points`"),
        (onemax_ga(mutation=gx.BitFlip(rate=1.5)), "`BitFlip.rate`"),
        (onemax_ga(scheme=gx.MuPlusLambda(0)), "`scheme`: offspring must be at least 1"),
        (
            gx.Ga(
                gx.Real((0.0, 1.0), length=3),
                population_size=10,
                select=gx.Tournament(2),
                crossover=gx.SimulatedBinaryCrossover(-1.0),
                mutation=gx.GaussianMutation(0.1, rate=0.5),
            ),
            "`SimulatedBinaryCrossover.eta`",
        ),
        (
            gx.Ga(
                gx.Real((0.0, 1.0), length=3),
                population_size=10,
                select=gx.Tournament(2),
                crossover=gx.ArithmeticCrossover(),
                mutation=gx.GaussianMutation(0.0, count=1),
            ),
            "`GaussianMutation.sigma`",
        ),
        (
            gx.LocalSearch(gx.Permutation(0), neighbor=gx.SwapMutation()),
            "`Permutation.length`",
        ),
        (
            gx.LocalSearch(gx.Integer([(3, 1)]), neighbor=gx.UniformMutation(count=1)),
            "`Integer.bounds`",
        ),
        (
            gx.LocalSearch(gx.Binary(8), neighbor=gx.BitFlip(count=1), acceptance=gx.Tabu(0)),
            "`Tabu.tenure`",
        ),
        (gx.Pso(gx.Real((0.0, 1.0), length=2), population_size=10, ring=0), "`ring`"),
        (gx.De(gx.Real((0.0, 1.0), length=2), l_shade=0), "`l_shade`"),
        (gx.De(gx.Real((0.0, 1.0), length=2), strategy={"p": 0.0, "archive": 1.0}), "`strategy.p`"),
        (gx.De(gx.Real((0.0, 1.0), length=2), strategy={"max_p": 1.5, "archive": 1.0}), "`strategy.max_p`"),
        (gx.De(gx.Real((0.0, 1.0), length=2), strategy={"p": 0.1, "archive": -1.0}), "`strategy.archive`"),
        (gx.De(gx.Real((0.0, 1.0), length=2), control={"f": 2.5, "cr": 0.9}), "`control.f`"),
        (gx.De(gx.Real((0.0, 1.0), length=2), control={"min_f": 0.8, "max_f": 0.5, "cr": 0.9}), "`control.f`"),
        (gx.De(gx.Real((0.0, 1.0), length=2), control={"f": 0.5, "cr": 1.5}), "`control.cr`"),
        (gx.De(gx.Real((0.0, 1.0), length=2), control={"c": 0.0}), "`control.c`"),
        (gx.De(gx.Real((0.0, 1.0), length=2), control={"memory": 0}), "`control.memory`"),
        (gx.De(gx.Real((0.0, 1.0), length=2), restarts={"tolerance": -1.0, "patience": 200}), "`restarts`"),
        (gx.De(gx.Real((0.0, 1.0), length=2), restarts={"tolerance": 1e-12, "patience": 0}), "`restarts`"),
    ],
)
def test_a_wrong_setting_is_named_as_in_python(algorithm, name):
    with pytest.raises(ValueError, match=name):
        algorithm.run(lambda genome: 0.0, generations=1)


def test_a_negative_constraint_violation_is_an_error():
    with pytest.raises(ValueError, match="violation can't be negative"):
        onemax_ga().run(lambda bits: (bits.sum(), -1.0), generations=1)


def test_the_native_run_names_a_wrong_setting():
    ga = onemax_ga()
    run = {
        "genome": ga._genome._describe(),
        "algorithm": ga._describe(),
        "objectives": ["maximize"],
        "stop": {"generations": 3.5},
    }
    with pytest.raises(ValueError, match="`stop.generations`"):
        gx._genoxide.run(json.dumps(run), lambda bits: 0.0)


def test_integer_bounds_that_are_not_whole_numbers_are_an_error():
    # int() truncates: (-0.5, 3.7) would be (0, 3)
    with pytest.raises(ValueError, match="Integer.bounds are whole numbers, not -0.5"):
        gx.Integer((-0.5, 3.7), length=2)._describe()
    with pytest.raises(ValueError, match="Integer.bounds"):
        gx.Integer([(0, 5), (1.0, 3)])._describe()
    # numpy integers are whole numbers
    bounds = np.array([[0, 5], [-3, 3]], dtype=np.int64)
    assert gx.Integer(bounds)._describe()["bounds"] == [[0, 5], [-3, 3]]
    genome = gx.Integer((np.int32(1), np.int64(4)), length=np.int64(2))
    assert genome._describe()["bounds"] == [[1, 4], [1, 4]]


def test_a_bool_length_is_an_error():
    # range(True) is range(1): one gene
    with pytest.raises(ValueError, match="Real.length is a whole number, not True"):
        gx.Real((0.0, 1.0), length=True)._describe()
    with pytest.raises(ValueError, match="Binary.length is a whole number, not 8.0"):
        gx.Binary(8.0)._describe()
    with pytest.raises(ValueError, match="Permutation.length"):
        gx.Permutation(np.True_)._describe()


# --- fitness values that fail to convert keep their error ---------------------------------------


def test_a_fitness_value_too_large_for_a_float_raises_overflow_error():
    with pytest.raises(OverflowError):
        onemax_ga().run(lambda bits: 10**400, generations=1)
    nsga2 = gx.Nsga2(
        gx.Binary(8),
        objectives=["maximize", "maximize"],
        population_size=8,
        crossover=gx.UniformCrossover(),
        mutation=gx.BitFlip(count=1),
    )
    with pytest.raises(OverflowError):
        nsga2.run(lambda bits: (10**400, 1.0), generations=1)
    with pytest.raises(OverflowError):
        nsga2.run(lambda bits: [1.0, 10**400], generations=1)


def test_the_exception_of_a_fitness_values_float_conversion_is_kept():
    class Score:
        def __float__(self):
            raise ZeroDivisionError("the real cause")

    with pytest.raises(ZeroDivisionError, match="the real cause"):
        onemax_ga().run(lambda bits: Score(), generations=1)


def test_a_fitness_value_that_is_not_a_number_is_caused_by_its_conversion_error():
    with pytest.raises(TypeError, match="returns a number") as raised:
        onemax_ga().run(lambda bits: "one", generations=1)
    assert isinstance(raised.value.__cause__, TypeError)


# --- batch results ------------------------------------------------------------------------------


def test_batch_scores_and_violations_as_columns():
    def fitness(bits):
        return bits.sum(axis=1, keepdims=True), np.zeros((len(bits), 1))

    columns = onemax_ga().run(fitness, generations=3, batch=True)
    rows = onemax_ga().run(
        lambda bits: (bits.sum(axis=1), np.zeros(len(bits))), generations=3, batch=True
    )
    assert columns.generations == 3
    assert columns.best_fitness == rows.best_fitness


def test_a_1d_multi_objective_batch_result_is_a_clear_error():
    nsga2 = gx.Nsga2(
        gx.Binary(5),
        objectives=["minimize", "minimize"],
        population_size=10,
        crossover=gx.UniformCrossover(),
        mutation=gx.BitFlip(rate=0.1),
    )
    with pytest.raises(ValueError, match="2-D array, a row of objective values per genome"):
        nsga2.run(lambda bits: bits.sum(axis=1), generations=1, batch=True)


def zdt1_columns(x):
    g = 1 + 9 * x[:, 1:].sum(axis=1) / (x.shape[1] - 1)
    return x[:, 0], g * (1 - np.sqrt(x[:, 0] / g))


def test_a_multi_objective_batch_may_return_an_array_per_objective():
    def run(fitness, population_size, offspring):
        return gx.SmsEmoa(
            gx.Real((0.0, 1.0), length=5),
            objectives=["minimize", "minimize"],
            population_size=population_size,
            offspring=offspring,
            crossover=gx.SimulatedBinaryCrossover(),
            mutation=gx.PolynomialMutation(rate=0.2),
            seed=1,
        ).run(fitness, generations=20, batch=True)

    # a tuple of columns is the same as the rows, also for batches of 2 genomes and 2 objectives
    for population_size, offspring in [(2, 2), (20, 20)]:
        columns = run(zdt1_columns, population_size, offspring)
        rows = run(lambda x: np.column_stack(zdt1_columns(x)), population_size, offspring)
        assert columns.front_objectives.tolist() == rows.front_objectives.tolist()
        true_values = np.column_stack(zdt1_columns(columns.front_genomes))
        assert np.allclose(columns.front_objectives, true_values)
    # with constraint violations
    result = run(lambda x: (zdt1_columns(x), np.zeros(len(x))), 20, 20)
    assert np.allclose(result.front_objectives, np.column_stack(zdt1_columns(result.front_genomes)))


def test_a_transposed_multi_objective_batch_result_is_a_clear_error():
    nsga2 = gx.Nsga2(
        gx.Real((0.0, 1.0), length=5),
        objectives=["minimize", "minimize"],
        population_size=20,
        crossover=gx.SimulatedBinaryCrossover(),
        mutation=gx.PolynomialMutation(rate=0.2),
    )
    with pytest.raises(ValueError, match=r"returned 2 rows of 20 values.*np.column_stack"):
        nsga2.run(lambda x: np.array(zdt1_columns(x)), generations=1, batch=True)


# --- invalid solutions --------------------------------------------------------------------------


def test_an_invalid_solution_has_a_nan_violation():
    result = onemax_ga().run(lambda bits: None, generations=2)
    assert result.best_fitness is None
    assert math.isnan(result.violation)


def test_invalid_front_members_have_no_violation_of_0():
    result = gx.Nsga2(
        gx.Real((0.0, 1.0), length=3),
        objectives=["minimize", "minimize"],
        population_size=10,
        crossover=gx.SimulatedBinaryCrossover(),
        mutation=gx.PolynomialMutation(rate=0.3),
        seed=1,
    ).run(lambda x: None, generations=2)
    assert np.isnan(result.front_objectives).all()
    assert np.isnan(result.front_violations).all()


def test_a_run_that_can_only_make_copies_stops_as_stalled():
    # every child of a 1-gene permutation is a copy, so 100 evaluations are never reached
    ga = gx.Ga(
        gx.Permutation(1),
        population_size=4,
        select=gx.Tournament(2),
        crossover=gx.OrderCrossover(),
        mutation=gx.SwapMutation(),
        seed=0,
    )
    result = ga.run(lambda order: 1.0, evaluations=100)
    assert result.stop_reason == "stalled"
    assert result.evaluations == 4


# --- batch and parallel -------------------------------------------------------------------------


def test_batch_and_parallel_accept_ints():
    assert onemax_ga().run(lambda bits: 0.0, generations=1, parallel=1).generations == 1
    assert onemax_ga().run(lambda bits: bits.sum(axis=1), generations=1, batch=1).generations == 1
    assert onemax_ga().run(lambda bits: bits.sum(), generations=1, batch=0).generations == 1


# --- packaging ----------------------------------------------------------------------------------


def test_the_package_has_the_license_files():
    files = importlib.metadata.distribution("genoxide").files or []
    names = {pathlib.PurePath(str(file)).name for file in files}
    assert {"LICENSE-MIT", "LICENSE-APACHE"} <= names, sorted(names)
