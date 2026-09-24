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
    return gx.Ga(
        gx.Binary(64),
        population_size=40,
        select=gx.Tournament(3),
        crossover=gx.UniformCrossover(),
        mutation=gx.BitFlip(rate=1 / 64),
        seed=7,
        **settings,
    )


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


@pytest.mark.parametrize(
    "algorithm",
    [
        gx.De(gx.Real((-5.12, 5.12), length=5), objective="minimize", seed=1),
        gx.De(gx.Real((-5.12, 5.12), length=5), l_shade=40_000, objective="minimize", seed=1),
        gx.Cmaes(gx.Real((-5.12, 5.12), length=5), restarts="ipop", objective="minimize", seed=1),
        gx.Pso(gx.Real((-5.12, 5.12), length=5), ring=2, objective="minimize", seed=1),
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

    result = gx.De(gx.Real((0.0, 1.0), length=1), seed=9).run(fitness, generations=40, batch=True)
    assert result.violation == 0
    assert result.best_genome[0] == pytest.approx(0.5, abs=0.01)


def test_nsga2_finds_a_front():
    def zdt1(x):
        f1 = x[:, 0]
        g = 1 + 9 * x[:, 1:].mean(axis=1)
        return np.column_stack([f1, g * (1 - np.sqrt(f1 / g))])

    nsga2 = gx.Nsga2(
        gx.Real((0.0, 1.0), length=10),
        objectives=["minimize", "minimize"],
        population_size=40,
        crossover=gx.SimulatedBinaryCrossover(15),
        mutation=gx.PolynomialMutation(20, rate=0.1),
        seed=10,
    )
    result = nsga2.run(zdt1, evaluations=8_000, batch=True)
    front = result.front_objectives
    assert front.shape[1] == 2 and len(front) == len(result.front_genomes) > 5
    assert result.front_genomes.shape[1] == 10
    assert np.all(result.front_violations == 0)
    # no member dominates another
    for a in front:
        assert not np.any(np.all(front <= a, axis=1) & np.any(front < a, axis=1))
    # the same run, a genome at a time
    one = nsga2.run(lambda x: zdt1(x[None, :])[0], evaluations=8_000)
    assert np.allclose(np.sort(one.front_objectives, axis=0), np.sort(front, axis=0))


def test_nsga2_with_three_objectives_and_tuples():
    result = gx.Nsga2(
        gx.Binary(20),
        objectives=["maximize", "maximize", "minimize"],
        population_size=20,
        crossover=gx.UniformCrossover(),
        mutation=gx.BitFlip(rate=0.05),
        seed=11,
    ).run(lambda bits: (bits[:10].sum(), bits[10:].sum(), bits.sum()), generations=10)
    assert result.front_objectives.shape[1] == 3
    assert result.front_genomes.dtype == np.bool_


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
