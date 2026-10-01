"""gx.OpenEs, OpenAI's evolution strategy, with gx.Adam, gx.Sgd and gx.RunningOpenEs."""

import numpy as np
import pytest

import genoxide as gx


def sphere(x):
    return float(np.sum(x * x))


def open_es(**settings):
    defaults = {
        "population_size": 50,
        "sigma": 0.01,
        "optimizer": gx.Adam(0.003),
        "evaluate_mean": True,
        "objective": "minimize",
        "seed": 1,
    }
    return gx.OpenEs(gx.Real((-5.0, 5.0), length=100), **{**defaults, **settings})


def same(a, b):
    return (
        a.best_fitness == b.best_fitness
        and np.array_equal(a.best_genome, b.best_genome)
        and (a.generations, a.evaluations, a.stop_reason)
        == (b.generations, b.evaluations, b.stop_reason)
    )


def test_open_es_solves_the_sphere():
    # the Rust doc example's run: 100 genes, to below 0.1
    result = open_es().run(sphere, target=0.1, generations=2_000)
    assert result.stop_reason == "target"
    assert result.best_fitness <= 0.1
    assert result.best_genome.dtype == np.float64 and result.best_genome.shape == (100,)
    # 50 samples and the mean per generation, the initial one included
    assert result.evaluations == 51 * (result.generations + 1)


def test_a_problem_evaluated_in_rust_equals_the_same_python_function():
    problem = gx.problems.Sphere(10)
    result = gx.OpenEs(
        problem.genome, population_size=20, objective="minimize", seed=3
    ).run(problem, generations=30)
    again = gx.OpenEs(
        problem.genome, population_size=20, objective="minimize", seed=3
    ).run(lambda x: float(np.cumsum(x * x)[-1]), generations=30)
    assert result.evaluations == again.evaluations == 20 * 31
    assert np.array_equal(result.best_genome, again.best_genome)


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"parallel_breeding": True},
        {"optimizer": gx.Sgd(0.01, 0.9)},
        {"optimizer": gx.Adam(0.01, beta1=0.8, beta2=0.99)},
        {"weight_decay": 0.01},
        {"evaluate_mean": False},
    ],
)
def test_seeded_runs_repeat(settings):
    first = open_es(**settings).run(sphere, generations=20)
    assert same(first, open_es(**settings).run(sphere, generations=20))
    # one genome at a time, in batches or in parallel: the same run
    parallel = open_es(**settings).run(sphere, generations=20, parallel=True)
    assert same(first, parallel)
    batch = open_es(**settings).run(
        lambda x: np.sum(x * x, axis=1), generations=20, batch=True
    )
    assert batch.evaluations == first.evaluations
    assert batch.best_fitness == pytest.approx(first.best_fitness, rel=1e-12)


def test_parallel_breeding_draws_other_samples():
    serial = open_es().run(sphere, generations=10)
    parallel = open_es(parallel_breeding=True).run(sphere, generations=10)
    assert not np.array_equal(serial.best_genome, parallel.best_genome)


def test_the_initial_mean_is_respected():
    mean = np.linspace(-1.0, 1.0, 100)
    seen = []
    open_es(initial_mean=mean, evaluate_mean=True).run(
        sphere, generations=0, on_generation=lambda progress: seen.append(progress.population)
    )
    # the samples are mirrored around the mean, which comes last
    population = seen[0]
    assert np.array_equal(population[-1], mean)
    pairs = population[:-1]
    assert np.allclose((pairs[0::2] + pairs[1::2]) / 2, mean)
    # a list does too
    listed = open_es(initial_mean=list(mean)).run(sphere, generations=3)
    assert same(listed, open_es(initial_mean=mean).run(sphere, generations=3))


def test_a_random_genome_matches_rust():
    real = gx.Real((-0.25, 0.25), length=4)
    genome = real.random_genome(1)
    assert np.array_equal(genome, real.random_genome(1))
    assert not np.array_equal(genome, real.random_genome(2))
    assert np.all(np.abs(genome) <= 0.25)


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"population_size": 3}, "population_size"),
        ({"population_size": 0}, "population_size"),
        ({"population_size": 2.0}, "population_size"),
        ({"sigma": 0.0}, "sigma"),
        ({"sigma": float("inf")}, "sigma"),
        ({"weight_decay": -1.0}, "weight_decay"),
        ({"optimizer": gx.Adam(0.0)}, "optimizer"),
        ({"optimizer": gx.Adam(0.01, beta1=1.0)}, "optimizer"),
        ({"optimizer": gx.Sgd(0.01, -0.5)}, "optimizer"),
        ({"optimizer": gx.Adam("fast")}, "Adam.learning_rate"),
        ({"optimizer": "adam"}, "optimizer"),
        ({"evaluate_mean": 1}, "evaluate_mean"),
        ({"parallel_breeding": "yes"}, "parallel_breeding"),
        ({"initial_mean": np.zeros(99)}, "initial_mean"),
        ({"initial_mean": np.full(100, 6.0)}, "initial_mean"),
        ({"initial_mean": np.zeros((10, 10))}, "initial_mean"),
        ({"seed": -1}, "seed"),
        ({"objective": "max"}, "objective"),
    ],
)
def test_settings_errors_name_the_setting(settings, message):
    with pytest.raises(ValueError, match=message):
        open_es(**settings).run(sphere, generations=1)


def test_open_es_needs_a_real_genome():
    for genome in (gx.Binary(8), gx.Integer((0, 5), length=3), gx.Permutation(5)):
        with pytest.raises(ValueError, match="OpenEs needs a Real genome"):
            gx.OpenEs(genome, population_size=10).run(lambda x: 0.0, generations=1)


def test_running_open_es_reads_and_changes_sigma_and_the_learning_rate():
    seen = []

    def control(running, progress):
        assert isinstance(running, gx.RunningOpenEs)
        seen.append((running.sigma, running.learning_rate))
        running.sigma = running.sigma * 0.9
        running.learning_rate = running.learning_rate * 0.9

    result = open_es().run(sphere, generations=3, control=control)
    assert seen[0] == (0.01, 0.003)
    assert seen[1] == pytest.approx((0.009, 0.0027))
    assert seen[3] == pytest.approx((0.01 * 0.9**3, 0.003 * 0.9**3))
    # a decay changes the run
    assert not same(result, open_es().run(sphere, generations=3))


def test_running_open_es_rejects_wrong_values_and_changes_nothing():
    seen = []

    def control(running, progress):
        with pytest.raises(ValueError, match="sigma"):
            running.sigma = 0.0
        with pytest.raises(ValueError, match="optimizer"):
            running.learning_rate = -1.0
        seen.append((running.sigma, running.learning_rate))

    result = open_es().run(sphere, generations=3, control=control)
    assert seen == [(0.01, 0.003)] * 4
    # a control that changes nothing leaves the run as it is
    assert same(result, open_es().run(sphere, generations=3))


def test_running_open_es_reevaluates():
    shift = {"value": 0.0}

    def moving(x):
        return float(np.sum((x - shift["value"]) ** 2))

    progress = []

    def control(running, state):
        if state.generation == 5:
            shift["value"] = 1.0
            running.reevaluate()

    open_es().run(moving, generations=10, on_generation=progress.append, control=control)
    generations = [state.generation for state in progress]
    assert generations == list(range(6)) + [5] + list(range(6, 11))
    rescored = progress[6]
    # the same samples, scored again by the new function
    assert np.array_equal(rescored.population, progress[5].population)
    assert np.allclose(rescored.scores, [moving(genome) for genome in rescored.population])
