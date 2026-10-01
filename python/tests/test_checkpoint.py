"""Checkpoints: run(..., checkpoint=..., checkpoint_every=...) and run(..., resume=...)."""

import pathlib

import numpy as np
import pytest

import genoxide as gx


def sphere(x):
    return float(np.sum(x * x))


def same(a, b):
    return (
        a.best_fitness == b.best_fitness
        and np.array_equal(a.best_genome, b.best_genome)
        and a.evaluations == b.evaluations
        and a.generations == b.generations
    )


real = gx.Real((-5.0, 5.0), length=6)


def ga(genome=real, mutation=None, crossover=None, seed=1, **settings):
    return gx.Ga(
        genome,
        population_size=20,
        select=gx.Tournament(3),
        crossover=crossover or gx.UniformCrossover(),
        mutation=mutation or gx.PolynomialMutation(20.0, rate=0.2),
        objective="minimize",
        seed=seed,
        **settings,
    )


ALGORITHMS = {
    "ga": lambda: ga(),
    "binary ga": lambda: ga(gx.Binary(30), gx.BitFlip(rate=1 / 30)),
    "permutation ga": lambda: ga(gx.Permutation(8), gx.SwapMutation(), gx.OrderCrossover()),
    "self-adaptive ga": lambda: ga(
        gx.AdaptiveReal(real, 0.3), gx.SelfAdaptiveMutation(), scheme=gx.MuCommaLambda(40)
    ),
    "de": lambda: gx.De(real, population_size=20, objective="minimize", seed=1),
    "es": lambda: gx.Es(real, parents=4, offspring=28, objective="minimize", seed=1),
    "cmaes": lambda: gx.Cmaes(real, restarts="ipop", objective="minimize", seed=1),
    "open es": lambda: gx.OpenEs(
        real, population_size=20, evaluate_mean=True, objective="minimize", seed=1
    ),
    "neat": lambda: gx.Neat(2, 1, population_size=30, seed=1),
    "pso": lambda: gx.Pso(real, population_size=20, objective="minimize", seed=1),
    "local search": lambda: gx.LocalSearch(
        real, neighbor=gx.GaussianMutation(0.1, rate=0.5), neighbors=3, objective="minimize", seed=1
    ),
    "islands": lambda: gx.Islands(
        [ga(seed=seed) for seed in range(3)], topology="random", interval=3, seed=5
    ),
}


def xor(network):
    evaluator = network.feed_forward()
    outputs = evaluator.activate(np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]))
    return float((4.0 - np.sum(np.abs(outputs[:, 0] - [0.0, 1.0, 1.0, 0.0]))) ** 2)


def fitness_for(algorithm):
    if isinstance(algorithm, gx.Neat):
        return xor
    genome = algorithm._genome
    if isinstance(genome, gx.Permutation):
        return lambda order: float(np.sum(np.abs(np.diff(order))))
    if isinstance(genome, gx.Binary):
        return lambda bits: float(bits.sum())
    return sphere


@pytest.mark.parametrize("name", ALGORITHMS)
def test_a_resumed_run_equals_an_uninterrupted_one(name, tmp_path):
    algorithm = ALGORITHMS[name]()
    fitness = fitness_for(algorithm)
    path = tmp_path / "run.ckpt"
    whole = algorithm.run(fitness, generations=40)
    first = algorithm.run(fitness, generations=17, checkpoint=path, checkpoint_every=5)
    assert first.generations == 17
    # saved when it stopped, at generation 17
    resumed = algorithm.run(fitness, generations=40, resume=path)
    assert same(whole, resumed)
    # and from generation 15, saved every 5
    algorithm.run(
        fitness,
        generations=17,
        checkpoint=path,
        checkpoint_every=5,
        on_generation=lambda p: p.generation < 15,
    )
    progress = []
    resumed = algorithm.run(
        fitness, generations=40, resume=path, on_generation=lambda p: progress.append(p.generation)
    )
    assert progress[0] == 16
    assert same(whole, resumed)


def test_a_resumed_multi_objective_run_equals_an_uninterrupted_one(tmp_path):
    nsga2 = gx.Nsga2(
        gx.Real((0.0, 1.0), length=5),
        objectives=["minimize", "minimize"],
        population_size=20,
        crossover=gx.SimulatedBinaryCrossover(15),
        mutation=gx.PolynomialMutation(20, rate=0.2),
        seed=1,
    )

    def zdt1(x):
        g = 1 + 9 * float(np.sum(x[1:])) / 4
        return x[0], g * (1 - np.sqrt(x[0] / g))

    path = tmp_path / "front.ckpt"
    whole = nsga2.run(zdt1, generations=30)
    nsga2.run(zdt1, generations=12, checkpoint=str(path), checkpoint_every=4)
    resumed = nsga2.run(zdt1, generations=30, resume=str(path))
    assert np.array_equal(whole.front_objectives, resumed.front_objectives)
    assert np.array_equal(whole.front_genomes, resumed.front_genomes)
    assert (whole.evaluations, whole.generations) == (resumed.evaluations, resumed.generations)


def test_a_resumed_run_with_a_problem_evaluated_in_rust(tmp_path):
    problem = gx.problems.Rastrigin(5)
    de = gx.De(problem.genome, population_size=20, objective="minimize", seed=2)
    whole = de.run(problem, evaluations=3_000, parallel=True)
    de.run(problem, evaluations=1_000, checkpoint=tmp_path / "de.ckpt", checkpoint_every=10)
    # the stop conditions, parallel and the fitness function (the same one, in Python) can change
    resumed = de.run(lambda x: problem(x), evaluations=3_000, resume=tmp_path / "de.ckpt")
    assert same(whole, resumed)


def test_a_control_continues_after_a_resume(tmp_path):
    def anneal(ga, progress):
        ga.mutation = gx.GaussianMutation(0.1 / (1 + progress.generation), rate=0.5)

    whole = ga().run(sphere, generations=30, control=anneal)
    ga().run(sphere, generations=10, control=anneal, checkpoint=tmp_path / "a", checkpoint_every=10)
    resumed = ga().run(sphere, generations=30, control=anneal, resume=tmp_path / "a")
    assert same(whole, resumed)


def test_the_checkpoint_file(tmp_path):
    path = tmp_path / "run.ckpt"
    ga().run(sphere, generations=3, checkpoint=path, checkpoint_every=1)
    content = path.read_bytes()
    # genoxide's checkpoint format: its magic bytes, then the version
    assert content.startswith(b"genoxide")
    assert gx.__version__.encode() in content[:40]
    # saved atomically: no temporary file left
    assert [file.name for file in tmp_path.iterdir()] == ["run.ckpt"]


def test_a_checkpoint_resumes_only_its_settings(tmp_path):
    path = tmp_path / "run.ckpt"
    ga().run(sphere, generations=5, checkpoint=path, checkpoint_every=5)
    other = "was saved by a run with other settings"
    with pytest.raises(ValueError, match=other):
        ga(crossover_rate=0.5).run(sphere, generations=10, resume=path)
    with pytest.raises(ValueError, match=other):
        ga(gx.Real((-5.0, 5.0), length=7)).run(sphere, generations=10, resume=path)
    with pytest.raises(ValueError, match=other):
        gx.Ga(
            real,
            population_size=20,
            select=gx.Tournament(3),
            crossover=gx.UniformCrossover(),
            mutation=gx.PolynomialMutation(20.0, rate=0.2),
            objective="maximize",
            seed=1,
        ).run(sphere, generations=10, resume=path)
    # another algorithm, and another genome's
    with pytest.raises(ValueError, match=other):
        gx.De(real, population_size=20, objective="minimize", seed=1).run(
            sphere, generations=10, resume=path
        )
    with pytest.raises(ValueError, match=other):
        ga(gx.Binary(6), gx.BitFlip(rate=0.2)).run(sphere, generations=10, resume=path)


def test_broken_checkpoints(tmp_path):
    path = tmp_path / "run.ckpt"
    with pytest.raises(FileNotFoundError):
        ga().run(sphere, generations=5, resume=path)
    path.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="can't resume from .*not a genoxide checkpoint"):
        ga().run(sphere, generations=5, resume=path)
    ga().run(sphere, generations=5, checkpoint=path, checkpoint_every=5)
    content = bytearray(path.read_bytes())
    content[len(content) // 2] ^= 0xFF
    path.write_bytes(bytes(content))
    with pytest.raises(ValueError, match="corrupted or truncated"):
        ga().run(sphere, generations=5, resume=path)
    path.write_bytes(bytes(content[:40]))
    with pytest.raises(ValueError, match="can't resume from"):
        ga().run(sphere, generations=5, resume=path)


def test_checkpoint_settings_are_checked(tmp_path):
    path = tmp_path / "run.ckpt"
    with pytest.raises(ValueError, match="checkpoint and checkpoint_every go together"):
        ga().run(sphere, generations=5, checkpoint=path)
    with pytest.raises(ValueError, match="checkpoint and checkpoint_every go together"):
        ga().run(sphere, generations=5, checkpoint_every=5)
    with pytest.raises(ValueError, match="checkpoint_every is at least 1, not 0"):
        ga().run(sphere, generations=5, checkpoint=path, checkpoint_every=0)
    with pytest.raises(ValueError, match="checkpoint_every is a whole number"):
        ga().run(sphere, generations=5, checkpoint=path, checkpoint_every=2.5)
    with pytest.raises(ValueError, match="checkpoint is a path"):
        ga().run(sphere, generations=5, checkpoint=5, checkpoint_every=1)
    with pytest.raises(ValueError, match="resume is a path"):
        ga().run(sphere, generations=5, resume=b"run.ckpt")
    # a checkpoint that can't be saved stops the run
    with pytest.raises(OSError, match="checkpoint"):
        ga().run(sphere, generations=5, checkpoint=tmp_path / "no" / "run.ckpt", checkpoint_every=1)


def test_an_exception_keeps_the_last_good_checkpoint(tmp_path):
    path = tmp_path / "run.ckpt"
    calls = 0

    def failing(x):
        nonlocal calls
        calls += 1
        if calls == 150:
            raise RuntimeError("the fitness failed")
        return sphere(x)

    with pytest.raises(RuntimeError, match="the fitness failed"):
        ga().run(failing, generations=40, checkpoint=path, checkpoint_every=2)
    # the checkpoint is of a generation before the failure, and resumes as if nothing happened
    progress = []
    resumed = ga().run(
        sphere, generations=40, resume=path, on_generation=lambda p: progress.append(p.generation)
    )
    assert same(resumed, ga().run(sphere, generations=40))
    assert progress[0] % 2 == 1


def test_a_finished_run_resumes_to_its_end(tmp_path):
    path = tmp_path / "run.ckpt"
    first = ga().run(sphere, generations=10, checkpoint=path, checkpoint_every=3)
    again = ga().run(sphere, generations=10, resume=pathlib.Path(path))
    assert same(first, again)
