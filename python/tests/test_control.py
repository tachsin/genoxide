"""Parameter control and re-evaluation: ``run(..., control=...)`` and the ``Running`` handles."""

import threading

import numpy as np
import pytest

import genoxide as gx


def sphere(x):
    return float(np.sum(x * x))


def real_ga(**settings):
    defaults = {
        "population_size": 30,
        "select": gx.Tournament(3),
        "crossover": gx.UniformCrossover(),
        "mutation": gx.GaussianMutation(0.1, rate=0.2),
        "objective": "minimize",
        "seed": 1,
    }
    return gx.Ga(gx.Real((-5, 5), length=5), **{**defaults, **settings})


def algorithms():
    """An algorithm of each kind, on 5 real genes, and a run of it."""
    genome = gx.Real((-5, 5), length=5)
    return {
        "ga": real_ga(),
        "de": gx.De(genome, population_size=20, objective="minimize", seed=1),
        "cmaes": gx.Cmaes(genome, objective="minimize", seed=1),
        "pso": gx.Pso(genome, population_size=20, objective="minimize", seed=1),
        "local_search": gx.LocalSearch(
            genome, neighbor=gx.GaussianMutation(0.05, rate=0.5), objective="minimize", seed=1
        ),
        "nelder_mead": gx.NelderMead(genome, objective="minimize", seed=1),
        "first_order": gx.FirstOrder(
            genome, step="adam", learning_rate=0.05, objective="minimize", seed=1
        ),
    }


RUNNING = {
    "ga": gx.RunningGa,
    "de": gx.RunningDe,
    "cmaes": gx.RunningCmaes,
    "pso": gx.RunningPso,
    "local_search": gx.RunningLocalSearch,
    "nelder_mead": gx.RunningNelderMead,
    "first_order": gx.RunningFirstOrder,
}


def same(a, b):
    return (
        a.best_fitness == b.best_fitness
        and np.array_equal(a.best_genome, b.best_genome)
        and (a.generations, a.evaluations, a.stop_reason)
        == (b.generations, b.evaluations, b.stop_reason)
    )


@pytest.mark.parametrize("name", list(RUNNING))
def test_control_is_called_once_per_generation_after_on_generation(name):
    algorithm = algorithms()[name]
    main = threading.get_ident()
    events = []
    handles = set()

    def control(running, progress):
        assert isinstance(running, RUNNING[name])
        assert threading.get_ident() == main
        handles.add(id(running))
        events.append(("control", progress))

    result = algorithm.run(
        sphere,
        generations=10,
        parallel=True,
        on_generation=lambda progress: events.append(("progress", progress)),
        control=control,
    )
    # generation 0 to the last, each after on_generation, with the same progress
    assert [kind for kind, _ in events] == ["progress", "control"] * 11
    assert [progress.generation for kind, progress in events if kind == "control"] == list(
        range(11)
    )
    for (_, seen), (_, given) in zip(events[::2], events[1::2]):
        assert (seen.generation, seen.evaluations, seen.best_fitness) == (
            given.generation,
            given.evaluations,
            given.best_fitness,
        )
        assert np.array_equal(seen.population, given.population)
        assert np.array_equal(seen.scores, given.scores, equal_nan=True)
        assert np.array_equal(seen.best_genome, given.best_genome)
    assert events[-1][1].best_fitness == result.best_fitness
    # the same handle every generation
    assert len(handles) == 1


def test_control_is_called_after_the_generation_that_aborts():
    generations = []
    result = real_ga().run(
        sphere,
        generations=100,
        on_generation=lambda progress: progress.generation < 3,
        control=lambda running, progress: generations.append(progress.generation),
    )
    assert result.stop_reason == "aborted"
    assert generations == [0, 1, 2, 3]


def keep(name):
    """A control that sets every setting to what it is."""

    def control(running, progress):
        if name == "ga":
            running.crossover_rate = running.crossover_rate
            running.mutation_rate = running.mutation_rate
            running.select = running.select
            running.crossover = running.crossover
            running.mutation = running.mutation
        elif name == "de":
            running.strategy = running.strategy
            running.control = running.control
        elif name == "pso":
            running.inertia = running.inertia
            running.acceleration = running.acceleration
        elif name == "local_search":
            running.neighbor = running.neighbor
            running.neighbors = running.neighbors

    return control


@pytest.mark.parametrize("name", list(RUNNING))
def test_a_control_that_changes_nothing_leaves_the_run_as_it_is(name):
    algorithm = algorithms()[name]
    alone = algorithm.run(sphere, generations=40)
    assert same(alone, algorithm.run(sphere, generations=40, control=lambda *_: None))
    assert same(alone, algorithm.run(sphere, generations=40, control=keep(name)))
    # and with a problem evaluated in Rust
    problem = gx.problems.Sphere(5)
    alone = algorithm.run(problem, generations=40)
    assert same(alone, algorithm.run(problem, generations=40, control=keep(name)))


def test_the_settings_in_use_are_read_back():
    seen = {}

    def read(name):
        def control(running, progress):
            if progress.generation == 0:
                seen[name] = {
                    attribute: getattr(running, attribute)
                    for attribute in dir(type(running))
                    if isinstance(getattr(type(running), attribute), property)
                }

        return control

    for name, algorithm in algorithms().items():
        algorithm.run(sphere, generations=1, control=read(name))
    ga = seen["ga"]
    assert (ga["crossover_rate"], ga["mutation_rate"]) == (0.9, 1.0)
    assert ga["select"] == gx.Tournament(3)
    assert ga["crossover"] == gx.UniformCrossover()
    assert ga["mutation"] == gx.GaussianMutation(0.1, rate=0.2)
    # SHADE's defaults
    assert seen["de"] == {"strategy": {"max_p": 0.2, "archive": 1.0}, "control": {"memory": 100}}
    assert seen["cmaes"] == {}
    assert seen["pso"] == {"inertia": 0.7298, "acceleration": (1.49618, 1.49618)}
    assert seen["local_search"] == {
        "neighbor": gx.GaussianMutation(0.05, rate=0.5),
        "neighbors": 1,
    }
    # the first simplex: one initial step in each gene
    assert seen["nelder_mead"] == {
        "converged": False,
        "size": pytest.approx(1.0),
        "iterations": 0,
        "restart_count": 0,
    }

    # the start's gradient, by forward differences: no step yet
    first_order = seen["first_order"]
    assert np.allclose(first_order.pop("gradient"), 2 * first_order_start(), atol=1e-6)
    assert first_order.pop("gradient_norm") == pytest.approx(
        np.max(np.abs(2 * first_order_start())), abs=1e-6
    )
    assert first_order == {
        "learning_rate": 0.05,
        "multiplier": 1.0,
        "converged": None,
        "iterations": 0,
        "steps": 0,
        "restart_count": 0,
        "gradients": "forward",
    }

    de = gx.De(
        gx.Real((-5, 5), length=5), strategy="rand1", control={"f": 0.5, "cr": 0.9}, seed=1
    )
    de.run(sphere, generations=1, control=read("de"))
    assert seen["de"] == {"strategy": "rand1", "control": {"f": 0.5, "cr": 0.9}}


def test_an_annealed_mutation_step():
    generations = 150

    def anneal(ga, progress):
        # from 10% to 0.1% of each gene's range
        sigma = 0.1 * 0.01 ** (progress.generation / generations)
        ga.mutation = gx.GaussianMutation(sigma, rate=0.2)

    fixed = real_ga().run(sphere, generations=generations)
    annealed = real_ga().run(sphere, generations=generations, control=anneal)
    assert annealed.best_fitness < fixed.best_fitness / 100
    assert annealed.best_fitness < 1e-5


def test_a_change_applies_from_the_next_generation():
    # mutation only, from generation 5 on: until then, the children of a GA without mutation and
    # crossover would be copies, so a crossover recombines them
    populations = []

    def control(ga, progress):
        populations.append(progress.population)
        if progress.generation == 4:
            ga.crossover_rate = 0.0

    real_ga(mutation_rate=1.0).run(sphere, generations=8, control=control)
    assert len(populations) == 9


def test_each_setting_changes_the_run():
    alone = real_ga().run(sphere, generations=30)

    def at(generation, change):
        def control(running, progress):
            if progress.generation == generation:
                change(running)

        return control

    changes = [
        lambda ga: setattr(ga, "crossover_rate", 0.5),
        lambda ga: setattr(ga, "mutation_rate", 0.5),
        lambda ga: setattr(ga, "select", gx.Rank()),
        lambda ga: setattr(ga, "crossover", gx.SimulatedBinaryCrossover(15)),
        lambda ga: setattr(ga, "mutation", gx.PolynomialMutation(20, rate=0.2)),
    ]
    for change in changes:
        changed = real_ga().run(sphere, generations=30, control=at(10, change))
        assert not same(alone, changed)

    de = algorithms()["de"]
    alone = de.run(sphere, generations=30)
    for change in [
        lambda de: setattr(de, "strategy", "rand1"),
        lambda de: setattr(de, "control", {"f": 0.5, "cr": 0.9}),
    ]:
        assert not same(alone, de.run(sphere, generations=30, control=at(10, change)))

    pso = algorithms()["pso"]
    alone = pso.run(sphere, generations=30)
    for change in [
        lambda pso: setattr(pso, "inertia", 0.4),
        lambda pso: setattr(pso, "acceleration", (2.0, 1.0)),
    ]:
        assert not same(alone, pso.run(sphere, generations=30, control=at(10, change)))

    search = algorithms()["local_search"]
    alone = search.run(sphere, generations=30)
    for change in [
        lambda search: setattr(search, "neighbors", 4),
        lambda search: setattr(search, "neighbor", gx.UniformMutation(count=1)),
    ]:
        assert not same(alone, search.run(sphere, generations=30, control=at(10, change)))


def test_a_decreasing_inertia_weight():
    generations = 300

    def inertia(pso, progress):
        pso.inertia = 0.9 - 0.5 * progress.generation / generations

    pso = gx.Pso(gx.Real((-5, 5), length=10), population_size=30, objective="minimize", seed=1)
    result = pso.run(sphere, generations=generations, control=inertia)
    assert result.best_fitness < 1e-6


def test_a_wrong_value_raises_and_changes_nothing():
    wrong = {
        "ga": [
            ("crossover_rate", 1.5, "crossover_rate"),
            ("mutation_rate", -0.1, "mutation_rate"),
            ("mutation_rate", float("nan"), "mutation_rate"),
            ("select", gx.Tournament(0), "Tournament.size"),
            ("select", "tournament", "select"),
            ("crossover", gx.OrderCrossover(), "OrderCrossover doesn't work with real genomes"),
            ("crossover", gx.SimulatedBinaryCrossover(-1), "SimulatedBinaryCrossover.eta"),
            ("mutation", gx.BitFlip(rate=0.1), "BitFlip doesn't work with real genomes"),
            ("mutation", gx.GaussianMutation(0, rate=0.1), "GaussianMutation.sigma"),
            ("mutation", gx.GaussianMutation, "an instance, not the class"),
        ],
        "de": [
            ("strategy", {"p": 0, "archive": 1.0}, "strategy.p"),
            ("strategy", "rand2", "strategy"),
            ("strategy", None, "not None"),
            ("control", {"f": 3.0, "cr": 0.9}, "control.f"),
            ("control", {"memory": 0}, "control.memory"),
        ],
        "pso": [
            ("inertia", -1.0, "inertia"),
            ("inertia", True, "inertia"),
            ("acceleration", (1.0, -1.0), "acceleration"),
            ("acceleration", 1.0, "a pair"),
        ],
        "local_search": [
            ("neighbors", 0, "neighbors"),
            ("neighbors", 2.0, "neighbors"),
            ("neighbor", gx.SwapMutation(), "SwapMutation doesn't work with real genomes"),
        ],
    }
    for name, cases in wrong.items():
        algorithm = algorithms()[name]
        alone = algorithm.run(sphere, generations=20)
        for attribute, value, message in cases:
            errors = []

            def control(running, progress):
                if progress.generation == 5:
                    before = getattr(running, attribute)
                    with pytest.raises(ValueError, match=message):
                        setattr(running, attribute, value)
                    errors.append(attribute)
                    assert getattr(running, attribute) == before

            result = algorithm.run(sphere, generations=20, control=control)
            assert errors == [attribute]
            assert same(alone, result), (name, attribute, value)


def test_no_crossover_needs_a_mutation_rate_above_zero_in_a_run_too():
    def control(ga, progress):
        if progress.generation == 3:
            ga.mutation_rate = 0.0
            with pytest.raises(ValueError, match="doesn't recombine"):
                ga.crossover = gx.NoCrossover()
            assert ga.crossover == gx.UniformCrossover()
            with pytest.raises(ValueError, match="both 0"):
                ga.crossover_rate = 0.0
            assert ga.crossover_rate == 0.9

    real_ga().run(sphere, generations=5, control=control)


def test_an_exception_in_control_is_raised():
    calls = []

    def control(ga, progress):
        calls.append(progress.generation)
        if progress.generation == 3:
            ga.mutation_rate = 2.0

    with pytest.raises(ValueError, match="mutation_rate"):
        real_ga().run(sphere, generations=10, control=control)
    assert calls == [0, 1, 2, 3]

    class Stop(Exception):
        pass

    def fail(ga, progress):
        raise Stop

    with pytest.raises(Stop):
        real_ga().run(sphere, generations=10, control=fail)
    with pytest.raises(TypeError, match="control isn't callable"):
        real_ga().run(sphere, generations=10, control=3)


def test_the_handle_works_only_during_the_call():
    held = []
    real_ga().run(sphere, generations=1, control=lambda ga, progress: held.append(ga))
    with pytest.raises(RuntimeError, match="only during its control call"):
        held[0].mutation_rate = 0.5
    with pytest.raises(RuntimeError, match="only during its control call"):
        held[0].reevaluate()
    with pytest.raises(RuntimeError, match="only during its control call"):
        _ = held[0].crossover_rate


def test_a_misspelled_setting_is_an_error():
    def control(ga, progress):
        with pytest.raises(AttributeError):
            ga.mutaton_rate = 0.5

    real_ga().run(sphere, generations=1, control=control)


def test_multi_objective_runs_have_no_control():
    nsga2 = gx.Nsga2(
        gx.Real((0, 1), length=3),
        objectives=["minimize", "minimize"],
        population_size=10,
        crossover=gx.SimulatedBinaryCrossover(),
        mutation=gx.PolynomialMutation(rate=0.3),
    )
    with pytest.raises(TypeError, match="control"):
        nsga2.run(lambda x: [x[0], 1 - x[0]], generations=1, control=lambda *_: None)


@pytest.mark.parametrize("name", list(RUNNING))
def test_reevaluate_scores_the_algorithm_again(name):
    algorithm = algorithms()[name]
    # the optimum moves from 0 to 1 at generation 20
    shift = {"value": 0.0}

    def moving(x):
        return float(np.sum((x - shift["value"]) ** 2))

    progress = []
    controls = []

    def control(running, state):
        controls.append(state.generation)
        if state.generation == 20:
            shift["value"] = 1.0
            running.reevaluate()

    result = algorithm.run(
        moving, generations=60, on_generation=progress.append, control=control
    )
    generations = [state.generation for state in progress]
    # on_generation sees the re-evaluation, with the same generation, and control doesn't
    assert generations == list(range(21)) + [20] + list(range(21, 61))
    assert controls == list(range(61))
    rescored = progress[21]
    assert rescored.evaluations > progress[20].evaluations
    # the best by the new function, not the old best score
    assert rescored.best_fitness == pytest.approx(moving(rescored.best_genome))
    assert rescored.best_fitness > progress[20].best_fitness
    assert np.allclose(rescored.scores, [moving(genome) for genome in rescored.population])
    assert result.best_fitness == pytest.approx(moving(result.best_genome))
    # then it moves toward the new optimum
    assert result.best_fitness < rescored.best_fitness
    # the local methods evaluate a point or two per generation, or take steps of a set length:
    # they move, but don't get there
    if name not in ("local_search", "nelder_mead", "first_order"):
        assert np.allclose(result.best_genome, 1.0, atol=0.3)


def first_order_start():
    """The random start of the first-order method of `algorithms()`."""
    seen = []
    algorithms()["first_order"].run(
        sphere, generations=0, on_generation=lambda progress: seen.append(progress.population[0])
    )
    return seen[0]


def test_an_adaptive_penalty():
    # maximize the ones, with at most 10 of them allowed: the weight of the penalty rises while
    # the best breaks the limit
    weight = {"value": 0.1}

    def penalized(bits):
        ones = int(bits.sum())
        return ones - weight["value"] * max(ones - 10, 0)

    def control(ga, progress):
        if progress.generation % 20 == 19 and progress.best_genome.sum() > 10:
            weight["value"] *= 4
            ga.reevaluate()

    ga = gx.Ga(
        gx.Binary(32),
        population_size=30,
        select=gx.Tournament(3),
        crossover=gx.UniformCrossover(),
        mutation=gx.BitFlip(rate=1 / 32),
        seed=2,
    )
    result = ga.run(penalized, generations=200, control=control)
    assert result.best_genome.sum() == 10
    assert weight["value"] > 0.1


def test_a_seeded_control_schedule_repeats():
    def schedule(ga, progress):
        ga.mutation = gx.GaussianMutation(0.1 / (1 + progress.generation), rate=0.2)
        if progress.generation == 10:
            ga.reevaluate()

    first = real_ga().run(sphere, generations=30, control=schedule)
    for batch, parallel in [(False, False), (True, False), (False, True)]:
        function = (lambda x: np.sum(x * x, axis=1)) if batch else sphere
        again = real_ga().run(
            function, generations=30, batch=batch, parallel=parallel, control=schedule
        )
        assert same(first, again)


def test_a_control_schedule_matches_rust():
    # tests/engine.rs has the same run in Rust: a GA on the sphere function, evaluated in Rust,
    # with a mutation step and rates that change every generation, and a re-evaluation at
    # generation 10
    problem = gx.problems.Sphere(5)
    ga = gx.Ga(
        problem.genome,
        population_size=20,
        select=gx.Tournament(3),
        crossover=gx.SimulatedBinaryCrossover(15),
        mutation=gx.GaussianMutation(0.1, rate=0.2),
        objective="minimize",
        seed=1,
    )

    def schedule(ga, progress):
        generation = progress.generation
        ga.mutation = gx.GaussianMutation(0.1 / (1 + generation), rate=0.2)
        ga.crossover_rate = 0.5 if generation % 2 == 0 else 0.9
        ga.mutation_rate = 1.0 / (1 + generation % 3)
        if generation == 10:
            ga.reevaluate()

    result = ga.run(problem, generations=50, control=schedule)
    assert result.evaluations == 767
    assert result.best_fitness == 1109.9904210975287
