"""gx.Es, gx.AdaptiveReal with gx.SelfAdaptiveMutation, and gx.Islands."""

import numpy as np
import pytest

import genoxide as gx


def sphere(x):
    return float(np.sum(x * x))


def rastrigin(x):
    return 10 * len(x) + float(np.sum(x * x - 10 * np.cos(2 * np.pi * x)))


def same(a, b):
    return (
        a.best_fitness == b.best_fitness
        and np.array_equal(a.best_genome, b.best_genome)
        and a.evaluations == b.evaluations
        and a.generations == b.generations
    )


# --- Es ------------------------------------------------------------------------------------------


def es(**settings):
    defaults = {"parents": 5, "offspring": 35, "objective": "minimize", "seed": 1}
    return gx.Es(gx.Real((-5.0, 5.0), length=5), **{**defaults, **settings})


def test_es_reaches_the_target():
    result = es().run(sphere, target=1e-10, evaluations=100_000)
    assert result.stop_reason == "target"
    assert result.best_fitness <= 1e-10
    assert result.best_genome.dtype == np.float64 and result.best_genome.shape == (5,)


def test_es_matches_rust():
    # tests/engine.rs has the same run in Rust, evaluated in Rust
    problem = gx.problems.Sphere(5)
    result = gx.Es(
        problem.genome,
        parents=4,
        offspring=20,
        recombination="dominant",
        rho=2,
        selection="plus",
        step_sizes="one",
        initial_step=0.2,
        objective="minimize",
        seed=3,
    ).run(problem, generations=40)
    assert result.evaluations == 804
    assert result.best_fitness == 6.424989845860214e-05


def test_es_settings():
    for settings in (
        {"recombination": "intermediate", "rho": 1},
        {"recombination": "dominant"},
        {"selection": "plus", "offspring": 3},
        {"step_sizes": "one"},
        {"initial_step": 1.0},
        {"parallel_breeding": True},
    ):
        first = es(**settings).run(sphere, generations=10)
        assert same(first, es(**settings).run(sphere, generations=10)), settings
        assert first.evaluations == 5 + 10 * es(**settings).offspring


def test_es_is_the_same_one_at_a_time_in_batches_and_in_parallel():
    one = es().run(sphere, generations=30)
    batch = es().run(lambda x: np.sum(x * x, axis=1), generations=30, batch=True)
    parallel = es().run(sphere, generations=30, parallel=True)
    for other in (batch, parallel):
        assert other.best_fitness == pytest.approx(one.best_fitness, rel=1e-12)
        assert other.evaluations == one.evaluations


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"parents": 0}, "invalid setting `parents`: must be at least 1"),
        ({"offspring": 0}, "invalid setting `offspring`"),
        ({"offspring": 4}, "at least the 5 parents with comma selection, got 4"),
        ({"rho": 6}, "invalid setting `rho`: must be between 1 and the 5 parents, got 6"),
        ({"rho": 0}, "invalid setting `rho`"),
        ({"initial_step": 0.0}, "invalid setting `initial_step`"),
        ({"initial_step": 11.0}, "invalid setting `initial_step`"),
        ({"parents": 1.5}, "parents is a whole number"),
        ({"recombination": "global"}, 'recombination is "intermediate" or "dominant"'),
        ({"selection": "best"}, 'selection is "comma" or "plus"'),
        ({"step_sizes": "two"}, 'step_sizes is "per_gene" or "one"'),
        ({"parents": 2**25}, "invalid setting `parents`"),
    ],
)
def test_es_settings_are_checked(settings, message):
    with pytest.raises(ValueError, match=message):
        es(**settings).run(sphere, generations=1)


def test_es_needs_a_real_genome():
    with pytest.raises(ValueError, match="Es needs a Real genome"):
        gx.Es(gx.Binary(8), parents=2, offspring=4).run(lambda x: x.sum(), generations=1)
    with pytest.raises(ValueError, match="a gene with more than one value"):
        gx.Es(gx.Real((1.0, 1.0), length=3), parents=2, offspring=4).run(sphere, generations=1)
    with pytest.raises(TypeError):
        gx.Es(gx.Real((0.0, 1.0), length=3), parents=2)  # type: ignore[call-arg]


def test_es_control_reevaluates():
    handles = []

    def control(algorithm, progress):
        handles.append(type(algorithm))
        if progress.generation == 5:
            algorithm.reevaluate()

    result = es().run(sphere, generations=10, control=control)
    assert set(handles) == {gx.RunningEs}
    # the re-evaluation scores the 5 parents again
    assert result.evaluations == 5 + 10 * 35 + 5


# --- AdaptiveReal and SelfAdaptiveMutation -------------------------------------------------------


def adaptive_ga(genome=None, **settings):
    defaults = {
        "population_size": 10,
        "select": gx.Tournament(2),
        "crossover": gx.NoCrossover(),
        "mutation": gx.SelfAdaptiveMutation(),
        "scheme": gx.MuCommaLambda(60),
        "objective": "minimize",
        "seed": 1,
    }
    genome = genome or gx.AdaptiveReal(gx.Real((-5.0, 5.0), length=5), 0.3)
    return gx.Ga(genome, **{**defaults, **settings})


def test_a_ga_with_self_adaptation_is_an_evolution_strategy():
    genomes = []

    def fitness(x):
        genomes.append(x)
        return sphere(x)

    result = adaptive_ga().run(fitness, target=1e-10, evaluations=200_000)
    assert result.stop_reason == "target"
    # the genes only, as a Real's, without the step size
    assert all(genome.dtype == np.float64 and genome.shape == (5,) for genome in genomes)
    assert result.best_genome.shape == (5,)


def test_self_adaptation_matches_rust():
    # tests/engine.rs has the same run in Rust, evaluated in Rust
    problem = gx.problems.Sphere(5)
    result = gx.Ga(
        gx.AdaptiveReal(problem.genome, 0.2),
        population_size=10,
        select=gx.Tournament(2),
        crossover=gx.UniformCrossover(),
        mutation=gx.SelfAdaptiveMutation(learning_rate=0.5, min_step=1e-6),
        scheme=gx.MuCommaLambda(40),
        objective="minimize",
        seed=4,
    ).run(problem, generations=40)
    assert result.evaluations == 1610
    assert result.best_fitness == 2.5135888040149383e-08
    # the same with the problem in Python
    python = gx.Ga(
        gx.AdaptiveReal(problem.genome, 0.2),
        population_size=10,
        select=gx.Tournament(2),
        crossover=gx.UniformCrossover(),
        mutation=gx.SelfAdaptiveMutation(learning_rate=0.5, min_step=1e-6),
        scheme=gx.MuCommaLambda(40),
        objective="minimize",
        seed=4,
    ).run(lambda x: problem(x), generations=40)
    assert same(result, python)


def test_adaptive_reals_with_other_algorithms():
    genome = gx.AdaptiveReal(gx.Real((0.0, 1.0), length=4), 0.1)
    search = gx.LocalSearch(
        genome, neighbor=gx.SelfAdaptiveMutation(), neighbors=4, objective="minimize", seed=1
    )
    assert search.run(sphere, generations=200).best_fitness < 0.1
    for crossover in (gx.UniformCrossover(), gx.PointCrossover(2)):
        result = adaptive_ga(genome, crossover=crossover).run(sphere, generations=20)
        assert result.best_fitness < 0.1
    nsga2 = gx.Nsga2(
        genome,
        objectives=["minimize", "minimize"],
        population_size=20,
        crossover=gx.UniformCrossover(),
        mutation=gx.SelfAdaptiveMutation(),
        seed=1,
    )
    front = nsga2.run(lambda x: (x[0], 1 - x[0]), generations=10)
    assert front.front_genomes.shape[1] == 4


def test_adaptive_real_settings_are_checked():
    real = gx.Real((0.0, 1.0), length=3)
    for step in (0.0, -1.0):
        with pytest.raises(ValueError, match="invalid setting `AdaptiveReal.initial_step`"):
            adaptive_ga(gx.AdaptiveReal(real, step)).run(sphere, generations=1)
    with pytest.raises(ValueError, match="AdaptiveReal.initial_step is a finite number"):
        adaptive_ga(gx.AdaptiveReal(real, float("inf"))).run(sphere, generations=1)
    with pytest.raises(ValueError, match="AdaptiveReal.real is a gx.Real"):
        adaptive_ga(gx.AdaptiveReal(gx.Binary(3), 0.3)).run(sphere, generations=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Real.bounds"):
        adaptive_ga(gx.AdaptiveReal(gx.Real([(1.0, 0.0)]), 0.3)).run(sphere, generations=1)
    for learning_rate in (0.0, -1.0):
        with pytest.raises(ValueError, match="SelfAdaptiveMutation.learning_rate"):
            mutation = gx.SelfAdaptiveMutation(learning_rate=learning_rate)
            adaptive_ga(mutation=mutation).run(sphere, generations=1)
    with pytest.raises(ValueError, match="SelfAdaptiveMutation.min_step"):
        adaptive_ga(mutation=gx.SelfAdaptiveMutation(min_step=0.0)).run(sphere, generations=1)


def test_operators_must_fit_adaptive_reals():
    wrong = "SimulatedBinaryCrossover doesn't work with adaptive real"
    with pytest.raises(ValueError, match=wrong):
        adaptive_ga(crossover=gx.SimulatedBinaryCrossover()).run(sphere, generations=1)
    with pytest.raises(ValueError, match="GaussianMutation doesn't work with adaptive real"):
        adaptive_ga(mutation=gx.GaussianMutation(0.1, rate=0.5)).run(sphere, generations=1)
    real = gx.Ga(
        gx.Real((0.0, 1.0), length=3),
        population_size=10,
        select=gx.Tournament(2),
        crossover=gx.UniformCrossover(),
        mutation=gx.SelfAdaptiveMutation(),
    )
    with pytest.raises(ValueError, match="SelfAdaptiveMutation doesn't work with real genomes"):
        real.run(sphere, generations=1)
    for algorithm in (gx.De, gx.Cmaes):
        genome = gx.AdaptiveReal(gx.Real((0.0, 1.0), length=3), 0.3)
        with pytest.raises(ValueError, match="needs a Real genome"):
            algorithm(genome).run(sphere, generations=1)  # type: ignore[arg-type]


# --- Islands -------------------------------------------------------------------------------------


def ga_islands(count=4, **settings):
    return gx.Islands(
        [
            gx.Ga(
                gx.Real((-5.12, 5.12), length=10),
                population_size=25,
                select=gx.Tournament(3),
                crossover=gx.UniformCrossover(),
                mutation=gx.PolynomialMutation(20.0, rate=0.1),
                objective="minimize",
                seed=seed,
            )
            for seed in range(count)
        ],
        **settings,
    )


def test_islands_reach_the_target():
    result = ga_islands(topology="ring", interval=10, migrants=2).run(
        rastrigin, target=0.01, evaluations=500_000
    )
    assert result.stop_reason == "target"


def test_islands_match_rust():
    # tests/engine.rs has the same runs in Rust, evaluated in Rust
    problem = gx.problems.Rastrigin(6)
    gas = gx.Islands(
        [
            gx.Ga(
                problem.genome,
                population_size=20,
                select=gx.Tournament(3),
                crossover=gx.SimulatedBinaryCrossover(15),
                mutation=gx.PolynomialMutation(20, rate=1 / 6),
                objective="minimize",
                seed=seed,
            )
            for seed in range(3)
        ],
        topology="random",
        interval=5,
        migrants=2,
        seed=9,
    )
    result = gas.run(problem, generations=50)
    assert result.evaluations == 2684
    assert result.best_fitness == 0.5250156249865867
    des = gx.Islands(
        [
            gx.De(problem.genome, population_size=20, objective="minimize", seed=seed)
            for seed in range(3)
        ],
        topology="fully_connected",
        interval=7,
        migrants=1,
    )
    result = des.run(problem, generations=50)
    assert result.evaluations == 3060
    assert result.best_fitness == 5.771504985092051


def test_islands_are_the_same_one_at_a_time_in_batches_and_in_parallel():
    sizes = []

    def batch(x):
        sizes.append(len(x))
        return 10 * x.shape[1] + np.sum(x * x - 10 * np.cos(2 * np.pi * x), axis=1)

    islands = ga_islands(topology="random", seed=3)
    one = islands.run(rastrigin, generations=30)
    assert same(one, islands.run(rastrigin, generations=30))
    batched = islands.run(batch, generations=30, batch=True)
    parallel = islands.run(rastrigin, generations=30, parallel=True)
    for other in (batched, parallel):
        assert other.best_fitness == pytest.approx(one.best_fitness, rel=1e-12)
        assert other.evaluations == one.evaluations
    # a call per generation, with the candidates of all four islands
    assert sizes[0] == 100 and len(sizes) <= 31


def test_the_progress_has_every_island():
    sizes = []
    ga_islands(count=3, topology="isolated").run(
        rastrigin, generations=3, on_generation=lambda p: sizes.append(p.population.shape)
    )
    assert sizes == [(75, 10)] * 4


def test_islands_settings_are_checked():
    real = gx.Real((0.0, 1.0), length=3)

    def ga(genome=real, objective="minimize"):
        return gx.Ga(
            genome,
            population_size=10,
            select=gx.Tournament(2),
            crossover=gx.UniformCrossover(),
            mutation=gx.UniformMutation(rate=0.5),
            objective=objective,
        )

    for islands, message in (
        ([ga()], "invalid setting `islands`: at least 2 islands, got 1"),
        ([], "invalid setting `islands`: at least 2 islands, got 0"),
        ([ga(), gx.De(real, objective="minimize")], "the islands are all Ga or all De"),
        ([ga(), gx.Cmaes(real)], "islands are Ga or De"),
        ([ga(), ga(gx.Real((0.0, 2.0), length=3))], "island 1 has another one than island 0"),
        ([ga(), ga(objective="maximize")], "the islands must share an objective"),
    ):
        with pytest.raises(ValueError, match=message):
            gx.Islands(islands).run(sphere, generations=1)
    for settings, message in (
        ({"interval": 0}, "invalid setting `interval`: at least 1 generation"),
        ({"migrants": 0}, "invalid setting `migrants`: at least 1"),
        ({"topology": "star"}, 'topology is "ring", "fully_connected", "random" or "isolated"'),
        ({"interval": -1}, "interval is at least 0"),
        ({"seed": 1.5}, "seed is a whole number"),
    ):
        with pytest.raises(ValueError, match=message):
            gx.Islands([ga(), ga()], **settings).run(sphere, generations=1)
    # isolated islands don't migrate: their interval and migrants don't apply
    gx.Islands([ga(), ga()], topology="isolated", interval=0, migrants=0).run(sphere, generations=1)
    with pytest.raises(ValueError, match="De needs a Real genome"):
        binary = gx.De(gx.Binary(3))  # type: ignore[arg-type]
        gx.Islands([binary, binary]).run(lambda x: x.sum(), generations=1)
    # an island's own settings are checked as its own
    with pytest.raises(ValueError, match="invalid setting `Tournament.size`"):
        bad = gx.Ga(
            real,
            population_size=10,
            select=gx.Tournament(0),
            crossover=gx.UniformCrossover(),
            mutation=gx.UniformMutation(rate=0.5),
            objective="minimize",
        )
        gx.Islands([ga(), bad]).run(sphere, generations=1)


def test_a_control_changes_each_island():
    seen = []

    def control(islands, progress):
        assert isinstance(islands, gx.RunningIslands)
        assert [type(island) for island in islands.islands] == [gx.RunningGa] * 4
        if progress.generation == 0:
            for index, island in enumerate(islands.islands):
                island.mutation = gx.PolynomialMutation(20.0 + index, rate=0.1)
                island.crossover_rate = 0.5 + index / 10
        seen.append([island.crossover_rate for island in islands.islands])
        with pytest.raises(ValueError, match="re-evaluated together"):
            islands.islands[0].reevaluate()
        with pytest.raises(ValueError, match="crossover_rate"):
            islands.islands[1].crossover_rate = 2.0
        if progress.generation == 3:
            islands.reevaluate()

    islands = ga_islands()
    result = islands.run(rastrigin, generations=5, control=control)
    assert seen[-1] == [0.5, 0.6, 0.7, 0.8]
    assert result.generations == 5
    # the operators are the islands' own, changed only in the run
    assert islands.islands[1].mutation == gx.PolynomialMutation(20.0, rate=0.1)


def test_a_control_of_de_islands():
    def control(islands, progress):
        assert [type(island) for island in islands.islands] == [gx.RunningDe] * 2
        islands.islands[1].strategy = "rand1"
        assert islands.islands[1].strategy == "rand1"
        assert islands.islands[0].control == {"memory": 100}

    real = gx.Real((-5.0, 5.0), length=4)
    des = gx.Islands(
        [gx.De(real, population_size=10, objective="minimize", seed=s) for s in range(2)]
    )
    des.run(sphere, generations=3, control=control)


def test_a_control_that_changes_nothing_changes_no_result():
    islands = ga_islands(topology="random", seed=2)
    first = islands.run(rastrigin, generations=20)
    again = islands.run(rastrigin, generations=20, control=lambda islands, progress: None)
    assert same(first, again)
