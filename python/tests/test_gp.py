"""Genetic programming (gx.gp): trees of the built-in primitives, their operators, symbolic
regression, the Boolean problems, and the selections against bloat."""

import math
import pickle

import numpy as np
import pytest

import genoxide as gx

KOZA = gx.gp.regression.problems.Koza1()
GP = gx.gp.Gp(KOZA.primitives())


def ga(genome=GP, **settings):
    options = {
        "population_size": 200,
        "select": gx.Tournament(7),
        "crossover": gx.gp.SubtreeCrossover(),
        "mutation": gx.gp.SubtreeMutation(),
        "mutation_rate": 0.1,
        "objective": "minimize",
        "seed": 1,
    }
    options.update(settings)
    return gx.Ga(genome, **options)


def same(a, b):
    return (
        a.best_fitness == b.best_fitness
        and a.best_genome == b.best_genome
        and (a.generations, a.evaluations, a.stop_reason)
        == (b.generations, b.evaluations, b.stop_reason)
    )


def rmse(predictions, targets):
    """The RMSE summed in the order of the points, as Rust sums it."""
    total = 0.0
    for p, y in zip(predictions, targets):
        total += (p - y) * (p - y)
    return math.sqrt(total / len(targets))


# --- the primitive set, trees and the representation -------------------------------------------


def test_a_primitive_set_of_regression():
    primitives = gx.gp.regression.primitives(
        ["add", "mul", "sin", "aq"], ["x", "y"], gx.gp.Constants.normal(0.0, 5.0)
    )
    assert primitives.functions == ["add", "mul", "sin", "aq"]
    assert primitives.terminals == ["x", "y"]
    assert primitives.constants == "Constants.normal(0.0, 5.0)"
    assert primitives == gx.gp.regression.primitives(
        ["add", "mul", "sin", "aq"], ["x", "y"], gx.gp.Constants.normal(0.0, 5.0)
    )
    assert primitives != gx.gp.regression.primitives(["add"], ["x", "y"])
    assert pickle.loads(pickle.dumps(primitives)) == primitives
    assert set(gx.gp.regression.FUNCTIONS) >= {"add", "pdiv", "plog", "psqrt", "aq", "inv"}
    for constants in (
        gx.gp.Constants.uniform(-1.0, 1.0),
        gx.gp.Constants.integers(-5, 5),
        gx.gp.Constants.choice([0.5, 1.0, 2.0]),
    ):
        assert gx.gp.regression.primitives(["add"], ["x"], constants).constants is not None


def test_parse_and_display_round_trip():
    primitives = gx.gp.regression.primitives(
        ["add", "sub", "mul", "sin"], ["x", "y"], gx.gp.Constants.normal(0.0, 5.0)
    )
    gp = gx.gp.Gp(primitives, init=gx.gp.Grow((1, 5)))
    for seed in range(50):
        tree = gp.random_genome(seed)
        text = str(tree)
        assert tree.display() == text
        again = primitives.parse(text)
        assert again == tree and hash(again) == hash(tree)
        assert gp.parse(text) == tree
        assert repr(tree) == f"Tree({text!r})"
        assert pickle.loads(pickle.dumps(tree)) == tree
        gp.validate(tree)
    tree = primitives.parse("add(mul(x, y), sin(-1.5))")
    assert (len(tree), tree.depth, str(tree)) == (6, 2, "add(mul(x, y), sin(-1.5))")
    assert tree.primitives == primitives


def test_parse_errors():
    primitives = KOZA.primitives()
    for text in ("add(x)", "foo(x, x)", "add(x, x", "1.5", "add(x, x) x"):
        with pytest.raises(ValueError):
            primitives.parse(text)
    # within the set, beyond the Gp's limits
    deep = gx.gp.Gp(primitives, max_depth=2, init=gx.gp.Full((1, 2)))
    with pytest.raises(ValueError, match="depth 3, above max_depth"):
        deep.parse("sin(sin(sin(x)))")
    with pytest.raises(ValueError, match="another primitive set"):
        deep.validate(gx.gp.regression.problems.Nguyen9().primitives().parse("y"))


def test_ramped_half_and_half_is_rusts():
    trees = GP.ramped_half_and_half(500, 100)
    assert len(trees) == 500 and len(set(trees)) == 500  # no duplicates
    assert trees == GP.ramped_half_and_half(500, 100)
    # grow stops a branch early: depths of at most 6
    assert max(tree.depth for tree in trees) == 6
    for tree in trees:
        GP.validate(tree)
    full = gx.gp.Gp(KOZA.primitives(), init=gx.gp.Full((3, 3)))
    assert all(full.random_genome(seed).depth == 3 for seed in range(20))


@pytest.mark.parametrize(
    ("gp", "setting"),
    [
        (gx.gp.Gp(KOZA.primitives(), max_depth=2**24 + 1), "Gp.max_depth"),
        (gx.gp.Gp(KOZA.primitives(), max_size=0), "Gp.max_size"),
        (gx.gp.Gp(KOZA.primitives(), max_depth=4), "Gp.init"),
        (gx.gp.Gp(KOZA.primitives(), init=gx.gp.Grow((3, 2))), "Gp.init"),
        (gx.gp.Gp(KOZA.primitives(), max_depth=-1), "Gp.max_depth"),
        (gx.gp.Gp(KOZA.primitives(), init=gx.gp.Full(4)), "Full.depths"),
    ],
)
def test_wrong_gp_settings_are_named(gp, setting):
    with pytest.raises(ValueError, match=setting.replace(".", r"\.")):
        gp.check()
    with pytest.raises(ValueError, match=setting.replace(".", r"\.")):
        ga(gp).run(KOZA, generations=1)


def test_a_gp_needs_a_primitive_set():
    with pytest.raises(ValueError, match="Gp.primitives"):
        gx.gp.Gp("add").check()
    with pytest.raises(ValueError, match="Gp.init"):
        gx.gp.Gp(KOZA.primitives(), init="full").check()


# --- evaluation ---------------------------------------------------------------------------------


def test_a_tree_evaluates_on_columns():
    primitives = gx.gp.regression.primitives(["add", "mul", "sin", "pdiv"], ["x", "y"])
    tree = primitives.parse("add(mul(x, y), sin(pdiv(x, y)))")
    points = np.array([[0.5, 2.0], [1.0, 0.0], [-3.0, 4.0]])
    values = tree.evaluate(points)
    expected = [
        x * y + gx.math.sin(x / y if y != 0.0 else 1.0) for x, y in points.tolist()
    ]
    assert values.tolist() == expected
    one = gx.gp.regression.primitives(["mul"], ["x"]).parse("mul(x, x)")
    assert one.evaluate([1.0, 2.0, 3.0]).tolist() == [1.0, 4.0, 9.0]
    with pytest.raises(ValueError, match="2 variables, but x has 1 columns"):
        tree.evaluate([[1.0], [2.0]])
    with pytest.raises(ValueError):
        tree.evaluate(np.zeros((2, 2, 2)))


def test_a_regression_problem_is_rusts_regression():
    training, test = KOZA.dataset().training, KOZA.dataset().test
    assert (training.points, test.points, training.variables, len(training)) == (20, 101, 1, 20)
    assert KOZA.name == "Koza-1" and KOZA.formula == "x^4 + x^3 + x^2 + x"
    assert KOZA.reference.startswith("Koza")
    x = training.x[:, 0]
    assert training.y.tolist() == [KOZA.target([v]) for v in x.tolist()]
    # the quartic itself: an error at the level of rounding
    quartic = KOZA.primitives().parse("add(mul(x, add(x, mul(x, add(x, mul(x, x))))), x)")
    assert KOZA(quartic) < 1e-15
    # a tree's error, by hand: without linear scaling, the RMSE of its values
    tree = KOZA.primitives().parse("mul(x, sin(x))")
    raw = KOZA.regression(linear_scaling=False)
    values = tree.evaluate(training.x)
    assert raw.values(tree, training).tolist() == values.tolist()
    assert raw.evaluate(tree) == rmse(values.tolist(), training.y.tolist())
    assert raw(tree) == raw.evaluate(tree) == raw.error(tree, training)
    assert raw.scaling(tree) == (0.0, 1.0)
    assert raw.display(tree) == "mul(x, sin(x))"
    # with linear scaling, the RMSE of a + b f, a and b by least squares
    scaled = KOZA.regression()
    a, b = scaled.scaling(tree)
    predictions = scaled.predict(tree, training)
    assert predictions.tolist() == [a + b * v for v in values.tolist()]
    assert scaled(tree) == pytest.approx(rmse(predictions.tolist(), training.y.tolist()), 1e-12)
    assert KOZA(tree) == scaled(tree)
    assert scaled.display(tree).endswith(" * (mul(x, sin(x)))")
    mae = KOZA.regression(metric="mae", linear_scaling=False)
    assert mae.metric == "mae" and not mae.linear_scaling
    assert mae(tree) == pytest.approx(np.mean(np.abs(values - training.y)), 1e-12)
    # a tree that isn't finite at a point is invalid
    primitives = gx.gp.regression.primitives(["div", "sub"], ["x"])
    nan = primitives.parse("div(x, sub(x, x))")
    dataset = gx.gp.regression.Dataset(gx.gp.regression.Sample([0.0, 1.0], [0.0, 1.0]))
    assert gx.gp.regression.Regression(primitives, dataset)(nan) is None


def test_every_regression_problem():
    problems = gx.gp.regression.problems.all()
    assert [problem.name for problem in problems] == [
        "Koza-1",
        "Koza-2",
        "Koza-3",
        *(f"Nguyen-{n}" for n in range(1, 13)),
    ]
    for problem in problems:
        dataset = problem.dataset()
        variables = len(problem.primitives().terminals)
        assert dataset.variables == variables
        point = dataset.training.x[0].tolist()
        assert problem.target(point) == dataset.training.y[0]
        assert type(problem).__name__ == problem.name.replace("-", "")
    nguyen9 = gx.gp.regression.problems.Nguyen9()
    tree = nguyen9.primitives().parse("add(sin(x), sin(mul(y, y)))")
    assert nguyen9.regression(linear_scaling=False)(tree) == 0.0
    with pytest.raises(ValueError, match="2 values, not 1"):
        nguyen9.target([1.0])


def test_samples_and_datasets():
    x = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    sample = gx.gp.regression.Sample(x, [1.0, 2.0, 3.0])
    assert np.array_equal(sample.x, x) and sample.y.tolist() == [1.0, 2.0, 3.0]
    assert sample.deviation() == pytest.approx(np.std([1.0, 2.0, 3.0]), 1e-15)
    dataset = gx.gp.regression.Dataset(sample, sample)
    assert dataset.test is not None and dataset.variables == 2
    assert gx.gp.regression.Dataset(sample).test is None
    for wrong in (
        lambda: gx.gp.regression.Sample(x, [1.0, 2.0]),
        lambda: gx.gp.regression.Sample(x, [1.0, 2.0, float("nan")]),
        lambda: gx.gp.regression.Dataset(sample, gx.gp.regression.Sample([1.0], [1.0])),
    ):
        with pytest.raises(ValueError):
            wrong()


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda: gx.gp.regression.primitives(["foo"], ["x"]), "no function \"foo\""),
        (lambda: gx.gp.regression.primitives(["add"], []), "at least one variable"),
        (lambda: gx.gp.regression.primitives(["add"], ["x", "x"]), "used twice"),
        (lambda: gx.gp.regression.primitives("add", ["x"]), "sequences of names"),
        (
            lambda: gx.gp.regression.primitives(["add"], ["x"], gx.gp.Constants.uniform(1, 0)),
            "constants",
        ),
        (
            lambda: gx.gp.regression.primitives(["add"], ["x"], gx.gp.Constants.normal(0, 0)),
            "constants",
        ),
        (lambda: gx.gp.regression.primitives(["add"], ["x"], 1.0), "gx.gp.Constants"),
        (
            lambda: gx.gp.regression.Regression(
                gx.gp.boolean.Multiplexer(2).primitives(), KOZA.dataset()
            ),
            "gx.gp.regression's functions",
        ),
        (
            lambda: gx.gp.regression.Regression(
                gx.gp.regression.problems.Nguyen9().primitives(), KOZA.dataset()
            ),
            "beyond the dataset's",
        ),
        (lambda: KOZA.regression(metric="max"), "metric"),
        (lambda: gx.gp.boolean.Multiplexer(0), r"Multiplexer\.address_bits"),
        (lambda: gx.gp.boolean.EvenParity(1), r"EvenParity\.inputs"),
        (lambda: gx.gp.WithSize(lambda tree: 0.0), "WithSize takes"),
        (lambda: KOZA(gx.gp.boolean.Multiplexer(2).primitives().parse("d0")), "another primitive"),
    ],
)
def test_wrong_fitness_settings(make, message):
    with pytest.raises(ValueError, match=message):
        make()


def test_the_boolean_problems():
    problem = gx.gp.boolean.Multiplexer(3)
    assert (problem.inputs, problem.cases, problem.address_bits) == (11, 2048, 3)
    primitives = problem.primitives()
    assert primitives.functions == ["and", "or", "not", "if"]
    assert primitives.terminals == ["a0", "a1", "a2", *(f"d{bit}" for bit in range(8))]
    right = primitives.parse(
        "if(a2, if(a1, if(a0, d7, d6), if(a0, d5, d4)), if(a1, if(a0, d3, d2), if(a0, d1, d0)))"
    )
    assert problem.errors(right) == 0 and problem(right) == 0.0
    assert problem.outputs(right) == problem.targets()
    # Rust's doc: one data bit is right in the cases that select it, and in half the others
    d0 = primitives.parse("d0")
    assert problem.errors(d0) == 896
    # the truth table by evaluate: input i of case c is bit i of c
    cases = np.array([[c >> i & 1 for i in range(11)] for c in range(2048)], dtype=bool)
    outputs = right.evaluate(cases)
    assert outputs.dtype == bool
    targets = problem.targets()
    assert outputs.tolist() == [bool(targets[c // 64] >> (c % 64) & 1) for c in range(2048)]
    wrong = d0.evaluate(cases) != outputs
    assert int(wrong.sum()) == 896
    parity = gx.gp.boolean.EvenParity(3)
    assert (parity.cases, parity.address_bits) == (8, None)
    odd = "and(or(d0, d1), nand(d0, d1))"
    even = parity.primitives().parse(f"or(and({odd}, d2), nor({odd}, d2))")
    assert parity.errors(even) == 0 and parity.errors(parity.primitives().parse("d0")) == 4


# --- runs -----------------------------------------------------------------------------------------


def test_a_seeded_regression_solves_koza1():
    # the koza_quartic example's search: 8 islands of 500 trees
    tolerance = 1e-10 * KOZA.dataset().training.deviation()
    regression = KOZA.regression(linear_scaling=False)
    search = gx.Islands(
        [
            ga(
                population_size=500,
                initial_genomes=GP.ramped_half_and_half(500, seed),
                crossover_rate=0.9,
                seed=seed,
            )
            for seed in range(100, 108)
        ],
        interval=10,
        migrants=2,
    )
    result = search.run(regression, target=tolerance, generations=50)
    assert (result.generations, result.evaluations) == (4, 17061)  # as in Rust
    assert result.stop_reason == "target"
    assert result.best_fitness <= tolerance
    assert regression.error(result.best_genome, KOZA.dataset().test) <= 1e-9
    # the same seed repeats the run, in parallel too, and a Python function gets the same trees
    assert same(result, search.run(regression, target=tolerance, generations=50, parallel=True))
    in_python = search.run(lambda tree: regression(tree), target=tolerance, generations=50)
    assert same(result, in_python)


def test_a_python_fitness_gets_trees():
    seen = []

    def size(tree):
        assert isinstance(tree, gx.gp.Tree) and tree.primitives == GP.primitives
        seen.append(tree)
        return len(tree)

    result = ga(population_size=20).run(size, generations=3)
    assert len(seen) == result.evaluations
    assert result.best_fitness == len(result.best_genome) == min(map(len, seen))

    def sizes(trees):
        assert isinstance(trees, tuple) and all(isinstance(t, gx.gp.Tree) for t in trees)
        return [len(tree) for tree in trees]

    assert same(result, ga(population_size=20).run(sizes, generations=3, batch=True))


def test_progress_and_control_see_trees():
    populations = []

    def on_generation(progress):
        populations.append(progress.population)
        assert isinstance(progress.best_genome, gx.gp.Tree)
        assert progress.best_fitness == KOZA(progress.best_genome)

    def control(running, progress):
        if progress.generation == 0:
            assert isinstance(running.mutation, gx.gp.SubtreeMutation)
        running.mutation = gx.gp.Mutations([(1.0, gx.gp.HoistMutation())])
        running.select = gx.DoubleTournament(7, 1.4)

    ga(population_size=30).run(KOZA, generations=4, on_generation=on_generation, control=control)
    assert len(populations) == 5
    for population in populations:
        assert isinstance(population, tuple) and len(population) == 30
        for tree in population:
            GP.validate(tree)
    assert pickle.loads(pickle.dumps(populations[-1])) == populations[-1]


@pytest.mark.parametrize(
    ("crossover", "mutation"),
    [
        (gx.gp.SubtreeCrossover(), gx.gp.SubtreeMutation(max_depth=2)),
        (gx.gp.SubtreeCrossover(internal_rate=0.5), gx.gp.PointMutation(rate=0.2)),
        (gx.gp.OnePointCrossover(), gx.gp.PointMutation(count=2)),
        (gx.NoCrossover(), gx.gp.HoistMutation()),
        (gx.NoCrossover(), gx.gp.ShrinkMutation()),
        (gx.gp.SubtreeCrossover(), gx.gp.ConstantMutation.gaussian(0.1)),
        (
            gx.gp.OnePointCrossover(),
            gx.gp.Mutations(
                [
                    (0.5, gx.gp.SubtreeMutation()),
                    (0.2, gx.gp.PointMutation(count=1)),
                    (0.1, gx.gp.HoistMutation()),
                    (0.1, gx.gp.ShrinkMutation()),
                    (0.1, gx.gp.ConstantMutation(0.1)),
                ]
            ),
        ),
    ],
)
def test_the_operators_make_valid_trees(crossover, mutation):
    primitives = gx.gp.regression.primitives(
        ["add", "mul", "sin"], ["x"], gx.gp.Constants.uniform(-1.0, 1.0)
    )
    gp = gx.gp.Gp(primitives, max_depth=8, max_size=40)
    data = gx.gp.regression.Dataset(gx.gp.regression.Sample([-1.0, 0.0, 1.0], [1.0, 0.0, 1.0]))
    trees = []
    search = ga(
        gp,
        population_size=40,
        crossover=crossover,
        mutation=mutation,
        mutation_rate=1.0,
        select=gx.LexicographicTournament(3),
    )
    result = search.run(
        gx.gp.regression.Regression(primitives, data),
        generations=10,
        on_generation=lambda progress: trees.extend(progress.population),
    )
    for tree in trees:
        gp.validate(tree)
        assert len(tree) <= 40 and tree.depth <= 8
    assert result.generations == 10


def test_hoist_and_shrink_make_trees_smaller():
    gp = gx.gp.Gp(KOZA.primitives(), init=gx.gp.Full((4, 4)))
    sizes = {}
    for mutation in (gx.gp.HoistMutation(), gx.gp.ShrinkMutation()):
        mean_sizes = []
        ga(
            gp,
            population_size=50,
            crossover=gx.NoCrossover(),
            mutation=mutation,
            mutation_rate=1.0,
            select=gx.RandomSelection(),
            scheme=gx.Generational(elitism=0),
        ).run(
            lambda tree: 0.0,
            generations=3,
            on_generation=lambda p: mean_sizes.append(np.mean([len(t) for t in p.population])),
        )
        sizes[type(mutation).__name__] = mean_sizes
        assert all(later < earlier for earlier, later in zip(mean_sizes, mean_sizes[1:]))
    assert len(sizes) == 2


@pytest.mark.parametrize(
    ("operator", "setting"),
    [
        ({"crossover": gx.gp.SubtreeCrossover(internal_rate=1.5)}, "SubtreeCrossover.internal_rate"),
        ({"mutation": gx.gp.SubtreeMutation(max_depth=2**25)}, "SubtreeMutation.max_depth"),
        ({"mutation": gx.gp.PointMutation(rate=0.0)}, "PointMutation.rate"),
        ({"mutation": gx.gp.PointMutation(count=0)}, "PointMutation.count"),
        ({"mutation": gx.gp.PointMutation()}, "PointMutation needs"),
        ({"mutation": gx.gp.ConstantMutation(0.0)}, "ConstantMutation.sigma"),
        (
            {"mutation": gx.gp.Mutations([(-1.0, gx.gp.HoistMutation())])},
            "Mutations.mutations",
        ),
        ({"mutation": gx.gp.Mutations([(0.0, gx.gp.HoistMutation())])}, "Mutations.mutations"),
        ({"mutation": gx.gp.Mutations([(1.0, gx.BitFlip(rate=0.1))])}, "Mutations.mutations"),
        ({"mutation": gx.gp.Mutations([gx.gp.HoistMutation()])}, "Mutations.mutations"),
        ({"select": gx.DoubleTournament(7, 2.5)}, "DoubleTournament.parsimony"),
        ({"select": gx.DoubleTournament(0, 1.4)}, "DoubleTournament.fitness_size"),
        ({"select": gx.LexicographicTournament(0)}, "LexicographicTournament.size"),
        (
            {"select": gx.LexicographicTournament(2, bucket_ratio=0.0)},
            "LexicographicTournament.bucket_ratio",
        ),
        ({"select": gx.Tarpeian(gx.Tournament(3), 0.0)}, "Tarpeian.rate"),
        ({"select": gx.Tarpeian(gx.Tournament(0), 0.5)}, "Tournament.size"),
        (
            {"select": gx.Tarpeian(gx.Tarpeian(gx.Tournament(3), 0.5), 0.5)},
            "Tarpeian.select",
        ),
        ({"crossover": gx.UniformCrossover()}, "UniformCrossover doesn't work with tree"),
        ({"mutation": gx.BitFlip(rate=0.1)}, "BitFlip doesn't work with tree"),
        ({"initial_genomes": GP.ramped_half_and_half(201, 1)}, "initial_genomes"),
        (
            {"initial_genomes": [gx.gp.boolean.Multiplexer(2).primitives().parse("d0")]},
            "initial_genomes",
        ),
        ({"initial_genomes": ["x"]}, "initial_genomes"),
    ],
)
def test_wrong_tree_settings_are_named(operator, setting):
    with pytest.raises(ValueError, match=setting.replace(".", r"\.")):
        ga(**operator).run(KOZA, generations=1)


def test_trees_and_other_genomes_dont_mix():
    real = gx.Real((-1.0, 1.0), length=3)
    with pytest.raises(ValueError, match="SubtreeCrossover doesn't work with real"):
        ga(
            real, crossover=gx.gp.SubtreeCrossover(), mutation=gx.PolynomialMutation(20, rate=0.2)
        ).run(
            lambda x: 0.0, generations=1
        )
    with pytest.raises(ValueError, match="SubtreeMutation doesn't work with real"):
        ga(real, crossover=gx.UniformCrossover()).run(lambda x: 0.0, generations=1)
    with pytest.raises(ValueError, match="initial_genomes are the trees of a gx.gp.Gp"):
        ga(
            real,
            crossover=gx.UniformCrossover(),
            mutation=gx.PolynomialMutation(20, rate=0.2),
            initial_genomes=GP.ramped_half_and_half(2, 1),
        ).run(lambda x: 0.0, generations=1)
    with pytest.raises(ValueError, match="needs a gx.gp.Gp genome"):
        ga(
            real, crossover=gx.UniformCrossover(), mutation=gx.PolynomialMutation(20, rate=0.2)
        ).run(
            KOZA, generations=1
        )
    with pytest.raises(ValueError, match="trees run with Ga"):
        gx.LocalSearch(GP, neighbor=gx.gp.SubtreeMutation(), objective="minimize").run(
            KOZA, generations=1
        )
    # a fitness of another primitive set, or of the wrong objectives
    with pytest.raises(ValueError, match="its own primitive set"):
        ga(gx.gp.Gp(gx.gp.regression.problems.Nguyen9().primitives())).run(KOZA, generations=1)
    with pytest.raises(ValueError, match="minimizes its error"):
        ga(objective="maximize").run(KOZA, generations=1)
    with pytest.raises(ValueError, match="2 objectives"):
        ga().run(gx.gp.WithSize(KOZA), generations=1)


def test_selections_against_bloat_take_any_genome():
    for select in (
        gx.DoubleTournament(7, 1.4),
        gx.DoubleTournament(3, 2.0, size_first=True),
        gx.LexicographicTournament(2),
        gx.LexicographicTournament(2, bucket_ratio=0.5),
        gx.Tarpeian(gx.Tournament(3), 0.3),
        gx.Tarpeian(gx.DoubleTournament(7, 1.4), 0.3),
    ):
        one_max = gx.Ga(
            gx.Binary(20),
            population_size=30,
            select=select,
            crossover=gx.UniformCrossover(),
            mutation=gx.BitFlip(rate=0.05),
            seed=1,
        )
        result = one_max.run(lambda bits: bits.sum(), target=20, generations=200)
        assert result.stop_reason == "target", select
        trees = ga(select=select, population_size=50).run(KOZA, generations=5)
        assert trees.generations == 5


def test_double_tournament_keeps_trees_smaller():
    def mean_size(select):
        sizes = []
        ga(select=select, population_size=200, mutation_rate=0.1).run(
            KOZA,
            generations=15,
            on_generation=lambda p: sizes.append(np.mean([len(t) for t in p.population])),
        )
        return sizes[-1]

    assert mean_size(gx.DoubleTournament(7, 2.0)) < mean_size(gx.Tournament(7))


def test_islands_of_trees():
    islands = gx.Islands(
        [
            ga(population_size=100, seed=seed, initial_genomes=GP.ramped_half_and_half(100, seed))
            for seed in range(3)
        ],
        interval=5,
        migrants=2,
    )
    result = islands.run(KOZA, generations=12)
    assert isinstance(result.best_genome, gx.gp.Tree)
    assert result.best_fitness == KOZA(result.best_genome)
    assert same(result, islands.run(KOZA, generations=12, parallel=True))
    with pytest.raises(ValueError, match="share a genome"):
        gx.Islands([ga(), ga(gx.gp.Gp(KOZA.primitives(), max_size=500))]).run(KOZA, generations=1)


def test_nsga2_on_error_and_size(tmp_path):
    problem = gx.gp.regression.problems.Nguyen7()
    gp = gx.gp.Gp(problem.primitives())

    def nsga2(seed=1):
        return gx.Nsga2(
            gp,
            objectives=["minimize", "minimize"],
            population_size=100,
            initial_genomes=gp.ramped_half_and_half(100, seed),
            crossover=gx.gp.SubtreeCrossover(),
            mutation=gx.gp.SubtreeMutation(),
            mutation_rate=0.1,
            seed=seed,
        )

    objectives = gx.gp.WithSize(problem)
    assert objectives.fitness is problem
    result = nsga2().run(objectives, generations=20)
    assert isinstance(result.front_genomes, tuple)
    assert len(result.front_genomes) == len(result.front_objectives) > 1
    for tree, values in zip(result.front_genomes, result.front_objectives.tolist()):
        assert tuple(values) == objectives(tree) == (problem(tree), float(len(tree)))
    # a front: no point dominates another
    points = result.front_objectives.tolist()
    for a in points:
        assert not any(b[0] <= a[0] and b[1] <= a[1] and b != a for b in points)
    # the same in Python, and in parallel
    def in_python(tree):
        error = problem(tree)
        return None if error is None else (error, len(tree))

    in_python = nsga2().run(in_python, generations=20)
    assert np.array_equal(in_python.front_objectives, result.front_objectives)
    parallel = nsga2().run(objectives, generations=20, parallel=True)
    assert parallel.front_genomes == result.front_genomes
    # a checkpoint resumes the run
    path = tmp_path / "front.ckpt"
    nsga2().run(objectives, generations=8, checkpoint=path, checkpoint_every=4)
    resumed = nsga2().run(objectives, generations=20, resume=path)
    assert resumed.front_genomes == result.front_genomes
    assert resumed.evaluations == result.evaluations
    # the wrong objectives
    with pytest.raises(ValueError, match="has one objective"):
        nsga2().run(problem, generations=1)
    with pytest.raises(ValueError, match="minimizes the error and the size"):
        gx.Nsga2(
            gp,
            objectives=["minimize", "maximize"],
            population_size=10,
            crossover=gx.gp.SubtreeCrossover(),
            mutation=gx.gp.SubtreeMutation(),
        ).run(objectives, generations=1)
    with pytest.raises(ValueError, match="Nsga2 of 2 objectives"):
        gx.Nsga2(
            gp,
            objectives=["minimize"] * 3,
            population_size=10,
            crossover=gx.gp.SubtreeCrossover(),
            mutation=gx.gp.SubtreeMutation(),
        ).run(lambda tree: (0.0, 0.0, 0.0), generations=1)
    with pytest.raises(ValueError, match="trees run with Ga"):
        gx.Spea2(
            gp,
            objectives=["minimize", "minimize"],
            population_size=10,
            crossover=gx.gp.SubtreeCrossover(),
            mutation=gx.gp.SubtreeMutation(),
        ).run(objectives, generations=1)


def test_a_boolean_problem_is_solved():
    problem = gx.gp.boolean.Multiplexer(2)
    gp = gx.gp.Gp(problem.primitives())
    result = gx.Ga(
        gp,
        population_size=500,
        initial_genomes=gp.ramped_half_and_half(500, 1),
        select=gx.DoubleTournament(7, 1.4),
        crossover=gx.gp.SubtreeCrossover(),
        mutation=gx.gp.Mutations(
            [(0.5, gx.gp.SubtreeMutation()), (0.5, gx.gp.PointMutation(count=1))]
        ),
        mutation_rate=0.1,
        objective="minimize",
        seed=1,
    ).run(problem, target=0.0, generations=50)
    assert result.stop_reason == "target"
    assert problem.errors(result.best_genome) == 0
