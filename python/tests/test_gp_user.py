"""Genetic programming with your own primitives (gx.gp.PrimitiveSetBuilder): the set and its
errors, typed trees, parse and display, evaluation by numpy and per point, and runs with Ga,
Islands, Nsga2 and checkpoints."""

import operator
import pickle

import numpy as np
import pytest

import genoxide as gx

# the signatures of the typed set below: (argument types, return type)
SIGNATURES = {
    "add": (["real", "real"], "real"),
    "mul": (["real", "real"], "real"),
    "less": (["real", "real"], "bool"),
    "not": (["bool"], "bool"),
    "if": (["bool", "real", "real"], "real"),
    "x": ([], "real"),
}

NUMPY = {
    "add": np.add,
    "mul": np.multiply,
    "less": np.less,
    "not": np.logical_not,
    "if": np.where,
}


def typed(constants=gx.gp.Constants.integers(-2, 2)):
    builder = gx.gp.PrimitiveSetBuilder()
    real = builder.new_type("real")
    boolean = builder.new_type("bool")
    names = {"real": real, "bool": boolean}
    for name, (args, returns) in SIGNATURES.items():
        if args:
            builder.function(name, [names[arg] for arg in args], names[returns])
        else:
            builder.terminal(name, names[returns])
    if constants is not None:
        builder.constants(real, constants)
    return builder.build(real)


TYPED = typed()
GP = gx.gp.Gp(TYPED)


def well_typed(tree, root="real"):
    """Whether every child of ``tree`` has its parent's argument type, and the root ``root``."""
    nodes = tree.nodes()
    # the types expected of the nodes still to come, the next on top
    expected = [root]
    for node in nodes:
        if not expected or node.type != expected.pop():
            return False
        if node.kind == "constant":
            assert node.value is not None and node.arity == 0 and node.type == "real"
            continue
        args, returns = SIGNATURES[node.name]
        if node.type != returns or node.arity != len(args):
            return False
        expected.extend(reversed(args))
    return not expected


# --- the set and its errors ------------------------------------------------------------------


def test_a_set_of_your_own():
    assert TYPED.types == ["real", "bool"]
    assert TYPED.root_type == "real"
    assert TYPED.functions == ["add", "mul", "less", "not", "if"]
    assert TYPED.terminals == ["x"]
    assert TYPED.constants == "Constants.integers(-2, 2)"
    assert TYPED == typed() and hash(TYPED) == hash(typed())
    assert TYPED != typed(constants=None)
    assert pickle.loads(pickle.dumps(TYPED)) == TYPED
    assert repr(TYPED).startswith("PrimitiveSet(types=[real, bool], functions=[add, mul,")
    # the methods chain, and a type is its name
    builder = gx.gp.PrimitiveSetBuilder()
    assert builder.new_type("number") == "number"
    single = builder.function("neg", ["number"], "number").terminal("t", "number").build("number")
    assert single.functions == ["neg"] and single.terminals == ["t"]
    assert single.constants is None


def build(*steps, root="real"):
    builder = gx.gp.PrimitiveSetBuilder()
    for step in steps:
        method, *args = step
        getattr(builder, method)(*args)
    return builder.build(root)


@pytest.mark.parametrize(
    "steps, root, message",
    [
        ((), "real", "the trees return the type `real`, which isn't a type of the set"),
        (
            (("new_type", "real"), ("function", "less", ["real", "real"], "boolean")),
            "real",
            "`less` returns the type `boolean`, which isn't a type of the set \\(declared: real\\)",
        ),
        (
            (("new_type", "real"), ("function", "f", ["int"], "real"), ("terminal", "x", "real")),
            "real",
            "`f` takes the type `int`",
        ),
        (
            (("new_type", "real"), ("terminal", "x", "real"), ("constants", "int", None)),
            "real",
            "constants is a gx.gp.Constants",
        ),
        (
            (("new_type", "real"), ("terminal", "x", "real"), ("terminal", "x", "real")),
            "real",
            "the name `x` is used twice",
        ),
        (
            (("new_type", "real"), ("new_type", "real"), ("terminal", "x", "real")),
            "real",
            "the type name `real` is used twice",
        ),
        ((("new_type", "real"), ("terminal", "a b", "real")), "real", "whitespace"),
        ((("new_type", "real"), ("terminal", "1.5", "real")), "real", "number"),
        (
            (
                ("new_type", "real"),
                ("terminal", "x", "real"),
                ("constants", "real", gx.gp.Constants.uniform(0.0, 1.0)),
                ("constants", "real", gx.gp.Constants.uniform(0.0, 1.0)),
            ),
            "real",
            "given constants twice",
        ),
        (
            (
                ("new_type", "real"),
                ("terminal", "x", "real"),
                ("constants", "real", gx.gp.Constants.uniform(1.0, 0.0)),
            ),
            "real",
            "invalid setting",
        ),
        (
            # a Boolean is needed, and nothing makes one
            (
                ("new_type", "real"),
                ("new_type", "bool"),
                ("function", "if", ["bool", "real", "real"], "real"),
                ("terminal", "x", "real"),
                ("function", "not", ["bool"], "bool"),
            ),
            "real",
            "no tree can be made of the types \\[\"bool\"\\]",
        ),
    ],
)
def test_set_errors_name_the_problem(steps, root, message):
    with pytest.raises(ValueError, match=message):
        build(*steps, root=root)


def test_argument_types_are_a_sequence():
    builder = gx.gp.PrimitiveSetBuilder()
    real = builder.new_type("real")
    with pytest.raises(ValueError, match="sequence of type names"):
        builder.function("neg", real, real)


# --- typed trees, parse and display -------------------------------------------------------------


def test_every_generated_tree_is_well_typed():
    trees = GP.ramped_half_and_half(300, seed=1)
    assert len(set(trees)) == 300
    assert all(well_typed(tree) for tree in trees)
    # Booleans appear, and only where a Boolean goes
    assert any(node.type == "bool" for tree in trees for node in tree.nodes())
    assert all(well_typed(GP.random_genome(seed)) for seed in range(50))


@pytest.mark.parametrize(
    "crossover, mutation",
    [
        (gx.gp.SubtreeCrossover(), gx.gp.SubtreeMutation()),
        (gx.gp.OnePointCrossover(), gx.gp.PointMutation(rate=0.2)),
        (gx.NoCrossover(), gx.gp.HoistMutation()),
        (gx.gp.SubtreeCrossover(0.5), gx.gp.ShrinkMutation()),
        (gx.gp.SubtreeCrossover(), gx.gp.ConstantMutation(0.3)),
        (
            gx.gp.SubtreeCrossover(),
            gx.gp.Mutations(
                [
                    (0.4, gx.gp.SubtreeMutation()),
                    (0.3, gx.gp.PointMutation(count=1)),
                    (0.1, gx.gp.HoistMutation()),
                    (0.1, gx.gp.ShrinkMutation()),
                    (0.1, gx.gp.ConstantMutation(0.5)),
                ]
            ),
        ),
    ],
)
def test_the_operators_keep_trees_typed(crossover, mutation):
    trees = []
    for select in (gx.DoubleTournament(7, 1.4), gx.LexicographicTournament(3)):
        ga = gx.Ga(
            GP,
            population_size=40,
            select=select,
            crossover=crossover,
            mutation=mutation,
            mutation_rate=0.5,
            objective="minimize",
            seed=3,
        )
        ga.run(len, generations=10, on_generation=lambda p: trees.extend(p.population))
    assert all(well_typed(tree) for tree in trees)
    for tree in trees[::50]:
        GP.validate(tree)


def test_parse_and_display_round_trip():
    text = "if(not(less(x, 1.0)), mul(x, -2.0), add(x, 1.0))"
    tree = TYPED.parse(text)
    assert str(tree) == tree.display() == text
    assert TYPED.parse(str(tree)) == tree == GP.parse(text)
    assert tree.primitives == TYPED
    assert (len(tree), tree.depth) == (11, 3)
    assert pickle.loads(pickle.dumps(tree)) == tree
    for tree in GP.ramped_half_and_half(100, seed=2):
        assert TYPED.parse(str(tree)) == tree
    # a Boolean where a real goes, and the root a Boolean
    with pytest.raises(ValueError):
        TYPED.parse("add(x, less(x, x))")
    with pytest.raises(ValueError):
        TYPED.parse("less(x, x)")
    with pytest.raises(ValueError):
        TYPED.parse("sub(x, x)")


def test_nodes_in_prefix_order():
    tree = TYPED.parse("if(less(x, 1.0), x, 2.0)")
    nodes = tree.nodes()
    assert [(n.kind, n.name, n.arity, n.type, n.value) for n in nodes] == [
        ("function", "if", 3, "real", None),
        ("function", "less", 2, "bool", None),
        ("terminal", "x", 0, "real", None),
        ("constant", "1.0", 0, "real", 1.0),
        ("terminal", "x", 0, "real", None),
        ("constant", "2.0", 0, "real", 2.0),
    ]
    assert repr(nodes[0]) == "Node(kind='function', name='if', arity=3, type='real')"
    assert isinstance(nodes[0], gx.gp.Node)

    # a recursive interpreter of the nodes, top-down
    def value(nodes, at, x):
        node = nodes[at]
        at += 1
        args = []
        for _ in range(node.arity):
            arg, at = value(nodes, at, x)
            args.append(arg)
        if node.kind == "constant":
            return node.value, at
        if node.kind == "terminal":
            return x, at
        return {"if": lambda c, a, b: a if c else b, "less": operator.lt}[node.name](*args), at

    assert value(nodes, 0, 0.25) == (0.25, len(nodes))
    assert value(nodes, 0, 1.5) == (2.0, len(nodes))


# --- evaluation ----------------------------------------------------------------------------------


def test_numpy_evaluation_against_hand_computed_values():
    x = np.array([-2.0, -0.5, 0.0, 0.5, 3.0])
    tree = TYPED.parse("if(less(x, 0.0), mul(-1.0, x), add(x, mul(x, x)))")
    expected = np.array([2.0, 0.5, 0.0, 0.75, 12.0])
    assert np.array_equal(tree.evaluate({"x": x}, NUMPY), expected)
    # x as an array: one terminal, a 1-D array or a column
    assert np.array_equal(tree.evaluate(x, NUMPY), expected)
    assert np.array_equal(tree.evaluate(x[:, None], NUMPY), expected)
    # one call per node, its children before it
    calls = []

    def counted(name):
        def call(*args):
            calls.append(name)
            return NUMPY[name](*args)

        return call

    tree.evaluate({"x": x}, {name: counted(name) for name in NUMPY})
    assert sorted(calls) == ["add", "if", "less", "mul", "mul"]
    assert calls[-1] == "if"
    # a constant tree's value is a float
    assert TYPED.parse("2.0").evaluate({"x": x}, NUMPY) == 2.0


def test_per_point_fallback():
    tree = TYPED.parse("if(less(x, 0.0), mul(-1.0, x), add(x, mul(x, x)))")
    plain = {
        "add": operator.add,
        "mul": operator.mul,
        "less": operator.lt,
        "not": operator.not_,
        "if": lambda condition, a, b: a if condition else b,
    }
    xs = [-2.0, -0.5, 0.0, 0.5, 3.0]
    values = [tree.evaluate({"x": x}, plain) for x in xs]
    assert values == [2.0, 0.5, 0.0, 0.75, 12.0]
    assert all(type(value) is float for value in values)
    # the same bits as on columns
    columns = tree.evaluate({"x": np.array(xs)}, NUMPY)
    assert columns.tolist() == values


def test_several_terminals_by_name_or_column():
    builder = gx.gp.PrimitiveSetBuilder()
    real = builder.new_type("real")
    builder.function("sub", [real, real], real).terminal("a", real).terminal("b", real)
    primitives = builder.build(real)
    tree = primitives.parse("sub(b, a)")
    points = np.array([[1.0, 10.0], [2.0, 20.0]])
    assert tree.evaluate(points, {"sub": np.subtract}).tolist() == [9.0, 18.0]
    assert tree.evaluate({"a": 1.0, "b": 10.0}, {"sub": operator.sub}) == 9.0
    with pytest.raises(ValueError, match="2 terminals, but x has 1 columns"):
        tree.evaluate(points[:, :1], {"sub": np.subtract})
    with pytest.raises(ValueError, match="x has no value for the terminal `b`"):
        tree.evaluate({"a": 1.0}, {"sub": operator.sub})
    with pytest.raises(ValueError, match="a 2-D array"):
        tree.evaluate(np.zeros((2, 2, 2)), {"sub": np.subtract})


def test_evaluation_errors():
    tree = TYPED.parse("add(x, 1.0)")
    with pytest.raises(ValueError, match="functions has no function `mul`"):
        tree.evaluate({"x": 1.0}, {"add": operator.add})
    with pytest.raises(ValueError, match="evaluate\\(x, functions\\)"):
        tree.evaluate([1.0, 2.0])
    with pytest.raises(ValueError, match="a mapping"):
        tree.evaluate({"x": 1.0}, [operator.add])
    # a function's exception propagates
    with pytest.raises(ZeroDivisionError):
        tree.evaluate({"x": 1.0}, {**NUMPY, "add": lambda a, b: 1 / 0})
    # built-in primitives are evaluated by genoxide
    koza = gx.gp.regression.problems.Koza1().primitives().parse("mul(x, x)")
    with pytest.raises(ValueError, match="functions are for trees of your own primitives"):
        koza.evaluate([1.0], NUMPY)
    # and a user set doesn't fit a built-in fitness
    regression = gx.gp.regression.problems.Koza1().regression()
    with pytest.raises(ValueError):
        regression(tree)


# --- runs ----------------------------------------------------------------------------------------

# x^2 + x from 21 points, with add, sub and mul: exact recovery
X = np.linspace(-1.0, 1.0, 21)
Y = X * X + X
ARITHMETIC = {"add": np.add, "sub": np.subtract, "mul": np.multiply}


def arithmetic():
    builder = gx.gp.PrimitiveSetBuilder()
    real = builder.new_type("real")
    for name in ARITHMETIC:
        builder.function(name, [real, real], real)
    builder.terminal("x", real)
    builder.constants(real, gx.gp.Constants.integers(-1, 1))
    return builder.build(real)


ARITHMETIC_GP = gx.gp.Gp(arithmetic())


def error(tree):
    values = tree.evaluate({"x": X}, ARITHMETIC)
    return float(np.sqrt(np.mean((values - Y) ** 2)))


def ga(seed=1, **settings):
    options = {
        "population_size": 200,
        "initial_genomes": ARITHMETIC_GP.ramped_half_and_half(200, seed),
        "select": gx.DoubleTournament(7, 1.4),
        "crossover": gx.gp.SubtreeCrossover(),
        "mutation": gx.gp.Mutations(
            [(0.6, gx.gp.SubtreeMutation()), (0.4, gx.gp.PointMutation(count=1))]
        ),
        "mutation_rate": 0.2,
        "objective": "minimize",
        "seed": seed,
    }
    options.update(settings)
    return gx.Ga(ARITHMETIC_GP, **options)


def test_a_seeded_run_solves_a_regression():
    result = ga().run(error, target=1e-12, generations=50)
    assert result.stop_reason == "target"
    assert error(result.best_genome) <= 1e-12
    assert result.best_genome.primitives == arithmetic()
    # seeded: the same run again, and in parallel
    again = ga().run(error, target=1e-12, generations=50)
    assert again.best_genome == result.best_genome
    assert again.evaluations == result.evaluations
    parallel = ga().run(error, target=1e-12, generations=50, parallel=True)
    assert parallel.best_genome == result.best_genome


def test_islands_of_your_own_trees():
    islands = gx.Islands([ga(seed) for seed in range(3)], interval=5, migrants=2)
    result = islands.run(error, target=1e-12, generations=50)
    assert result.stop_reason == "target"
    assert error(result.best_genome) <= 1e-12


def test_nsga2_on_error_and_size(tmp_path):
    def nsga2():
        return gx.Nsga2(
            ARITHMETIC_GP,
            objectives=["minimize", "minimize"],
            population_size=60,
            initial_genomes=ARITHMETIC_GP.ramped_half_and_half(60, 1),
            crossover=gx.gp.SubtreeCrossover(),
            mutation=gx.gp.SubtreeMutation(),
            mutation_rate=0.2,
            seed=1,
        )

    def objectives(tree):
        return error(tree), len(tree)

    result = nsga2().run(objectives, generations=15)
    points = result.front_objectives.tolist()
    assert len(points) > 1
    for tree, values in zip(result.front_genomes, points):
        assert tuple(values) == (error(tree), float(len(tree)))
    # a single node is the smallest tree
    assert min(size for _, size in points) == 1.0
    path = tmp_path / "front.ckpt"
    nsga2().run(objectives, generations=6, checkpoint=path, checkpoint_every=3)
    resumed = nsga2().run(objectives, generations=15, resume=path)
    assert resumed.front_genomes == result.front_genomes


def test_a_checkpoint_resumes_with_the_functions_given_again(tmp_path):
    path = tmp_path / "run.ckpt"
    whole = ga().run(error, generations=12)
    ga().run(error, generations=5, checkpoint=path, checkpoint_every=5)

    # the set is in the checkpoint; what its primitives mean is the fitness function's
    def again(tree):
        return float(np.sqrt(np.mean((tree.evaluate(X, dict(ARITHMETIC)) - Y) ** 2)))

    resumed = ga().run(again, generations=12, resume=path)
    assert resumed.best_genome == whole.best_genome
    assert (resumed.best_fitness, resumed.evaluations) == (whole.best_fitness, whole.evaluations)
