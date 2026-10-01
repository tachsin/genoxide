"""|x| by strongly typed genetic programming: find the absolute value from 20 points of it, with a
comparison that returns a Boolean and a conditional that takes one.

Two types, real numbers and Booleans (Montana 1995): ``less`` compares two reals and returns a
Boolean, ``if`` takes a Boolean and two reals, and every tree genoxide makes puts a Boolean where
a Boolean goes. The run stops at exact recovery: an error at the level of rounding, on the 20
training points and on 101 test points across [-1, 1], with the expression printed.

The primitives are the program's own (``gx.gp.PrimitiveSetBuilder``), evaluated by numpy: each
function is called once per node, on the columns of all the points. Their arithmetic is IEEE's,
the same bits as Rust's point by point, so the run is the Rust example's, to the bit.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/abs_typed/main.py
"""

import math

import numpy as np

import genoxide as gx

from trace import Trace

POPULATION = 1000


def primitives():
    """The primitives: arithmetic on reals, a comparison, Boolean functions and a conditional."""
    builder = gx.gp.PrimitiveSetBuilder()
    real = builder.new_type("real")
    boolean = builder.new_type("bool")
    builder.function("add", [real, real], real)
    builder.function("sub", [real, real], real)
    builder.function("mul", [real, real], real)
    builder.function("less", [real, real], boolean)
    builder.function("and", [boolean, boolean], boolean)
    builder.function("or", [boolean, boolean], boolean)
    builder.function("not", [boolean], boolean)
    builder.function("if", [boolean, real, real], real)
    builder.terminal("x", real)
    # ephemeral random constants: the integers -2 to 2
    builder.constants(real, gx.gp.Constants.integers(-2, 2))
    return builder.build(real)


# what the functions mean, on numpy columns: each is called once per node, on all the points
FUNCTIONS = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "less": np.less,
    "and": np.logical_and,
    "or": np.logical_or,
    "not": np.logical_not,
    "if": np.where,
}


class Data:
    """Points and |x| at them."""

    def __init__(self, xs):
        self.xs = np.asarray(xs, dtype=float)
        self.ys = np.abs(self.xs)

    def predict(self, tree):
        """The tree's values at the points (a constant broadcast to all of them)."""
        values = tree.evaluate({"x": self.xs}, FUNCTIONS)
        return np.broadcast_to(np.asarray(values, dtype=float), self.xs.shape)

    def rmse(self, tree):
        """The root mean squared error of the tree, summed in order as in Rust."""
        total = 0.0
        for error in (self.predict(tree) - self.ys).tolist():
            total += error * error
        return math.sqrt(total / len(self.ys))

    def deviation(self):
        """The standard deviation of the values."""
        ys = self.ys.tolist()
        mean = 0.0
        for y in ys:
            mean += y
        mean /= len(ys)
        squares = 0.0
        for y in ys:
            squares += (y - mean) * (y - mean)
        return math.sqrt(squares / len(ys))


def scientific(value):
    """``value`` with 2 decimals and an exponent, as Rust's ``{:.2e}`` writes it: 1.27e-10,
    0.00e0."""
    if value != value:
        return "NaN"
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"{mantissa}e{int(exponent)}"


# 20 training points uniform in [-1, 1], from a fixed seed (Koza's sampling, the points of his
# quartic), and 101 test points evenly spaced
training = Data(gx.gp.regression.problems.Koza1().dataset().training.x[:, 0])
test = Data([i / 50.0 - 1.0 for i in range(101)])
# exact recovery: an error of at most 1e-10 of the values' standard deviation
tolerance = 1e-10 * training.deviation()
print("|x| from 20 points in [-1, 1], with real and Boolean types")
print(f"{POPULATION} trees, until the RMSE is at most {scientific(tolerance)}")
print()

gp = gx.gp.Gp(primitives())
ga = gx.Ga(
    gp,
    population_size=POPULATION,
    initial_genomes=gp.ramped_half_and_half(POPULATION, 1),
    select=gx.DoubleTournament(7, 1.4),
    crossover=gx.gp.SubtreeCrossover(),
    mutation=gx.gp.Mutations(
        [
            (0.5, gx.gp.SubtreeMutation()),
            (0.3, gx.gp.PointMutation(count=1)),
            (0.1, gx.gp.HoistMutation()),
            (0.1, gx.gp.ShrinkMutation()),
        ]
    ),
    crossover_rate=0.9,
    mutation_rate=0.1,
    objective="minimize",
    seed=1,
)
trace = Trace(training)
result = ga.run(
    training.rmse, target=tolerance, generations=100, on_generation=trace.on_generation
)

best = result.best_genome
print(
    f"{result.stop_reason.capitalize()} after {result.generations} generations and "
    f"{result.evaluations} evaluations"
)
print(f"RMSE on the 20 training points: {scientific(training.rmse(best))}")
print(f"RMSE on the 101 test points:    {scientific(test.rmse(best))}")
print()
print(f"the expression, {len(best)} nodes of depth {best.depth}:")
print(best)
trace.write()
