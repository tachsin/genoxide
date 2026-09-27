"""Branin, Goldstein-Price and the six-hump camel: find the minima of three two-dimensional
functions by restarting a local search from random points.

Each search is a hill climber with Gaussian steps; it ends in the minimum whose basin it started
in. The searches that end close together are grouped, and the table gives each group's best point
and value. The functions, their bounds and their global minima come from genoxide's problems,
which run evaluates in Rust.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/minima_2d/main.py
"""

import genoxide as gx

from trace import Trace

SEARCHES = 30
STEPS = 1_000


def significant(value):
    """The value rounded to 5 significant digits."""
    return float(f"{value:.4e}")


def five_digits(value):
    """5 significant digits, trailing zeros kept, e.g. 0.39789, 3.0000 or 840.00."""
    return f"{significant(value):#.5g}"


# each function, its name, and its name for the contour plot
functions = [
    (gx.problems.Branin(), "Branin", "branin"),
    (gx.problems.GoldsteinPrice(), "Goldstein-Price", "goldstein_price"),
    (gx.problems.SixHumpCamel(), "Six-hump camel", "six_hump_camel"),
]

print(f"{SEARCHES} local searches per function from random points, {STEPS} steps each")
# with GENOXIDE_TRACE=<file>, a trace of the searches for the plot on the example's page
trace = Trace()
for problem, name, function in functions:
    optimum = problem.optimum
    genome = problem.genome
    bounds = [genome.bounds] * genome.length if genome.length else list(genome.bounds)
    width = [high - low for low, high in bounds]
    trace.function(name, function, bounds, optimum)
    # per group of searches that ended close together: its best point and value, and its
    # searches
    minima = []
    for seed in range(1, SEARCHES + 1):
        search = gx.LocalSearch(
            genome,
            neighbor=gx.GaussianMutation(0.001, rate=1.0),
            neighbors=10,
            acceptance=gx.Improving(),
            objective="minimize",
            seed=seed,
        )
        result = search.run(problem, generations=STEPS, on_generation=trace.on_generation)
        point = result.best_genome.tolist()
        value = result.best_fitness
        # the same minimum: within 2% of the bounds' width in both genes
        close = (
            minimum
            for minimum in minima
            if all(abs(point[i] - minimum["point"][i]) <= 0.02 * width[i] for i in range(2))
        )
        minimum = next(close, None)
        if minimum is None:
            minima.append({"point": point, "value": value, "searches": 1})
        else:
            minimum["searches"] += 1
            if value < minimum["value"]:
                minimum["point"], minimum["value"] = point, value
    # by value, to 5 significant digits, then by the first gene
    minima.sort(key=lambda minimum: (significant(minimum["value"]), minimum["point"][0]))

    print()
    (low1, high1), (low2, high2) = bounds
    print(f"{name} in [{low1:.0f}, {high1:.0f}] x [{low2:.0f}, {high2:.0f}]")
    print("minimum reached   searches  best value")
    for minimum in minima:
        x1, x2 = minimum["point"]
        is_global = "  global" if abs(minimum["value"] - optimum.value) < 1e-3 else ""
        print(
            f"({x1:>6.3f}, {x2:>6.3f})  {minimum['searches']:>8}  "
            f"{five_digits(minimum['value']):>10}{is_global}"
        )
trace.write()
