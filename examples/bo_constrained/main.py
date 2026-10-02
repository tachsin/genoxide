"""Constrained Bayesian optimization of the toy problem of Gramacy et al. (2016): a linear objective
on [0, 1]^2 with two constraints whose values the fitness function gives one by one, each modeled
by a Gaussian process. The search maximizes the log expected improvement over the best feasible
point plus the logarithm of the probability of feasibility, and reaches the global minimum, on the
boundary of a wavy constraint, to within 1e-5. Then, as a contrast, the same search told only the
total violation, which it ignores.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/bo_constrained/main.py
"""

import math

import numpy as np

import genoxide as gx

from trace import Trace

# the global minimum and its point (tests/reference/gramacy_toy.py, with mpmath)
MINIMUM = 0.5997880520100676
MINIMIZER = [0.19512268347207176, 0.4046653685379958]
# how close to the minimum, and the evaluations the search may take at most
TOLERANCE = 1e-5
BUDGET = 60


def scientific(value):
    """Two significant digits, e.g. 1.2e-7."""
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def constraints(x):
    """The values of the two constraints g(x) <= 0, with genoxide's portable sine."""
    wave = gx.math.sin(2.0 * math.pi * (x[0] * x[0] - 2.0 * x[1]))
    return np.array([1.5 - x[0] - 2.0 * x[1] - 0.5 * wave, x[0] * x[0] + x[1] * x[1] - 1.5])


def toy(x):
    """x1 + x2, and the constraints' values."""
    return x[0] + x[1], constraints(x)


def feasible(g):
    return bool(np.all(g <= 0.0))


print("Gramacy et al.'s toy problem: minimize x1 + x2 on [0, 1]^2 subject to")
print("  c1 = 1.5 - x1 - 2 x2 - sin(2 pi (x1^2 - 2 x2)) / 2 <= 0, c2 = x1^2 + x2^2 - 1.5 <= 0")
print(
    f"global minimum {MINIMUM:.6f} at ({MINIMIZER[0]:.6f}, {MINIMIZER[1]:.6f}), on the boundary "
    "c1 = 0"
)
print("6 points of a Latin hypercube, then a point per step by log-EI x P(feasible)")
print("evaluation        x1        x2         f         c1         c2  best feasible f - f*")
# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(MINIMIZER, MINIMUM, constraints)
shown = {"printed": 0, "best": math.inf}


def on_generation(progress):
    # the points evaluated in this generation
    for index in range(shown["printed"], len(progress.population)):
        x = progress.population[index]
        value, g = toy(x)
        if feasible(g):
            shown["best"] = min(shown["best"], float(value))
        gap = scientific(shown["best"] - MINIMUM) if math.isfinite(shown["best"]) else "none yet"
        print(
            f"{index + 1:>10} {x[0]:>9.6f} {x[1]:>9.6f} {value:>9.6f} {g[0]:>10.6f} "
            f"{g[1]:>10.6f} {gap:>20}"
        )
    shown["printed"] = len(progress.population)


space = gx.Real((0.0, 1.0), length=2)
bo = gx.Bo(space, objective="minimize", seed=1)
result = bo.run(
    toy,
    constraints=2,
    target=MINIMUM + TOLERANCE,
    evaluations=BUDGET,
    on_generation=on_generation,
    control=trace.record,
)
x = result.best_genome
value = float(x[0] + x[1])
assert feasible(constraints(x))
distance = math.hypot(x[0] - MINIMIZER[0], x[1] - MINIMIZER[1])
print(
    f"{result.evaluations} evaluations: the best feasible point ({x[0]:.6f}, {x[1]:.6f}), "
    f"{value:.6f}, {scientific(value - MINIMUM)} above the minimum, {scientific(distance)} from "
    "its point"
)
assert value - MINIMUM <= TOLERANCE
trace.write()


# the contrast: the same problem as (score, violation), without the constraints' values
def violation(x):
    value, g = toy(x)
    return value, max(g[0], 0.0) + max(g[1], 0.0)


evaluated = {}


def keep(progress):
    evaluated["population"] = progress.population


blind = gx.Bo(space, objective="minimize", seed=1)
contrast = blind.run(violation, evaluations=result.evaluations, on_generation=keep)
count = sum(feasible(constraints(x)) for x in evaluated["population"])
best = contrast.best_genome
g = constraints(best)
if feasible(g):
    score = float(best[0] + best[1])
    summary = f"{score:.6f}, {scientific(score - MINIMUM)} above the minimum"
else:
    summary = f"infeasible, by {scientific(max(g[0], 0.0) + max(g[1], 0.0))}"
print(
    f"without the constraints' values: {count} feasible points of {contrast.evaluations}, the "
    f"best {summary}"
)
