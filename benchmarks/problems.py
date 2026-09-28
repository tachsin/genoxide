"""The reference definition of the benchmark problems (docs/benchmarks/rules.md, rule 1.1): OneMax,
the shifted Rastrigin function and Rosenbrock's.

Every adapter implements these functions in its own language. `run.py check` compares the adapters'
values with these at fixed points, and re-evaluates every solution a run reports with them.
"""

import math
import random

# the target of Rosenbrock: reached when the best value is at most this. OneMax's is all ones.
# Rastrigin has none: every run uses a fixed budget, and it's measured by the time for it and the
# error at the end
REAL_TARGET = 0.01

# (lower, upper) of every variable
REAL_BOUNDS = {
    "rastrigin": (-5.12, 5.12),
    "rosenbrock": (-5.0, 10.0),
}


def shift(problem, n):
    """The optimum of the shifted Rastrigin function (rule 1.4): s_i = 0.8 * upper * (2 * ((37 i +
    11) mod 101) / 101 - 1), with `upper` the box's upper bound, so within 80% of the box. Computed
    in this order in every language, for the same doubles."""
    upper = REAL_BOUNDS[problem][1]
    return [0.8 * upper * (2 * ((37 * i + 11) % 101) / 101 - 1) for i in range(n)]


# ------------------------------------------------------------------------------------------------
# OneMax is maximized, the others minimized
# ------------------------------------------------------------------------------------------------


def onemax(bits):
    return sum(1 for bit in bits if bit)


def rastrigin(x):
    d = [v - s for v, s in zip(x, shift("rastrigin", len(x)))]
    return 10 * len(d) + sum(v * v - 10 * math.cos(2 * math.pi * v) for v in d)


def rosenbrock(x):
    return sum(100 * (x[i + 1] - x[i] * x[i]) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1))


SINGLE = {"onemax": onemax, "rastrigin": rastrigin, "rosenbrock": rosenbrock}


def maximized(problem):
    return problem == "onemax"


def target(problem, size):
    """The problem's target, or None for a fixed budget (Rastrigin)."""
    if problem == "rastrigin":
        return None
    return size if problem == "onemax" else REAL_TARGET


def reached(problem, size, best):
    goal = target(problem, size)
    if goal is None:
        return False
    return best >= goal if maximized(problem) else best <= goal


def value(problem, size, solution):
    """The reference value of a solution."""
    return SINGLE[problem](solution)


# ------------------------------------------------------------------------------------------------
# The points `run.py check` evaluates: the optimum, and fixed random points
# ------------------------------------------------------------------------------------------------


def check_points(problem, size, count=20):
    """Solutions of the problem, the same every time."""
    rng = random.Random(f"{problem}-{size}")
    if problem == "onemax":
        return [[1] * size, [0] * size] + [[rng.randint(0, 1) for _ in range(size)] for _ in range(count)]
    lower, upper = REAL_BOUNDS[problem]
    optimum = [1.0] * size if problem == "rosenbrock" else shift(problem, size)
    return [optimum] + [[rng.uniform(lower, upper) for _ in range(size)] for _ in range(count)]


def close(a, b, tolerance=1e-9):
    """Whether two values agree to `tolerance` relative (absolute near 0)."""
    return abs(a - b) <= tolerance * max(1.0, abs(a), abs(b))
