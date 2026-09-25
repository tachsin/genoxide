"""The reference definition of the benchmark problems (docs/benchmarks/rules.md, rule 1.1).

Every adapter implements these functions in its own language. `run.py check` compares the adapters'
values with these at fixed points, and re-evaluates every solution a run reports with them.
"""

import math
import random

# the target of the single-objective problems: reached when the best value is at least (OneMax)
# or at most (the others) this
REAL_TARGET = 0.01

# (lower, upper) of every variable
REAL_BOUNDS = {
    "rastrigin": (-5.12, 5.12),
    "rosenbrock": (-5.0, 10.0),
    "ackley": (-32.768, 32.768),
}

# the multi-objective problems: the number of variables and objectives for the scenario's size, the
# number of objectives
FRONT_VARIABLES = {
    "zdt1": lambda size: size,
    "zdt2": lambda size: size,
    "zdt3": lambda size: size,
    # DTLZ with k = 10 (DTLZ2) and 5 (DTLZ1) distance variables
    "dtlz2": lambda size: size + 9,
    "dtlz1": lambda size: size + 4,
}
FRONT_OBJECTIVES = {"zdt1": lambda size: 2, "zdt2": lambda size: 2, "zdt3": lambda size: 2,
                    "dtlz2": lambda size: size, "dtlz1": lambda size: size}


def shift(n):
    """The optimum of Rastrigin and Ackley: s_i = 2 ((37 i + 11) mod 101) / 101 - 1, in [-1, 1]."""
    return [2 * ((37 * i + 11) % 101) / 101 - 1 for i in range(n)]


# ------------------------------------------------------------------------------------------------
# Single-objective: OneMax is maximized, the others minimized
# ------------------------------------------------------------------------------------------------


def onemax(bits):
    return sum(1 for bit in bits if bit)


def nqueens(order):
    """Diagonal conflicts of queens at (i, order[i]): for each diagonal, its queens minus one."""
    n = len(order)
    left, right = [0] * (2 * n - 1), [0] * (2 * n - 1)
    for i, column in enumerate(order):
        left[i + column] += 1
        right[n - 1 - i + column] += 1
    return sum(max(count - 1, 0) for count in left + right)


def rastrigin(x):
    d = [v - s for v, s in zip(x, shift(len(x)))]
    return 10 * len(d) + sum(v * v - 10 * math.cos(2 * math.pi * v) for v in d)


def rosenbrock(x):
    return sum(100 * (x[i + 1] - x[i] * x[i]) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1))


def ackley(x):
    n = len(x)
    d = [v - s for v, s in zip(x, shift(n))]
    return (-20 * math.exp(-0.2 * math.sqrt(sum(v * v for v in d) / n))
            - math.exp(sum(math.cos(2 * math.pi * v) for v in d) / n) + 20 + math.e)


SINGLE = {"onemax": onemax, "nqueens": nqueens, "rastrigin": rastrigin, "rosenbrock": rosenbrock,
          "ackley": ackley}


def maximized(problem):
    return problem == "onemax"


def target(problem, size):
    return {"onemax": size, "nqueens": 0}.get(problem, REAL_TARGET)


def reached(problem, size, best):
    return best >= size if maximized(problem) else best <= target(problem, size)


# ------------------------------------------------------------------------------------------------
# Multi-objective, all objectives minimized, all variables in [0, 1]
# ------------------------------------------------------------------------------------------------


def zdt_g(x):
    return 1 + 9 * sum(x[1:]) / (len(x) - 1)


def zdt1(x):
    g = zdt_g(x)
    return [x[0], g * (1 - math.sqrt(x[0] / g))]


def zdt2(x):
    g = zdt_g(x)
    return [x[0], g * (1 - (x[0] / g) ** 2)]


def zdt3(x):
    g = zdt_g(x)
    return [x[0], g * (1 - math.sqrt(x[0] / g) - x[0] / g * math.sin(10 * math.pi * x[0]))]


def dtlz2(x, objectives):
    g = sum((v - 0.5) ** 2 for v in x[objectives - 1:])
    values = []
    for m in range(objectives):
        f = 1 + g
        for v in x[:objectives - 1 - m]:
            f *= math.cos(v * math.pi / 2)
        if m > 0:
            f *= math.sin(x[objectives - 1 - m] * math.pi / 2)
        values.append(f)
    return values


def dtlz1(x, objectives):
    tail = x[objectives - 1:]
    g = 100 * (len(tail) + sum((v - 0.5) ** 2 - math.cos(20 * math.pi * (v - 0.5)) for v in tail))
    values = []
    for m in range(objectives):
        f = 0.5 * (1 + g)
        for v in x[:objectives - 1 - m]:
            f *= v
        if m > 0:
            f *= 1 - x[objectives - 1 - m]
        values.append(f)
    return values


def objectives(problem, size, x):
    if problem.startswith("dtlz"):
        return {"dtlz2": dtlz2, "dtlz1": dtlz1}[problem](x, size)
    return {"zdt1": zdt1, "zdt2": zdt2, "zdt3": zdt3}[problem](x)


def value(problem, size, solution):
    """The reference value of a solution: a number, or a list of objectives."""
    if problem in FRONT_VARIABLES:
        return objectives(problem, size, solution)
    return SINGLE[problem](solution)


# ------------------------------------------------------------------------------------------------
# The points `run.py check` evaluates: the optimum, and fixed random points
# ------------------------------------------------------------------------------------------------


def check_points(problem, size, count=20):
    """Solutions of the problem, the same every time."""
    rng = random.Random(f"{problem}-{size}")
    if problem == "onemax":
        return [[1] * size, [0] * size] + [[rng.randint(0, 1) for _ in range(size)] for _ in range(count)]
    if problem == "nqueens":
        points = [list(range(size))]
        for _ in range(count):
            order = list(range(size))
            rng.shuffle(order)
            points.append(order)
        return points
    if problem in REAL_BOUNDS:
        lower, upper = REAL_BOUNDS[problem]
        optimum = [1.0] * size if problem == "rosenbrock" else shift(size)
        return [optimum] + [[rng.uniform(lower, upper) for _ in range(size)] for _ in range(count)]
    n, m = FRONT_VARIABLES[problem](size), FRONT_OBJECTIVES[problem](size)
    # points on the optimal front: ZDT with the distance variables at 0, DTLZ at 0.5
    rest = 0.0 if problem.startswith("zdt") else 0.5
    front = [[rng.random() for _ in range(m - 1)] + [rest] * (n - m + 1) for _ in range(count // 2)]
    return front + [[rng.random() for _ in range(n)] for _ in range(count)]


def close(a, b, tolerance=1e-9):
    """Whether two values, or lists of objectives, agree to `tolerance` relative (absolute near 0)."""
    if isinstance(a, list) or isinstance(b, list):
        return (isinstance(a, list) and isinstance(b, list) and len(a) == len(b)
                and all(close(x, y, tolerance) for x, y in zip(a, b)))
    return abs(a - b) <= tolerance * max(1.0, abs(a), abs(b))
