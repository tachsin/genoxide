"""The reference definition of the benchmark problems (docs/benchmarks/rules.md, rule 1.1).

Every adapter implements these functions in its own language. `run.py check` compares the adapters'
values with these at fixed points, and re-evaluates every solution a run reports with them.
"""

import math
import random

# the target of the problems: reached when the best value is at least (OneMax)
# or at most (the others) this
REAL_TARGET = 0.01

# (lower, upper) of every variable
REAL_BOUNDS = {
    "rastrigin": (-5.12, 5.12),
    "rosenbrock": (-5.0, 10.0),
    "ackley": (-32.768, 32.768),
}


def shift(problem, n):
    """The optimum of Rastrigin and Ackley (rule 1.4): s_i = 0.8 * upper * (2 * ((37 i + 11) mod 101)
    / 101 - 1), with `upper` the box's upper bound, so within 80% of the box. Computed in this order
    in every language, for the same doubles."""
    upper = REAL_BOUNDS[problem][1]
    return [0.8 * upper * (2 * ((37 * i + 11) % 101) / 101 - 1) for i in range(n)]


# ------------------------------------------------------------------------------------------------
# OneMax is maximized, the others minimized
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
    d = [v - s for v, s in zip(x, shift("rastrigin", len(x)))]
    return 10 * len(d) + sum(v * v - 10 * math.cos(2 * math.pi * v) for v in d)


def rosenbrock(x):
    return sum(100 * (x[i + 1] - x[i] * x[i]) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1))


def ackley(x):
    n = len(x)
    d = [v - s for v, s in zip(x, shift("ackley", n))]
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


def value(problem, size, solution):
    """The reference value of a solution."""
    return SINGLE[problem](solution)


# ------------------------------------------------------------------------------------------------
# The points `run.py check` evaluates: the optimum, and fixed random points
# ------------------------------------------------------------------------------------------------


def nqueens_solution(size):
    """A board of `size` queens without conflicts, for any size but 2 and 3: the explicit
    construction (Hoffman, Loessi and Moore, 1969; Bernhardsson, 1991) that puts the queens of rows
    0, 1, ... in the even columns 2, 4, ... and then the odd columns 1, 3, ... (counting from 1),
    with the swaps the sizes 6k + 2 and 6k + 3 need."""
    evens = list(range(2, size + 1, 2))
    odds = list(range(1, size + 1, 2))
    if size % 6 == 2:
        # 3 and 1 swap, 5 goes last
        odds = [3, 1] + [c for c in odds if c not in (1, 3, 5)] + [5]
    elif size % 6 == 3:
        # 2 goes last among the evens, 1 and 3 last among the odds
        evens = evens[1:] + [2]
        odds = [c for c in odds if c not in (1, 3)] + [1, 3]
    order = [c - 1 for c in evens + odds]
    assert sorted(order) == list(range(size)) and nqueens(order) == 0, f"no solution for {size}"
    return order


def check_points(problem, size, count=20):
    """Solutions of the problem, the same every time."""
    rng = random.Random(f"{problem}-{size}")
    if problem == "onemax":
        return [[1] * size, [0] * size] + [[rng.randint(0, 1) for _ in range(size)] for _ in range(count)]
    if problem == "nqueens":
        # a solution, the target (0 conflicts), and the identity: every queen on one diagonal
        points = [nqueens_solution(size), list(range(size))]
        for _ in range(count):
            order = list(range(size))
            rng.shuffle(order)
            points.append(order)
        return points
    lower, upper = REAL_BOUNDS[problem]
    optimum = [1.0] * size if problem == "rosenbrock" else shift(problem, size)
    return [optimum] + [[rng.uniform(lower, upper) for _ in range(size)] for _ in range(count)]


def close(a, b, tolerance=1e-9):
    """Whether two values agree to `tolerance` relative (absolute near 0)."""
    return abs(a - b) <= tolerance * max(1.0, abs(a), abs(b))
