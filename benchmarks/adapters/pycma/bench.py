"""Benchmark adapter for pycma (CMA-ES).

Usage: python bench.py <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
Prints one JSON line per solver per seed, see ../../README.md for the fields.

pycma only optimizes real-valued functions: rastrigin, rosenbrock and ackley. It prints nothing for
the other problems.
"""

import os

# single-threaded numpy (BLAS), before numpy is imported
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import functools
import json
import math
import sys
import time

import cma
import numpy as np


class BudgetExhausted(Exception):
    pass


class Budget:
    """Counts fitness evaluations and tracks the best, raises BudgetExhausted at max_evaluations
    or max_seconds (pycma has no evaluation budget across its restarts)."""

    def __init__(self, function, max_evaluations, max_seconds):
        self.function = function
        self.evaluations = 0
        self.generations = 0
        self.best = math.inf
        self.max_evaluations = max_evaluations
        self.deadline = time.perf_counter() + max_seconds

    def __call__(self, x):
        if self.evaluations >= self.max_evaluations or time.perf_counter() >= self.deadline:
            raise BudgetExhausted
        self.evaluations += 1
        value = self.function(x)
        if value < self.best:
            self.best = value
        return value

    def next_generation(self, es):
        self.generations += 1


# -------------------------------------------------------------------------------------------------
# Fitness functions, identical to the other adapters (minimized)
# -------------------------------------------------------------------------------------------------


@functools.cache
def shift(n):
    """The optimum of rastrigin and ackley, away from the origin: s_i = 2 ((37 i + 11) mod 101) / 101 - 1."""
    return tuple(2 * ((37 * i + 11) % 101) / 101 - 1 for i in range(n))


def rastrigin(x):
    """Shifted: 10 n + sum((x_i - s_i)^2 - 10 cos(2 pi (x_i - s_i)))."""
    d = [v - s for v, s in zip(x, shift(len(x)))]
    return float(10 * len(d) + sum(v * v - 10 * math.cos(2 * math.pi * v) for v in d))


def rosenbrock(x):
    return float(sum(100 * (x[i + 1] - x[i] * x[i]) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1)))


def ackley(x):
    """Shifted: Ackley of x - s."""
    n = len(x)
    d = [v - s for v, s in zip(x, shift(n))]
    return float(
        -20 * math.exp(-0.2 * math.sqrt(sum(v * v for v in d) / n))
        - math.exp(sum(math.cos(2 * math.pi * v) for v in d) / n)
        + 20 + math.e
    )


TARGET = 0.01

# (fitness function, (lower, upper) of every variable)
PROBLEMS = {
    "rastrigin": (rastrigin, (-5.12, 5.12)),
    "rosenbrock": (rosenbrock, (-5.0, 10.0)),
    "ackley": (ackley, (-32.768, 32.768)),
}
UNSUPPORTED = {"onemax", "nqueens", "zdt1", "zdt2", "zdt3", "dtlz1", "dtlz2"}


# -------------------------------------------------------------------------------------------------
# Solver
# -------------------------------------------------------------------------------------------------


def solve_cmaes(size, lower, upper, seed, budget):
    """IPOP-CMA-ES with the library's defaults, as the fmin2 docstring recommends for multimodal
    functions: 9 restarts with the population doubled each time (incpopsize=2), each restart from
    a new random x0 in the domain (x0 as a callable), sigma0 a quarter of the domain width, and
    the domain as the bounds option (pycma's default BoundTransform). The ftarget option stops
    the run and the restarts after the generation that reaches the target."""
    options = {
        "bounds": [lower, upper],
        "ftarget": TARGET,
        # pycma treats seed 0 as "seed from the clock", so shift it by one
        "seed": seed + 1,
        # no output, no log files (outcmaes/) in the working directory
        "verbose": -9,
        "verb_disp": 0,
        "verb_log": 0,
    }
    try:
        cma.fmin2(
            budget, lambda: np.random.uniform(lower, upper, size), (upper - lower) / 4, options,
            restarts=9, incpopsize=2, callback=budget.next_generation,
        )
    except BudgetExhausted:
        pass
    return budget.best, budget.generations


def main():
    if len(sys.argv) != 8:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    problem, size, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    seed_from, seed_to = int(sys.argv[4]), int(sys.argv[5])
    max_evaluations, max_seconds = int(sys.argv[6]), float(sys.argv[7])

    if problem in UNSUPPORTED:
        return
    if problem not in PROBLEMS:
        print(f"unknown problem {problem}", file=sys.stderr)
        sys.exit(2)
    function, (lower, upper) = PROBLEMS[problem]

    for seed in range(seed_from, seed_to + 1):
        np.random.seed(seed)
        budget = Budget(function, max_evaluations, max_seconds)
        start = time.perf_counter()
        best, generations = solve_cmaes(size, lower, upper, seed, budget)
        elapsed = time.perf_counter() - start
        print(json.dumps({
            "library": "pycma",
            "solver": "cma_es",
            "problem": problem,
            "size": size,
            "mode": mode,
            "seed": seed,
            "time_s": round(elapsed, 6),
            "generations": generations,
            "evaluations": budget.evaluations,
            "best": best,
            "target": TARGET,
            "success": bool(best <= TARGET),
        }), flush=True)


if __name__ == "__main__":
    main()
