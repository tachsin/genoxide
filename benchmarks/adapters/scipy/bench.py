"""Benchmark adapter for SciPy's differential evolution (scipy.optimize.differential_evolution).

Usage: python bench.py <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
Prints one JSON line per solver per seed, see ../../README.md for the fields.

differential_evolution only optimizes real-valued functions: rastrigin, rosenbrock and ackley. It
prints nothing for the other problems.
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

import numpy as np
from scipy.optimize import differential_evolution


class BudgetExhausted(Exception):
    pass


class Budget:
    """Counts fitness evaluations (the L-BFGS-B polish included) and tracks the best, raises
    BudgetExhausted at max_evaluations or max_seconds."""

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

    def next_generation(self, intermediate_result):
        """Called after each generation: returning True stops at the target (the polish still runs)."""
        self.generations += 1
        return intermediate_result.fun <= TARGET


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


def solve_de(size, lower, upper, seed, budget):
    """differential_evolution with its defaults, as a user calls it: best1bin, popsize 15 (15 * n
    individuals), mutation (0.5, 1) with dithering, recombination 0.7, Latin hypercube
    initialization, immediate updating, maxiter 1000 and its own convergence test (tol 0.01), and
    polish=True: a final L-BFGS-B from the best, with finite-difference gradients whose
    evaluations are counted like the others. The callback stops after the generation that
    reaches the target."""
    try:
        differential_evolution(budget, [(lower, upper)] * size, rng=seed, callback=budget.next_generation)
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
        best, generations = solve_de(size, lower, upper, seed, budget)
        elapsed = time.perf_counter() - start
        print(json.dumps({
            "library": "scipy",
            "solver": "de",
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
