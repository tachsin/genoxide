"""Benchmark adapter for SciPy's differential_evolution (scipy.optimize).

Usage:
    python bench.py <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
    python bench.py values <problem> <size>

The first prints one JSON line per solver per seed, see ../../README.md for the fields. The second
reads one JSON solution per line and prints its value with this adapter's fitness function.

The matched suite (docs/benchmarks/rules.md): SciPy runs only matched Rastrigin 30, with its
differential_evolution set to DE/rand/1/bin as the suite defines it, and prints nothing for any
other problem, size or mode. The settings, their sources in SciPy's docs and the differences are on
the page docs/benchmarks/libraries/scipy.md.

Rastrigin 30 has no target: a run ends at its budget or the time cap, where the Budget wrapper
stops the solver, or where SciPy's convergence test, which can't be turned off, ends it, with
"ended_by" (rule 2.2). differential_evolution evaluates each generation in one call
(vectorized=True, rule 3.4); the Budget counts every solution, and those outside the bounds
(rule 2.4).
"""

import os

# single-threaded numpy (BLAS), before numpy is imported
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import json
import math
import sys
import time

import numpy as np
from scipy.optimize import differential_evolution


class Stop(Exception):
    """Raised by Budget at the budget or the time cap: it ends the run."""


class Budget:
    """The objective function, vectorized as differential_evolution calls it with vectorized=True:
    x has one solution per column. Counts the evaluations itself (rule 3.2), one per column, and
    keeps the best solution. Rastrigin 30 has no target: raises Stop before a generation that
    starts past the budget or the time cap; a generation that would go past the budget is evaluated
    only up to it. Counts the evaluated solutions outside the bounds, as the solver gave them
    (rule 2.4)."""

    def __init__(self, function, lower, upper, max_evaluations, start, max_seconds):
        self.function = function
        self.lower, self.upper = lower, upper
        self.max_evaluations = max_evaluations
        self.start = start
        self.deadline = start + max_seconds
        self.evaluations = 0
        self.generations = 0
        # the evaluations at the end of the last generation, and that generation's (rule 2.3)
        self.generation_end = 0
        self.last = 0
        self.outside = 0
        self.best = math.inf
        self.best_x = None

    def __call__(self, x):
        if self.evaluations >= self.max_evaluations or time.perf_counter() >= self.deadline:
            raise Stop
        # never past the budget
        columns = x[:, :self.max_evaluations - self.evaluations]
        values = self.function(columns)
        self.evaluations += columns.shape[1]
        self.outside += int(np.sum(np.any((columns < self.lower) | (columns > self.upper), axis=0)))
        best = int(np.argmin(values))
        if values[best] < self.best:
            self.best = float(values[best])
            self.best_x = columns[:, best].copy()
        if columns.shape[1] < x.shape[1]:
            raise Stop
        return values

    def next_generation(self, *args, **kwargs):
        """differential_evolution's callback, called after every generation."""
        self.generations += 1
        self.last = self.evaluations - self.generation_end
        self.generation_end = self.evaluations

    def last_generation(self):
        """The evaluations since the start of the last generation (rule 2.3): of one cut short, or
        of the last one that ended."""
        return self.evaluations - self.generation_end or self.last


# -------------------------------------------------------------------------------------------------
# The fitness function, identical to benchmarks/problems.py (minimized). It takes the solutions as
# the columns of a numpy array, as differential_evolution passes them with vectorized=True, like
# SciPy's own scipy.optimize.rosen.
# -------------------------------------------------------------------------------------------------

LOWER, UPPER = -5.12, 5.12


def shift(n):
    """The optimum of rastrigin, away from the origin:
    s_i = 0.8 upper (2 ((37 i + 11) mod 101) / 101 - 1), computed in this order."""
    return np.array([0.8 * UPPER * (2 * ((37 * i + 11) % 101) / 101 - 1) for i in range(n)])


SHIFT = shift(1000)


def rastrigin(x):
    """Shifted: 10 n + sum((x_i - s_i)^2 - 10 cos(2 pi (x_i - s_i))), of each column of x."""
    n = x.shape[0]
    d = x - SHIFT[:n, None]
    return 10 * n + np.sum(d * d - 10 * np.cos(2 * np.pi * d), axis=0)


# -------------------------------------------------------------------------------------------------
# The method. It runs until Budget raises Stop at the budget or the time cap.
# -------------------------------------------------------------------------------------------------


def solve_de(size, seed, budget):
    """DE/rand/1/bin, the matched suite's (docs/benchmarks/libraries/scipy.md): 100 individuals
    drawn uniformly in the box, F 0.5 fixed, CR 0.9, one-to-one replacement when not worse,
    generational ('deferred'), no polish, each generation evaluated in one call (vectorized=True).
    Bounds (rule 2.4): SciPy redraws a trial gene outside the box uniformly in it.

    Rule 2.2: maxiter, a limit that's only a budget, is lifted. The convergence test can't be
    turned off: tol=0 and atol=0 end the run only when all 100 values are equal, when DE can no
    longer move. The run ends there, before the budget; the function returns why, for
    "ended_by"."""
    # a generator seeded with the run's seed: it draws the initial population, then
    # differential_evolution draws from it (rng=)
    rng = np.random.default_rng(seed)
    population = rng.uniform(LOWER, UPPER, (100, size))
    differential_evolution(
        budget,
        [(LOWER, UPPER)] * size,
        strategy="rand1bin",
        mutation=0.5,
        recombination=0.9,
        init=population,
        updating="deferred",
        vectorized=True,
        polish=False,
        tol=0,
        atol=0,
        maxiter=budget.max_evaluations,
        rng=rng,
        callback=budget.next_generation,
    )
    # it returned by itself, not by Stop: its convergence test
    return "convergence test (tol=0, atol=0): all 100 values equal"


# -------------------------------------------------------------------------------------------------


def values(problem, size):
    """Prints the value of each solution read from stdin, one JSON list per line (rule 1.2)."""
    if problem != "rastrigin":
        print(f"unknown problem {problem}", file=sys.stderr)
        sys.exit(2)
    for line in sys.stdin:
        if line.strip():
            x = np.array(json.loads(line), dtype=float)[:, None]
            print(json.dumps(float(rastrigin(x)[0])), flush=True)


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "values":
        values(sys.argv[2], int(sys.argv[3]))
        return
    if len(sys.argv) != 8:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    problem, size, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    seed_from, seed_to = int(sys.argv[4]), int(sys.argv[5])
    max_evaluations, max_seconds = int(sys.argv[6]), float(sys.argv[7])

    # the matched suite: SciPy runs only matched Rastrigin 30
    if (problem, size, mode) != ("rastrigin", 30, "matched"):
        return

    for seed in range(seed_from, seed_to + 1):
        # the clock starts before the initial population (rule 4.1)
        start = time.perf_counter()
        budget = Budget(rastrigin, LOWER, UPPER, max_evaluations, start, max_seconds)
        ended_by = None
        try:
            ended_by = solve_de(size, seed, budget)
        except Stop:
            pass
        # the clock stops when the run ends (rule 4.1)
        elapsed = time.perf_counter() - start
        print(json.dumps({
            "library": "scipy",
            "solver": "de",
            "problem": problem,
            "size": size,
            "mode": mode,
            "seed": seed,
            "time_s": round(elapsed, 6),
            "generations": budget.generations,
            "evaluations": budget.evaluations,
            "last_generation": budget.last_generation(),
            "outside": budget.outside,
            "best": budget.best,
            # Rastrigin 30 has no target: a fixed budget, measured by its time and final error
            "target": None,
            "success": False,
            "first_hit": None,
            **({"ended_by": ended_by} if ended_by else {}),
            "solution": budget.best_x.tolist(),
        }), flush=True)


if __name__ == "__main__":
    main()
