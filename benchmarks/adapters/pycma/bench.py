"""Benchmark adapter for pycma (CMA-ES).

Usage:
    python bench.py <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
    python bench.py values <problem> <size>

The first prints one JSON line per solver per seed, see ../../README.md for the fields. The second
reads one JSON solution per line and prints its value with this adapter's fitness function.

The matched suite (docs/benchmarks/rules.md): pycma runs only matched Rosenbrock 10, with its
CMAEvolutionStrategy set to the suite's CMA-ES, and prints nothing for any other problem, size or
mode. The settings, their sources in pycma's docs and the differences are on the page
docs/benchmarks/libraries/pycma.md.

Every run ends only at the target, the budget or the time cap (rules 2.1 and 2.2): the ask-and-tell
loop never consults es.stop(), and the Budget wrapper ends the run. Each generation is evaluated in
one numpy call (rule 3.4); the Budget counts every solution.
"""

import os

# single-threaded numpy (BLAS), before numpy is imported
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import json
import math
import sys
import time
import traceback

import cma
import numpy as np


class Stop(Exception):
    """Raised by Budget at the target, the budget or the time cap: it ends the run."""


class Budget:
    """Counts the evaluations itself (rule 3.2): every solution evaluated, one per row of a
    generation. Keeps the best solution and the first evaluation that reaches the target
    (first_hit). Raises Stop after the generation whose evaluations reach the target, and before
    one that starts past the budget or the time cap; a generation that would go past the budget is
    evaluated only up to it. Counts the evaluated solutions outside the bounds, as pycma gave them
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
        self.first_hit = None
        # why pycma ended the run with an error, if it did (rule 8.4)
        self.ended_by = None

    def batch(self, solutions):
        """The values of a generation's solutions, in one numpy call."""
        if self.evaluations >= self.max_evaluations or time.perf_counter() >= self.deadline:
            raise Stop
        x = np.asarray(solutions, dtype=float)
        # never past the budget
        rows = x[:self.max_evaluations - self.evaluations]
        values = self.function(rows)
        now = time.perf_counter()
        before = self.evaluations
        self.evaluations += len(rows)
        # outside: a gene below, above or not a number (pycma samples nan once sigma is nan)
        self.outside += int(np.sum(np.any(~((rows >= self.lower) & (rows <= self.upper)), axis=1)))
        best = int(np.argmin(values))
        if values[best] < self.best:
            self.best = float(values[best])
            self.best_x = rows[best].copy()
        if self.first_hit is None and self.best <= TARGET:
            hit = int(np.flatnonzero(values <= TARGET)[0])
            self.first_hit = {"evaluations": before + hit + 1,
                              "time_s": round(now - self.start, 6)}
        if self.first_hit is not None or len(rows) < len(x):
            raise Stop
        return values.tolist()

    def next_generation(self):
        """Called after every generation."""
        self.generations += 1
        self.last = self.evaluations - self.generation_end
        self.generation_end = self.evaluations

    def last_generation(self):
        """The evaluations since the start of the last generation (rule 2.3): of one cut short, or
        of the last one that ended."""
        return self.evaluations - self.generation_end or self.last


# -------------------------------------------------------------------------------------------------
# The fitness function, identical to benchmarks/problems.py (minimized). It takes the solutions as
# the rows of a numpy array, a generation at once, and returns their values, written with numpy like
# pycma's own test functions (cma.ff).
# -------------------------------------------------------------------------------------------------


def rosenbrock(x):
    return np.sum(100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2, axis=1)


TARGET = 0.01
LOWER, UPPER = -5.0, 10.0


# -------------------------------------------------------------------------------------------------
# The method. It runs until Budget raises Stop at the target, the budget or the time cap.
# -------------------------------------------------------------------------------------------------


def solve_cma_es(size, seed, budget):
    """CMA-ES, the matched suite's (docs/benchmarks/libraries/pycma.md): pycma's
    CMAEvolutionStrategy with its default population (10) and recombination weights (5 positive),
    CMA_active=False (no negative weights), the mean drawn uniformly in the box, sigma0 0.3 of the
    box width, the box as bounds with pycma's default BoundTransform (rule 2.4), and no cap on the
    standard deviations (maxstd_boundrange, by default a third of the box width with bounds).

    Rule 2.2: the ask-and-tell loop never consults es.stop(), so no criterion ends the run, and
    there are no restarts. Seeds: pycma samples from numpy's global random state (its option randn,
    np.random.randn), which the adapter seeds with the run's seed; x0 is drawn from it too. The seed
    option is np.nan, "do nothing": pycma would read a seed of 0 as "seed from the clock".

    Rule 8.4: long after converging to a local optimum, the distribution can degenerate until
    pycma raises an error (sigma nan, then an AssertionError in its sampler). The run ends there,
    not reached, and says so in "ended_by"."""
    np.random.seed(seed)
    x0 = np.random.uniform(LOWER, UPPER, size)
    es = cma.CMAEvolutionStrategy(x0, 0.3 * (UPPER - LOWER), {
        "bounds": [LOWER, UPPER],
        "CMA_active": False,
        "maxstd_boundrange": np.inf,
        "seed": np.nan,
        # no output, no log files (outcmaes/) in the working directory
        "verbose": -9,
        "verb_disp": 0,
        "verb_log": 0,
    })
    try:
        while True:
            solutions = es.ask()
            es.tell(solutions, budget.batch(solutions))
            budget.next_generation()
    except Stop:
        raise
    except Exception as error:  # noqa: BLE001, an error of pycma ends the run (rule 8.4)
        frame = traceback.extract_tb(error.__traceback__)[-1]
        budget.ended_by = (f"{type(error).__name__} in {os.path.basename(frame.filename)}, "
                           f"line {frame.lineno} ({frame.name})")


# -------------------------------------------------------------------------------------------------


def values(problem, size):
    """Prints the value of each solution read from stdin, one JSON list per line (rule 1.2)."""
    if problem != "rosenbrock":
        print(f"unknown problem {problem}", file=sys.stderr)
        sys.exit(2)
    for line in sys.stdin:
        if line.strip():
            print(json.dumps(float(rosenbrock(np.array([json.loads(line)], dtype=float))[0])), flush=True)


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

    # the matched suite: pycma runs only matched Rosenbrock 10
    if (problem, size, mode) != ("rosenbrock", 10, "matched"):
        return

    for seed in range(seed_from, seed_to + 1):
        # the clock starts before the initial distribution (rule 4.1)
        start = time.perf_counter()
        budget = Budget(rosenbrock, LOWER, UPPER, max_evaluations, start, max_seconds)
        try:
            solve_cma_es(size, seed, budget)
        except Stop:
            pass
        # the clock stops when the run ends (rule 4.1)
        elapsed = time.perf_counter() - start
        print(json.dumps({
            "library": "pycma",
            "solver": "cma_es",
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
            "target": TARGET,
            "success": bool(budget.best <= TARGET),
            "first_hit": budget.first_hit,
            **({"ended_by": budget.ended_by} if budget.ended_by else {}),
            "solution": budget.best_x.tolist(),
        }), flush=True)


if __name__ == "__main__":
    main()
