"""Benchmark adapter for pycma (CMA-ES).

Usage:
    python bench.py <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
    python bench.py values <problem> <size>

The first prints one JSON line per solver per seed, see ../../README.md for the fields. The second
reads one JSON solution per line and prints its value with this adapter's fitness functions.

pycma only optimizes real-valued functions: it runs rastrigin, rosenbrock and ackley, and prints
nothing for the other problems. Which methods run and why, with the links to pycma's docs, is on
the page docs/benchmarks/libraries/pycma.md:
- Rastrigin and Ackley (multimodal): cma.fmin2 with IPOP restarts, and with BIPOP restarts;
- Rosenbrock: cma.fmin2 as in its docstring's Rosenbrock example, and cma.fmin_lq_surr2 (lq-CMA-ES),
  whose docstring example is Rosenbrock too, both with IPOP restarts only if a run
  stops before the target (rule 2.2).

Every run ends only at the target, the budget or the time cap (rules 2.1 and 2.2 of
docs/benchmarks/rules.md): the Budget wrapper stops fmin2 there, and fmin2's restarts continue the
run when CMA-ES stops by its own criteria, see solve.
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


class Stop(Exception):
    """Raised by Budget at the target, the budget or the time cap: it ends the run."""


class Budget:
    """Counts the evaluations itself (rule 3.2): every call of the fitness function, including
    fmin2's evaluation of the final mean of each run (option eval_final_mean). Keeps the best
    solution, and raises Stop right after the evaluation that reaches the target, and before an
    evaluation past the budget or the time cap. (pycma's own ftarget and maxfevals options stop
    only after the generation.) Counts the evaluated solutions outside the bounds, as pycma gave
    them (rule 2.4)."""

    def __init__(self, function, lower, upper, max_evaluations, deadline):
        self.function = function
        self.lower, self.upper = lower, upper
        self.max_evaluations = max_evaluations
        self.deadline = deadline
        self.evaluations = 0
        self.generations = 0
        self.outside = 0
        self.best = math.inf
        self.best_x = None

    def __call__(self, x):
        if self.evaluations >= self.max_evaluations or time.perf_counter() >= self.deadline:
            raise Stop
        self.evaluations += 1
        self.outside += bool(np.any(x < self.lower) or np.any(x > self.upper))
        value = float(self.function(x))
        if value < self.best:
            self.best = value
            self.best_x = np.array(x, dtype=float)
            if value <= TARGET:
                raise Stop
        return value

    def next_generation(self, es):
        self.generations += 1


# -------------------------------------------------------------------------------------------------
# Fitness functions, identical to benchmarks/problems.py (minimized). pycma calls them with one
# solution, a numpy array, and its own test functions (cma.ff) are written with numpy like these.
# -------------------------------------------------------------------------------------------------


@functools.cache
def shift(n):
    """The optimum of rastrigin and ackley, away from the origin: s_i = 2 ((37 i + 11) mod 101) / 101 - 1."""
    return np.array([2 * ((37 * i + 11) % 101) / 101 - 1 for i in range(n)])


def rastrigin(x):
    """Shifted: 10 n + sum((x_i - s_i)^2 - 10 cos(2 pi (x_i - s_i)))."""
    d = x - shift(len(x))
    return 10 * len(x) + np.sum(d * d - 10 * np.cos(2 * np.pi * d))


def rosenbrock(x):
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def ackley(x):
    """Shifted: Ackley of x - s."""
    n = len(x)
    d = x - shift(n)
    return (-20 * np.exp(-0.2 * np.sqrt(np.sum(d * d) / n))
            - np.exp(np.sum(np.cos(2 * np.pi * d)) / n) + 20 + math.e)


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


def solve(size, lower, upper, seed, budget, method):
    """cma.fmin2 with restarts, as its docstring describes them (help(cma.fmin2), "restarts",
    "incpopsize", "bipop"), or cma.fmin_lq_surr2, lq-CMA-ES, with the same arguments:
    - restarts=9, the setting the docstring recommends ("restarts <= 9"), and incpopsize=2, its
      default: each IPOP restart doubles the population. With bipop=True, BIPOP-CMA-ES, whose
      docstring example is the Rastrigin function, also interleaves restarts with small
      populations and smaller initial step-sizes, which don't count as restarts.
    - x0 a callable that draws a uniform random point in the domain before each restart, "to
      restart from different points (recommended)".
    - sigma0 a quarter of the domain width: "sigma0 should be about 1/4th of the search domain
      width".
    - the domain as the bounds option, with pycma's default boundary handling, BoundTransform
      (CMAOptions: "BoundaryHandler = BoundTransform"): it maps every sampled point into the
      bounds before the evaluation, so every solution evaluated is inside them (rule 2.4).
    - every other option at its default (population 4 + 3 ln n, the tolerances tolfun, tolx,
      tolstagnation etc. that end each run and trigger the next restart).
    lq-CMA-ES (method "lq") builds a linear or quadratic model of the function from the evaluated
    points and evaluates only part of each population ("to circumvent evaluations of the objective
    function", help(cma.fmin_lq_surr2)); its restarts double the population like IPOP.

    Rule 2.2: the restarts continue a run when CMA-ES stops by its criteria. If all 9 end before
    the budget, the function starts again with its first population size. pycma seeds numpy's
    global random state with the seed option, and each call starts from seed * 100000 + 1000 *
    call + 1 (seed 0 would mean "seed from the clock"). fmin2 adds 1 to it at each restart, so no
    two of its runs share a seed; fmin_lq_surr2 keeps it, and its restarts differ by their x0,
    drawn before the seeding, and their population size."""
    call = 0
    while True:
        options = {
            "bounds": [lower, upper],
            "seed": seed * 100_000 + 1000 * call + 1,
            # no output, no log files (outcmaes/) in the working directory
            "verbose": -9,
            "verb_disp": 0,
            "verb_log": 0,
        }
        x0 = lambda: np.random.uniform(lower, upper, size)
        sigma0 = (upper - lower) / 4
        # pycma draws each x0 before it seeds: the first one comes from this seeding, so a run
        # doesn't depend on the runs before it
        np.random.seed(options["seed"])
        if method == "lq":
            cma.fmin_lq_surr2(budget, x0, sigma0, options, restarts=9, incpopsize=2,
                              callback=budget.next_generation)
        else:
            cma.fmin2(budget, x0, sigma0, options, restarts=9, incpopsize=2, bipop=method == "bipop",
                      callback=budget.next_generation)
        print(f"pycma {method} seed {seed}: call {call} ended its 9 restarts after "
              f"{budget.evaluations} evaluations, best {budget.best}", file=sys.stderr)
        call += 1


def solvers(problem):
    """(solver name, method) of the problem type, see the page."""
    if problem == "rosenbrock":
        return [("cma_es", "ipop"), ("lq_cma_es", "lq")]
    return [("ipop_cma_es", "ipop"), ("bipop_cma_es", "bipop")]


# -------------------------------------------------------------------------------------------------


def values(problem, size):
    """Prints the value of each solution read from stdin, one JSON list per line (rule 1.2)."""
    function = PROBLEMS[problem][0]
    for line in sys.stdin:
        if line.strip():
            print(json.dumps(float(function(np.array(json.loads(line), dtype=float)))), flush=True)


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

    if problem in UNSUPPORTED:
        return
    if problem not in PROBLEMS:
        print(f"unknown problem {problem}", file=sys.stderr)
        sys.exit(2)
    function, (lower, upper) = PROBLEMS[problem]

    for seed in range(seed_from, seed_to + 1):
        for solver, method in solvers(problem):
            # the clock starts before the initial population (rule 4.1)
            start = time.perf_counter()
            budget = Budget(function, lower, upper, max_evaluations, start + max_seconds)
            try:
                solve(size, lower, upper, seed, budget, method)
            except Stop:
                pass
            elapsed = time.perf_counter() - start
            print(json.dumps({
                "library": "pycma",
                "solver": solver,
                "problem": problem,
                "size": size,
                "mode": mode,
                "seed": seed,
                "time_s": round(elapsed, 6),
                "generations": budget.generations,
                "evaluations": budget.evaluations,
                "outside": budget.outside,
                "best": budget.best,
                "target": TARGET,
                "success": bool(budget.best <= TARGET),
                "solution": budget.best_x.tolist(),
            }), flush=True)


if __name__ == "__main__":
    main()
