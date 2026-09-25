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
run when CMA-ES stops by its own criteria, see solve. fmin2 evaluates each population in one call
of its parallel_objective, a numpy function of all its solutions; the Budget counts every one.
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
    """Counts the evaluations itself (rule 3.2): every solution evaluated, one per row of a
    population (batch) and one per single call, including fmin2's evaluation of the final mean of
    each run (option eval_final_mean). Keeps the best solution and the first evaluation that
    reaches the target (first_hit). Raises Stop after the call whose evaluations reach the target,
    and before one that starts past the budget or the time cap; a population that would go past
    the budget is evaluated only up to it. (pycma's own ftarget and maxfevals options stop only
    after the generation.) Counts the evaluated solutions outside the bounds, as pycma gave them
    (rule 2.4)."""

    def __init__(self, function, lower, upper, max_evaluations, start, max_seconds):
        self.function = function
        self.lower, self.upper = lower, upper
        self.max_evaluations = max_evaluations
        self.start = start
        self.deadline = start + max_seconds
        self.evaluations = 0
        self.generations = 0
        self.outside = 0
        self.best = math.inf
        self.best_x = None
        self.first_hit = None

    def batch(self, solutions):
        """fmin2's parallel_objective: the values of a list of solutions, in one numpy call."""
        if self.evaluations >= self.max_evaluations or time.perf_counter() >= self.deadline:
            raise Stop
        x = np.asarray(solutions, dtype=float)
        # never past the budget
        rows = x[:self.max_evaluations - self.evaluations]
        values = self.function(rows)
        now = time.perf_counter()
        before = self.evaluations
        self.evaluations += len(rows)
        self.outside += int(np.sum(np.any((rows < self.lower) | (rows > self.upper), axis=1)))
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

    def __call__(self, x):
        """The objective function: the value of one solution."""
        return self.batch([x])[0]

    def next_generation(self, es):
        self.generations += 1


# -------------------------------------------------------------------------------------------------
# Fitness functions, identical to benchmarks/problems.py (minimized). They take the solutions as
# the rows of a numpy array, all of a population at once (fmin2's parallel_objective), and return
# their values, written with numpy like pycma's own test functions (cma.ff).
# -------------------------------------------------------------------------------------------------


@functools.cache
def shift(n, upper):
    """The optimum of rastrigin and ackley, away from the origin:
    s_i = 0.8 upper (2 ((37 i + 11) mod 101) / 101 - 1), computed in this order."""
    return np.array([0.8 * upper * (2 * ((37 * i + 11) % 101) / 101 - 1) for i in range(n)])


def rastrigin(x):
    """Shifted: 10 n + sum((x_i - s_i)^2 - 10 cos(2 pi (x_i - s_i)))."""
    n = x.shape[1]
    d = x - shift(n, 5.12)
    return 10 * n + np.sum(d * d - 10 * np.cos(2 * np.pi * d), axis=1)


def rosenbrock(x):
    return np.sum(100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2, axis=1)


def ackley(x):
    """Shifted: Ackley of x - s."""
    n = x.shape[1]
    d = x - shift(n, 32.768)
    return (-20 * np.exp(-0.2 * np.sqrt(np.sum(d * d, axis=1) / n))
            - np.exp(np.sum(np.cos(2 * np.pi * d), axis=1) / n) + 20 + math.e)


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
    - parallel_objective, fmin2's batch interface ("an objective function that accepts a list of
      numpy.ndarray as input and returns a list"): each population is evaluated in one numpy call.
      The search is the same: the same solutions, in the same order. The objective function
      still evaluates the final mean.
    lq-CMA-ES (method "lq") builds a linear or quadratic model of the function from the evaluated
    points and evaluates only part of each population ("to circumvent evaluations of the objective
    function", help(cma.fmin_lq_surr2)); its restarts double the population like IPOP. It has no
    parallel_objective: it evaluates one solution at a time.

    Rule 2.2: the restarts continue a run when CMA-ES stops by its criteria. If all 9 end before
    the budget, the function starts again with its first population size. Seeds: pycma samples
    from numpy's global random state (option randn, np.random.randn), which the adapter seeds with
    the run's seed, and with (seed + 1) * 1_000_000 + call for a later call. The seed option is
    np.nan, which pycma documents as "do nothing": pycma reads a seed of 0 as "seed from the
    clock". So pycma doesn't seed again, and its restarts continue the same random stream. x0 is
    drawn from it too."""
    call = 0
    while True:
        options = {
            "bounds": [lower, upper],
            "seed": np.nan,
            # no output, no log files (outcmaes/) in the working directory
            "verbose": -9,
            "verb_disp": 0,
            "verb_log": 0,
        }
        x0 = lambda: np.random.uniform(lower, upper, size)
        sigma0 = (upper - lower) / 4
        np.random.seed(seed if call == 0 else (seed + 1) * 1_000_000 + call)
        if method == "lq":
            cma.fmin_lq_surr2(budget, x0, sigma0, options, restarts=9, incpopsize=2,
                              callback=budget.next_generation)
        else:
            cma.fmin2(budget, x0, sigma0, options, restarts=9, incpopsize=2, bipop=method == "bipop",
                      parallel_objective=budget.batch, callback=budget.next_generation)
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
            print(json.dumps(float(function(np.array([json.loads(line)], dtype=float))[0])), flush=True)


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
            budget = Budget(function, lower, upper, max_evaluations, start, max_seconds)
            try:
                solve(size, lower, upper, seed, budget, method)
            except Stop:
                pass
            # the clock stops when the run ends (rule 4.1)
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
                "first_hit": budget.first_hit,
                "solution": budget.best_x.tolist(),
            }), flush=True)


if __name__ == "__main__":
    main()
