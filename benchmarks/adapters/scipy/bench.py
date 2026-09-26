"""Benchmark adapter for SciPy's optimizers (scipy.optimize).

Usage:
    python bench.py <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
    python bench.py values <problem> <size>

The first prints one JSON line per solver per seed, see ../../README.md for the fields. The second
reads one JSON solution per line and prints its value with this adapter's fitness functions.

SciPy's optimizers minimize one objective of real numbers: this adapter runs rastrigin, rosenbrock
and ackley, and prints nothing for the other problems. Which methods run and why, with the links to
SciPy's docs, is on the page docs/benchmarks/libraries/scipy.md:
- Rastrigin and Ackley (many local minima): differential_evolution, dual_annealing and direct, the
  bounded global optimizers of the tutorial's "Global optimization" section;
- Rosenbrock: minimize with L-BFGS-B (its default with bounds) and Nelder-Mead, the tutorial's
  "Local minimization" section, and differential_evolution, whose docstring example is rosen.

Every run ends only at the target, the budget or the time cap (rules 2.1 and 2.2 of
docs/benchmarks/rules.md): the Budget wrapper stops the solver there; each solver's limits that are
only budgets are lifted, and its convergence criteria end an attempt and start it again, see the
solver functions. Every solution evaluated lies in the bounds, by SciPy's own bound handling
(rule 2.4); the wrapper counts those outside.
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
from scipy.optimize import differential_evolution, direct, dual_annealing, minimize


class Stop(Exception):
    """Raised by Budget at the target, the budget or the time cap: it ends the run."""


class Budget:
    """Counts the evaluations itself (rule 3.2), every call of the fitness function on one solution,
    whoever calls it: the solver, its restarts, its local searches, its polish and its finite
    differences. Keeps the best solution and the first evaluation that reaches the target
    (first_hit), and raises Stop right after it, and before an evaluation past the budget or the
    time cap. Counts the evaluated solutions outside the bounds, as the solver gave them (rule
    2.4)."""

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
                self.first_hit = {"evaluations": self.evaluations,
                                  "time_s": round(time.perf_counter() - self.start, 6)}
                raise Stop
        return value

    def next_generation(self, *args, **kwargs):
        """The callback of differential_evolution and minimize, called after every generation or
        iteration."""
        self.generations += 1
        self.last = self.evaluations - self.generation_end
        self.generation_end = self.evaluations

    def last_generation(self, stepwise):
        """The evaluations since the start of the last generation (rule 2.3): of one cut short, or
        of the last one that ended. `stepwise`: a solver without the callback, whose generation is
        an evaluation."""
        if stepwise:
            return min(self.evaluations, 1)
        return self.evaluations - self.generation_end or self.last


# -------------------------------------------------------------------------------------------------
# Fitness functions, identical to benchmarks/problems.py (minimized). SciPy calls them with one
# solution, a numpy array, and they're written with numpy like SciPy's own scipy.optimize.rosen.
# -------------------------------------------------------------------------------------------------


@functools.cache
def shift(n, upper):
    """The optimum of rastrigin and ackley, away from the origin:
    s_i = 0.8 upper (2 ((37 i + 11) mod 101) / 101 - 1), computed in this order."""
    return np.array([0.8 * upper * (2 * ((37 * i + 11) % 101) / 101 - 1) for i in range(n)])


def rastrigin(x):
    """Shifted: 10 n + sum((x_i - s_i)^2 - 10 cos(2 pi (x_i - s_i)))."""
    d = x - shift(len(x), 5.12)
    return 10 * len(x) + np.sum(d * d - 10 * np.cos(2 * np.pi * d))


def rosenbrock(x):
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def ackley(x):
    """Shifted: Ackley of x - s."""
    n = len(x)
    d = x - shift(n, 32.768)
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
# Solvers. Each runs until Budget raises Stop at the target, the budget or the time cap.
# -------------------------------------------------------------------------------------------------


def solve_de(size, lower, upper, seed, budget):
    """differential_evolution with its defaults, as its docstring's examples call it (rosen, and
    ackley): best1bin, popsize 15 so 15 n individuals, mutation (0.5, 1) with dithering,
    recombination 0.7, Latin hypercube initialization, updating='immediate', and the final
    L-BFGS-B polish, whose finite-difference evaluations count. Bounds (rule 2.4): the population
    lives in the box (a mutant outside it is replaced by a random point), and the polish is bounded.

    Rule 2.2: maxiter (1000 generations by default), a limit that's only a budget, is lifted. Its
    convergence test, the default tol 0.01 (std of the population's values <= 0.01 * |mean|),
    ends an attempt: it polishes, and starts again (SciPy has no restart mechanism) with the seed
    restart_seed(seed, restart)."""
    bounds = [(lower, upper)] * size
    restart = 0
    while True:
        differential_evolution(budget, bounds, rng=restart_seed(seed, restart),
                               maxiter=budget.max_evaluations, callback=budget.next_generation)
        restart += 1


def restart_seed(seed, restart):
    """The seed of attempt `restart` of a run (rule 2.2): the run's seed first, then
    (seed + 1) * 1_000_000 + restart, so no two runs share a seed."""
    return seed if restart == 0 else (seed + 1) * 1_000_000 + restart


def solve_dual_annealing(size, lower, upper, seed, budget):
    """dual_annealing with its defaults (its docstring's example is Rastrigin in 10 dimensions):
    generalized simulated annealing with an L-BFGS-B local search, whose finite-difference
    evaluations count. Bounds (rule 2.4): a visit outside the box is wrapped back into it (modulo
    the range), and the local search is bounded.

    Rule 2.2: it has no convergence criterion; it restarts by itself (re-annealing from a new random
    point when the temperature falls to initial_temp * restart_temp_ratio). Its limits that are
    only budgets, maxiter global iterations and maxfun evaluations, are lifted: maxfun is twice the
    budget, so that the Budget wrapper, not dual_annealing's own count, ends the run."""
    dual_annealing(budget, [(lower, upper)] * size, rng=seed, maxiter=budget.max_evaluations,
                   maxfun=2 * budget.max_evaluations)
    # it can't end before max_evaluations global iterations, each at least one evaluation
    raise AssertionError("dual_annealing ended before the budget")


def solve_direct(size, lower, upper, seed, budget):
    """direct with locally_biased=False, which its docs recommend "for hard problems with many
    local minima". DIRECT is deterministic: every seed gives the same run. Bounds (rule 2.4): it
    samples the centres of hyperrectangles that divide the box.

    vol_tol = 0: its docs say vol_tol, the volume of the best hyperrectangle, "decreases
    exponentially with increasing dimensionality of the problem. Therefore vol_tol should be
    decreased to avoid premature termination of the algorithm for higher dimensions"; with the
    default 1e-16 it ended Rastrigin 10 after 1,965 evaluations, at 33.

    Rule 2.2: its limits that are only budgets are lifted: maxiter, and maxfun, set to twice the
    budget, because DIRECT ends before an iteration that could go past maxfun (with maxfun at the
    budget it ended Rastrigin 30 after 1,886,939 of 2,000,000 evaluations); the Budget wrapper
    ends it at the budget. Its convergence criterion, len_tol (default 1e-6), stays; DIRECT has no
    random start and no restart mechanism, so a restart would repeat the same run. In the separate
    tests it never converged before the target or the budget; if it did, the run would end there,
    with a message on stderr."""
    direct(budget, [(lower, upper)] * size, locally_biased=False, maxfun=2 * budget.max_evaluations,
           maxiter=budget.max_evaluations, vol_tol=0.0)
    print(f"direct converged (len_tol) after {budget.evaluations} evaluations", file=sys.stderr)


def solve_minimize(method):
    def solve(size, lower, upper, seed, budget):
        """minimize with bounds from a random point, with the method's default tolerances.
        L-BFGS-B is minimize's default when there are bounds; without a gradient it estimates one
        by finite differences, whose evaluations count. Nelder-Mead is the tutorial's first
        example, on the Rosenbrock function. Bounds (rule 2.4): both take them (bounds=): L-BFGS-B
        projects onto the box, and its finite differences step inside it; Nelder-Mead clips its
        points to the box.

        Rule 2.2: the limits that are only budgets (maxiter, maxfun / maxfev) are lifted. Its
        convergence tests (ftol, gtol; xatol, fatol) end an attempt, and it starts again from a new
        random point, drawn with restart_seed(seed, restart) (SciPy has no restart mechanism)."""
        limits = ({"maxiter": budget.max_evaluations, "maxfun": budget.max_evaluations}
                  if method == "L-BFGS-B" else
                  {"maxiter": budget.max_evaluations, "maxfev": budget.max_evaluations})
        restart = 0
        while True:
            x0 = np.random.default_rng(restart_seed(seed, restart)).uniform(lower, upper, size)
            minimize(budget, x0, method=method, bounds=[(lower, upper)] * size, options=limits,
                     callback=budget.next_generation)
            restart += 1
    return solve


# the solvers without a callback: a generation is an evaluation
STEPWISE = {"dual_annealing", "direct"}


def solvers(problem):
    """(solver name, function) of the problem type, see the page."""
    if problem == "rosenbrock":
        return [
            ("lbfgsb", solve_minimize("L-BFGS-B")),
            ("nelder_mead", solve_minimize("Nelder-Mead")),
            ("de", solve_de),
        ]
    return [
        ("de", solve_de),
        ("dual_annealing", solve_dual_annealing),
        ("direct", solve_direct),
    ]


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
        for solver, solve in solvers(problem):
            # the clock starts before the initial population (rule 4.1)
            start = time.perf_counter()
            budget = Budget(function, lower, upper, max_evaluations, start, max_seconds)
            try:
                solve(size, lower, upper, seed, budget)
            except Stop:
                pass
            # the clock stops when the run ends (rule 4.1)
            elapsed = time.perf_counter() - start
            print(json.dumps({
                "library": "scipy",
                "solver": solver,
                "problem": problem,
                "size": size,
                "mode": mode,
                "seed": seed,
                "time_s": round(elapsed, 6),
                # generations of differential_evolution, iterations of minimize; dual_annealing and
                # direct report their evaluations
                "generations": budget.generations or budget.evaluations,
                "evaluations": budget.evaluations,
                "last_generation": budget.last_generation(solver in STEPWISE),
                "outside": budget.outside,
                "best": budget.best,
                "target": TARGET,
                "success": bool(budget.best <= TARGET),
                "first_hit": budget.first_hit,
                "solution": budget.best_x.tolist(),
            }), flush=True)


if __name__ == "__main__":
    main()
