"""Benchmark adapter for pygmo (the Python bindings of pagmo): the matched suite.

Usage:
    python bench.py <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
    python bench.py values <problem> <size>     (one JSON solution per line on stdin)

Prints one JSON line per solver per seed, see ../../README.md for the fields. pygmo runs two
matched scenarios, and prints nothing for any other problem, size or mode:
- Rastrigin 30: DE/rand/1/bin, pygmo's de with variant 7 ("de"), with no target: every run uses
  the whole budget (or the time cap), and prints "target": null, "success": false and
  "first_hit": null;
- Rosenbrock 10: CMA-ES, pygmo's cmaes ("cma_es"), to the target 0.01.
pagmo's only GA, sga, can't run the matched OneMax (see the page). The settings, their sources and
the differences from the definitions are on the library's page, docs/benchmarks/libraries/pygmo.md.

pagmo's algorithms run in C++ and call the fitness of a Python user-defined problem (UDP),
single-threaded (no islands or archipelagos). cmaes accepts a batch fitness evaluator: it gets
pygmo's member_bfe, which evaluates a generation in one call of the UDP's batch_fitness, with
numpy; de takes none and calls its fitness, one decision vector at a time. The UDP counts every
decision vector evaluated (rule 3), the initial population included, keeps the best solution and
the first evaluation that reaches the target, and counts the evaluated solutions outside the bounds
(rule 2.4). How a run ends (rule 2): the counter raises Stop from inside the fitness at the first of
the target (if the scenario has one), the budget and the time limit, after the evaluation, or the
batch, that reaches the target (a batch that would go past the budget is evaluated only up to it). The algorithms' gen is a
budget only, set to cover the whole budget, and their ftol and xtol are 0, so their stopping tests
never fire: one call of evolve runs to the end. An error of pagmo ends the run, reported in
"ended_by" (rule 8.4).
"""

import os

# single-threaded, also for any BLAS behind numpy
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import functools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pygmo as pg  # noqa: E402


class _Trivial:
    def fitness(self, x):
        return [0.0]

    def batch_fitness(self, dvs):
        return np.zeros(len(dvs))

    def has_batch_fitness(self):
        return True

    def get_bounds(self):
        return [0.0], [1.0]


def start_tbb_with_one_thread():
    """One thread (rule 4.3): pagmo's problem.batch_fitness, which member_bfe calls, runs TBB
    worker threads besides the evaluation, and pygmo has no setting for their number. TBB sizes
    its thread pool from the CPUs the process may run on when it starts: this starts it with one
    CPU, with a batch evaluation of a trivial problem, then gives the process all its CPUs back.
    TBB keeps its single thread."""
    cpus = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {min(cpus)})
    pg.bfe(pg.member_bfe())(pg.problem(_Trivial()), np.zeros(2))
    os.sched_setaffinity(0, cpus)


start_tbb_with_one_thread()


class Stop(Exception):
    """Raised by the fitness to end a run."""


class Counter:
    """Counts fitness evaluations (rule 3), one per decision vector, and keeps the best (minimized)
    value, its solution and the first evaluation that reaches the target. pagmo deep-copies the
    problem, so the problems count here, in the module's current counter."""

    def __init__(self, max_evaluations, start, max_seconds, target=-math.inf):
        self.evaluations = 0
        self.max_evaluations = max_evaluations
        self.start = start
        self.deadline = start + max_seconds
        self.target = target
        self.best = math.inf
        self.best_x = None
        self.first_hit = None
        self.stopped = False
        # evaluated solutions outside the bounds (rule 2.4)
        self.outside = 0

    def left(self):
        """The evaluations left in the budget."""
        return self.max_evaluations - self.evaluations

    def count(self, x, values):
        """The evaluations of the rows of x with their values: ends the run (rule 2.1) by raising
        Stop."""
        now = time.perf_counter()
        before = self.evaluations
        self.evaluations += len(x)
        best = int(np.argmin(values))
        if values[best] < self.best:
            self.best = float(values[best])
            self.best_x = x[best].copy()
        if self.first_hit is None and self.best <= self.target:
            hit = int(np.flatnonzero(values <= self.target)[0])
            self.first_hit = {"evaluations": before + hit + 1, "time_s": round(now - self.start, 6)}
        if self.first_hit is not None or self.evaluations >= self.max_evaluations or now >= self.deadline:
            self.stopped = True
            raise Stop()


counter = Counter(0, 0.0, 0.0)


def last_generation(evaluations, per_generation):
    """The evaluations since the start of the last generation (rule 2.3). pagmo's algorithms run
    in C++ and don't call back between generations, but both evaluate an initial population of
    per_generation solutions (pygmo.population), then per_generation solutions a generation."""
    return (evaluations - 1) % per_generation + 1 if evaluations else 0


# -------------------------------------------------------------------------------------------------
# Fitness functions, identical to problems.py. Each takes the decision vectors as the rows of a
# numpy array (a whole generation with a batch fitness evaluator, one row otherwise) and returns
# their values. pagmo passes decision vectors as numpy arrays, and its tutorials use numpy
# arithmetic on them (tutorials/coding_udp_simple: "It is important to remember that x is a NumPy
# array, so that the NumPy array arithmetic applies in the body of fitness()")
# -------------------------------------------------------------------------------------------------


TARGET = 0.01


@functools.cache
def shift(n, upper):
    """The optimum of rastrigin, away from the origin:
    s_i = 0.8 upper (2 ((37 i + 11) mod 101) / 101 - 1), computed in this order."""
    return np.array([0.8 * upper * (2 * ((37 * i + 11) % 101) / 101 - 1) for i in range(n)])


def rastrigin(x):
    n = x.shape[1]
    y = x - shift(n, 5.12)
    return 10 * n + np.sum(y * y - 10 * np.cos(2 * np.pi * y), axis=1)


def rosenbrock(x):
    return np.sum(100 * (x[:, 1:] - x[:, :-1] * x[:, :-1]) ** 2 + (1 - x[:, :-1]) ** 2, axis=1)


# the real-valued problems: fitness function and bounds
REAL_PROBLEMS = {
    "rastrigin": (rastrigin, -5.12, 5.12),
    "rosenbrock": (rosenbrock, -5.0, 10.0),
}


class Problem:
    """A pagmo user-defined problem: a box-bounded fitness that counts its evaluations, with a batch
    fitness (tutorials/coding_udp_simple; pygmo.problem.batch_fitness: "the decision vectors ...
    are all concatenated in a single array")."""

    def __init__(self, function, lower, upper):
        self.function = function
        self.lower = lower
        self.upper = upper
        self.low, self.high = np.array(lower, dtype=float), np.array(upper, dtype=float)

    def evaluate(self, x):
        """The fitness vectors of the rows of x, counted."""
        # never past the budget: the counter stops the run there
        x = x[:counter.left()]
        # rule 2.4: x as pagmo proposed it, not clipped here
        counter.outside += int(np.sum(np.any((x < self.low) | (x > self.high), axis=1)))
        values = self.function(x)
        counter.count(x, values)
        return values[:, None]

    def fitness(self, x):
        return self.evaluate(np.asarray(x, dtype=float)[None, :])[0]

    def batch_fitness(self, dvs):
        return self.evaluate(np.asarray(dvs, dtype=float).reshape(-1, len(self.lower))).ravel()

    def has_batch_fitness(self):
        return True

    def get_bounds(self):
        return self.lower, self.upper


def budget_generations(evaluations_per_generation):
    """Generations enough to use up the whole budget: the counter ends the run first (rule 2.2:
    gen is a limit that's only a budget, lifted)."""
    return counter.max_evaluations // evaluations_per_generation + 2


# -------------------------------------------------------------------------------------------------
# The methods: (solver, fitness function, population size, the run of one seed)
# -------------------------------------------------------------------------------------------------


# DE/rand/1/bin on Rastrigin 30 (pagmo's de, src/algorithms/de.cpp): NP 100, F 0.5, CR 0.9
DE_POPULATION = 100


def run_de(problem, seed):
    # a population of 100 drawn uniformly in the bounds (pygmo.population), evolved by de with
    # variant 7, rand/1/bin ("7 - rand/1/bin" in de's docstring), F 0.5 and CR 0.9. ftol and xtol 0:
    # its stopping tests (dx < xtol, df < ftol) never fire. de draws r1, r2, r3 from the whole
    # population, x_i included (see the page), redraws a trial gene outside the bounds uniformly
    # in them, and replaces x_i when its trial is not worse (<=); the trials of a
    # generation are built from the previous generation's population. It takes no bfe
    population = pg.population(problem, DE_POPULATION, seed=seed)
    algorithm = pg.algorithm(pg.de(gen=budget_generations(DE_POPULATION), F=0.5, CR=0.9, variant=7,
                                   ftol=0.0, xtol=0.0, seed=seed))
    algorithm.evolve(population)


# CMA-ES on Rosenbrock 10 (pagmo's cmaes, src/algorithms/cmaes.cpp): lambda 10
CMAES_POPULATION = 10


def run_cma_es(problem, seed):
    # lambda = the population's size, 10 (4 + floor(3 ln 10)); mu = lambda / 2 = 5, log weights;
    # cc, cs, c1, cmu at -1, their default: computed from Hansen's formulas. sigma0 is relative to
    # the bounds' width (cmaes starts from C = diag(width^2)): 0.3 gives a step of 4.5. The mean
    # starts at the best of the population, 10 points drawn uniformly in the bounds. ftol and xtol
    # 0: its stopping tests never fire. force_bounds=True, pagmo's only bound handling for cmaes,
    # clips each sample to the bounds before it's evaluated ("The fitness will never be called
    # outside the bounds"). It accepts a bfe: a generation in one batch (rule 3.4)
    population = pg.population(problem, CMAES_POPULATION, seed=seed)
    uda = pg.cmaes(gen=budget_generations(CMAES_POPULATION), sigma0=0.3, ftol=0.0, xtol=0.0,
                   force_bounds=True, seed=seed)
    uda.set_bfe(pg.bfe(pg.member_bfe()))
    pg.algorithm(uda).evolve(population)


def solvers_of(problem, size, mode):
    """[(solver, fitness function, population size, run, target)] of a scenario: none outside the
    matched suite. Rastrigin 30 has no target: its runs use the whole budget, measured by the time
    they take and the error at the end."""
    if mode != "matched":
        return []
    if problem == "rastrigin" and size == 30:
        return [("de", rastrigin, DE_POPULATION, run_de, None)]
    if problem == "rosenbrock" and size == 10:
        return [("cma_es", rosenbrock, CMAES_POPULATION, run_cma_es, TARGET)]
    return []


# -------------------------------------------------------------------------------------------------


def values(problem, size):
    """Evaluates the solutions on stdin with the adapter's fitness functions (rule 1.2)."""
    if problem not in REAL_PROBLEMS:
        print(f"pygmo doesn't run {problem}", file=sys.stderr)
        sys.exit(2)
    for line in sys.stdin:
        if not line.strip():
            continue
        x = np.array([json.loads(line)], dtype=float)
        print(json.dumps(float(REAL_PROBLEMS[problem][0](x)[0])), flush=True)


def main():
    global counter
    if len(sys.argv) == 4 and sys.argv[1] == "values":
        values(sys.argv[2], int(sys.argv[3]))
        return
    if len(sys.argv) != 8:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    problem, size, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    seed_from, seed_to = int(sys.argv[4]), int(sys.argv[5])
    max_evaluations, max_seconds = int(sys.argv[6]), float(sys.argv[7])

    for seed in range(seed_from, seed_to + 1):
        for solver, function, population_size, run, target in solvers_of(problem, size, mode):
            _, low, high = REAL_PROBLEMS[problem]
            problem_ = pg.problem(Problem(function, [low] * size, [high] * size))
            start = time.perf_counter()
            # without a target, the counter never stops at a value: only the budget and the time
            counter = Counter(max_evaluations, start, max_seconds, -math.inf if target is None else target)
            error = None
            try:
                run(problem_, seed)
            except Exception as e:
                # Stop, raised by the fitness, reaches Python through pagmo's C++; any other error
                # is the library's, and ends the run there (rule 8.4)
                if not counter.stopped:
                    error = f"{type(e).__name__}: {e}"[:300]
            else:
                raise RuntimeError(f"{solver} ended before the budget")
            # the clock stops when the run ends (rule 4.1)
            elapsed = time.perf_counter() - start
            result = {
                "library": "pygmo",
                "solver": solver,
                "problem": problem,
                "size": size,
                "mode": mode,
                "seed": seed,
                "time_s": round(elapsed, 6),
                # generations of population_size evaluations, the initial population included
                "generations": counter.evaluations // population_size,
                "evaluations": counter.evaluations,
                "last_generation": last_generation(counter.evaluations, population_size),
                "best": counter.best,
                "target": target,
                "success": target is not None and counter.best <= target,
                "first_hit": counter.first_hit,
                "solution": [float(v) for v in counter.best_x],
                "outside": counter.outside,
            }
            if error is not None:
                result["ended_by"] = error
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
