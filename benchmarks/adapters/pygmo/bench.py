"""Benchmark adapter for pygmo (the Python bindings of pagmo).

Usage:
    python bench.py <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
    python bench.py values <problem> <size>     (one JSON solution per line on stdin)

Prints one JSON line per solver per seed, see ../../README.md for the fields. Every method, setting
and where pygmo recommends it is on the library's page, docs/benchmarks/libraries/pygmo.md, and
next to the code below.

pagmo's algorithms run in C++ and call the fitness of a Python user-defined problem (UDP),
single-threaded (no islands or archipelagos). The algorithms that accept a batch fitness evaluator
(cmaes, gaco) get pygmo's member_bfe, which evaluates a whole generation in one call of the
UDP's batch_fitness, with numpy; the others call its fitness, one decision vector at a time. The
UDP counts every decision vector evaluated (rule 3), the initial populations and every restart
included, and keeps the best solution and the first evaluation that reaches the target. It also
counts the evaluated solutions outside the bounds, as pagmo proposed them (rule 2.4: pagmo's own
bound handling keeps them inside, see the page). How a run ends (rule 2):
- the counter raises Stop from inside the fitness at the first of the target, the budget and the
  time limit: after the evaluation, or the batch, that reaches the target; a batch that would go
  past the budget is evaluated only up to it;
- a limit that's only a budget (every algorithm's gen) is lifted: gen covers the whole budget;
- an algorithm that ends on convergence (sade's, CMA-ES' and xNES' ftol and xtol, GACO's impstop and
  evalstop) starts again from a new random population, the procedure of pygmo's cmaes_vs_xnes
  tutorial for algorithms "with well defined exit conditions", with the run's seed first and
  (seed + 1) * 1_000_000 + restart for restart 1 on;
- simulated annealing's cooling schedule has a fixed length; it's annealed again from its best
  point, as in the solving_schwefel_20 tutorial.
"""

import os

# single-threaded, also for any BLAS behind numpy
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import functools  # noqa: E402
import itertools  # noqa: E402
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
    """Raised by the fitness to end a single-objective run."""


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
        """The evaluations of a single-objective run, the rows of x with their values: ends the run
        (rule 2.1) by raising Stop."""
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


def last_generation(evaluations, initial, per_generation):
    """The evaluations since the start of the last generation (rule 2.3). pagmo's algorithms run
    in C++ and don't call back between generations, but each of these evaluates an initial
    population of `initial` solutions, then per_generation solutions a generation; a restart
    starts after a whole generation, with a population of per_generation."""
    if evaluations <= initial:
        return evaluations
    return (evaluations - initial - 1) % per_generation + 1


def restart_seed(seed, restart):
    """The seed of attempt `restart` of a run (rule 2.2): the run's seed first, then
    (seed + 1) * 1_000_000 + restart, so no two runs share a seed."""
    return seed if restart == 0 else (seed + 1) * 1_000_000 + restart


# -------------------------------------------------------------------------------------------------
# Fitness functions, identical to problems.py. Each takes the decision vectors as the rows of a
# numpy array (a whole generation with a batch fitness evaluator, one row otherwise) and returns
# their values. pagmo passes decision vectors as numpy arrays, and its tutorials use numpy
# arithmetic on them (tutorials/coding_udp_simple: "It is important to remember that x is a NumPy
# array, so that the NumPy array arithmetic applies in the body of fitness()")
# -------------------------------------------------------------------------------------------------


def onemax(x):
    return np.sum(x, axis=1)  # the number of ones, maximized: the UDP minimizes its negative


TARGET = 0.01


@functools.cache
def shift(n, upper):
    """The optimum of rastrigin and ackley, away from the origin:
    s_i = 0.8 upper (2 ((37 i + 11) mod 101) / 101 - 1), computed in this order."""
    return np.array([0.8 * upper * (2 * ((37 * i + 11) % 101) / 101 - 1) for i in range(n)])


def rastrigin(x):
    n = x.shape[1]
    y = x - shift(n, 5.12)
    return 10 * n + np.sum(y * y - 10 * np.cos(2 * np.pi * y), axis=1)


def rosenbrock(x):
    return np.sum(100 * (x[:, 1:] - x[:, :-1] * x[:, :-1]) ** 2 + (1 - x[:, :-1]) ** 2, axis=1)


def ackley(x):
    n = x.shape[1]
    y = x - shift(n, 32.768)
    return (-20 * np.exp(-0.2 * np.sqrt(np.sum(y * y, axis=1) / n))
            - np.exp(np.sum(np.cos(2 * np.pi * y), axis=1) / n) + 20 + np.e)


# the real-valued problems: fitness function and bounds
REAL_PROBLEMS = {
    "rastrigin": (rastrigin, -5.12, 5.12),
    "rosenbrock": (rosenbrock, -5.0, 10.0),
    "ackley": (ackley, -32.768, 32.768),
}


class Problem:
    """A pagmo user-defined problem: a box-bounded fitness that counts its evaluations, with a batch
    fitness (tutorials/coding_udp_simple; pygmo.problem.batch_fitness: "the decision vectors ...
    are all concatenated in a single array"). `sign` is -1 for OneMax, which is maximized (pagmo
    minimizes)."""

    def __init__(self, function, lower, upper, integers=0, sign=1.0):
        self.function = function
        self.lower = lower
        self.upper = upper
        self.integers = integers
        self.sign = sign
        self.low, self.high = np.array(lower, dtype=float), np.array(upper, dtype=float)

    def evaluate(self, x):
        """The fitness vectors of the rows of x, counted."""
        # never past the budget: the counter stops the run there
        x = x[:counter.left()]
        # rule 2.4: x as pagmo proposed it, not clipped here
        counter.outside += int(np.sum(np.any((x < self.low) | (x > self.high), axis=1)))
        values = self.sign * self.function(x)
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

    def get_nix(self):
        return self.integers


def with_bfe(uda):
    """The algorithm with pygmo's member_bfe, which evaluates a generation in one call of the UDP's
    batch_fitness, in this thread (rule 3.4). The search is the same as with fitness."""
    uda.set_bfe(pg.bfe(pg.member_bfe()))
    return uda


# -------------------------------------------------------------------------------------------------
# How the single-objective methods run
# -------------------------------------------------------------------------------------------------


def budget_generations(evaluations_per_generation):
    """Generations enough to use up the whole budget: the counter ends the run first."""
    return counter.max_evaluations // evaluations_per_generation + 2


class ToBudget:
    """One call of evolve with the generations of the whole budget, for a method with no stop
    criterion of its own besides its generations (sga, ihs)."""

    def __init__(self, population_size, make_algorithm):
        self.population_size = population_size
        self.make_algorithm = make_algorithm  # seed -> UDA

    def run(self, problem, seed):
        population = pg.population(problem, self.population_size, seed=seed)
        pg.algorithm(self.make_algorithm(seed)).evolve(population)
        raise RuntimeError("the algorithm ended before its budget")


class Restarts:
    """Evolves a random population until the algorithm stops on convergence (its gen covers the
    whole budget), then starts again from a new random population; the counter keeps the best
    (rule 2.2). pygmo has no restart mechanism of its own; this is the procedure of
    tutorials/cmaes_vs_xnes, "the best practice ... when algorithms have well defined exit
    conditions", which assembles "the results in single runs containing multiple restarts".
    Restart r uses the seed restart_seed(seed, r) for its population and its algorithm."""

    def __init__(self, population_size, make_algorithm):
        self.population_size = population_size
        self.make_algorithm = make_algorithm  # seed -> UDA
        self.restarts = 0

    def run(self, problem, seed):
        for restart in itertools.count():
            self.restarts = restart
            attempt_seed = restart_seed(seed, restart)
            population = pg.population(problem, self.population_size, seed=attempt_seed)
            pg.algorithm(self.make_algorithm(attempt_seed)).evolve(population)


class Reanneal:
    """Simulated annealing, annealed again from its best point after each annealing schedule, as in
    tutorials/solving_schwefel_20 ("since we will be using some reannealing": 5 calls of evolve on
    the same population). Each call starts from the population's best and puts its best back. The
    schedule's length is part of its cooling rate ((Tf / Ts)^(1 / n_T_adj) per adjustment), so it
    can't be lifted without changing the method."""

    def __init__(self, population_size, make_algorithm):
        self.population_size = population_size
        self.make_algorithm = make_algorithm  # seed -> UDA
        self.restarts = 0

    def run(self, problem, seed):
        population = pg.population(problem, self.population_size, seed=seed)
        algorithm = pg.algorithm(self.make_algorithm(seed))
        for restart in itertools.count():
            self.restarts = restart
            population = algorithm.evolve(population)


# the population of 20 of pygmo's tutorials: tutorials/evolving_a_population (sade on Rosenbrock
# 10), tutorials/solving_schwefel_20 (sade, de, de1220, pso, simulated annealing on Schwefel 20)
# and pagmo's quick start (tutorials/getting_started.cpp: sade on Schwefel 30, islands of 20)
TUTORIAL_POPULATION = 20


def cmaes_population(problem, size):
    """CMA-ES and xNES: the population sizes of tutorials/cmaes_vs_xnes, whose figures run each
    function with three sizes and state no preference, so the first listed is taken (rule 6.2):
    Rosenbrock 10 [10, 20, 30] (the tutorial's code), Rastrigin 10 [40, 60, 100], Ackley 10 [10,
    20, 30], and above 10 variables the figures' largest dimension, 20: Rastrigin 20 [100, 150,
    200], Ackley 20 [20, 30, 40]."""
    if problem == "rosenbrock":
        return 10
    if problem == "rastrigin":
        return 40 if size <= 10 else 100
    return 10 if size <= 10 else 20


def single_solvers(problem, size, mode):
    """[(solver, problem, method, evaluations per generation)]"""
    if problem == "onemax":
        if mode == "matched":
            # pagmo's only GA, sga, can't run the matched OneMax: its reinsertion is elitist (the
            # best of parents and children), which can't be turned off ("the only reinsertion
            # strategy provided is what we call pure elitism"), and it has no two-point crossover
            return []
        # binary genes as integers in [0, 1] (the integer dimension of tutorials/coding_udp_minlp)
        udp = Problem(onemax, [0] * size, [1] * size, integers=size, sign=-1.0)

        # the three single-objective algorithms that the algorithm list of pygmo's docs
        # (overview.rst, "Heuristic Global Optimization") flags for integer programming (I), with
        # their defaults
        return [
            # sga's defaults: exponential crossover 0.9, mutation 0.02 per gene (its polynomial
            # mutation draws an integer gene again from its bounds), tournament of 2
            ("ga", udp, ToBudget(TUTORIAL_POPULATION, lambda seed: pg.sga(
                gen=budget_generations(TUTORIAL_POPULATION), seed=seed)), TUTORIAL_POPULATION),
            # ihs' defaults; one evaluation per generation
            ("ihs", udp, ToBudget(TUTORIAL_POPULATION, lambda seed: pg.ihs(
                gen=budget_generations(1), seed=seed)), 1),
            # gaco's defaults, whose kernel of 63 solutions needs a population of at least 63; its
            # impstop and evalstop (100,000 generations or evaluations without improvement) end
            # it, and it restarts. It accepts a bfe: a generation in one batch
            ("gaco", udp, Restarts(63, lambda seed: with_bfe(pg.gaco(gen=budget_generations(63), seed=seed))),
             63),
        ]

    if problem in REAL_PROBLEMS:
        function, low, high = REAL_PROBLEMS[problem]
        udp = Problem(function, [low] * size, [high] * size)
        solvers = [
            # sade (jDE) with its defaults (variant rand/1/exp, jDE adaptation, ftol and xtol 1e-6)
            # and the tutorials' population of 20: pagmo's quick start, tutorials/evolving_a_population
            # (Rosenbrock 10), tutorials/solving_schwefel_20, and tutorials/cec2013_comp, where
            # "cmaes and sade (jDE) are performing particularly well". It restarts when it converges
            ("sade", udp, Restarts(TUTORIAL_POPULATION, lambda seed: pg.sade(
                gen=budget_generations(TUTORIAL_POPULATION), seed=seed)), TUTORIAL_POPULATION),
        ]
        # CMA-ES as in tutorials/cmaes_vs_xnes, the example in code on Rosenbrock 10 and, in its
        # figures, on Rastrigin and Ackley: cmaes(gen=4000, ftol=1e-8, xtol=1e-10), with the
        # population of cmaes_population, restarted as the tutorial does. With force_bounds,
        # pagmo's only bound handling for it, which clips each sample to the bounds ("The fitness
        # will never be called outside the bounds"). Its gen is only a budget, lifted (rule 2.2);
        # ftol and xtol end an attempt. It accepts a bfe: a generation in one batch
        population = cmaes_population(problem, size)
        solvers.append(("cma_es", udp, Restarts(population, lambda seed: with_bfe(pg.cmaes(
            gen=budget_generations(population), ftol=1e-8, xtol=1e-10, force_bounds=True, seed=seed))),
            population))
        if problem == "rosenbrock":
            # xNES, the other method of the same example, with the same settings
            solvers.append(("xnes", udp, Restarts(population, lambda seed: pg.xnes(
                gen=budget_generations(population), ftol=1e-8, xtol=1e-10, force_bounds=True,
                seed=seed)), population))
        else:
            # simulated annealing, one of "the two most successful algorithms" of
            # tutorials/solving_schwefel_20, with its settings (Ts=10, Tf=0.01, n_T_adj=5, the
            # other parameters default), its population of 20 and its reannealing; one evaluation
            # per step
            solvers.append(("simulated_annealing", udp, Reanneal(TUTORIAL_POPULATION, lambda seed: (
                pg.simulated_annealing(Ts=10.0, Tf=0.01, n_T_adj=5, seed=seed))), 1))
        return solvers

    if problem == "nqueens":
        # pagmo has no permutation representation
        return []

    print(f"unknown problem {problem}", file=sys.stderr)
    sys.exit(2)


# -------------------------------------------------------------------------------------------------


def values(problem, size):
    """Evaluates the solutions on stdin with the adapter's fitness functions (rule 1.2)."""
    for line in sys.stdin:
        if not line.strip():
            continue
        x = np.array([json.loads(line)], dtype=float)
        if problem == "onemax":
            value = int(onemax(x)[0])
        elif problem in REAL_PROBLEMS:
            value = float(REAL_PROBLEMS[problem][0](x)[0])
        else:
            print(f"pygmo can't evaluate {problem}", file=sys.stderr)
            sys.exit(2)
        print(json.dumps(value), flush=True)


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

    target = -size if problem == "onemax" else TARGET
    for seed in range(seed_from, seed_to + 1):
        for solver, udp, method, per_generation in single_solvers(problem, size, mode):
            problem_ = pg.problem(udp)
            start = time.perf_counter()
            counter = Counter(max_evaluations, start, max_seconds, target)
            try:
                method.run(problem_, seed)
            except Exception:
                # Stop, raised by the fitness, reaches Python through pagmo's C++
                if not counter.stopped:
                    raise
            # the clock stops when the run ends (rule 4.1)
            elapsed = time.perf_counter() - start
            if problem == "onemax":
                best, solution = -int(counter.best), [int(v) for v in counter.best_x]
            else:
                best, solution = counter.best, [float(v) for v in counter.best_x]
            run = {
                "library": "pygmo",
                "solver": solver,
                "problem": problem,
                "size": size,
                "mode": mode,
                "seed": seed,
                "time_s": round(elapsed, 6),
                # generations of per_generation evaluations (1 for ihs and simulated annealing)
                "generations": counter.evaluations // per_generation,
                "evaluations": counter.evaluations,
                "last_generation": last_generation(counter.evaluations, method.population_size, per_generation),
                "best": best,
                "target": size if problem == "onemax" else TARGET,
                "success": counter.best <= target,
                "first_hit": counter.first_hit,
                "solution": solution,
            }
            if problem in REAL_PROBLEMS:
                run["outside"] = counter.outside
            if hasattr(method, "restarts"):
                # restarts, or reannealings of simulated annealing
                run["restarts"] = method.restarts
            print(json.dumps(run), flush=True)


if __name__ == "__main__":
    main()
