"""Benchmark adapter for pygmo (the Python bindings of pagmo).

Usage:
    python bench.py <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
    python bench.py values <problem> <size>     (one JSON solution per line on stdin)

Prints one JSON line per solver per seed, see ../../README.md for the fields. Every method, setting
and where pygmo recommends it is on the library's page, docs/benchmarks/libraries/pygmo.md, and
next to the code below.

pagmo's algorithms run in C++ and call the fitness of a Python user-defined problem (UDP), one
decision vector at a time, single-threaded (no islands, archipelagos or batch evaluators). The UDP
counts every call (rule 3), the initial populations and every restart included, and keeps the best
solution. It also counts the evaluated solutions outside the bounds, as pagmo proposed them (rule
2.4: pagmo's own bound handling keeps them inside, see the page). How a run ends (rule 2):
- single-objective runs: the counter raises Stop from inside the fitness at the first of the target,
  the budget and the time limit, so a run ends at that evaluation, never past it;
- a limit that's only a budget (every algorithm's gen) is lifted: gen covers the whole budget;
- an algorithm that ends on convergence (sade's, CMA-ES' and xNES' ftol and xtol, GACO's impstop and
  evalstop) starts again from a new random population with the seed seed * 1000 + restart, the
  procedure of pygmo's cmaes_vs_xnes tutorial for algorithms "with well defined exit conditions";
- simulated annealing's cooling schedule has a fixed length; it's annealed again from its best
  point, as in the solving_schwefel_20 tutorial;
- multi-objective runs have no target and run the generations of the budget, see evolve_front.
"""

import os

# single-threaded, also for any BLAS behind numpy
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import functools  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pygmo as pg  # noqa: E402


class Stop(Exception):
    """Raised by the fitness to end a single-objective run."""


class Counter:
    """Counts fitness evaluations (rule 3) and keeps the best (minimized) value and its solution.
    pagmo deep-copies the problem, so the problems count here, in the module's current counter."""

    def __init__(self, max_evaluations, max_seconds, target=-math.inf):
        self.evaluations = 0
        self.max_evaluations = max_evaluations
        self.max_seconds = max_seconds
        self.deadline = math.inf
        self.target = target
        self.best = math.inf
        self.best_x = None
        self.stopped = False
        # evaluated solutions outside the bounds (rule 2.4)
        self.outside = 0

    def start(self):
        self.deadline = time.perf_counter() + self.max_seconds

    def count(self, x, value):
        """One evaluation of a single-objective run: ends the run (rule 2.1) by raising Stop."""
        self.evaluations += 1
        if value < self.best:
            self.best = value
            self.best_x = x.copy()
        if (self.best <= self.target or self.evaluations >= self.max_evaluations
                or time.perf_counter() >= self.deadline):
            self.stopped = True
            raise Stop()

    def count_front(self):
        """One evaluation of a multi-objective run, which evolve_front ends."""
        self.evaluations += 1

    def out_of_time(self):
        return time.perf_counter() >= self.deadline


counter = Counter(0, 0.0)


# -------------------------------------------------------------------------------------------------
# Fitness functions, identical to problems.py. pagmo passes x as a numpy array, and its tutorials
# use numpy arithmetic on it (tutorials/coding_udp_simple: "It is important to remember that x is a
# NumPy array, so that the NumPy array arithmetic applies in the body of fitness()")
# -------------------------------------------------------------------------------------------------


def onemax(x):
    return float(np.sum(x))  # the number of ones, maximized: the UDP minimizes its negative


TARGET = 0.01

# Rastrigin and Ackley are shifted, so an optimum at the origin can't favour operators that drift
# towards 0: gene i is measured from s_i = 2 ((37 i + 11) mod 101) / 101 - 1, in [-1, 1]
SHIFT = np.array([2 * ((37 * i + 11) % 101) / 101 - 1 for i in range(1000)])


def rastrigin(x):
    y = x - SHIFT[:len(x)]
    return float(10 * len(x) + np.sum(y * y - 10 * np.cos(2 * np.pi * y)))


def rosenbrock(x):
    return float(np.sum(100 * (x[1:] - x[:-1] * x[:-1]) ** 2 + (1 - x[:-1]) ** 2))


def ackley(x):
    n = len(x)
    y = x - SHIFT[:n]
    return float(-20 * np.exp(-0.2 * np.sqrt(np.sum(y * y) / n)) - np.exp(np.sum(np.cos(2 * np.pi * y)) / n)
                 + 20 + np.e)


# the real-valued problems: fitness function and bounds
REAL_PROBLEMS = {
    "rastrigin": (rastrigin, -5.12, 5.12),
    "rosenbrock": (rosenbrock, -5.0, 10.0),
    "ackley": (ackley, -32.768, 32.768),
}


def zdt_g(x):
    return 1 + 9 * np.sum(x[1:]) / (len(x) - 1)


def zdt1(x):
    g = zdt_g(x)
    return [float(x[0]), float(g * (1 - np.sqrt(x[0] / g)))]


def zdt2(x):
    g = zdt_g(x)
    return [float(x[0]), float(g * (1 - (x[0] / g) ** 2))]


def zdt3(x):
    g = zdt_g(x)
    return [float(x[0]), float(g * (1 - np.sqrt(x[0] / g) - x[0] / g * np.sin(10 * np.pi * x[0])))]


def dtlz2(x, objectives):
    k = objectives - 1
    g = np.sum((x[k:] - 0.5) ** 2)
    angles = x[:k] * np.pi / 2
    # f_m = (1 + g) cos(x_0) ... cos(x_(k-m-1)) sin(x_(k-m)), without the sine for m = 0
    f = (1 + g) * np.concatenate(([1.0], np.cumprod(np.cos(angles))))[::-1]
    f[1:] *= np.sin(angles)[::-1]
    return f.tolist()


def dtlz1(x, objectives):
    k = objectives - 1
    tail = x[k:] - 0.5
    g = 100 * (len(tail) + np.sum(tail * tail - np.cos(20 * np.pi * tail)))
    # f_m = (1 + g) / 2 x_0 ... x_(k-m-1) (1 - x_(k-m)), without the last factor for m = 0
    f = 0.5 * (1 + g) * np.concatenate(([1.0], np.cumprod(x[:k])))[::-1]
    f[1:] *= 1 - x[:k][::-1]
    return f.tolist()


# (fitness function of x and the size, variables, objectives, population size of NSGA-II, a
# multiple of 4)
FRONT_PROBLEMS = {
    "zdt1": (lambda x, size: zdt1(x), lambda size: size, 2, 100),
    "zdt2": (lambda x, size: zdt2(x), lambda size: size, 2, 100),
    "zdt3": (lambda x, size: zdt3(x), lambda size: size, 2, 100),
    # size: the number of objectives, with k = 10 (DTLZ2) and 5 (DTLZ1)
    "dtlz2": (lambda x, size: dtlz2(x, size), lambda size: size + 9, 3, 92),
    "dtlz1": (lambda x, size: dtlz1(x, size), lambda size: size + 4, 3, 92),
}


class Problem:
    """A pagmo user-defined problem: a box-bounded fitness that counts its evaluations. `sign` is -1
    for OneMax, which is maximized (pagmo minimizes)."""

    def __init__(self, function, lower, upper, objectives=1, integers=0, sign=1.0):
        self.function = function
        self.lower = lower
        self.upper = upper
        self.objectives = objectives
        self.integers = integers
        self.sign = sign
        self.low, self.high = np.array(lower, dtype=float), np.array(upper, dtype=float)

    def fitness(self, x):
        # rule 2.4: x as pagmo proposed it, not clipped here
        if np.any(x < self.low) or np.any(x > self.high):
            counter.outside += 1
        if self.objectives == 1:
            value = self.sign * self.function(x)
            counter.count(x, value)
            return [value]
        counter.count_front()
        return self.function(x)

    def get_bounds(self):
        return self.lower, self.upper

    def get_nobj(self):
        return self.objectives

    def get_nix(self):
        return self.integers


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
    Restart r uses the seed seed * 1000 + r for its population and its algorithm."""

    def __init__(self, population_size, make_algorithm):
        self.population_size = population_size
        self.make_algorithm = make_algorithm  # seed -> UDA
        self.restarts = 0

    def run(self, problem, seed):
        for restart in itertools.count():
            self.restarts = restart
            restart_seed = seed * 1000 + restart
            population = pg.population(problem, self.population_size, seed=restart_seed)
            pg.algorithm(self.make_algorithm(restart_seed)).evolve(population)


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
# CMA-ES and xNES: the first of the population sizes of tutorials/cmaes_vs_xnes (Rosenbrock 10), and
# the population of tutorials/cec2013_comp
ROSENBROCK_POPULATION = 10
CEC2013_POPULATION = 50


def single_solvers(problem, size, mode):
    """[(solver, problem, method, evaluations per generation)]"""
    if problem == "onemax":
        # binary genes as integers in [0, 1] (the integer dimension of tutorials/coding_udp_minlp)
        udp = Problem(onemax, [0] * size, [1] * size, integers=size, sign=-1.0)
        if mode == "matched":
            # population 300, tournament 3, crossover 0.5, bit-flip ~0.2 bits per child.
            # Differences: pagmo's sga has no two-point crossover (single point here, one child per
            # selected parent with a random partner), mutates every child with 0.2 / size per bit
            # instead of 20% of the children with 1 / size, and its reinsertion is elitist (the
            # best 300 of parents and children), which can't be turned off. Its uniform mutation
            # draws an integer gene again from its bounds, which flips a bit half the time, hence
            # 0.4 / size
            return [("ga", udp, ToBudget(300, lambda seed: pg.sga(
                gen=budget_generations(300), cr=0.5, m=0.4 / size, param_s=3, crossover="single",
                mutation="uniform", selection="tournament", seed=seed)), 300)]

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
            # it, and it restarts
            ("gaco", udp, Restarts(63, lambda seed: pg.gaco(gen=budget_generations(63), seed=seed)), 63),
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
        # CMA-ES and xNES with force_bounds, pagmo's only bound handling for them, which clips each
        # sample to the bounds ("The fitness will never be called outside the bounds"). Their gen
        # (4000 or 1000 in the tutorials) is only a budget, lifted (rule 2.2); ftol and xtol end an
        # attempt, and they restart, as both tutorials say to
        if problem == "rosenbrock":
            # the example of tutorials/cmaes_vs_xnes, which runs exactly Rosenbrock 10: ftol=1e-8,
            # xtol=1e-10, and popsizes = [10, 20, 30]. It states no preference between the three
            # sizes, so the adapter takes the first, 10
            solvers.append(("cma_es", udp, Restarts(ROSENBROCK_POPULATION, lambda seed: pg.cmaes(
                gen=budget_generations(ROSENBROCK_POPULATION), ftol=1e-8, xtol=1e-10, force_bounds=True,
                seed=seed)), ROSENBROCK_POPULATION))
            # xNES, the other method of the same example, with the same settings
            solvers.append(("xnes", udp, Restarts(ROSENBROCK_POPULATION, lambda seed: pg.xnes(
                gen=budget_generations(ROSENBROCK_POPULATION), ftol=1e-8, xtol=1e-10, force_bounds=True,
                seed=seed)), ROSENBROCK_POPULATION))
        else:
            # the example of tutorials/cec2013_comp, on the CEC 2013 suite of multimodal functions:
            # cmaes(gen=1000, ftol=1e-9, xtol=1e-9), "we choose a population of 50", and "a proper
            # comparison ... should allow for restarts"
            solvers.append(("cma_es", udp, Restarts(CEC2013_POPULATION, lambda seed: pg.cmaes(
                gen=budget_generations(CEC2013_POPULATION), ftol=1e-9, xtol=1e-9, force_bounds=True,
                seed=seed)), CEC2013_POPULATION))
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
# Multi-objective
# -------------------------------------------------------------------------------------------------


def front_solvers(problem, size):
    """[(solver, problem, population size, algorithm factory (gen, seed) -> UDA)]"""
    function, variables, objectives, population = FRONT_PROBLEMS[problem]
    n = variables(size)
    udp = Problem(functools.partial(function, size=size), [0.0] * n, [1.0] * n, objectives=objectives)
    # The matched scenarios run only NSGA-II, NSGA-III, SPEA2, MOEA/D and SMS-EMOA (rule 6.1).
    # pagmo 2.19.1 has no NSGA-III, SPEA2 or SMS-EMOA, and its MOEA/D (moead, moead_gen) is the DE
    # variant, which can't use SBX; NSPSO and MACO aren't matched algorithms
    return [
        # the matched settings: SBX with eta 15 at 0.9, polynomial mutation with eta 20 at 1 / n.
        # Both of pagmo's operators keep the genes within the bounds: its SBX clips the children,
        # and its polynomial mutation is Deb's bounded one
        ("nsga2", udp, population,
         lambda gen, seed: pg.nsga2(gen=gen, cr=0.9, eta_c=15, m=1.0 / n, eta_m=20, seed=seed)),
    ]


def evolve_front(udp, population_size, make_algorithm, seed):
    """Multi-objective runs have no target: one call of evolve with the generations of the budget,
    as pagmo is used, when they fit in the time limit with room to spare (3 times the time of the
    initial population's evaluations per generation). Otherwise one generation per call, with the
    time checked after each. Returns (population, generations)."""
    initial = time.perf_counter()
    population = pg.population(udp, population_size, seed=seed)
    seconds_per_generation = time.perf_counter() - initial
    generations = max(1, math.ceil((counter.max_evaluations - counter.evaluations) / population_size))
    if generations * seconds_per_generation * 3 < counter.deadline - time.perf_counter():
        algorithm = pg.algorithm(make_algorithm(generations, seed))
        return algorithm.evolve(population), generations
    algorithm = pg.algorithm(make_algorithm(1, seed))
    done = 0
    while done < generations and not counter.out_of_time():
        population = algorithm.evolve(population)
        done += 1
    return population, done


# -------------------------------------------------------------------------------------------------


def values(problem, size):
    """Evaluates the solutions on stdin with the adapter's fitness functions (rule 1.2)."""
    for line in sys.stdin:
        if not line.strip():
            continue
        x = np.array(json.loads(line), dtype=float)
        if problem in FRONT_PROBLEMS:
            value = FRONT_PROBLEMS[problem][0](x, size)
        elif problem == "onemax":
            value = int(onemax(x))
        elif problem in REAL_PROBLEMS:
            value = REAL_PROBLEMS[problem][0](x)
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

    if problem in FRONT_PROBLEMS:
        for seed in range(seed_from, seed_to + 1):
            for solver, udp, population_size, make_algorithm in front_solvers(problem, size):
                problem_ = pg.problem(udp)
                counter = Counter(max_evaluations, max_seconds)
                start = time.perf_counter()
                counter.start()
                population, generations = evolve_front(problem_, population_size, make_algorithm, seed)
                # the non-dominated part of the final population (rule 7.2)
                x, f = population.get_x(), population.get_f()
                first = pg.fast_non_dominated_sorting(f)[0][0]
                elapsed = time.perf_counter() - start
                print(json.dumps({
                    "library": "pygmo",
                    "solver": solver,
                    "problem": problem,
                    "size": size,
                    "mode": mode,
                    "seed": seed,
                    "time_s": round(elapsed, 6),
                    "generations": generations,
                    "evaluations": counter.evaluations,
                    "front": [[float(v) for v in f[i]] for i in first],
                    "solutions": [[float(v) for v in x[i]] for i in first],
                    "outside": counter.outside,
                }), flush=True)
        return

    target = -size if problem == "onemax" else TARGET
    for seed in range(seed_from, seed_to + 1):
        for solver, udp, method, per_generation in single_solvers(problem, size, mode):
            problem_ = pg.problem(udp)
            counter = Counter(max_evaluations, max_seconds, target)
            start = time.perf_counter()
            counter.start()
            try:
                method.run(problem_, seed)
            except Exception:
                # Stop, raised by the fitness, reaches Python through pagmo's C++
                if not counter.stopped:
                    raise
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
                "best": best,
                "target": size if problem == "onemax" else TARGET,
                "success": counter.best <= target,
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
