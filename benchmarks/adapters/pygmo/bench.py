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
solution. How a run ends (rule 2):
- single-objective runs: the counter raises Stop from inside the fitness at the first of the target,
  the budget and the time limit, so a run ends at that evaluation, never past it;
- an algorithm that ends by its own criteria (sade's and CMA-ES' ftol and xtol, CMA-ES' and xNES'
  generations, GACO's impstop and evalstop) starts again from a new random population, the
  procedure of pygmo's cmaes_vs_xnes tutorial for algorithms "with well defined exit conditions";
  simulated annealing is annealed again from its best point, as in the solving_schwefel_20 tutorial;
- the others (sga, ihs) are given the generations of the whole budget and run to it by themselves;
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


# (fitness function of x and the size, variables, objectives, population size of NSGA-II and NSPSO,
# MOEA/D decomposition). NSGA-II needs a multiple of 4; MOEA/D's population is its 100 (2
# objectives, 99 divisions) or 91 (3 objectives, 12 divisions) grid weights
FRONT_PROBLEMS = {
    "zdt1": (lambda x, size: zdt1(x), lambda size: size, 2, 100, "tchebycheff"),
    "zdt2": (lambda x, size: zdt2(x), lambda size: size, 2, 100, "tchebycheff"),
    "zdt3": (lambda x, size: zdt3(x), lambda size: size, 2, 100, "tchebycheff"),
    # size: the number of objectives, with k = 10 (DTLZ2) and 5 (DTLZ1); pagmo's "bi" is PBI with θ 5
    "dtlz2": (lambda x, size: dtlz2(x, size), lambda size: size + 9, 3, 92, "bi"),
    "dtlz1": (lambda x, size: dtlz1(x, size), lambda size: size + 4, 3, 92, "bi"),
}
MOEAD_WEIGHTS = {2: 100, 3: 91}


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

    def fitness(self, x):
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
    """Evolves a random population until the algorithm stops by its own criteria, then starts
    again from a new random population; the counter keeps the best. This is the procedure of
    tutorials/cmaes_vs_xnes, "the best practice ... when algorithms have well defined exit
    conditions", which assembles "the results in single runs containing multiple restarts".
    Restart r uses the seed seed * 100000 + r for its population and its algorithm."""

    def __init__(self, population_size, make_algorithm):
        self.population_size = population_size
        self.make_algorithm = make_algorithm  # seed -> UDA
        self.restarts = 0

    def run(self, problem, seed):
        for restart in itertools.count():
            self.restarts = restart
            restart_seed = seed * 100_000 + restart
            population = pg.population(problem, self.population_size, seed=restart_seed)
            pg.algorithm(self.make_algorithm(restart_seed)).evolve(population)


class Reanneal:
    """Simulated annealing, annealed again from its best point after each annealing schedule, as in
    tutorials/solving_schwefel_20 ("since we will be using some reannealing": 5 calls of evolve on
    the same population). Each call starts from the population's best and puts its best back."""

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


# The population sizes of CMA-ES and xNES in tutorials/cmaes_vs_xnes. Its figures compare 3 sizes
# per function and dimension (Rosenbrock 10: 10, 20, 30; Rosenbrock 20: 40, 60, 100; Rastrigin 10:
# 40, 60, 100; Rastrigin 20: 100, 150, 200; Ackley 10: 10, 20, 30; Ackley 20: 20, 30, 40), and this
# is the one with the fewest median evaluations to the target in each figure (where its curve
# crosses 0.5). A dimension the tutorial doesn't show uses the nearest one it shows (Rastrigin 30
# and Ackley 30: their dimension 20).
TUTORIAL_POPULATIONS = {
    # function: {dimension: (CMA-ES, xNES)}
    "rosenbrock": {10: (10, 10), 20: (40, 40)},
    "rastrigin": {10: (100, 100), 20: (200, 150)},
    "ackley": {10: (10, 10), 20: (20, 20)},
}


def tutorial_population(problem, size, method):
    dimensions = TUTORIAL_POPULATIONS[problem]
    nearest = min(dimensions, key=lambda dimension: (abs(dimension - size), -dimension))
    return dimensions[nearest][0 if method == "cmaes" else 1]


# the population of 20 of pygmo's tutorials: tutorials/evolving_a_population (sade on Rosenbrock
# 10), tutorials/solving_schwefel_20 (sade, de, de1220, pso, simulated annealing on Schwefel 20)
# and pagmo's quick start (tutorials/getting_started.cpp: sade on Schwefel 30, islands of 20)
TUTORIAL_POPULATION = 20


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
        # CMA-ES with the settings of tutorials/cmaes_vs_xnes (gen=4000, ftol=1e-8, xtol=1e-10,
        # restarts), its population for the function, and force_bounds: the only way pagmo offers
        # to keep it within the bounds ("The fitness will never be called outside the bounds")
        cmaes_population = tutorial_population(problem, size, "cmaes")
        solvers.append(("cma_es", udp, Restarts(cmaes_population, lambda seed: pg.cmaes(
            gen=4000, ftol=1e-8, xtol=1e-10, force_bounds=True, seed=seed)), cmaes_population))
        if problem == "rosenbrock":
            # xNES, the other method of tutorials/cmaes_vs_xnes, with the same settings
            xnes_population = tutorial_population(problem, size, "xnes")
            solvers.append(("xnes", udp, Restarts(xnes_population, lambda seed: pg.xnes(
                gen=4000, ftol=1e-8, xtol=1e-10, force_bounds=True, seed=seed)), xnes_population))
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
# Multi-objective
# -------------------------------------------------------------------------------------------------


def front_solvers(problem, size):
    """[(solver, problem, population size, algorithm factory (gen, seed) -> UDA)]"""
    function, variables, objectives, population, decomposition = FRONT_PROBLEMS[problem]
    n = variables(size)
    udp = Problem(functools.partial(function, size=size), [0.0] * n, [1.0] * n, objectives=objectives)
    return [
        # the matched settings: SBX with eta 15 at 0.9, polynomial mutation with eta 20 at 1 / n
        ("nsga2", udp, population,
         lambda gen, seed: pg.nsga2(gen=gen, cr=0.9, eta_c=15, m=1.0 / n, eta_m=20, seed=seed)),
        # 100 or 91 grid weights, 20 neighbors, parents from the neighborhood with probability
        # 0.9, at most 2 replacements, Tchebycheff or PBI with θ 5. Difference: pagmo's MOEA/D
        # is the DE variant (DE with CR 1 and F 0.5, then polynomial mutation with eta 20), there's
        # no SBX
        ("moead", udp, MOEAD_WEIGHTS[objectives],
         lambda gen, seed: pg.moead(gen=gen, weight_generation="grid", decomposition=decomposition,
                                    neighbours=20, CR=1.0, F=0.5, eta_m=20, realb=0.9, limit=2,
                                    preserve_diversity=True, seed=seed)),
        # the settings of tutorials/nspso_tutorial_zdt1_2 ("as recommended in the original paper"),
        # with NSGA-II's population; memory keeps the velocities when evolve_front calls it once per
        # generation ("NSPSO can be called either in one single call or iteratively in a for loop,
        # by maintaining the same results")
        ("nspso", udp, population,
         lambda gen, seed: pg.nspso(gen=gen, omega=0.001, c1=2.0, c2=2.0, chi=1.0, v_coeff=0.5,
                                    leader_selection_range=100, diversity_mechanism="crowding distance",
                                    memory=True, seed=seed)),
    ]


def evolve_front(udp, population_size, make_algorithm, seed):
    """Multi-objective runs have no target: one call of evolve with the generations of the budget,
    as pagmo is used, when they fit in the time limit with room to spare (3 times the time of the
    initial population's evaluations per generation). Otherwise one generation per call, with the
    time checked after each. (One call per generation would change moead, which recomputes its
    weights, neighborhoods and ideal point in every call, and cost it ~50% more time.) Returns
    (population, generations)."""
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
            if hasattr(method, "restarts"):
                # restarts, or reannealings of simulated annealing
                run["restarts"] = method.restarts
            print(json.dumps(run), flush=True)


if __name__ == "__main__":
    main()
