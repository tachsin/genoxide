"""Benchmark adapter for pygmo (the Python bindings of pagmo).

Usage: python bench.py <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
Prints one JSON line per solver per seed, see ../../README.md for the fields.

pagmo's algorithms run in C++ and call the fitness of a Python user-defined problem (UDP),
single-threaded (no islands, archipelagos or batch evaluators). How a run stops:
- sade, cmaes and sga evolve one generation per call of evolve (gen=1, memory=True for sade and
  cmaes), checked after each generation for the target, the budget and the time limit, and for
  the algorithm's own stop criteria. This gives exactly the same populations, evaluations and stop
  as one call with gen=N (checked).
- pso can't be split this way (every call restarts the particles from their best positions), so
  it runs in one call and the fitness stops it at the end of a generation (by raising).
- the multi-objective algorithms have no target and run in one call with the generations of the
  budget, see evolve_front.
"""

import os

# single-threaded, also for any BLAS behind numpy
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import pygmo as pg  # noqa: E402


class Stop(Exception):
    """Raised by the fitness to stop an algorithm that runs in one call of evolve."""


class Budget:
    """Counts fitness evaluations and tracks the best (minimized) value. pagmo deep-copies the
    problem, so the problems count here, in the module's current budget."""

    def __init__(self, max_evaluations, max_seconds, target=-math.inf):
        self.evaluations = 0
        self.max_evaluations = max_evaluations
        self.max_seconds = max_seconds
        self.deadline = math.inf
        self.target = target
        self.best = math.inf
        # when set, check the stop condition after every generation of this many evaluations
        self.generation_size = 0
        self.stopped = False

    def start(self):
        self.deadline = time.perf_counter() + self.max_seconds

    def count(self, value):
        self.evaluations += 1
        if value < self.best:
            self.best = value
        if self.generation_size and self.evaluations % self.generation_size == 0 and self.done():
            self.stopped = True
            raise Stop()

    def done(self):
        return (self.best <= self.target or self.evaluations >= self.max_evaluations
                or time.perf_counter() >= self.deadline)


budget = Budget(0, 0.0)


# -------------------------------------------------------------------------------------------------
# Fitness functions, identical to the other adapters (pagmo minimizes)
# -------------------------------------------------------------------------------------------------


def onemax(x):
    return -sum(x)  # maximize the number of ones


TARGET = 0.01


# Rastrigin and Ackley are shifted, so an optimum at the origin can't favour operators that drift
# towards 0: gene i is measured from s_i = 2 ((37 i + 11) mod 101) / 101 - 1, in [-1, 1]
SHIFT = [2 * ((37 * i + 11) % 101) / 101 - 1 for i in range(1000)]


def rastrigin(x):
    return 10 * len(x) + sum((v - s) ** 2 - 10 * math.cos(2 * math.pi * (v - s)) for v, s in zip(x, SHIFT))


def rosenbrock(x):
    return sum(100 * (b - a * a) ** 2 + (1 - a) ** 2 for a, b in zip(x, x[1:]))


def ackley(x):
    n = len(x)
    squares = sum((v - s) ** 2 for v, s in zip(x, SHIFT)) / n
    cosines = sum(math.cos(2 * math.pi * (v - s)) for v, s in zip(x, SHIFT)) / n
    return -20 * math.exp(-0.2 * math.sqrt(squares)) - math.exp(cosines) + 20 + math.e


# the real-valued problems: fitness function and bounds
REAL_PROBLEMS = {
    "rastrigin": (rastrigin, -5.12, 5.12),
    "rosenbrock": (rosenbrock, -5.0, 10.0),
    "ackley": (ackley, -32.768, 32.768),
}


def zdt_g(x):
    return 1 + 9 * sum(x[1:]) / (len(x) - 1)


def zdt1(x):
    g = zdt_g(x)
    return [x[0], g * (1 - math.sqrt(x[0] / g))]


def zdt2(x):
    g = zdt_g(x)
    return [x[0], g * (1 - (x[0] / g) ** 2)]


def zdt3(x):
    g = zdt_g(x)
    return [x[0], g * (1 - math.sqrt(x[0] / g) - x[0] / g * math.sin(10 * math.pi * x[0]))]


def dtlz2(x, objectives=3):
    g = sum((v - 0.5) ** 2 for v in x[objectives - 1:])
    values = []
    for m in range(objectives):
        f = 1 + g
        for v in x[:objectives - 1 - m]:
            f *= math.cos(v * math.pi / 2)
        if m > 0:
            f *= math.sin(x[objectives - 1 - m] * math.pi / 2)
        values.append(f)
    return values


def dtlz1(x, objectives=3):
    tail = x[objectives - 1:]
    g = 100 * (len(tail) + sum((v - 0.5) ** 2 - math.cos(20 * math.pi * (v - 0.5)) for v in tail))
    values = []
    for m in range(objectives):
        f = 0.5 * (1 + g)
        for v in x[:objectives - 1 - m]:
            f *= v
        if m > 0:
            f *= 1 - x[objectives - 1 - m]
        values.append(f)
    return values


# (fitness function, variables, objectives, population size of NSGA-II and NSPSO, MOEA/D decomposition)
# NSGA-II needs a multiple of 4; MOEA/D's population is its 100 (2 objectives, 99 divisions) or
# 91 (3 objectives, 12 divisions) grid weights
FRONT_PROBLEMS = {
    "zdt1": (zdt1, lambda size: size, 2, 100, "tchebycheff"),
    "zdt2": (zdt2, lambda size: size, 2, 100, "tchebycheff"),
    "zdt3": (zdt3, lambda size: size, 2, 100, "tchebycheff"),
    # size: the number of objectives, with k = 10 (DTLZ2) and 5 (DTLZ1); pagmo's "bi" is PBI with θ 5
    "dtlz2": (dtlz2, lambda size: size + 9, 3, 92, "bi"),
    "dtlz1": (dtlz1, lambda size: size + 4, 3, 92, "bi"),
}
MOEAD_WEIGHTS = {2: 100, 3: 91}


class Problem:
    """A pagmo user-defined problem: a box-bounded fitness that counts its evaluations."""

    def __init__(self, function, lower, upper, objectives=1, integers=0):
        self.function = function
        self.lower = lower
        self.upper = upper
        self.objectives = objectives
        self.integers = integers

    def fitness(self, x):
        values = self.function(x.tolist())
        if self.objectives == 1:
            budget.count(values)
            return [values]
        budget.count(math.inf)
        return values

    def get_bounds(self):
        return self.lower, self.upper

    def get_nobj(self):
        return self.objectives

    def get_nix(self):
        return self.integers


# -------------------------------------------------------------------------------------------------
# Solvers
# -------------------------------------------------------------------------------------------------


def never(population):
    return False


def de_converged(population, xtol=1e-6, ftol=1e-6):
    """The exit check that sade (and de) make after every generation, which ends a call of evolve
    with gen=N: the best and the worst individual differ by less than xtol (the sum of the gene
    differences) or ftol."""
    best, worst = population.best_idx(), population.worst_idx()
    x, f = population.get_x(), population.get_f()
    return float(abs(x[worst] - x[best]).sum()) < xtol or abs(f[worst][0] - f[best][0]) < ftol


def single_solvers(problem, size, mode):
    """[(solver, problem, population size, algorithm factory (gen, seed) -> UDA, converged)]:
    converged(population) is the algorithm's own stop check after each generation, or None for an
    algorithm that runs in one call of evolve"""
    if problem == "onemax":
        # binary genes as integers in [0, 1]: sga mutates an integer gene by drawing it uniformly
        # from its bounds, so a mutation flips the bit with probability 1/2
        udp = Problem(onemax, [0] * size, [1] * size, integers=size)
        if mode == "matched":
            # population 300, tournament 3, crossover 0.5, bit-flip ~0.2 bits per child.
            # Differences: pagmo's sga has no two-point crossover (single point here, one child per
            # selected parent with a random partner), mutates every child with 0.2 / size per bit
            # instead of 20% of the children with 1 / size, and its reinsertion is elitist (the
            # best 300 of parents and children), which can't be turned off
            def algorithm(gen, seed):
                return pg.sga(gen=gen, cr=0.5, m=0.4 / size, param_s=3, crossover="single",
                              mutation="uniform", selection="tournament", seed=seed)
            return [("ga", udp, 300, algorithm, never)]

        # sga's defaults (exponential crossover 0.9, mutation 0.02 per gene, tournament 2), with the
        # population of 20 of pagmo's tutorials (tutorials/solving_schwefel_20)
        return [("ga", udp, 20, lambda gen, seed: pg.sga(gen=gen, seed=seed), never)]

    if problem in REAL_PROBLEMS:
        function, low, high = REAL_PROBLEMS[problem]
        udp = Problem(function, [low] * size, [high] * size)
        return [
            # the defaults, with the population of 20 of tutorials/solving_schwefel_20 (sade, de,
            # pso); memory keeps the adapted F and CR between the calls of evolve
            ("sade", udp, 20, lambda gen, seed: pg.sade(gen=gen, memory=True, seed=seed), de_converged),
            # the defaults, with the population of 20 of tutorials/cmaes_vs_xnes (10, 20 and 30),
            # within the bounds
            ("cma_es", udp, 20,
             lambda gen, seed: pg.cmaes(gen=gen, memory=True, force_bounds=True, seed=seed), never),
            ("pso", udp, 20, lambda gen, seed: pg.pso(gen=gen, seed=seed), None),
            # sga's defaults (exponential crossover 0.9, polynomial mutation 0.02, tournament 2)
            ("ga", udp, 20, lambda gen, seed: pg.sga(gen=gen, seed=seed), never),
        ]

    if problem == "nqueens":
        # pagmo has no permutation representation
        return []

    print(f"unknown problem {problem}", file=sys.stderr)
    sys.exit(2)


def front_solvers(problem, size):
    """[(solver, problem, population size, algorithm factory (gen, seed) -> UDA)]"""
    function, variables, objectives, population, decomposition = FRONT_PROBLEMS[problem]
    n = variables(size)
    udp = Problem(function, [0.0] * n, [1.0] * n, objectives=objectives)
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
        # with NSGA-II's population
        ("nspso", udp, population,
         lambda gen, seed: pg.nspso(gen=gen, omega=0.001, c1=2.0, c2=2.0, chi=1.0, v_coeff=0.5,
                                    leader_selection_range=100, diversity_mechanism="crowding distance",
                                    memory=True, seed=seed)),
    ]


def budget_generations(population_size):
    """The generations that use up the rest of the evaluation budget."""
    return max(1, math.ceil((budget.max_evaluations - budget.evaluations) / population_size))


def evolve_per_generation(algorithm, population, converged=never):
    """Evolves one generation per call until the budget says stop, or until the algorithm stops
    by its own criteria, as it would in one call with gen=N: cmaes' ftol and xtol (a call then
    evaluates nothing) or converged(population). Returns (population, generations)."""
    generations = 0
    while not budget.done():
        evaluations = budget.evaluations
        population = algorithm.evolve(population)
        if budget.evaluations == evaluations:
            break
        generations += 1
        if converged(population):
            break
    return population, generations


def evolve_at_once(udp, population_size, make_algorithm, seed):
    """Evolves in one call with the generations of the budget, stopped by the fitness at the end
    of the first generation at the target or the time limit. Returns the generations."""
    population = pg.population(udp, population_size, seed=seed)
    if budget.done():
        return 0
    algorithm = pg.algorithm(make_algorithm(budget_generations(population_size), seed))
    budget.generation_size = population_size
    try:
        algorithm.evolve(population)
    except Exception:
        if not budget.stopped:
            raise
    return (budget.evaluations - population_size) // population_size


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
    generations = budget_generations(population_size)
    if generations * seconds_per_generation * 3 < budget.deadline - time.perf_counter():
        algorithm = pg.algorithm(make_algorithm(generations, seed))
        return algorithm.evolve(population), generations
    return evolve_per_generation(pg.algorithm(make_algorithm(1, seed)), population)


def main():
    global budget
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
                budget = Budget(max_evaluations, max_seconds)
                start = time.perf_counter()
                budget.start()
                population, generations = evolve_front(problem_, population_size, make_algorithm, seed)
                values = population.get_f()
                first = pg.fast_non_dominated_sorting(values)[0][0]
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
                    "evaluations": budget.evaluations,
                    "front": [[float(v) for v in values[i]] for i in first],
                }), flush=True)
        return

    target = -size if problem == "onemax" else TARGET
    for seed in range(seed_from, seed_to + 1):
        for solver, udp, population_size, make_algorithm, converged in single_solvers(problem, size, mode):
            problem_ = pg.problem(udp)
            budget = Budget(max_evaluations, max_seconds, target)
            start = time.perf_counter()
            budget.start()
            if converged is not None:
                algorithm = pg.algorithm(make_algorithm(1, seed))
                population = pg.population(problem_, population_size, seed=seed)
                _, generations = evolve_per_generation(algorithm, population, converged)
            else:
                generations = evolve_at_once(problem_, population_size, make_algorithm, seed)
            elapsed = time.perf_counter() - start
            best = -int(budget.best) if problem == "onemax" else budget.best
            print(json.dumps({
                "library": "pygmo",
                "solver": solver,
                "problem": problem,
                "size": size,
                "mode": mode,
                "seed": seed,
                "time_s": round(elapsed, 6),
                "generations": generations,
                "evaluations": budget.evaluations,
                "best": best,
                "target": size if problem == "onemax" else TARGET,
                "success": budget.best <= target,
            }), flush=True)


if __name__ == "__main__":
    main()
