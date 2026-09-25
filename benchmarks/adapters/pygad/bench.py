"""Benchmark adapter for PyGAD.

Usage:
    python bench.py <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
    python bench.py values <problem> <size>    (one JSON solution per line on stdin)

The first prints one JSON line per solver per seed, see ../../README.md for the fields; the second
prints the value (or objectives) of each solution with the fitness functions below.

The methods, their settings, where PyGAD's docs and examples show them, what was left out and the
separate test runs are in docs/benchmarks/libraries/pygad.md. Every setting below cites the PyGAD
example or doc page it comes from (PyGAD 3.7.0:
https://github.com/ahmedfgad/GeneticAlgorithmPython/tree/3.7.0, https://pygad.readthedocs.io).
"""

import os

# one thread (rule 4.3): numpy's BLAS
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy  # noqa: E402
import pygad  # noqa: E402

REAL_TARGET = 0.01
# the number of generations PyGAD is given: a run ends at the target, the budget or the time cap,
# by on_generation returning "stop", never by this
GENERATIONS = 10_000_000


class Budget:
    """Counts the fitness evaluations (rule 3) and keeps the best solution evaluated. A run stops at
    the target, at max_evaluations or at max_seconds (rule 2.1)."""

    def __init__(self, problem, size, max_evaluations, max_seconds):
        self.evaluations = 0
        self.max_evaluations = max_evaluations
        self.deadline = time.perf_counter() + max_seconds
        self.maximize = problem == "onemax"
        self.target = {"onemax": size, "nqueens": 0}.get(problem, REAL_TARGET)
        self.best = -math.inf if self.maximize else math.inf
        self.solution = None
        # the evaluations when the last generation started: PyGAD calls on_fitness at the start of
        # every generation
        self.generation_start = 0
        # the box of a continuous or multi-objective problem, and the evaluated solutions outside it
        # (rule 2.4)
        if problem in REAL_PROBLEMS:
            self.bounds = REAL_PROBLEMS[problem][1:]
        elif problem in FRONT_PROBLEMS:
            self.bounds = (0.0, 1.0)
        else:
            self.bounds = None
        self.outside = 0

    def count(self, solution):
        """Counts one evaluation (rule 3), and whether the solution is outside the box (rule 2.4)."""
        self.evaluations += 1
        if self.bounds is not None:
            low, high = self.bounds
            if numpy.any(solution < low) or numpy.any(solution > high):
                self.outside += 1

    def on_fitness(self, ga, fitness):
        self.generation_start = self.evaluations

    def last_generation(self):
        """The evaluations of the last generation: PyGAD doesn't evaluate a child identical to an
        elite or a parent again, so generations differ in size, and the budget check (rule 2.3)
        allows the last one's."""
        return self.evaluations - self.generation_start

    def keep(self, value, solution):
        if value > self.best if self.maximize else value < self.best:
            self.best = value
            self.solution = solution.copy()

    def reached(self):
        return bool(self.best >= self.target if self.maximize else self.best <= self.target)

    def exhausted(self):
        return self.evaluations >= self.max_evaluations or time.perf_counter() >= self.deadline

    def done(self):
        return self.reached() or self.exhausted()


# -------------------------------------------------------------------------------------------------
# Fitness functions, identical to benchmarks/problems.py. PyGAD calls the fitness function with one
# solution, a numpy array, at a time, and PyGAD's own benchmark problems (pygad/benchmarks) are
# numpy functions of that array, as these are.
# -------------------------------------------------------------------------------------------------


def onemax(solution):
    return int(numpy.sum(solution))


def nqueens(solution):
    """Diagonal conflicts of the queens at (i, solution[i]): for each diagonal, its queens minus one."""
    n = len(solution)
    rows = numpy.arange(n)
    columns = numpy.asarray(solution, dtype=int)
    left = numpy.bincount(rows + columns, minlength=2 * n - 1)
    right = numpy.bincount(n - 1 - rows + columns, minlength=2 * n - 1)
    return int(numpy.maximum(left - 1, 0).sum() + numpy.maximum(right - 1, 0).sum())


# Rastrigin and Ackley are shifted, so an optimum at the origin can't favour operators that drift
# towards 0: gene i is measured from s_i = 2 ((37 i + 11) mod 101) / 101 - 1, in [-1, 1]
SHIFT = numpy.array([2 * ((37 * i + 11) % 101) / 101 - 1 for i in range(1000)])


def rastrigin(solution):
    d = numpy.asarray(solution, dtype=float) - SHIFT[:len(solution)]
    return float(10 * len(d) + numpy.sum(d ** 2 - 10 * numpy.cos(2 * numpy.pi * d)))


def rosenbrock(solution):
    x = numpy.asarray(solution, dtype=float)
    a, b = x[:-1], x[1:]
    return float(numpy.sum(100 * (b - a * a) ** 2 + (1 - a) ** 2))


def ackley(solution):
    n = len(solution)
    d = numpy.asarray(solution, dtype=float) - SHIFT[:n]
    squares = numpy.sum(d ** 2) / n
    cosines = numpy.sum(numpy.cos(2 * numpy.pi * d)) / n
    return float(-20 * math.exp(-0.2 * math.sqrt(squares)) - math.exp(cosines) + 20 + math.e)


# the real-valued problems: fitness function and bounds
REAL_PROBLEMS = {
    "rastrigin": (rastrigin, -5.12, 5.12),
    "rosenbrock": (rosenbrock, -5.0, 10.0),
    "ackley": (ackley, -32.768, 32.768),
}


# multi-objective problems, minimized, all variables in [0, 1]
def zdt_g(x):
    return 1 + 9 * numpy.sum(x[1:]) / (len(x) - 1)


def zdt1(x):
    g = zdt_g(x)
    return [x[0], g * (1 - math.sqrt(x[0] / g))]


def zdt2(x):
    g = zdt_g(x)
    return [x[0], g * (1 - (x[0] / g) ** 2)]


def zdt3(x):
    g = zdt_g(x)
    return [x[0], g * (1 - math.sqrt(x[0] / g) - x[0] / g * math.sin(10 * math.pi * x[0]))]


def dtlz2(x, objectives):
    g = numpy.sum((x[objectives - 1:] - 0.5) ** 2)
    values = []
    for m in range(objectives):
        f = 1 + g
        for v in x[:objectives - 1 - m]:
            f *= math.cos(v * math.pi / 2)
        if m > 0:
            f *= math.sin(x[objectives - 1 - m] * math.pi / 2)
        values.append(f)
    return values


def dtlz1(x, objectives):
    tail = x[objectives - 1:]
    g = 100 * (len(tail) + numpy.sum((tail - 0.5) ** 2 - numpy.cos(20 * numpy.pi * (tail - 0.5))))
    values = []
    for m in range(objectives):
        f = 0.5 * (1 + g)
        for v in x[:objectives - 1 - m]:
            f *= v
        if m > 0:
            f *= 1 - x[objectives - 1 - m]
        values.append(f)
    return values


# (fitness function of the size, number of variables, number of objectives); the size of DTLZ is
# its number of objectives, with k = 10 (DTLZ2) and 5 (DTLZ1) distance variables
FRONT_PROBLEMS = {
    "zdt1": (lambda size: zdt1, lambda size: size, lambda size: 2),
    "zdt2": (lambda size: zdt2, lambda size: size, lambda size: 2),
    "zdt3": (lambda size: zdt3, lambda size: size, lambda size: 2),
    "dtlz2": (lambda size: lambda x: dtlz2(x, size), lambda size: size + 9, lambda size: size),
    "dtlz1": (lambda size: lambda x: dtlz1(x, size), lambda size: size + 4, lambda size: size),
}


# -------------------------------------------------------------------------------------------------
# Single-objective: PyGAD's GA
# -------------------------------------------------------------------------------------------------


def two_points_at(probability):
    """The matched crossover: PyGAD's own two_points crossover (utils/crossover.py), applied to a
    pair with the matched probability 0.5, else the child is a copy of its first parent. PyGAD's
    crossover_probability can't give this: it makes each parent eligible with that probability
    and crosses two eligible parents, so with 300 parents nearly every child is crossed. With
    crossover_probability unset, two_points_crossover makes child k from parents k and k + 1."""

    def crossover(parents, offspring_size, ga):
        offspring = ga.two_points_crossover(parents, offspring_size)
        copies = numpy.random.random(offspring_size[0]) >= probability
        for k in numpy.flatnonzero(copies):
            offspring[k] = parents[k % parents.shape[0]]
        return offspring

    return crossover


def single_objective(problem, size, mode):
    """(PyGAD's keyword arguments, the problem's value of a solution, whether a solution is valid)."""
    if problem == "onemax":
        # binary genes as PyGAD documents them (docs benchmarks.md, "Knapsack": gene_space [0, 1],
        # gene_type int); mutation by space then sets a gene to the other value of its space
        # (helper/misc.py, generate_gene_value_from_space), a bit flip
        common = dict(num_genes=size, gene_space=[0, 1], gene_type=int)
        if mode == "matched":
            # the matched GA of examples/ga/onemax.py in DEAP: 300 individuals, generational
            # without elitism, tournament of 3, two-point crossover at 0.5, about 0.2 bits flipped
            # per offspring
            config = dict(
                common, sol_per_pop=300, num_parents_mating=300,
                parent_selection_type="tournament", K_tournament=3,
                keep_parents=0, keep_elitism=0,
                crossover_type=two_points_at(0.5),
                mutation_type="random", mutation_probability=0.2 / size,
            )
        else:
            # examples/benchmarks/example_knapsack.py, PyGAD's binary example: 30 solutions, 10
            # parents, and PyGAD's defaults: steady-state selection ("sss"), single_point crossover,
            # random mutation of 10% of the genes (mutation_percent_genes "default"), keep_elitism 1
            config = dict(common, sol_per_pop=30, num_parents_mating=10)
        return config, onemax, None

    if problem == "nqueens":
        # examples/benchmarks/example_tsp.py, PyGAD's permutation example (docs benchmarks.md,
        # "Travelling Salesman Problem": gene_space range(n), gene_type int,
        # allow_duplicate_genes False "keep the permutation constraint"): 30 solutions, 10 parents,
        # PyGAD's defaults otherwise
        config = dict(
            num_genes=size, gene_space=list(range(size)), gene_type=int, allow_duplicate_genes=False,
            sol_per_pop=30, num_parents_mating=10,
        )

        def valid(solution):
            return numpy.unique(solution).shape[0] == size

        return config, nqueens, valid

    # examples/benchmarks/example_classic_rastrigin.py, example_classic_ackley.py (and the docs'
    # benchmarks.md): 40 solutions, 10 parents, sbx crossover with eta 20, polynomial mutation with
    # eta 20 (PyGAD mutates each gene with probability 1 / n when mutation_probability isn't set),
    # the initial range at the bounds, which also bound sbx and polynomial; PyGAD's defaults
    # otherwise: steady-state selection, keep_elitism 1. example_classic_rosenbrock.py: the same with
    # sbx eta 30.
    function, low, high = REAL_PROBLEMS[problem]
    config = dict(
        num_genes=size, sol_per_pop=40, num_parents_mating=10,
        init_range_low=low, init_range_high=high,
        crossover_type="sbx", sbx_crossover_eta=30 if problem == "rosenbrock" else 20,
        mutation_type="polynomial", polynomial_mutation_eta=20,
    )
    return config, function, None


def run_single(problem, size, mode, seed, budget):
    """Runs PyGAD's GA; returns the number of generations."""
    config, function, valid = single_objective(problem, size, mode)
    # an invalid permutation (PyGAD couldn't remove a duplicate gene) scores worse than any
    # permutation, scaled by its missing values, as pygad/benchmarks/tsp.py does for tours
    worst = 2 * size

    def fitness_func(ga, solution, solution_idx):
        budget.count(solution)
        if valid is not None and not valid(solution):
            return -float(worst * (1 + size - numpy.unique(solution).shape[0]))
        value = function(solution)
        budget.keep(value, solution)
        # PyGAD maximizes
        return value if budget.maximize else -value

    def on_generation(ga):
        if budget.done():
            return "stop"

    ga = pygad.GA(num_generations=GENERATIONS, fitness_func=fitness_func, on_generation=on_generation,
                  on_fitness=budget.on_fitness, random_seed=seed, suppress_warnings=True, **config)
    ga.run()
    return ga.generations_completed


# -------------------------------------------------------------------------------------------------
# Multi-objective: NSGA-II and NSGA-III with the matched settings
#
# PyGAD 3.7.0 runs NSGA-II or NSGA-III when the fitness function returns several values and
# parent_selection_type is one of theirs (docs multi_objective.md; utils/parent_selection.py,
# nsga.py, nsga2.py, nsga3.py). But NSGA-II and NSGA-III are parent selections there: a generation
# (utils/engine.py, run) selects the parents from the population, crosses and mutates them, and the
# next population is either the keep_elitism best of the current one, by PyGAD's NSGA-II sort (front,
# then crowding distance), followed by the offspring, or, with keep_parents -1, the parents followed
# by the offspring. Each algorithm's survival is built from these, with N the matched population, 100
# (92 with 3 objectives), and a PyGAD population of 2N:
#
# - NSGA-II: keep_elitism N, and N offspring. The N elites of each generation are the best N of the
#   previous elites and their offspring by the NSGA-II sort, which is NSGA-II's survival. The parents
#   come from PyGAD's own binary tournament ("tournament_nsga2", K_tournament 2: the lower front
#   wins, then the larger crowding distance, then a random one). Because PyGAD selects the parents
#   before the elites, it draws them from all 2N, the N survivors and the N offspring that won't all
#   survive, not from the N survivors only, and its two contestants are drawn with replacement.
# - NSGA-III: parent_selection_type "nsga3" with Das-Dennis reference points of 12 divisions
#   (nsga3_num_divisions), num_parents_mating N, keep_elitism 0 and keep_parents -1. The parents are
#   the N survivors of the 2N by NSGA-III's niching, and the next population is those survivors and
#   their N offspring: NSGA-III's survival. Its mating is random, as in NSGA-III: the crossover below
#   pairs the survivors in a random order.
#
# Differences from the textbook algorithms and from the other libraries' runs:
# - The initial population has 2N random individuals, so the first generation costs N more
#   evaluations.
# - PyGAD's crowding distance normalizes each objective by its range over the whole population, not
#   over the front.
# - An offspring identical to an elite or a parent of the previous generation takes its fitness
#   without an evaluation (utils/engine.py, cal_pop_fitness). The evaluations printed are the true
#   number of calls to the fitness function.
# - Crossover: PyGAD's own "sbx" can't take the matched settings. It makes one child from two parents
#   drawn at random among those that pass crossover_probability, so a pair isn't crossed with
#   probability 0.9, and that child is always the one below the parents' midpoint
#   (0.5 * ((y1 + y2) - beta_q * (y2 - y1)) with beta_q > 0, utils/crossover.py), which pulls every
#   gene towards the lower bound, where ZDT's optimum is (ahmedfgad/GeneticAlgorithmPython#369). So
#   the crossover is make_sbx below, given to PyGAD as a crossover function (docs
#   user_defined_operators.md): bounded SBX as DEAP's cxSimulatedBinaryBounded, each variable with
#   probability 0.5, η 15 with each pair crossed with probability 0.9 for NSGA-II, η 30 with every
#   pair crossed for NSGA-III.
# - Mutation: PyGAD's own "polynomial" (utils/mutation.py, polynomial_mutation), Deb's bounded
#   polynomial mutation: η 20, each gene with probability 1 / n (mutation_probability), within
#   init_range_low and init_range_high.
# - PyGAD maximizes, so the fitness function returns the negated objectives. The front is printed
#   minimized.
# - Its non-dominated sorting compares every pair of individuals in Python (utils/nsga.py), and it
#   sorts the population two or three times per generation. The time cap can stop a run before the
#   budget.
# The front printed is the non-dominated part of the final N survivors: the elites (NSGA-II) or the
# parents (NSGA-III) that PyGAD selects from the last population after the last generation.
# -------------------------------------------------------------------------------------------------


def sbx_pair(a, b, eta, low=0.0, high=1.0):
    """Bounded SBX of two parents, as DEAP's cxSimulatedBinaryBounded: each variable crosses with
    probability 0.5, and the two children swap sides with probability 0.5."""
    y1, y2 = numpy.minimum(a, b), numpy.maximum(a, b)
    crossed = (numpy.random.random(a.size) <= 0.5) & (y2 - y1 > 1e-14)
    rand = numpy.random.random(a.size)
    # 1 where a variable doesn't cross, to avoid dividing by 0; those values are discarded
    delta = numpy.where(crossed, y2 - y1, 1.0)
    power = 1.0 / (eta + 1.0)

    def child(beta, sign):
        alpha = 2.0 - beta ** -(eta + 1.0)
        beta_q = numpy.where(rand <= 1.0 / alpha, (rand * alpha) ** power, (1.0 / (2.0 - rand * alpha)) ** power)
        return numpy.clip(0.5 * (y1 + y2 + sign * beta_q * delta), low, high)

    lower_child = child(1.0 + 2.0 * (y1 - low) / delta, -1.0)
    upper_child = child(1.0 + 2.0 * (high - y2) / delta, 1.0)
    swap = numpy.random.random(a.size) <= 0.5
    first = numpy.where(crossed, numpy.where(swap, upper_child, lower_child), a)
    second = numpy.where(crossed, numpy.where(swap, lower_child, upper_child), b)
    return first, second


def make_sbx(eta, probability, random_pairs):
    """PyGAD's crossover function: the parents in pairs, (0, 1), (2, 3), ... in their order or in a
    random one, each pair crossed by SBX with the probability, else copied."""

    def crossover(parents, offspring_size, ga):
        count = offspring_size[0]
        order = numpy.random.permutation(len(parents)) if random_pairs else numpy.arange(len(parents))
        offspring = numpy.empty(offspring_size, dtype=float)
        for k in range(0, count, 2):
            a = numpy.array(parents[order[k % len(parents)]], dtype=float)
            b = numpy.array(parents[order[(k + 1) % len(parents)]], dtype=float)
            if numpy.random.random() <= probability:
                a, b = sbx_pair(a, b, eta)
            offspring[k] = a
            if k + 1 < count:
                offspring[k + 1] = b
        return offspring

    return crossover


def non_dominated(points):
    """The indices of the points (minimized) that no other point dominates."""
    return [
        i for i, p in enumerate(points)
        if not any(all(a <= b for a, b in zip(q, p)) and any(a < b for a, b in zip(q, p)) for q in points)
    ]


def run_front(problem, size, solver, seed, budget):
    """Runs NSGA-II or NSGA-III; returns (front, solutions, generations)."""
    function = FRONT_PROBLEMS[problem][0](size)
    n = FRONT_PROBLEMS[problem][1](size)
    objectives = FRONT_PROBLEMS[problem][2](size)
    population_size = 100 if objectives == 2 else 92

    def fitness_func(ga, solution, solution_idx):
        budget.count(solution)
        return [-float(v) for v in function(solution)]

    def on_generation(ga):
        if budget.exhausted():
            return "stop"

    common = dict(
        num_generations=GENERATIONS, fitness_func=fitness_func, on_generation=on_generation,
        on_fitness=budget.on_fitness,
        num_genes=n, gene_type=float, init_range_low=0.0, init_range_high=1.0,
        # N survivors and N offspring per generation
        sol_per_pop=2 * population_size, num_parents_mating=population_size,
        mutation_type="polynomial", polynomial_mutation_eta=20.0, mutation_probability=1.0 / n,
        random_seed=seed, suppress_warnings=True,
    )
    if solver == "nsga2":
        ga = pygad.GA(**common, keep_elitism=population_size,
                      parent_selection_type="tournament_nsga2", K_tournament=2,
                      crossover_type=make_sbx(15.0, 0.9, random_pairs=False))
    else:
        ga = pygad.GA(**common, keep_elitism=0, keep_parents=-1,
                      parent_selection_type="nsga3", nsga3_num_divisions=12,
                      crossover_type=make_sbx(30.0, 1.0, random_pairs=True))
    ga.run()

    # the final survivors: the N elites, or the N parents, PyGAD selects from the last population
    if solver == "nsga2":
        survivors = numpy.asarray(ga.last_generation_elitism_indices, dtype=int)
    else:
        survivors = numpy.asarray(ga.last_generation_parents_indices, dtype=int)
    fitness = numpy.asarray(ga.last_generation_fitness, dtype=float)[survivors]
    points = [[-float(v) for v in row] for row in fitness]
    solutions = [[float(v) for v in ga.population[i]] for i in survivors]
    front = non_dominated(points)
    return [points[i] for i in front], [solutions[i] for i in front], ga.generations_completed


# -------------------------------------------------------------------------------------------------

# rule 5.3: a solver whose first EARLY_SEEDS runs all hit the time cap (a run that took CAPPED of
# it) without reaching the target runs no more seeds
EARLY_SEEDS = 3
CAPPED = 0.98


def values(problem, size):
    """Prints the value, or the objectives, of each solution read from stdin."""
    if problem in FRONT_PROBLEMS:
        function = FRONT_PROBLEMS[problem][0](size)
    else:
        function = {"onemax": onemax, "nqueens": nqueens}.get(problem) or REAL_PROBLEMS[problem][0]
    for line in sys.stdin:
        if line.strip():
            result = function(numpy.asarray(json.loads(line)))
            print(json.dumps([float(v) for v in result] if problem in FRONT_PROBLEMS else result), flush=True)


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
    if problem in FRONT_PROBLEMS:
        solvers = ["nsga2"] + (["nsga3"] if FRONT_PROBLEMS[problem][2](size) > 2 else [])
    elif problem in ("onemax", "nqueens") or problem in REAL_PROBLEMS:
        solvers = ["ga"]
    else:
        print(f"unknown problem {problem}", file=sys.stderr)
        sys.exit(2)

    capped = {solver: 0 for solver in solvers}
    for index, seed in enumerate(range(seed_from, seed_to + 1)):
        for solver in solvers:
            if index >= EARLY_SEEDS and capped[solver] == EARLY_SEEDS:
                continue
            random.seed(seed)
            numpy.random.seed(seed)
            budget = Budget(problem, size, max_evaluations, max_seconds)
            # the clock starts before PyGAD's constructor, which creates the initial population
            start = time.perf_counter()
            if problem in FRONT_PROBLEMS:
                front, solutions, generations = run_front(problem, size, solver, seed, budget)
            else:
                generations = run_single(problem, size, mode, seed, budget)
            elapsed = time.perf_counter() - start

            result = {
                "library": "pygad", "solver": solver, "problem": problem, "size": size, "mode": mode,
                "seed": seed, "time_s": round(elapsed, 6), "generations": generations,
                "evaluations": budget.evaluations, "last_generation": budget.last_generation(),
            }
            if problem in FRONT_PROBLEMS:
                result.update(outside=budget.outside, front=front, solutions=solutions)
                success = False
            else:
                real = problem in REAL_PROBLEMS
                success = budget.reached()
                if real:
                    result.update(outside=budget.outside)
                result.update(
                    best=float(budget.best) if real else int(budget.best), target=budget.target,
                    success=success,
                    solution=[float(v) if real else int(v) for v in budget.solution],
                )
            capped[solver] += index < EARLY_SEEDS and not success and elapsed >= CAPPED * max_seconds
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
