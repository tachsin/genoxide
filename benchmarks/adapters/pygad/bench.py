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
    """Counts the fitness evaluations (rule 3), one per row of a batch, and keeps the best solution
    evaluated. A run stops at the target, at max_evaluations or at max_seconds (rule 2.1). `start`
    is the run's clock."""

    def __init__(self, problem, size, max_evaluations, max_seconds, start):
        self.start = start
        self.evaluations = 0
        self.max_evaluations = max_evaluations
        self.deadline = start + max_seconds
        self.maximize = problem == "onemax"
        self.target = {"onemax": size, "nqueens": 0}.get(problem, REAL_TARGET)
        self.best = -math.inf if self.maximize else math.inf
        self.solution = None
        # the first evaluation that reaches the target: (its number, seconds since the start)
        self.first_hit = None
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

    def count(self, X):
        """Counts the evaluations of the rows of X (rule 3), and those outside the box (rule 2.4)."""
        self.evaluations += len(X)
        if self.bounds is not None:
            low, high = self.bounds
            self.outside += int(numpy.count_nonzero(numpy.any((X < low) | (X > high), axis=1)))

    def keep(self, X, values, valid):
        """Keeps the best valid row of a batch just counted, and the first one that reaches the
        target (its number counts the rows before it)."""
        rows = numpy.flatnonzero(valid)
        if rows.size == 0:
            return
        before = self.evaluations - len(X)
        reached = values[rows] >= self.target if self.maximize else values[rows] <= self.target
        if self.first_hit is None and reached.any():
            self.first_hit = (before + int(rows[numpy.argmax(reached)]) + 1, time.perf_counter() - self.start)
        best = rows[numpy.argmax(values[rows]) if self.maximize else numpy.argmin(values[rows])]
        if values[best] > self.best if self.maximize else values[best] < self.best:
            self.best = values[best].item()
            self.solution = X[best].copy()

    def on_fitness(self, ga, fitness):
        self.generation_start = self.evaluations

    def last_generation(self):
        """The evaluations of the last generation: PyGAD doesn't evaluate a child identical to an
        elite or a parent again, so generations differ in size, and the budget check (rule 2.3)
        allows the last one's."""
        return self.evaluations - self.generation_start

    def reached(self):
        return bool(self.best >= self.target if self.maximize else self.best <= self.target)

    def exhausted(self):
        return self.evaluations >= self.max_evaluations or time.perf_counter() >= self.deadline

    def done(self):
        return self.reached() or self.exhausted()


# -------------------------------------------------------------------------------------------------
# Fitness functions, identical to benchmarks/problems.py, vectorized with numpy over a batch of
# solutions, one per row: PyGAD's batch fitness (fitness_batch_size, docs fitness_calculation.md,
# "Batch Fitness Calculation") passes the solutions of a generation to one call, and changes
# nothing else in the algorithm (utils/engine.py, cal_pop_fitness).
# -------------------------------------------------------------------------------------------------


def onemax(X):
    return numpy.sum(X, axis=1)


def nqueens(X):
    """Diagonal conflicts of the queens at (i, X[i]): for each diagonal, its queens minus one. That's
    n minus the number of occupied diagonals, in each direction."""
    X = numpy.asarray(X, dtype=int)
    n = X.shape[1]
    column = numpy.arange(n)

    def occupied(diagonals):
        diagonals = numpy.sort(diagonals, axis=1)
        return 1 + numpy.count_nonzero(numpy.diff(diagonals, axis=1), axis=1)

    return 2 * n - occupied(X + column) - occupied(X + (n - 1 - column))


def distinct(X):
    """The number of distinct values in each row."""
    return 1 + numpy.count_nonzero(numpy.diff(numpy.sort(X, axis=1), axis=1), axis=1)


def shift(upper):
    """Rastrigin and Ackley are shifted, so an optimum at the origin can't favour operators that
    drift towards 0: gene i is measured from s_i = 0.8 upper (2 ((37 i + 11) mod 101) / 101 - 1),
    within 80% of the box, computed in this order, as problems.py."""
    return numpy.array([0.8 * upper * (2 * ((37 * i + 11) % 101) / 101 - 1) for i in range(1000)])


RASTRIGIN_SHIFT = shift(5.12)
ACKLEY_SHIFT = shift(32.768)


def rastrigin(X):
    D = numpy.asarray(X, dtype=float) - RASTRIGIN_SHIFT[:X.shape[1]]
    return 10 * X.shape[1] + numpy.sum(D * D - 10 * numpy.cos(2 * numpy.pi * D), axis=1)


def rosenbrock(X):
    X = numpy.asarray(X, dtype=float)
    a, b = X[:, :-1], X[:, 1:]
    return numpy.sum(100 * (b - a * a) ** 2 + (1 - a) ** 2, axis=1)


def ackley(X):
    n = X.shape[1]
    D = numpy.asarray(X, dtype=float) - ACKLEY_SHIFT[:n]
    return (-20 * numpy.exp(-0.2 * numpy.sqrt(numpy.sum(D * D, axis=1) / n))
            - numpy.exp(numpy.sum(numpy.cos(2 * numpy.pi * D), axis=1) / n) + 20 + math.e)


# the real-valued problems: fitness function and bounds
REAL_PROBLEMS = {
    "rastrigin": (rastrigin, -5.12, 5.12),
    "rosenbrock": (rosenbrock, -5.0, 10.0),
    "ackley": (ackley, -32.768, 32.768),
}


# multi-objective problems, minimized, all variables in [0, 1]
def zdt_g(X):
    return 1 + 9 * numpy.sum(X[:, 1:], axis=1) / (X.shape[1] - 1)


def zdt1(X):
    g = zdt_g(X)
    return numpy.column_stack([X[:, 0], g * (1 - numpy.sqrt(X[:, 0] / g))])


def zdt2(X):
    g = zdt_g(X)
    return numpy.column_stack([X[:, 0], g * (1 - (X[:, 0] / g) ** 2)])


def zdt3(X):
    g = zdt_g(X)
    h = 1 - numpy.sqrt(X[:, 0] / g) - X[:, 0] / g * numpy.sin(10 * numpy.pi * X[:, 0])
    return numpy.column_stack([X[:, 0], g * h])


def dtlz2(X, objectives):
    g = numpy.sum((X[:, objectives - 1:] - 0.5) ** 2, axis=1)
    columns = []
    for m in range(objectives):
        f = 1 + g
        for i in range(objectives - 1 - m):
            f = f * numpy.cos(X[:, i] * numpy.pi / 2)
        if m > 0:
            f = f * numpy.sin(X[:, objectives - 1 - m] * numpy.pi / 2)
        columns.append(f)
    return numpy.column_stack(columns)


def dtlz1(X, objectives):
    tail = X[:, objectives - 1:]
    g = 100 * (tail.shape[1] + numpy.sum((tail - 0.5) ** 2 - numpy.cos(20 * numpy.pi * (tail - 0.5)), axis=1))
    columns = []
    for m in range(objectives):
        f = 0.5 * (1 + g)
        for i in range(objectives - 1 - m):
            f = f * X[:, i]
        if m > 0:
            f = f * (1 - X[:, objectives - 1 - m])
        columns.append(f)
    return numpy.column_stack(columns)


# (fitness function of the size, number of variables, number of objectives); the size of DTLZ is
# its number of objectives, with k = 10 (DTLZ2) and 5 (DTLZ1) distance variables
FRONT_PROBLEMS = {
    "zdt1": (lambda size: zdt1, lambda size: size, lambda size: 2),
    "zdt2": (lambda size: zdt2, lambda size: size, lambda size: 2),
    "zdt3": (lambda size: zdt3, lambda size: size, lambda size: 2),
    "dtlz2": (lambda size: lambda X: dtlz2(X, size), lambda size: size + 9, lambda size: size),
    "dtlz1": (lambda size: lambda X: dtlz1(X, size), lambda size: size + 4, lambda size: size),
}


# -------------------------------------------------------------------------------------------------
# Single-objective: PyGAD's GA
# -------------------------------------------------------------------------------------------------


def single_objective(problem, size, mode):
    """(PyGAD's keyword arguments, the problem's values of a batch, whether a solution must be a
    permutation)."""
    if problem == "onemax":
        # binary genes as PyGAD documents them (docs benchmarks.md, "Knapsack": gene_space [0, 1],
        # gene_type int); mutation by space then sets a gene to the other value of its space
        # (helper/misc.py, generate_gene_value_from_space), a bit flip
        common = dict(num_genes=size, gene_space=[0, 1], gene_type=int)
        if mode == "matched":
            # the matched GA of examples/ga/onemax.py in DEAP with PyGAD's own components:
            # 300 individuals, generational without elitism (keep_elitism 0, keep_parents 0),
            # tournament of 3, PyGAD's two_points crossover with its crossover_probability 0.5,
            # random mutation of each gene with probability 0.2 / n (the matched mean of 0.2 bits
            # per child). The differences are on the library's page.
            config = dict(
                common, sol_per_pop=300, num_parents_mating=300,
                parent_selection_type="tournament", K_tournament=3,
                keep_parents=0, keep_elitism=0,
                crossover_type="two_points", crossover_probability=0.5,
                mutation_type="random", mutation_probability=0.2 / size,
            )
        else:
            # examples/benchmarks/example_knapsack.py, PyGAD's binary example: 30 solutions, 10
            # parents, and PyGAD's defaults: steady-state selection ("sss"), single_point crossover,
            # random mutation of 10% of the genes (mutation_percent_genes "default"), keep_elitism 1
            config = dict(common, sol_per_pop=30, num_parents_mating=10)
        return config, onemax, False

    if problem == "nqueens":
        # examples/benchmarks/example_tsp.py, PyGAD's permutation example (docs benchmarks.md,
        # "Travelling Salesman Problem": gene_space range(n), gene_type int,
        # allow_duplicate_genes False "keep the permutation constraint"): 30 solutions, 10 parents,
        # PyGAD's defaults otherwise
        config = dict(
            num_genes=size, gene_space=list(range(size)), gene_type=int, allow_duplicate_genes=False,
            sol_per_pop=30, num_parents_mating=10,
        )
        return config, nqueens, True

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
    return config, function, False


def run_single(problem, size, mode, seed, budget):
    """Runs PyGAD's GA; returns the number of generations."""
    config, function, permutation = single_objective(problem, size, mode)
    # an invalid permutation (PyGAD couldn't remove a duplicate gene) scores worse than any
    # permutation, scaled by its missing values, as pygad/benchmarks/tsp.py does for tours
    worst = 2 * size

    def fitness_func(ga, solutions, indices):
        X = numpy.asarray(solutions)
        budget.count(X)
        values = function(X)
        if permutation:
            counts = distinct(X)
            valid = counts == size
        else:
            valid = numpy.ones(len(X), dtype=bool)
        budget.keep(X, values, valid)
        # PyGAD maximizes
        fitness = values if budget.maximize else -values
        if permutation:
            fitness = numpy.where(valid, fitness, -worst * (1 + size - counts))
        return fitness.astype(float)

    def on_generation(ga):
        if budget.done():
            return "stop"

    ga = pygad.GA(num_generations=GENERATIONS, fitness_func=fitness_func, on_generation=on_generation,
                  on_fitness=budget.on_fitness, fitness_batch_size=config["sol_per_pop"],
                  random_seed=seed, suppress_warnings=True, **config)
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
# by the offspring. Each algorithm's survival is built from these settings, with N the matched
# population, 100 (92 with 3 objectives), and a PyGAD population of 2N:
#
# - NSGA-II: keep_elitism N, and N offspring. The N elites of each generation are the best N of the
#   previous elites and their offspring by the NSGA-II sort, which is NSGA-II's survival. The parents
#   come from PyGAD's own binary tournament ("tournament_nsga2", K_tournament 2: the lower front
#   wins, then the larger crowding distance, then a random one). Because PyGAD selects the parents
#   before the elites, it draws them from all 2N, the N survivors and the N offspring that won't all
#   survive, not from the N survivors only, and its two contestants are drawn with replacement.
# - NSGA-III: parent_selection_type "nsga3" with Das-Dennis reference points of 99 divisions with 2
#   objectives, 12 with 3 (nsga3_num_divisions), num_parents_mating N, keep_elitism 0 and
#   keep_parents -1. The parents are the N survivors of the 2N by NSGA-III's niching, and the next
#   population is those survivors and their N offspring: NSGA-III's survival.
#
# The operators are PyGAD's own (rule 6.1 and the matched settings), bugs included:
# - Crossover: "sbx" (utils/crossover.py, sbx_crossover) with sbx_crossover_eta 15 and
#   crossover_probability 0.9 (NSGA-II), 30 and every child crossed (NSGA-III, crossover_probability
#   unset). PyGAD's sbx makes one child per pair, always the one below the parents' midpoint
#   (ahmedfgad/GeneticAlgorithmPython#369), crosses every gene, and its crossover_probability makes
#   each parent eligible with that probability, and crosses two parents drawn from the eligible ones.
# - Mutation: "polynomial" (utils/mutation.py, polynomial_mutation), Deb's bounded polynomial
#   mutation: η 20, each gene with probability 1 / n (mutation_probability), within init_range_low
#   and init_range_high.
#
# Other differences from the textbook algorithms and from the other libraries' runs:
# - The initial population has 2N random individuals, so the first generation costs N more
#   evaluations.
# - PyGAD's crowding distance normalizes each objective by its range over the whole population, not
#   over the front.
# - An offspring identical to an elite or a parent of the previous generation takes its fitness
#   without an evaluation (utils/engine.py, cal_pop_fitness). The evaluations printed are the true
#   number of rows evaluated.
# - PyGAD maximizes, so the fitness function returns the negated objectives. The front is printed
#   minimized.
# - Its non-dominated sorting compares every pair of individuals in Python (utils/nsga.py), and it
#   sorts the population two or three times per generation. The time cap can stop a run before the
#   budget.
# The front printed is the non-dominated part of the final N survivors: the elites (NSGA-II) or the
# parents (NSGA-III) that PyGAD selects from the last population after the last generation.
# -------------------------------------------------------------------------------------------------


def non_dominated(points):
    """The indices of the points (minimized) that no other point dominates."""
    return [
        i for i, p in enumerate(points)
        if not any(all(a <= b for a, b in zip(q, p)) and any(a < b for a, b in zip(q, p)) for q in points)
    ]


def run_front(problem, size, solver, seed, budget):
    """Runs NSGA-II or NSGA-III; returns PyGAD's GA after the run."""
    function = FRONT_PROBLEMS[problem][0](size)
    n = FRONT_PROBLEMS[problem][1](size)
    objectives = FRONT_PROBLEMS[problem][2](size)
    population_size = 100 if objectives == 2 else 92

    def fitness_func(ga, solutions, indices):
        X = numpy.asarray(solutions, dtype=float)
        budget.count(X)
        return -function(X)

    def on_generation(ga):
        if budget.exhausted():
            return "stop"

    common = dict(
        num_generations=GENERATIONS, fitness_func=fitness_func, on_generation=on_generation,
        on_fitness=budget.on_fitness, fitness_batch_size=2 * population_size,
        num_genes=n, gene_type=float, init_range_low=0.0, init_range_high=1.0,
        # N survivors and N offspring per generation
        sol_per_pop=2 * population_size, num_parents_mating=population_size,
        crossover_type="sbx",
        mutation_type="polynomial", polynomial_mutation_eta=20.0, mutation_probability=1.0 / n,
        random_seed=seed, suppress_warnings=True,
    )
    if solver == "nsga2":
        ga = pygad.GA(**common, keep_elitism=population_size,
                      parent_selection_type="tournament_nsga2", K_tournament=2,
                      sbx_crossover_eta=15.0, crossover_probability=0.9)
    else:
        ga = pygad.GA(**common, keep_elitism=0, keep_parents=-1,
                      parent_selection_type="nsga3", nsga3_num_divisions=99 if objectives == 2 else 12,
                      sbx_crossover_eta=30.0)
    ga.run()
    return ga


def front_of(ga, solver):
    """The non-dominated part of the final survivors: the N elites, or the N parents, PyGAD selects
    from the last population (rule 7.2)."""
    if solver == "nsga2":
        survivors = numpy.asarray(ga.last_generation_elitism_indices, dtype=int)
    else:
        survivors = numpy.asarray(ga.last_generation_parents_indices, dtype=int)
    fitness = numpy.asarray(ga.last_generation_fitness, dtype=float)[survivors]
    points = [[-float(v) for v in row] for row in fitness]
    solutions = [[float(v) for v in ga.population[i]] for i in survivors]
    front = non_dominated(points)
    return [points[i] for i in front], [solutions[i] for i in front]


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
            result = function(numpy.asarray([json.loads(line)]))[0]
            if problem in FRONT_PROBLEMS:
                print(json.dumps([float(v) for v in result]), flush=True)
            else:
                print(json.dumps(result.item()), flush=True)


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
        solvers = ["nsga2", "nsga3"]
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
            # the clock starts before PyGAD's constructor, which creates the initial population
            start = time.perf_counter()
            budget = Budget(problem, size, max_evaluations, max_seconds, start)
            if problem in FRONT_PROBLEMS:
                ga = run_front(problem, size, solver, seed, budget)
            else:
                generations = run_single(problem, size, mode, seed, budget)
            elapsed = time.perf_counter() - start

            result = {
                "library": "pygad", "solver": solver, "problem": problem, "size": size, "mode": mode,
                "seed": seed, "time_s": round(elapsed, 6),
                "generations": ga.generations_completed if problem in FRONT_PROBLEMS else generations,
                "evaluations": budget.evaluations, "last_generation": budget.last_generation(),
            }
            if problem in FRONT_PROBLEMS:
                # after the clock (rule 4.1)
                front, solutions = front_of(ga, solver)
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
                    first_hit=budget.first_hit and {"evaluations": budget.first_hit[0],
                                                    "time_s": round(budget.first_hit[1], 6)},
                )
            capped[solver] += index < EARLY_SEEDS and not success and elapsed >= CAPPED * max_seconds
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
