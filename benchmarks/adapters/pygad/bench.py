"""Benchmark adapter for PyGAD.

Usage: python bench.py <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
Prints one JSON line per solver per seed, see ../../README.md for the fields.
"""

import json
import math
import random
import sys
import time

import numpy
import pygad


class Budget:
    """Counts fitness evaluations and tracks the best, stops at max_evaluations or max_seconds."""

    def __init__(self, max_evaluations, max_seconds):
        self.evaluations = 0
        self.max_evaluations = max_evaluations
        self.deadline = time.perf_counter() + max_seconds
        self.best = -math.inf  # PyGAD maximizes

    def count(self, value):
        self.evaluations += 1
        if value > self.best:
            self.best = value
        return value

    def exhausted(self):
        return self.evaluations >= self.max_evaluations or time.perf_counter() >= self.deadline


# -------------------------------------------------------------------------------------------------
# Fitness functions, identical to the other adapters (PyGAD maximizes, so minimized ones are negated)
# -------------------------------------------------------------------------------------------------


def onemax(solution):
    return int(solution.sum())


def nqueens(solution):
    """Number of diagonal conflicts, O(n) (DEAP examples/ga/nqueens.py)."""
    individual = [int(v) for v in solution]
    size = len(individual)
    left_diagonal = [0] * (2 * size - 1)
    right_diagonal = [0] * (2 * size - 1)
    for i in range(size):
        left_diagonal[i + individual[i]] += 1
        right_diagonal[size - 1 - i + individual[i]] += 1
    conflicts = 0
    for i in range(2 * size - 1):
        if left_diagonal[i] > 1:
            conflicts += left_diagonal[i] - 1
        if right_diagonal[i] > 1:
            conflicts += right_diagonal[i] - 1
    return conflicts


RASTRIGIN_BOUND = 5.12
RASTRIGIN_TARGET = 0.01


# Rastrigin and Ackley are shifted, so an optimum at the origin can't favour operators that drift
# towards 0: gene i is measured from s_i = 2 ((37 i + 11) mod 101) / 101 - 1, in [-1, 1]
SHIFT = numpy.array([2 * ((37 * i + 11) % 101) / 101 - 1 for i in range(1000)])


def rastrigin(solution):
    d = solution - SHIFT[:len(solution)]
    return 10 * len(solution) + numpy.sum(d ** 2 - 10 * numpy.cos(2 * numpy.pi * d))


def rosenbrock(solution):
    a, b = solution[:-1], solution[1:]
    return numpy.sum(100 * (b - a * a) ** 2 + (1 - a) ** 2)


def ackley(solution):
    n = len(solution)
    d = solution - SHIFT[:n]
    squares = numpy.sum(d ** 2) / n
    cosines = numpy.sum(numpy.cos(2 * numpy.pi * d)) / n
    return -20 * math.exp(-0.2 * math.sqrt(squares)) - math.exp(cosines) + 20 + math.e


# the real-valued problems: fitness function and bounds
REAL_PROBLEMS = {
    "rastrigin": (rastrigin, -RASTRIGIN_BOUND, RASTRIGIN_BOUND),
    "rosenbrock": (rosenbrock, -5.0, 10.0),
    "ackley": (ackley, -32.768, 32.768),
}


# multi-objective problems, minimized, all variables in [0, 1]
def zdt_g(x):
    return 1 + 9 * numpy.sum(x[1:]) / (len(x) - 1)


def zdt1(x):
    g = zdt_g(x)
    return x[0], g * (1 - math.sqrt(x[0] / g))


def zdt2(x):
    g = zdt_g(x)
    return x[0], g * (1 - (x[0] / g) ** 2)


def zdt3(x):
    g = zdt_g(x)
    return x[0], g * (1 - math.sqrt(x[0] / g) - x[0] / g * math.sin(10 * math.pi * x[0]))


def dtlz2(x, objectives=3):
    g = numpy.sum((x[objectives - 1:] - 0.5) ** 2)
    values = []
    for m in range(objectives):
        f = 1 + g
        for v in x[:objectives - 1 - m]:
            f *= math.cos(v * math.pi / 2)
        if m > 0:
            f *= math.sin(x[objectives - 1 - m] * math.pi / 2)
        values.append(f)
    return tuple(values)


def dtlz1(x, objectives=3):
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
    return tuple(values)


# (fitness function, variables, objectives, population size)
FRONT_PROBLEMS = {
    "zdt1": (zdt1, lambda size: size, 2, 100),
    "zdt2": (zdt2, lambda size: size, 2, 100),
    "zdt3": (zdt3, lambda size: size, 2, 100),
    # size: the number of objectives, with k = 10 (DTLZ2) and 5 (DTLZ1)
    "dtlz2": (dtlz2, lambda size: size + 9, 3, 92),
    "dtlz1": (dtlz1, lambda size: size + 4, 3, 92),
}


# -------------------------------------------------------------------------------------------------
# Solvers
# -------------------------------------------------------------------------------------------------


def config_for(problem, size, mode, seed):
    """(PyGAD keyword arguments, fitness value -> maximized score, is_success on the best score)"""
    if problem == "onemax":
        # binary genes as the integer range [0, 2)
        common = dict(num_genes=size, gene_space={"low": 0, "high": 2}, gene_type=int)
        if mode == "matched":
            # population 300, tournament 3, two point crossover 0.5, mutation ~0.2 bits per
            # offspring, generational without elitism
            config = dict(
                common, sol_per_pop=300, num_parents_mating=300,
                parent_selection_type="tournament", K_tournament=3,
                keep_parents=0, keep_elitism=0,
                crossover_type="two_points", crossover_probability=0.5,
                mutation_type="random", mutation_by_replacement=True,
                mutation_probability=0.2 / size,
            )
        else:
            # PyGAD defaults (steady state selection, single point crossover, random mutation of
            # 10% of the genes, elitism 1), with half of the population mating
            config = dict(common, sol_per_pop=100, num_parents_mating=50, mutation_by_replacement=True)
        return config, onemax, lambda best: best >= size

    if problem == "nqueens":
        # permutation: start from permutations, no duplicate genes, swap mutation
        rng = random.Random(seed)
        initial_population = [rng.sample(range(size), size) for _ in range(100)]
        config = dict(
            initial_population=initial_population, gene_type=int, gene_space=list(range(size)),
            num_parents_mating=50, allow_duplicate_genes=False, mutation_type="swap",
        )
        return config, lambda solution: -nqueens(solution), lambda best: best == 0

    if problem in REAL_PROBLEMS:
        # PyGAD defaults, within the bounds
        function, low, high = REAL_PROBLEMS[problem]
        config = dict(
            num_genes=size, sol_per_pop=100, num_parents_mating=50,
            gene_space={"low": low, "high": high},
            init_range_low=low, init_range_high=high,
        )
        return config, lambda solution: -function(solution), lambda best: -best <= RASTRIGIN_TARGET

    print(f"unknown problem {problem}", file=sys.stderr)
    sys.exit(2)


# -------------------------------------------------------------------------------------------------
# Multi-objective: NSGA-II with the matched settings
#
# PyGAD 3.7.0 runs NSGA-II when the fitness function returns several values and
# parent_selection_type is "nsga2" or "tournament_nsga2" (pygad/utils/parent_selection.py, nsga.py,
# nsga2.py). A generation (utils/engine.py, run) selects parents from the population, crosses and
# mutates them, and the next population is the `keep_elitism` best of the current population (by
# PyGAD's NSGA-II sort: front, then crowding distance) followed by the offspring. With a population
# of 2N, keep_elitism N and N offspring, the N elites of each generation are the best N of the
# previous elites and their offspring, which is NSGA-II's survival. N is the matched 100 (92 with
# 3 objectives).
#
# Differences from textbook NSGA-II (Deb et al. 2002) and from the other libraries' runs:
# - Parents: PyGAD's binary tournament ("tournament_nsga2", K_tournament 2: the lower front wins,
#   then the larger crowding distance, then a random one). It draws from the whole population of
#   2N, the N survivors and the N offspring that won't all survive, not from the N survivors only,
#   because PyGAD selects the parents before the elites. The two contestants are drawn with
#   replacement.
# - The initial population has 2N random individuals, so the first generation costs N more
#   evaluations.
# - PyGAD's crowding distance normalizes each objective by its range over the whole population,
#   not over the front.
# - An offspring identical to an elite or a parent of the previous generation takes its fitness
#   without an evaluation (utils/engine.py, cal_pop_fitness). The evaluations printed are the true
#   number of calls to the fitness function.
# - Crossover: PyGAD's own "sbx" can't take the matched settings. It makes one child from two
#   parents drawn at random among those that pass crossover_probability, so a pair isn't crossed
#   with probability 0.9, and that child is always the one below the parents' midpoint
#   (0.5 * ((y1 + y2) - beta_q * (y2 - y1)) with beta_q > 0, utils/crossover.py), which pulls every
#   gene towards the lower bound, where ZDT's optimum is. So the crossover is sbx_crossover below,
#   given to PyGAD as a crossover function: bounded SBX as DEAP's cxSimulatedBinaryBounded, η 15,
#   each pair of parents with probability 0.9, each variable with probability 0.5.
# - Mutation: PyGAD's own "polynomial" (utils/mutation.py, polynomial_mutation), which is Deb's
#   bounded polynomial mutation: η 20, each gene with probability 1 / n (mutation_probability),
#   within init_range_low and init_range_high.
# - PyGAD maximizes, so the fitness function returns the negated objectives. The front is printed
#   minimized.
# - Its non-dominated sorting compares every pair of individuals in Python (utils/nsga.py), and it
#   sorts the population three times per generation: for the tournament, for the elites and for
#   best_solution. The time cap can stop a run before the budget.
# The front printed is the non-dominated set of the final N elites, which PyGAD selects after the
# last generation (last_generation_elitism_indices).
# -------------------------------------------------------------------------------------------------

SBX_ETA = 15.0
SBX_PROBABILITY = 0.9
POLYNOMIAL_ETA = 20.0


def sbx_pair(a, b, low=0.0, high=1.0):
    """Bounded SBX of two parents, as DEAP's cxSimulatedBinaryBounded: each variable crosses with
    probability 0.5, and the two children swap sides with probability 0.5."""
    y1, y2 = numpy.minimum(a, b), numpy.maximum(a, b)
    crossed = (numpy.random.random(a.size) <= 0.5) & (y2 - y1 > 1e-14)
    rand = numpy.random.random(a.size)
    # 1 where a variable doesn't cross, to avoid dividing by 0; those values are discarded
    delta = numpy.where(crossed, y2 - y1, 1.0)
    power = 1.0 / (SBX_ETA + 1.0)

    def child(beta, sign):
        alpha = 2.0 - beta ** -(SBX_ETA + 1.0)
        beta_q = numpy.where(rand <= 1.0 / alpha, (rand * alpha) ** power, (1.0 / (2.0 - rand * alpha)) ** power)
        return numpy.clip(0.5 * (y1 + y2 + sign * beta_q * delta), low, high)

    lower_child = child(1.0 + 2.0 * (y1 - low) / delta, -1.0)
    upper_child = child(1.0 + 2.0 * (high - y2) / delta, 1.0)
    swap = numpy.random.random(a.size) <= 0.5
    first = numpy.where(crossed, numpy.where(swap, upper_child, lower_child), a)
    second = numpy.where(crossed, numpy.where(swap, lower_child, upper_child), b)
    return first, second


def sbx_crossover(parents, offspring_size, ga):
    """PyGAD's crossover function: the parents of the tournament in pairs, (0, 1), (2, 3), ...,
    crossed by SBX with probability 0.9, else copied."""
    count = offspring_size[0]
    offspring = numpy.empty(offspring_size, dtype=float)
    for k in range(0, count, 2):
        a = numpy.array(parents[k % len(parents)], dtype=float)
        b = numpy.array(parents[(k + 1) % len(parents)], dtype=float)
        if numpy.random.random() <= SBX_PROBABILITY:
            a, b = sbx_pair(a, b)
        offspring[k] = a
        if k + 1 < count:
            offspring[k + 1] = b
    return offspring


def non_dominated(points):
    """The points (minimized) that no other point dominates."""
    return [
        p for p in points
        if not any(all(a <= b for a, b in zip(q, p)) and any(a < b for a, b in zip(q, p)) for q in points)
    ]


# run.py's early stop: a solver whose first EARLY_SEEDS runs all hit the time cap (a run that took
# CAPPED of it) without reaching the target runs no more seeds
EARLY_SEEDS = 3
CAPPED = 0.98


def run_fronts(problem, size, mode, seed_from, seed_to, max_evaluations, max_seconds):
    function, variables, objectives, population_size = FRONT_PROBLEMS[problem]
    n = variables(size)
    capped = 0
    for index, seed in enumerate(range(seed_from, seed_to + 1)):
        if index >= EARLY_SEEDS and capped == EARLY_SEEDS:
            break
        random.seed(seed)
        numpy.random.seed(seed)
        budget = Budget(max_evaluations, max_seconds)

        def fitness_func(ga, solution, solution_index):
            budget.evaluations += 1
            return [-float(v) for v in function(solution)]

        def on_generation(ga):
            if budget.exhausted():
                return "stop"

        ga = pygad.GA(
            num_generations=10_000_000,
            fitness_func=fitness_func,
            on_generation=on_generation,
            num_genes=n,
            gene_type=float,
            init_range_low=0.0,
            init_range_high=1.0,
            # N elites and N offspring per generation
            sol_per_pop=2 * population_size,
            keep_elitism=population_size,
            num_parents_mating=population_size,
            parent_selection_type="tournament_nsga2",
            K_tournament=2,
            crossover_type=sbx_crossover,
            mutation_type="polynomial",
            polynomial_mutation_eta=POLYNOMIAL_ETA,
            mutation_probability=1.0 / n,
            random_seed=seed,
            suppress_warnings=True,
        )
        start = time.perf_counter()
        ga.run()
        elapsed = time.perf_counter() - start

        capped += index < EARLY_SEEDS and elapsed >= CAPPED * max_seconds
        # the final survivors: the N elites PyGAD selects from the last population
        survivors = numpy.asarray(ga.last_generation_elitism_indices, dtype=int)
        fitness = numpy.asarray(ga.last_generation_fitness, dtype=float)[survivors]
        front = non_dominated([[-float(v) for v in row] for row in fitness])
        print(json.dumps({
            "library": "pygad",
            "solver": "nsga2",
            "problem": problem,
            "size": size,
            "mode": mode,
            "seed": seed,
            "time_s": round(elapsed, 6),
            "generations": ga.generations_completed,
            "evaluations": budget.evaluations,
            "front": front,
        }), flush=True)


def main():
    if len(sys.argv) != 8:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    problem, size, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    seed_from, seed_to = int(sys.argv[4]), int(sys.argv[5])
    max_evaluations, max_seconds = int(sys.argv[6]), float(sys.argv[7])
    if problem in FRONT_PROBLEMS:
        run_fronts(problem, size, mode, seed_from, seed_to, max_evaluations, max_seconds)
        return

    capped = 0
    for index, seed in enumerate(range(seed_from, seed_to + 1)):
        if index >= EARLY_SEEDS and capped == EARLY_SEEDS:
            break
        random.seed(seed)
        numpy.random.seed(seed)
        config, score, is_success = config_for(problem, size, mode, seed)
        budget = Budget(max_evaluations, max_seconds)

        def fitness_func(ga, solution, solution_index):
            return budget.count(score(solution))

        def on_generation(ga):
            if is_success(budget.best) or budget.exhausted():
                return "stop"

        ga = pygad.GA(
            num_generations=10_000_000,
            fitness_func=fitness_func,
            on_generation=on_generation,
            random_seed=seed,
            suppress_warnings=True,
            **config,
        )
        start = time.perf_counter()
        ga.run()
        elapsed = time.perf_counter() - start

        capped += index < EARLY_SEEDS and not is_success(budget.best) and elapsed >= CAPPED * max_seconds
        best = budget.best
        if problem != "onemax":
            best = -best  # back to the minimized value
        print(json.dumps({
            "library": "pygad",
            "solver": "ga",
            "problem": problem,
            "size": size,
            "mode": mode,
            "seed": seed,
            "time_s": round(elapsed, 6),
            "generations": ga.generations_completed,
            "evaluations": budget.evaluations,
            "best": best,
            "target": size if problem == "onemax" else (0 if problem == "nqueens" else RASTRIGIN_TARGET),
            "success": bool(is_success(budget.best)),
        }), flush=True)


if __name__ == "__main__":
    main()
