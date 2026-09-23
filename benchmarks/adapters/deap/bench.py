"""Benchmark adapter for DEAP.

Usage: python bench.py <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
Prints one JSON line per solver per seed, see ../../README.md for the fields.
"""

import json
import math
import random
import sys
import time

import numpy
from deap import algorithms, base, cma, creator, tools

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("IndividualMax", list, fitness=creator.FitnessMax)
creator.create("IndividualMin", list, fitness=creator.FitnessMin)


class Budget:
    """Counts fitness evaluations, stops at max_evaluations or max_seconds."""

    def __init__(self, max_evaluations, max_seconds):
        self.evaluations = 0
        self.max_evaluations = max_evaluations
        self.deadline = time.perf_counter() + max_seconds

    def evaluate(self, function, individual):
        self.evaluations += 1
        return function(individual)

    def exhausted(self):
        return self.evaluations >= self.max_evaluations or time.perf_counter() >= self.deadline


# -------------------------------------------------------------------------------------------------
# Fitness functions, identical to the ones in the genetic_algorithm adapter
# -------------------------------------------------------------------------------------------------


def onemax(individual):
    return (sum(individual),)


def nqueens(individual):
    """Number of diagonal conflicts, O(n) (examples/ga/nqueens.py)."""
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
    return (conflicts,)


def rastrigin(individual):
    return (
        10 * len(individual)
        + sum(x * x - 10 * math.cos(2 * math.pi * x) for x in individual),
    )


# -------------------------------------------------------------------------------------------------
# Solvers
# -------------------------------------------------------------------------------------------------


def ea_simple(toolbox, population_size, cxpb, mutpb, budget, is_success):
    """algorithms.eaSimple, with a stop at the target and at the budget."""
    hall_of_fame = tools.HallOfFame(1)
    population = toolbox.population(n=population_size)
    for individual in population:
        individual.fitness.values = budget.evaluate(toolbox.evaluate, individual)
    hall_of_fame.update(population)
    generations = 0
    while not is_success(hall_of_fame[0].fitness.values[0]) and not budget.exhausted():
        generations += 1
        offspring = toolbox.select(population, len(population))
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)
        for individual in offspring:
            if not individual.fitness.valid:
                individual.fitness.values = budget.evaluate(toolbox.evaluate, individual)
        hall_of_fame.update(offspring)
        population[:] = offspring
    return hall_of_fame[0].fitness.values[0], generations


def solve_onemax(size, mode, budget):
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.IndividualMax, toolbox.attr_bool, size)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", onemax)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("select", tools.selTournament, tournsize=3)
    if mode == "matched":
        # mutation probability 0.2 of ~1 bit, like MutateSingleGene(0.2)
        toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / size)
    else:
        # examples/ga/onemax.py
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    best, generations = ea_simple(toolbox, 300, 0.5, 0.2, budget, lambda best: best >= size)
    return [("ga", best, generations, size, best >= size)]


def solve_nqueens(size, mode, budget):
    # examples/ga/nqueens.py
    toolbox = base.Toolbox()
    toolbox.register("permutation", random.sample, range(size), size)
    toolbox.register("individual", tools.initIterate, creator.IndividualMin, toolbox.permutation)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", nqueens)
    toolbox.register("mate", tools.cxPartialyMatched)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=2.0 / size)
    toolbox.register("select", tools.selTournament, tournsize=3)
    best, generations = ea_simple(toolbox, 300, 0.5, 0.2, budget, lambda best: best == 0)
    return [("ga", best, generations, 0, best == 0)]


RASTRIGIN_BOUND = 5.12
RASTRIGIN_TARGET = 0.01


def solve_rastrigin_ga(size, budget):
    # SBX + polynomial mutation, as in examples/ga/nsga2.py
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -RASTRIGIN_BOUND, RASTRIGIN_BOUND)
    toolbox.register("individual", tools.initRepeat, creator.IndividualMin, toolbox.attr_float, size)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", rastrigin)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=-RASTRIGIN_BOUND, up=RASTRIGIN_BOUND, eta=20.0)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=-RASTRIGIN_BOUND, up=RASTRIGIN_BOUND, eta=20.0, indpb=1.0 / size)
    toolbox.register("select", tools.selTournament, tournsize=3)
    best, generations = ea_simple(toolbox, 100, 0.9, 1.0, budget, lambda best: best <= RASTRIGIN_TARGET)
    return best, generations


def solve_rastrigin_cmaes(size, budget):
    # examples/es/cma_minfct.py (which also uses rastrigin)
    strategy = cma.Strategy(centroid=[5.0] * size, sigma=5.0, lambda_=20 * size)
    best = math.inf
    generations = 0
    while not budget.exhausted():
        generations += 1
        population = strategy.generate(creator.IndividualMin)
        values = []
        for individual in population:
            individual.fitness.values = budget.evaluate(rastrigin, individual)
            values.append(individual.fitness.values[0])
        # cma.Strategy has no stop criteria: once it converges (e.g. in a local optimum of
        # rastrigin) the covariance matrix degenerates, producing NaN candidates and then a
        # LinAlgError in eigh, so stop there
        if any(math.isnan(value) for value in values):
            break
        best = min(best, min(values))
        if best <= RASTRIGIN_TARGET:
            break
        try:
            strategy.update(population)
        except numpy.linalg.LinAlgError:
            break
    return best, generations


def main():
    if len(sys.argv) != 8:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    problem, size, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    seed_from, seed_to = int(sys.argv[4]), int(sys.argv[5])
    max_evaluations, max_seconds = int(sys.argv[6]), float(sys.argv[7])

    for seed in range(seed_from, seed_to + 1):
        solvers = []
        if problem == "onemax":
            solvers.append(lambda budget: solve_onemax(size, mode, budget)[0])
        elif problem == "nqueens":
            solvers.append(lambda budget: solve_nqueens(size, mode, budget)[0])
        elif problem == "rastrigin":
            def ga(budget):
                best, generations = solve_rastrigin_ga(size, budget)
                return ("ga", best, generations, RASTRIGIN_TARGET, best <= RASTRIGIN_TARGET)

            def cmaes(budget):
                best, generations = solve_rastrigin_cmaes(size, budget)
                return ("cma_es", best, generations, RASTRIGIN_TARGET, best <= RASTRIGIN_TARGET)

            solvers += [ga, cmaes]
        else:
            print(f"unknown problem {problem}", file=sys.stderr)
            sys.exit(2)

        for solve in solvers:
            random.seed(seed)
            numpy.random.seed(seed)
            budget = Budget(max_evaluations, max_seconds)
            start = time.perf_counter()
            solver, best, generations, target, success = solve(budget)
            elapsed = time.perf_counter() - start
            print(json.dumps({
                "library": "deap",
                "solver": solver,
                "problem": problem,
                "size": size,
                "mode": mode,
                "seed": seed,
                "time_s": round(elapsed, 6),
                "generations": generations,
                "evaluations": budget.evaluations,
                "best": best,
                "target": target,
                "success": bool(success),
            }), flush=True)


if __name__ == "__main__":
    main()
