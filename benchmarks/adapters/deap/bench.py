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
creator.create("FitnessMin2", base.Fitness, weights=(-1.0, -1.0))
creator.create("FitnessMin3", base.Fitness, weights=(-1.0, -1.0, -1.0))
creator.create("Individual2", list, fitness=creator.FitnessMin2)
creator.create("Individual3", list, fitness=creator.FitnessMin3)


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


# Rastrigin and Ackley are shifted, so an optimum at the origin can't favour operators that drift
# towards 0: gene i is measured from s_i = 2 ((37 i + 11) mod 101) / 101 - 1, in [-1, 1]
SHIFT = [2 * ((37 * i + 11) % 101) / 101 - 1 for i in range(1000)]


def rastrigin(individual):
    return (
        10 * len(individual)
        + sum((x - s) ** 2 - 10 * math.cos(2 * math.pi * (x - s)) for x, s in zip(individual, SHIFT)),
    )


def rosenbrock(individual):
    return (
        sum(100 * (b - a * a) ** 2 + (1 - a) ** 2 for a, b in zip(individual, individual[1:])),
    )


def ackley(individual):
    n = len(individual)
    squares = sum((x - s) ** 2 for x, s in zip(individual, SHIFT)) / n
    cosines = sum(math.cos(2 * math.pi * (x - s)) for x, s in zip(individual, SHIFT)) / n
    return (-20 * math.exp(-0.2 * math.sqrt(squares)) - math.exp(cosines) + 20 + math.e,)


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
    return x[0], g * (1 - math.sqrt(x[0] / g))


def zdt2(x):
    g = zdt_g(x)
    return x[0], g * (1 - (x[0] / g) ** 2)


def zdt3(x):
    g = zdt_g(x)
    return x[0], g * (1 - math.sqrt(x[0] / g) - x[0] / g * math.sin(10 * math.pi * x[0]))


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
    return tuple(values)


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


def solve_front(problem, size, solver, budget):
    """NSGA-II (examples/ga/nsga2.py) or NSGA-III (examples/ga/nsga3.py) with the matched settings
    of every library: SBX with eta 15 at 0.9 and polynomial mutation with eta 20 at 1 / n for
    NSGA-II; SBX with eta 30 at 1 for NSGA-III."""
    function, variables, objectives, population_size = FRONT_PROBLEMS[problem]
    n = variables(size)
    individual = creator.Individual2 if objectives == 2 else creator.Individual3
    toolbox = base.Toolbox()
    toolbox.register("attribute", random.random)
    toolbox.register("individual", tools.initRepeat, individual, toolbox.attribute, n)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", function)
    eta = 15.0 if solver == "nsga2" else 30.0
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0.0, up=1.0, eta=eta)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=0.0, up=1.0, eta=20.0, indpb=1.0 / n)
    if solver == "nsga2":
        toolbox.register("select", tools.selNSGA2)
    else:
        reference = tools.uniform_reference_points(objectives, 12)
        toolbox.register("select", tools.selNSGA3, ref_points=reference)

    population = toolbox.population(n=population_size)
    for member in population:
        member.fitness.values = budget.evaluate(toolbox.evaluate, member)
    # assigns the crowding distance for the first tournament
    population = toolbox.select(population, len(population))
    generations = 0
    while not budget.exhausted():
        generations += 1
        if solver == "nsga2":
            offspring = [toolbox.clone(member) for member in tools.selTournamentDCD(population, len(population))]
            for a, b in zip(offspring[::2], offspring[1::2]):
                if random.random() <= 0.9:
                    toolbox.mate(a, b)
                toolbox.mutate(a)
                toolbox.mutate(b)
                del a.fitness.values, b.fitness.values
        else:
            offspring = algorithms.varAnd(population, toolbox, 1.0, 1.0)
        for member in offspring:
            if not member.fitness.valid:
                member.fitness.values = budget.evaluate(toolbox.evaluate, member)
        population = toolbox.select(population + offspring, population_size)
    front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    return [list(member.fitness.values) for member in front], generations


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


def solve_real_ga(problem, size, budget):
    # SBX + polynomial mutation, as in examples/ga/nsga2.py
    function, low, high = REAL_PROBLEMS[problem]
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, low, high)
    toolbox.register("individual", tools.initRepeat, creator.IndividualMin, toolbox.attr_float, size)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", function)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=low, up=high, eta=20.0)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=low, up=high, eta=20.0, indpb=1.0 / size)
    toolbox.register("select", tools.selTournament, tournsize=3)
    best, generations = ea_simple(toolbox, 100, 0.9, 1.0, budget, lambda best: best <= RASTRIGIN_TARGET)
    return best, generations


def solve_real_cmaes(problem, size, budget):
    # examples/es/cma_minfct.py, which minimizes rastrigin from [5.0] * n with sigma 5.0; the
    # other problems start from a random point with a quarter of the range as sigma
    function, low, high = REAL_PROBLEMS[problem]
    if problem == "rastrigin":
        centroid, sigma = [5.0] * size, 5.0
    else:
        centroid, sigma = [random.uniform(low, high) for _ in range(size)], (high - low) / 4
    strategy = cma.Strategy(centroid=centroid, sigma=sigma, lambda_=20 * size)
    best = math.inf
    generations = 0
    while not budget.exhausted():
        generations += 1
        population = strategy.generate(creator.IndividualMin)
        values = []
        for individual in population:
            individual.fitness.values = budget.evaluate(function, individual)
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

    if problem in FRONT_PROBLEMS:
        solvers = ["nsga2"] + (["nsga3"] if FRONT_PROBLEMS[problem][2] > 2 else [])
        for seed in range(seed_from, seed_to + 1):
            for solver in solvers:
                random.seed(seed)
                numpy.random.seed(seed)
                budget = Budget(max_evaluations, max_seconds)
                start = time.perf_counter()
                front, generations = solve_front(problem, size, solver, budget)
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
                    "front": front,
                }), flush=True)
        return

    for seed in range(seed_from, seed_to + 1):
        solvers = []
        if problem == "onemax":
            solvers.append(lambda budget: solve_onemax(size, mode, budget)[0])
        elif problem == "nqueens":
            solvers.append(lambda budget: solve_nqueens(size, mode, budget)[0])
        elif problem in REAL_PROBLEMS:
            def ga(budget):
                best, generations = solve_real_ga(problem, size, budget)
                return ("ga", best, generations, RASTRIGIN_TARGET, best <= RASTRIGIN_TARGET)

            def cmaes(budget):
                best, generations = solve_real_cmaes(problem, size, budget)
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
