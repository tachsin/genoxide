"""Benchmark adapter for DEAP: the matched suite.

Usage:
    python bench.py <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
    python bench.py values <problem> <size>    (one JSON solution per line on stdin)

The first prints one JSON line per solver per seed, see ../../README.md for the fields; the second
prints the value of each solution with the fitness functions below.

DEAP runs two matched scenarios, and prints nothing for any other problem, size or mode:
- OneMax 1000: the GA of examples/ga/onemax.py with algorithms.eaSimple ("ga");
- Rosenbrock 10: CMA-ES, deap.cma.Strategy with algorithms.eaGenerateUpdate ("cma_es").
DEAP has no differential evolution (only examples/de/*.py, not the library), so it doesn't run
Rastrigin 30. The settings, their sources and the differences from the definitions are on the
library's page, docs/benchmarks/libraries/deap.md (DEAP 1.4.4).
"""

import os

# one thread (rule 4.3): the CMA-ES's eigendecomposition is numpy's BLAS
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy  # noqa: E402
from deap import algorithms, base, cma, creator, tools  # noqa: E402

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("IndividualMax", list, fitness=creator.FitnessMax)
creator.create("IndividualMin", list, fitness=creator.FitnessMin)

REAL_TARGET = 0.01


class Budget:
    """Counts the fitness evaluations (rule 3) and keeps the best solution evaluated. A run stops at
    the target, at max_evaluations or at max_seconds (rule 2.1). `start` is the run's clock."""

    def __init__(self, problem, size, max_evaluations, max_seconds, start):
        self.start = start
        self.evaluations = 0
        self.max_evaluations = max_evaluations
        self.deadline = start + max_seconds
        self.maximize = problem == "onemax"
        self.target = size if self.maximize else REAL_TARGET
        self.best = -math.inf if self.maximize else math.inf
        self.solution = None
        # the first evaluation that reaches the target: (its number, seconds since the start)
        self.first_hit = None
        # the evaluations when the last generation started (rule 2.3), marked by the solvers below
        self.generation_start = 0
        # the box of a continuous problem, and the evaluated solutions outside it (rule 2.4)
        if problem in REAL_PROBLEMS:
            self.bounds = REAL_PROBLEMS[problem][1:]
        else:
            self.bounds = None
        self.outside = 0
        # the attempts after the first, of a GA that stalled (rule 2.2)
        self.restarts = 0
        # the library's error that ended the run, if any (rule 8.4)
        self.ended_by = None

    def count(self, solution):
        """Counts one evaluation (rule 3), and whether the solution is outside the box (rule 2.4)."""
        self.evaluations += 1
        if self.bounds is not None:
            low, high = self.bounds
            if any(v < low or v > high for v in solution):
                self.outside += 1

    def wrap(self, function):
        """The fitness function with the counter around it: every call on one solution counts."""

        def counted(individual):
            self.count(individual)
            values = function(individual)
            value = values[0]
            if value > self.best if self.maximize else value < self.best:
                self.best = value
                self.solution = list(individual)
                if self.first_hit is None and self.reached():
                    self.first_hit = (self.evaluations, time.perf_counter() - self.start)
            return values

        return counted

    def new_generation(self):
        """Marks the start of a generation: an initial population, or a loop of a solver below."""
        self.generation_start = self.evaluations

    def last_generation(self):
        """The evaluations since the start of the last generation (rule 2.3)."""
        return self.evaluations - self.generation_start

    def reached(self):
        return self.best >= self.target if self.maximize else self.best <= self.target

    def full(self):
        """Whether the evaluation budget is spent: the GA checks it before each evaluation, so a run
        never goes past it; the target and the time cap are checked between generations."""
        return self.evaluations >= self.max_evaluations

    def exhausted(self):
        return self.evaluations >= self.max_evaluations or time.perf_counter() >= self.deadline

    def done(self):
        return self.reached() or self.exhausted()


# -------------------------------------------------------------------------------------------------
# Fitness functions, identical to benchmarks/problems.py, in plain Python as DEAP's examples write
# them (examples/ga/onemax.py, deap/benchmarks). DEAP's fitness is a tuple.
# -------------------------------------------------------------------------------------------------


def onemax(individual):
    return (sum(individual),)


def rosenbrock(individual):
    return (
        sum(100 * (b - a * a) ** 2 + (1 - a) ** 2 for a, b in zip(individual, individual[1:])),
    )


# the real-valued problems: fitness function and bounds
REAL_PROBLEMS = {
    "rosenbrock": (rosenbrock, -5.0, 10.0),
}


# -------------------------------------------------------------------------------------------------
# OneMax 1000, matched: DEAP's GA example with algorithms.eaSimple
# -------------------------------------------------------------------------------------------------


# an attempt whose generations evaluate nothing for this many in a row has converged (rule 2.2)
STALL_GENERATIONS = 10


def ea_simple(toolbox, population_size, cxpb, mutpb, budget, seed):
    """algorithms.eaSimple (deap/algorithms.py), with a stop at the target and at the time cap
    between generations, and at the budget: select, varAnd, evaluate the individuals whose fitness
    is invalid (varAnd keeps the fitness of an individual it neither crossed nor mutated), replace
    the population.

    Rule 2.2: after STALL_GENERATIONS generations in a row without an evaluation (no offspring
    crossed or mutated), the attempt has converged, and the GA starts again from a new random
    population, with Python's random seeded (seed + 1) * 1_000_000 + restart. The budget keeps the
    best and counts every evaluation."""
    population = toolbox.population(n=population_size)
    budget.new_generation()
    for individual in population:
        individual.fitness.values = toolbox.evaluate(individual)
    generations = idle = 0
    while not budget.done():
        if idle >= STALL_GENERATIONS:
            budget.restarts += 1
            random.seed((seed + 1) * 1_000_000 + budget.restarts)
            population = toolbox.population(n=population_size)
            budget.new_generation()
            for individual in population:
                if budget.full():
                    break
                individual.fitness.values = toolbox.evaluate(individual)
            idle = 0
            continue
        generations += 1
        budget.new_generation()
        offspring = toolbox.select(population, len(population))
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)
        for individual in offspring:
            if not individual.fitness.valid:
                if budget.full():
                    break
                individual.fitness.values = toolbox.evaluate(individual)
        idle = idle + 1 if budget.last_generation() == 0 else 0
        population[:] = offspring
    return generations


def solve_onemax(size, budget, seed):
    # examples/ga/onemax.py (docs: "One Max Problem"): 300 individuals, two-point crossover,
    # tournament of 3, eaSimple with cxpb 0.5 and mutpb 0.2. The matched OneMax is defined from it:
    # the only difference is the bit-flip probability 1 / n, about 0.2 bits per offspring.
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.IndividualMax, toolbox.attr_bool, size)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", budget.wrap(onemax))
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / size)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return ea_simple(toolbox, 300, 0.5, 0.2, budget, seed)


# -------------------------------------------------------------------------------------------------
# Rosenbrock 10, matched: CMA-ES, deap.cma.Strategy
# -------------------------------------------------------------------------------------------------


def bounded_evaluate(budget, function, low, high, size):
    """The fitness function within the box, with DEAP's documented constraint handling
    (tutorials/advanced/constraints, "Constraint Handling"), as its box-bounded ES example
    examples/es/cma_mo.py does (cma.Strategy has no bounds): tools.ClosestValidPenalty evaluates a
    sample outside the box at its closest point inside (each gene clipped) and adds 1e6 times the
    squared distance to it. A sample inside the box is evaluated as it is. The counter is around the
    fitness function itself, inside DEAP's repair: it sees the evaluated point, so `outside` is 0 by
    construction (rule 2.4)."""
    lower, upper = numpy.full(size, low), numpy.full(size, high)

    def valid(individual):
        return not (any(individual < lower) or any(individual > upper))

    def closest_feasible(individual):
        return numpy.minimum(upper, numpy.maximum(lower, numpy.array(individual)))

    def distance(feasible_ind, original_ind):
        return sum((f - o) ** 2 for f, o in zip(feasible_ind, original_ind))

    toolbox = base.Toolbox()
    toolbox.register("evaluate", budget.wrap(function))
    toolbox.decorate("evaluate", tools.ClosestValidPenalty(valid, closest_feasible, 1.0e6, distance))
    return toolbox


def solve_cma_es(size, budget):
    """CMA-ES as DEAP's docs run it (examples/es/cma_minfct.py, docs: "Covariance Matrix Adaptation
    Evolution Strategy"): a cma.Strategy whose generate and update are registered in a toolbox, run
    by algorithms.eaGenerateUpdate. Every Strategy parameter at its default: lambda_ int(4 + 3 ln n)
    = 10, mu int(lambda_ / 2) = 5, weights "superlinear" (ln(mu + 0.5) - ln i), cmatrix the
    identity, and cs, damps, ccum, ccov1, ccovmu from the Strategy's formulas. The centroid is drawn
    uniformly in the box and sigma is 0.3 of its width, 4.5. Strategy has no stop criterion and no
    restarts: eaGenerateUpdate runs one generation per call, until the target, the budget or the
    time cap (a generation is evaluated whole: at most 9 evaluations past the budget, rule 2.3).
    Strategy.update raises numpy.linalg.LinAlgError if its covariance matrix degenerates: that
    error ends the run (rule 8.4)."""
    function, low, high = REAL_PROBLEMS["rosenbrock"]
    toolbox = bounded_evaluate(budget, function, low, high, size)
    strategy = cma.Strategy(centroid=numpy.random.uniform(low, high, size), sigma=0.3 * (high - low))
    toolbox.register("generate", strategy.generate, creator.IndividualMin)
    toolbox.register("update", strategy.update)
    generations = 0
    try:
        while not budget.done():
            generations += 1
            budget.new_generation()
            algorithms.eaGenerateUpdate(toolbox, ngen=1, verbose=False)
    except numpy.linalg.LinAlgError as error:
        budget.ended_by = f"LinAlgError: {error}"
    return generations


# -------------------------------------------------------------------------------------------------


def solvers_of(problem, size, mode):
    """(solver name, function of the budget and the seed returning the generations) of a scenario:
    none outside the matched suite."""
    if mode != "matched":
        return []
    if problem == "onemax" and size == 1000:
        return [("ga", lambda budget, seed: solve_onemax(size, budget, seed))]
    if problem == "rosenbrock" and size == 10:
        return [("cma_es", lambda budget, seed: solve_cma_es(size, budget))]
    return []


def values(problem, size):
    """Prints the value of each solution read from stdin."""
    function = {"onemax": onemax, "rosenbrock": rosenbrock}.get(problem)
    if function is None:
        print(f"deap doesn't run {problem}", file=sys.stderr)
        sys.exit(2)
    for line in sys.stdin:
        if line.strip():
            result = function(json.loads(line))
            print(json.dumps(result[0]), flush=True)


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

    for seed in range(seed_from, seed_to + 1):
        for solver, solve in solvers_of(problem, size, mode):
            # DEAP draws from Python's random; its CMA-ES from numpy's
            random.seed(seed)
            numpy.random.seed(seed)
            start = time.perf_counter()
            budget = Budget(problem, size, max_evaluations, max_seconds, start)
            generations = solve(budget, seed)
            elapsed = time.perf_counter() - start
            result = {
                "library": "deap", "solver": solver, "problem": problem, "size": size, "mode": mode,
                "seed": seed, "time_s": round(elapsed, 6), "generations": generations,
                "evaluations": budget.evaluations, "last_generation": budget.last_generation(),
            }
            if budget.restarts:
                result.update(restarts=budget.restarts)
            if budget.ended_by:
                result.update(ended_by=budget.ended_by)
            if problem in REAL_PROBLEMS:
                result.update(outside=budget.outside, best=float(budget.best),
                              solution=[float(v) for v in budget.solution])
            else:
                result.update(best=int(budget.best), solution=[int(v) for v in budget.solution])
            result.update(target=budget.target, success=bool(budget.reached()),
                          first_hit=budget.first_hit and {"evaluations": budget.first_hit[0],
                                                          "time_s": round(budget.first_hit[1], 6)})
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
