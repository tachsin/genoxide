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
    return int(sum(solution))


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


def rastrigin(solution):
    return 10 * len(solution) + sum(v * v - 10 * math.cos(2 * math.pi * v) for v in solution)


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

    if problem == "rastrigin":
        # PyGAD defaults, within the bounds
        config = dict(
            num_genes=size, sol_per_pop=100, num_parents_mating=50,
            gene_space={"low": -RASTRIGIN_BOUND, "high": RASTRIGIN_BOUND},
            init_range_low=-RASTRIGIN_BOUND, init_range_high=RASTRIGIN_BOUND,
        )
        return config, lambda solution: -rastrigin(solution), lambda best: -best <= RASTRIGIN_TARGET

    print(f"unknown problem {problem}", file=sys.stderr)
    sys.exit(2)


def main():
    if len(sys.argv) != 8:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    problem, size, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    seed_from, seed_to = int(sys.argv[4]), int(sys.argv[5])
    max_evaluations, max_seconds = int(sys.argv[6]), float(sys.argv[7])

    for seed in range(seed_from, seed_to + 1):
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
