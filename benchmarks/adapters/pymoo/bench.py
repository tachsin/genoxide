"""Benchmark adapter for pymoo.

Usage: python bench.py <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
Prints one JSON line per solver per seed, see ../../README.md for the fields.
"""

import json
import math
import sys
import time

import numpy as np
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.ga import GA, comp_by_cv_and_fitness
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.termination import Termination
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling, PermutationRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.optimize import minimize

# -------------------------------------------------------------------------------------------------
# Problems, with the same fitness functions as the other adapters (pymoo minimizes)
# -------------------------------------------------------------------------------------------------


class CountingProblem(ElementwiseProblem):
    def __init__(self, **kwargs):
        super().__init__(n_obj=1, **kwargs)
        self.evaluations = 0

    def _evaluate(self, x, out, *args, **kwargs):
        self.evaluations += 1
        out["F"] = self.fitness(x)


class OneMax(CountingProblem):
    def __init__(self, size):
        super().__init__(n_var=size, xl=0, xu=1, vtype=bool)

    def fitness(self, x):
        return -int(sum(x))  # maximize the number of ones


class NQueens(CountingProblem):
    def __init__(self, size):
        super().__init__(n_var=size, xl=0, xu=size - 1, vtype=int)

    def fitness(self, x):
        """Number of diagonal conflicts, O(n) (DEAP examples/ga/nqueens.py)."""
        individual = [int(v) for v in x]
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


class Rastrigin(CountingProblem):
    def __init__(self, size):
        super().__init__(n_var=size, xl=-RASTRIGIN_BOUND, xu=RASTRIGIN_BOUND)

    def fitness(self, x):
        return 10 * len(x) + sum(v * v - 10 * math.cos(2 * math.pi * v) for v in x)


class BudgetTermination(Termination):
    """Stops at the target, at max_evaluations or at max_seconds."""

    def __init__(self, problem, max_evaluations, max_seconds, target):
        super().__init__()
        self.problem = problem
        self.max_evaluations = max_evaluations
        self.deadline = time.perf_counter() + max_seconds
        self.target = target

    def _update(self, algorithm):
        if self.problem.evaluations >= self.max_evaluations or time.perf_counter() >= self.deadline:
            return 1.0
        if algorithm.opt is not None and len(algorithm.opt) > 0 and algorithm.opt[0].F[0] <= self.target:
            return 1.0
        return 0.0


def run(problem, algorithm, target, seed, max_evaluations, max_seconds):
    termination = BudgetTermination(problem, max_evaluations, max_seconds, target)
    result = minimize(
        problem, algorithm, termination, seed=seed, verbose=False,
        copy_algorithm=False, copy_termination=False,
    )
    best = float(result.F[0]) if result.F is not None else math.inf
    return best, result.algorithm.n_gen


# -------------------------------------------------------------------------------------------------
# Solvers
# -------------------------------------------------------------------------------------------------


def solvers_for(problem_name, size, mode):
    """[(solver name, problem factory, algorithm factory, target, to_best, is_success)]"""
    if problem_name == "onemax":
        if mode == "matched":
            # population 300, tournament 3, two point crossover 0.5, mutation 0.2 of ~1 bit.
            # Difference: pymoo's GA survival is elitist (best of parents + offspring)
            def algorithm():
                return GA(
                    pop_size=300,
                    sampling=BinaryRandomSampling(),
                    selection=TournamentSelection(func_comp=comp_by_cv_and_fitness, pressure=3),
                    crossover=TwoPointCrossover(prob=0.5),
                    mutation=BitflipMutation(prob=0.2, prob_var=1.0 / size),
                    eliminate_duplicates=False,
                )
        else:
            # binary GA as in the pymoo docs (knapsack example)
            def algorithm():
                return GA(
                    pop_size=200,
                    sampling=BinaryRandomSampling(),
                    crossover=TwoPointCrossover(),
                    mutation=BitflipMutation(),
                    eliminate_duplicates=True,
                )
        return [("ga", lambda: OneMax(size), algorithm, -size, lambda f: -f, lambda best: best >= size)]

    if problem_name == "nqueens":
        # permutation GA as in the pymoo docs (docs/source/customization/permutation.md)
        def algorithm():
            return GA(
                pop_size=20,
                sampling=PermutationRandomSampling(),
                mutation=InversionMutation(),
                crossover=OrderCrossover(),
                eliminate_duplicates=True,
            )
        return [("ga", lambda: NQueens(size), algorithm, 0, lambda f: f, lambda best: best == 0)]

    if problem_name == "rastrigin":
        success = lambda best: best <= RASTRIGIN_TARGET
        return [
            # defaults: population 100, SBX, polynomial mutation
            ("ga", lambda: Rastrigin(size), lambda: GA(), RASTRIGIN_TARGET, lambda f: f, success),
            ("de", lambda: Rastrigin(size), lambda: DE(), RASTRIGIN_TARGET, lambda f: f, success),
            ("cma_es", lambda: Rastrigin(size),
             # with restarts, as recommended for Rastrigin in docs/source/algorithms/soo/cmaes.md
             lambda: CMAES(x0=np.random.uniform(-RASTRIGIN_BOUND, RASTRIGIN_BOUND, size),
                           restarts=10, restart_from_best=True),
             RASTRIGIN_TARGET, lambda f: f, success),
        ]

    print(f"unknown problem {problem_name}", file=sys.stderr)
    sys.exit(2)


def main():
    if len(sys.argv) != 8:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    problem_name, size, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    seed_from, seed_to = int(sys.argv[4]), int(sys.argv[5])
    max_evaluations, max_seconds = int(sys.argv[6]), float(sys.argv[7])

    for seed in range(seed_from, seed_to + 1):
        for solver, make_problem, make_algorithm, target, to_best, is_success in solvers_for(problem_name, size, mode):
            np.random.seed(seed)
            problem = make_problem()
            algorithm = make_algorithm()
            start = time.perf_counter()
            f, generations = run(problem, algorithm, target, seed, max_evaluations, max_seconds)
            elapsed = time.perf_counter() - start
            best = to_best(f)
            print(json.dumps({
                "library": "pymoo",
                "solver": solver,
                "problem": problem_name,
                "size": size,
                "mode": mode,
                "seed": seed,
                "time_s": round(elapsed, 6),
                "generations": generations,
                "evaluations": problem.evaluations,
                "best": best,
                "target": -target if problem_name == "onemax" else target,
                "success": bool(is_success(best)),
            }), flush=True)


if __name__ == "__main__":
    main()
