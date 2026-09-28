"""Benchmark adapter for PyGAD.

Usage:
    python bench.py <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
    python bench.py values <problem> <size>    (one JSON solution per line on stdin)

The first prints one JSON line per seed, see ../../README.md for the fields; the second prints the
value of each solution with the fitness function below.

The suite is matched: each problem has one method, defined in docs/benchmarks/rules.md (rule 6),
and every library runs it with its own implementation. PyGAD, a GA library, runs one: the GA of
OneMax 1000 (onemax 1000 matched). Any other problem, size or mode prints nothing: PyGAD has no DE
or CMA-ES. How each setting maps to the definition, and the differences:
docs/benchmarks/libraries/pygad.md (PyGAD 3.7.0:
https://github.com/ahmedfgad/GeneticAlgorithmPython/tree/3.7.0, https://pygad.readthedocs.io).
"""

import os

# one thread (rule 4.3): numpy's BLAS
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import json  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy  # noqa: E402
import pygad  # noqa: E402

# the number of generations PyGAD is given: a run ends at the target, the budget or the time cap,
# by on_generation returning "stop", never by this
GENERATIONS = 10_000_000


class Budget:
    """Counts the fitness evaluations (rule 3), one per row of a batch, and keeps the best solution
    evaluated. A run stops at the target, at max_evaluations or at max_seconds (rule 2.1). `start`
    is the run's clock."""

    def __init__(self, size, max_evaluations, max_seconds, start):
        self.start = start
        self.evaluations = 0
        self.max_evaluations = max_evaluations
        self.deadline = start + max_seconds
        # OneMax: maximized, to all ones
        self.target = size
        self.best = -1
        self.solution = None
        # the first evaluation that reaches the target: (its number, seconds since the start)
        self.first_hit = None
        # the evaluations when the last generation started: PyGAD calls on_fitness at the start of
        # every generation
        self.generation_start = 0
        # the attempts after the first (rule 2.2)
        self.restarts = 0

    def count(self, X):
        """Counts the evaluations of the rows of X (rule 3)."""
        self.evaluations += len(X)

    def keep(self, X, values):
        """Keeps the best row of a batch just counted, and the first one that reaches the target
        (its number counts the rows before it)."""
        before = self.evaluations - len(X)
        reached = values >= self.target
        if self.first_hit is None and reached.any():
            self.first_hit = (before + int(numpy.argmax(reached)) + 1, time.perf_counter() - self.start)
        best = int(numpy.argmax(values))
        if values[best] > self.best:
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
        return bool(self.best >= self.target)

    def exhausted(self):
        return self.evaluations >= self.max_evaluations or time.perf_counter() >= self.deadline

    def done(self):
        return self.reached() or self.exhausted()


# -------------------------------------------------------------------------------------------------
# The fitness function, identical to benchmarks/problems.py, vectorized with numpy over a batch of
# solutions, one per row: PyGAD's batch fitness (fitness_batch_size, docs fitness_calculation.md,
# "Batch Fitness Calculation") passes the solutions of a generation to one call, and changes
# nothing else in the algorithm (utils/engine.py, cal_pop_fitness).
# -------------------------------------------------------------------------------------------------


def onemax(X):
    return numpy.sum(X, axis=1)


# -------------------------------------------------------------------------------------------------
# OneMax 1000: the GA of rule 6.2 (DEAP's eaSimple) with PyGAD's own components
# -------------------------------------------------------------------------------------------------


def onemax_ga(size):
    """PyGAD's keyword arguments for the GA of rule 6.2."""
    # binary genes as PyGAD documents them (docs benchmarks.md, "Knapsack": gene_space [0, 1],
    # gene_type int); mutation by space then sets a gene to the other value of its space
    # (helper/misc.py, generate_gene_value_from_space), a bit flip. The GA of DEAP's eaSimple with
    # PyGAD's own components: 300 individuals, generational without elitism (keep_elitism 0,
    # keep_parents 0), tournament of 3, PyGAD's two_points crossover with its
    # crossover_probability 0.5, random mutation of each gene with probability 0.2 / n (the
    # definition's mean of 0.2 bits per child). The differences are on the library's page.
    return dict(
        num_genes=size, gene_space=[0, 1], gene_type=int,
        sol_per_pop=300, num_parents_mating=300,
        parent_selection_type="tournament", K_tournament=3,
        keep_parents=0, keep_elitism=0,
        crossover_type="two_points", crossover_probability=0.5,
        mutation_type="random", mutation_probability=0.2 / size,
    )


def attempt_seed(seed, restart):
    """Rule 2.2: attempt 0 runs with the seed, restart r with (seed + 1) * 1_000_000 + r."""
    return seed if restart == 0 else (seed + 1) * 1_000_000 + restart


# an attempt whose generations evaluate nothing for this many in a row has converged (rule 2.2)
STALL_GENERATIONS = 10


def run_ga(size, seed, budget):
    """Runs PyGAD's GA; returns the number of generations, over all attempts.

    PyGAD doesn't evaluate a child identical to an elite or a parent (utils/engine.py,
    cal_pop_fitness), so a converged population can go on without evaluating anything. After
    STALL_GENERATIONS such generations in a row, on_generation ends the attempt ("stop"), and the
    GA starts again from a new random population, seeded by attempt_seed (rule 2.2); the budget
    keeps the best and counts every evaluation."""
    config = onemax_ga(size)

    def fitness_func(ga, solutions, indices):
        X = numpy.asarray(solutions)
        budget.count(X)
        values = onemax(X)
        budget.keep(X, values)
        # PyGAD maximizes
        return values.astype(float)

    # the evaluations at the end of the last generation, and the generations in a row without one
    stall = {"end": 0, "idle": 0}

    def on_generation(ga):
        stall["idle"] = stall["idle"] + 1 if budget.evaluations == stall["end"] else 0
        stall["end"] = budget.evaluations
        if budget.done() or stall["idle"] >= STALL_GENERATIONS:
            return "stop"

    generations = 0
    for restart in range(sys.maxsize):
        stall.update(end=budget.evaluations, idle=0)
        ga = pygad.GA(num_generations=GENERATIONS, fitness_func=fitness_func, on_generation=on_generation,
                      on_fitness=budget.on_fitness, fitness_batch_size=config["sol_per_pop"],
                      random_seed=attempt_seed(seed, restart), suppress_warnings=True, **config)
        ga.run()
        generations += ga.generations_completed
        if budget.done():
            return generations
        budget.restarts += 1


# -------------------------------------------------------------------------------------------------

# rule 5.3: a solver whose first EARLY_SEEDS runs all hit the time cap (a run that took CAPPED of
# it) without reaching the target runs no more seeds
EARLY_SEEDS = 3
CAPPED = 0.98


def values(problem, size):
    """Prints the value of each solution read from stdin."""
    if problem != "onemax":
        raise SystemExit(f"unknown problem {problem}")
    for line in sys.stdin:
        if line.strip():
            result = onemax(numpy.asarray([json.loads(line)]))[0]
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
    if (problem, size, mode) != ("onemax", 1000, "matched"):
        # not a scenario PyGAD runs in the matched suite
        return

    capped = 0
    for index, seed in enumerate(range(seed_from, seed_to + 1)):
        if index >= EARLY_SEEDS and capped == EARLY_SEEDS:
            continue
        random.seed(seed)
        numpy.random.seed(seed)
        # the clock starts before PyGAD's constructor, which creates the initial population
        start = time.perf_counter()
        budget = Budget(size, max_evaluations, max_seconds, start)
        generations = run_ga(size, seed, budget)
        elapsed = time.perf_counter() - start

        result = {
            "library": "pygad", "solver": "ga", "problem": problem, "size": size, "mode": mode,
            "seed": seed, "time_s": round(elapsed, 6),
            "generations": generations,
            "evaluations": budget.evaluations, "last_generation": budget.last_generation(),
        }
        if budget.restarts:
            result.update(restarts=budget.restarts)
        success = budget.reached()
        result.update(
            best=int(budget.best), target=budget.target, success=success,
            solution=[int(v) for v in budget.solution],
            first_hit=budget.first_hit and {"evaluations": budget.first_hit[0],
                                            "time_s": round(budget.first_hit[1], 6)},
        )
        capped += index < EARLY_SEEDS and not success and elapsed >= CAPPED * max_seconds
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
