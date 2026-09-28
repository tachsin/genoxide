"""Schwefel 1.2: minimize the sum of the squared partial sums of 30 genes, which interact.

Compares how fast CMA-ES, with a full and with a diagonal covariance matrix (sep-CMA-ES), particle
swarm optimization and a real-coded genetic algorithm close in on the minimum, 0 at the origin: the
evaluations each takes until its error is at most 1, 1e-2, 1e-4, 1e-6 and 1e-8. The function is
genoxide's problems.Schwefel1_2, which run evaluates in Rust.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of a run for the plot on the example's
page, with trace.py.

    python examples/schwefel_1_2/main.py
"""

import genoxide as gx

from trace import record_2d

DIMENSIONS = 30
BUDGET = 10_000 * DIMENSIONS
# the errors at which the table gives each run's evaluations
ERRORS = [1e0, 1e-2, 1e-4, 1e-6, 1e-8]
COLUMNS = ["1", "1e-2", "1e-4", "1e-6", "1e-8"]


class Reached:
    """The evaluations after the first generation whose best error was at most each of ERRORS."""

    def __init__(self):
        self.evaluations = [None] * len(ERRORS)

    def record(self, progress):
        if progress.best_fitness is None:
            return
        for i, error in enumerate(ERRORS):
            if self.evaluations[i] is None and progress.best_fitness <= error:
                self.evaluations[i] = progress.evaluations

    def print(self, name, result):
        """A row of the table: the evaluations, "-" for an error not reached, and the best
        error."""
        cells = ["-" if reached is None else str(reached) for reached in self.evaluations]
        mantissa, exponent = f"{result.best_fitness:.1e}".split("e")
        cells.append(f"{mantissa}e{int(exponent)}")
        print(f"{name:<10}" + "".join(f"{cell:>9}" for cell in cells))


problem = gx.problems.Schwefel1_2(DIMENSIONS)
print(f"Schwefel 1.2 in {DIMENSIONS} dimensions, {BUDGET} evaluations at most")
print("Evaluations until the error is at most")
print(f"{'algorithm':<10}" + "".join(f"{column:>9}" for column in COLUMNS) + f"{'best':>9}")
for name, algorithm in (
    ("CMA-ES", gx.Cmaes(problem.genome, objective="minimize", seed=1)),
    ("sep-CMA-ES", gx.Cmaes(problem.genome, covariance="diagonal", objective="minimize", seed=1)),
    ("PSO", gx.Pso(problem.genome, population_size=40, objective="minimize", seed=1)),
    (
        "GA",
        gx.Ga(
            problem.genome,
            population_size=100,
            select=gx.Tournament(3),
            crossover=gx.SimulatedBinaryCrossover(15.0),
            mutation=gx.PolynomialMutation(20.0, rate=1 / DIMENSIONS),
            objective="minimize",
            seed=1,
        ),
    ),
):
    reached = Reached()
    result = algorithm.run(problem, target=1e-8, evaluations=BUDGET, on_generation=reached.record)
    reached.print(name, result)

# with GENOXIDE_TRACE=<file>, a trace for the plot on the example's page, of a separate run in
# 2 dimensions: the plot is the function's contour
record_2d()
