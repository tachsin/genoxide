"""Powell: minimize Powell's singular function in 24 dimensions, six blocks of four genes, convex,
with a singular Hessian at the minimum.

Compares how fast CMA-ES, with a full and with a diagonal covariance matrix (sep-CMA-ES),
differential evolution, particle swarm optimization and a real-coded genetic algorithm close in on
the minimum, 0 at the origin: the evaluations each takes until its error is at most 1, 1e-2, 1e-4,
1e-6 and 1e-8. The function is genoxide's `problems::Powell`.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of a run for the plot on the example's page,
with trace.py.

    python examples/powell/main.py
"""

import genoxide as gx

from trace import record_small

DIMENSIONS = 24
BUDGET = 10_000 * DIMENSIONS
# the errors at which the table gives each run's evaluations
ERRORS = [1e0, 1e-2, 1e-4, 1e-6, 1e-8]
COLUMNS = ["1", "1e-2", "1e-4", "1e-6", "1e-8"]


class Reached:
    """The evaluations after the first generation whose best error was at most each of ERRORS,
    for a function whose minimum is ``minimum``."""

    def __init__(self, minimum):
        self.minimum = minimum
        self.evaluations = [None] * len(ERRORS)

    def record(self, progress):
        if progress.best_fitness is None:
            return
        error = progress.best_fitness - self.minimum
        for i, bound in enumerate(ERRORS):
            if self.evaluations[i] is None and error <= bound:
                self.evaluations[i] = progress.evaluations

    def print(self, name, result):
        """A row of the table: the evaluations, "-" for an error not reached, and the best
        error."""
        cells = ["-" if reached is None else str(reached) for reached in self.evaluations]
        # rounding can put a solution a few ulps below the minimum
        error = max(result.best_fitness - self.minimum, 0.0)
        mantissa, exponent = f"{error:.1e}".split("e")
        cells.append(f"{mantissa}e{int(exponent)}")
        print(f"{name:<10}" + "".join(f"{cell:>9}" for cell in cells))


problem = gx.problems.Powell(DIMENSIONS)
minimum = problem.optimum.value
print(f"Powell in {DIMENSIONS} dimensions, {BUDGET} evaluations at most")
print("Evaluations until the error is at most")
print(f"{'algorithm':<10}" + "".join(f"{column:>9}" for column in COLUMNS) + f"{'best':>9}")
for name, algorithm in (
    ("CMA-ES", gx.Cmaes(problem.genome, objective="minimize", seed=1)),
    ("sep-CMA-ES", gx.Cmaes(problem.genome, covariance="diagonal", objective="minimize", seed=1)),
    ("DE", gx.De(problem.genome, objective="minimize", seed=1)),
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
    reached = Reached(minimum)
    result = algorithm.run(
        problem, target=minimum + 1e-8, evaluations=BUDGET, on_generation=reached.record
    )
    reached.print(name, result)

# with GENOXIDE_TRACE=<file>, a trace for the plot on the example's page, of a separate run in
# 4 dimensions, Powell's own
record_small()
