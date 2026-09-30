"""Matyas: minimize Matyas' function, a quadratic valley 25 times flatter along the diagonal than
across it, with CMA-ES, differential evolution, particle swarm optimization and a genetic algorithm,
from 30 seeds each.

The table counts the runs that reach the minimum, to within 1e-8, and the evaluations they take. The
function, its bounds and its minimum come from genoxide's `problems::Matyas`.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its runs for the plot on the example's
page, with trace.py.

    python examples/matyas/main.py
"""

import genoxide as gx

from trace import Trace

SEEDS = 30
BUDGET = 10_000
# a run stops once its error to the minimum is at most this
ERROR = 1e-8
# the algorithms of the table, in its order
ALGORITHMS = ["CMA-ES", "DE", "PSO", "GA"]
# the algorithm whose runs the trace records
TRACED = "CMA-ES"


def median(evaluations):
    """The median of the evaluations, rounded down; 0 without any."""
    evaluations = sorted(evaluations)
    middle = len(evaluations) // 2
    if not evaluations:
        return 0
    if len(evaluations) % 2:
        return evaluations[middle]
    return (evaluations[middle - 1] + evaluations[middle]) // 2


def build(name, genome, seed):
    """The algorithm called ``name``, on ``genome``, from ``seed``."""
    algorithms = {
        "CMA-ES": lambda: gx.Cmaes(genome, objective="minimize", seed=seed),
        "DE": lambda: gx.De(genome, population_size=20, objective="minimize", seed=seed),
        "PSO": lambda: gx.Pso(genome, population_size=40, objective="minimize", seed=seed),
        "GA": lambda: gx.Ga(
            genome,
            population_size=50,
            select=gx.Tournament(3),
            crossover=gx.SimulatedBinaryCrossover(15.0),
            mutation=gx.PolynomialMutation(20.0, rate=1 / 2),
            objective="minimize",
            seed=seed,
        ),
    }
    return algorithms[name]()


problem = gx.problems.Matyas()
target = problem.optimum.value + ERROR
print(f"Matyas: minimum 0 at (0, 0), {SEEDS} seeds, {BUDGET} evaluations at most per run")
print("runs              at min  elsewhere  evaluations: median  largest")
# with GENOXIDE_TRACE=<file>, a trace of the runs of CMA-ES for the plot on the example's page
trace = Trace(problem)
for name in ALGORITHMS:
    reached, elsewhere = 0, 0
    # the evaluations of the runs that reach the target
    evaluations = []
    for seed in range(1, SEEDS + 1):
        traced = name == TRACED
        result = build(name, problem.genome, seed).run(
            problem,
            target=target,
            evaluations=BUDGET,
            on_generation=trace.on_generation if traced else None,
        )
        if result.stop_reason == "target":
            reached += 1
            evaluations.append(result.evaluations)
        else:
            elsewhere += 1
    largest = max(evaluations, default=0)
    print(f"{name:<16}  {reached:>6}  {elsewhere:>9}  {median(evaluations):>19}  {largest:>7}")
print("evaluations: of the runs that reach the minimum, to within 1e-8")
trace.write()
