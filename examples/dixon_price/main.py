"""Dixon-Price: minimize the Dixon-Price function, a chain of curved valleys, in 5 and 10
dimensions, with CMA-ES with IPOP restarts, differential evolution and particle swarms with a global
and a ring topology, from 30 seeds each.

In 3 dimensions or more, the function has a stationary point with the value 2/3 at (1/3, 0, …, 0),
where searches stall, and more of them the more dimensions. The table counts the runs that reach the
minimum, 0, to within 1e-8, those that end at the stationary point, and the evaluations of the runs
that reach the minimum. The function, its bounds and its minima come from genoxide's
problems.DixonPrice, which run evaluates in Rust.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of one of its runs for the plot on the
example's page, with trace.py.

    python examples/dixon_price/main.py
"""

import genoxide as gx

from trace import Trace

SEEDS = 30
# the evaluations of a run, at most, per dimension
BUDGET = 20_000
# a run stops once its error to the minimum is at most this
ERROR = 1e-8
# the value at the stationary point (1/3, 0, …, 0), in 3 dimensions or more
STATIONARY = 2 / 3
# the rows of the table: an algorithm, and the number of dimensions
ROWS = [
    ("CMA-ES with IPOP", 5),
    ("CMA-ES with IPOP", 10),
    ("DE", 10),
    ("PSO", 10),
    ("PSO (ring)", 10),
]
# the run that the trace records: its algorithm, dimensions and seed
TRACED = ("PSO (ring)", 10, 2)


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
        "CMA-ES with IPOP": lambda: gx.Cmaes(
            genome, restarts="ipop", objective="minimize", seed=seed
        ),
        "DE": lambda: gx.De(genome, objective="minimize", seed=seed),
        "PSO": lambda: gx.Pso(genome, population_size=40, objective="minimize", seed=seed),
        "PSO (ring)": lambda: gx.Pso(
            genome, population_size=40, ring=1, objective="minimize", seed=seed
        ),
    }
    return algorithms[name]()


print(
    f"Dixon-Price: minimum 0, stationary point 2/3, {SEEDS} seeds, {BUDGET} evaluations per "
    "dimension at most per run"
)
print("runs                      at 0  at 2/3  elsewhere  evaluations: median  largest")
# with GENOXIDE_TRACE=<file>, a trace of one of the runs for the plot on the example's page
trace = Trace(gx.problems.DixonPrice(TRACED[1]))
for name, dimensions in ROWS:
    problem = gx.problems.DixonPrice(dimensions)
    reached, stalled, elsewhere = 0, 0, 0
    # the evaluations of the runs that reach the target
    evaluations = []
    for seed in range(1, SEEDS + 1):
        traced = (name, dimensions, seed) == TRACED
        result = build(name, problem.genome, seed).run(
            problem,
            target=ERROR,
            evaluations=BUDGET * dimensions,
            on_generation=trace.on_generation if traced else None,
        )
        if result.stop_reason == "target":
            reached += 1
            evaluations.append(result.evaluations)
        elif abs(result.best_fitness - STATIONARY) <= ERROR:
            stalled += 1
        else:
            elsewhere += 1
    largest = max(evaluations, default=0)
    row = f"{name}, n = {dimensions}"
    print(
        f"{row:<24}  {reached:>4}  {stalled:>6}  {elsewhere:>9}  {median(evaluations):>19}  "
        f"{largest:>7}"
    )
print("evaluations: of the runs that reach the minimum, to within 1e-8")
trace.write()
