"""Shekel 10: minimize Shekel's function with 10 wells in 4 dimensions with particle swarms, from 30
seeds, with a global and a ring topology.

The function has a well at each of 10 points, the deepest at (4, 4, 4, 4). A swarm whose particles
all follow the best position found so far (the global topology) can gather in another well before
a particle falls into the deepest; a ring topology spreads good positions slowly and keeps
exploring longer. The table counts the runs that end in each well. The function, its bounds and
its best known minimum come from genoxide's problems.Shekel10, which run evaluates in Rust.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/shekel10/main.py
"""

import genoxide as gx

from trace import Trace

# the wells: Shekel's first points aᵢ
WELLS = [
    (4.0, 4.0, 4.0, 4.0),
    (1.0, 1.0, 1.0, 1.0),
    (8.0, 8.0, 8.0, 8.0),
    (6.0, 6.0, 6.0, 6.0),
    (3.0, 7.0, 3.0, 7.0),
    (2.0, 9.0, 2.0, 9.0),
    (5.0, 5.0, 3.0, 3.0),
    (8.0, 1.0, 8.0, 1.0),
    (6.0, 2.0, 6.0, 2.0),
    (7.0, 3.6, 7.0, 3.6),
]
SEEDS = 30
BUDGET = 10_000
PARTICLES = 40
# a run stops once its error to the best known minimum is at most this
ERROR = 1e-6


def nearest(x):
    """The index of the well nearest ``x``."""
    distances = [sum((x[j] - well[j]) ** 2 for j in range(4)) for well in WELLS]
    return distances.index(min(distances))


def coordinates(well):
    """A well's coordinates, e.g. 3, 7, 3, 7 or 7, 3.6, 7, 3.6."""
    return ", ".join(f"{x:g}" for x in well)


def median(evaluations):
    """The median of the evaluations of the runs that reach the target, rounded down; 0 without
    any."""
    evaluations = sorted(evaluations)
    middle = len(evaluations) // 2
    if not evaluations:
        return 0
    if len(evaluations) % 2:
        return evaluations[middle]
    return (evaluations[middle - 1] + evaluations[middle]) // 2


problem = gx.problems.Shekel10()
minimum = problem.optimum.value
target = minimum + ERROR
# per topology: the runs that end in each well, those that reach the target, and their evaluations
wells = [[0] * len(WELLS), [0] * len(WELLS)]
reached = [0, 0]
evaluations = [[], []]
# with GENOXIDE_TRACE=<file>, a trace of the runs with the global topology for the plot on the
# example's page
trace = Trace([problem.genome.bounds] * problem.genome.length, problem.optimum)
for t, ring in enumerate([None, 1]):
    for seed in range(1, SEEDS + 1):
        swarm = gx.Pso(
            problem.genome,
            population_size=PARTICLES,
            ring=ring,
            objective="minimize",
            seed=seed,
        )
        on_generation = trace.on_generation if t == 0 else None
        result = swarm.run(
            problem, target=target, evaluations=BUDGET, on_generation=on_generation
        )
        wells[t][nearest(result.best_genome)] += 1
        if result.stop_reason == "target":
            reached[t] += 1
            evaluations[t].append(result.evaluations)

print(
    f"Shekel 10: best known minimum {minimum:.5f}, {SEEDS} seeds, {BUDGET} evaluations at most "
    f"per run, {PARTICLES} particles"
)
print("runs ending in the well at  PSO, global  PSO, ring")
for i, well in enumerate(WELLS):
    if wells[0][i] + wells[1][i] == 0:
        continue
    at = f"a{i + 1} = ({coordinates(well)})"
    print(f"{at:<26}  {wells[0][i]:>11}  {wells[1][i]:>9}")
print(f"{'error below 1e-6':<26}  {reached[0]:>11}  {reached[1]:>9}")
print(f"{'evaluations (median)':<26}  {median(evaluations[0]):>11}  {median(evaluations[1]):>9}")
trace.write()
