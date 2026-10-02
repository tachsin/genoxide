"""Royal road R2: maximize 8 blocks of 8 ones, and the pairs, quadruples and whole string above
them, each scoring its number of bits when complete, with a genetic algorithm with the settings
of Mitchell, Forrest and Holland (1992).

The GA has their population of 128, single-point crossover at a rate of 0.7 and a mutation
probability of 0.005 per bit, with tournament selection of size 2 in place of their
fitness-proportionate selection with sigma scaling. A run from seed 1 prints the generations at
which its best improves; then runs from seeds 1 to 50, as in their Table 1, give the generations
to the optimum, 256. As a comparison, random-mutation hill climbing, one bit at a time. The
function is genoxide's problems.binary.RoyalRoad.r2(), which run evaluates in Rust.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/royal_road_r2/main.py
"""

import genoxide as gx

from trace import Trace

BITS = 64
# the runs of the GA, as in the paper's Table 1, and of hill climbing
RUNS = 50
CLIMBS = 200
# the most evaluations of a run, the paper's 2000 generations of 128
BUDGET = 256_000


def ga(problem, seed):
    """The genetic algorithm from ``seed``."""
    return gx.Ga(
        problem.genome,
        population_size=128,
        select=gx.Tournament(2),
        crossover=gx.PointCrossover(1),
        crossover_rate=0.7,
        mutation=gx.BitFlip(rate=0.005),
        seed=seed,
    )


def mean_and_median(values):
    """The mean and the median of ``values``."""
    values = sorted(values)
    middle = len(values) // 2
    median = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2
    return sum(values) / len(values), median


problem = gx.problems.binary.RoyalRoad.r2()
optimum = problem.optimum.value
print(f"Royal road R2: 8 blocks of 8 bits and the levels above, maximum {optimum:.0f}")
print("a GA with single-point crossover, from seed 1")
print("generation  best")
# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(BITS, optimum)
last = -1.0


def progress(progress):
    global last
    best = progress.best_fitness or 0.0
    # the generations at which the best improves
    if best > last:
        print(f"{progress.generation:>10}  {best:>4.0f}")
        last = best
    trace.record(progress)


result = ga(problem, 1).run(problem, target=optimum, evaluations=BUDGET, on_generation=progress)
print(
    f"{result.best_fitness:.0f} after {result.generations} generations and "
    f"{result.evaluations} evaluations"
)

# the GA from seeds 1 to 50, and hill climbing from seeds 1 to 200
generations = []
for seed in range(1, RUNS + 1):
    result = ga(problem, seed).run(problem, target=optimum, evaluations=BUDGET)
    if result.stop_reason == "target":
        generations.append(result.generations)
mean, median = mean_and_median(generations)
print(
    f"\nGA, seeds 1 to {RUNS}: {len(generations)} reach {optimum:.0f}; mean {mean:.0f}, "
    f"median {median:.0f} generations"
)
print("  (the paper's GA, 50 runs: mean 590, median 542)")
evaluations = []
for seed in range(1, CLIMBS + 1):
    search = gx.LocalSearch(problem.genome, neighbor=gx.BitFlip(count=1), seed=seed)
    result = search.run(problem, target=optimum, evaluations=BUDGET)
    if result.stop_reason == "target":
        evaluations.append(result.evaluations)
mean, median = mean_and_median(evaluations)
print(
    f"hill climbing, seeds 1 to {CLIMBS}: {len(evaluations)} reach {optimum:.0f}; "
    f"mean {mean:.0f}, median {median:.0f} evaluations"
)
trace.write()
