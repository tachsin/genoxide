"""Royal road R1: maximize 8 blocks of 8 ones, each scoring only when complete, with
random-mutation hill climbing, which Mitchell, Holland and Forrest (1994) found faster on it than
their genetic algorithm.

Random-mutation hill climbing (RMHC) flips one bit, chosen at random, and keeps the change if it's
no worse: genoxide's LocalSearch with BitFlip(count=1). A run from seed 1 prints the evaluations
at which each block is completed; then 200 runs, as in the paper's Table 1, give the mean and
median evaluations to the optimum, 64. As a comparison, a genetic algorithm with the paper's
population, crossover and mutation (and tournament selection) from 50 seeds. The function is
genoxide's problems.binary.RoyalRoad.r1(), which run evaluates in Rust.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/royal_road_r1/main.py
"""

import genoxide as gx

from trace import Trace

BITS = 64
# the runs of RMHC, as in the paper's Table 1, and of the genetic algorithm
RUNS = 200
GA_RUNS = 50
# the most evaluations of a run, the paper's
BUDGET = 256_000


def rmhc(problem, seed):
    """Random-mutation hill climbing from ``seed``."""
    return gx.LocalSearch(problem.genome, neighbor=gx.BitFlip(count=1), seed=seed)


def mean_and_median(values):
    """The mean and the median of ``values``."""
    values = sorted(values)
    middle = len(values) // 2
    median = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2
    return sum(values) / len(values), median


problem = gx.problems.binary.RoyalRoad.r1()
optimum = problem.optimum.value
print(f"Royal road R1: 8 blocks of 8 bits, maximum {optimum:.0f}")
print("random-mutation hill climbing from seed 1")
print("R1  evaluations")
# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(BITS, optimum)
last = 0.0


def progress(progress):
    global last
    best = progress.best_fitness or 0.0
    # the evaluations at which a block is completed
    if best > last:
        print(f"{best:>2.0f}  {progress.evaluations:>11}")
        last = best
    trace.record(progress)


result = rmhc(problem, 1).run(problem, target=optimum, evaluations=BUDGET, on_generation=progress)
print(f"{result.best_fitness:.0f} after {result.evaluations} evaluations")

# RMHC from seeds 1 to 200, and the genetic algorithm from seeds 1 to 50
evaluations = []
for seed in range(1, RUNS + 1):
    result = rmhc(problem, seed).run(problem, target=optimum, evaluations=BUDGET)
    if result.stop_reason == "target":
        evaluations.append(result.evaluations)
mean, median = mean_and_median(evaluations)
print(
    f"\nRMHC, seeds 1 to {RUNS}: {len(evaluations)} reach {optimum:.0f}; mean {mean:.0f}, "
    f"median {median:.0f} evaluations"
)
print("  (the paper's 200 runs: mean 6179, median 5775)")
evaluations = []
for seed in range(1, GA_RUNS + 1):
    ga = gx.Ga(
        problem.genome,
        population_size=128,
        select=gx.Tournament(2),
        crossover=gx.PointCrossover(1),
        crossover_rate=0.7,
        mutation=gx.BitFlip(rate=0.005),
        seed=seed,
    )
    result = ga.run(problem, target=optimum, evaluations=BUDGET)
    if result.stop_reason == "target":
        evaluations.append(result.evaluations)
mean, median = mean_and_median(evaluations)
print(
    f"GA, seeds 1 to {GA_RUNS}: {len(evaluations)} reach {optimum:.0f}; mean {mean:.0f}, "
    f"median {median:.0f} evaluations"
)
print("  (the paper's GA, 200 runs: mean 61334, median 54208)")
trace.write()
