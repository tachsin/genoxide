"""Deceptive trap: maximize 10 blocks of Deb and Goldberg's trap function of 4 bits, which leads
each block away from its optimum, with a genetic algorithm whose two-point crossover keeps the
blocks together.

A block of 4 bits scores 3 − u for u ones below 4, and 4 with all of them: fully deceptive, every
schema of order below 4 favoring all zeros. A run from seed 1 prints each generation; then runs
from seeds 1 to 20 count how often the GA reaches the optimum, 40. As contrasts: the same GA with
uniform crossover, which breaks the blocks apart, and hill climbing, one bit at a time, which
climbs to the deceptive attractor. The function is genoxide's problems.binary.Trap, which run
evaluates in Rust.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/deceptive_trap/main.py
"""

import genoxide as gx

from trace import Trace

BLOCKS = 10
K = 4
BITS = BLOCKS * K
SEEDS = 20
GENERATIONS = 300
# the evaluations of a hill climb
CLIMB = 10_000


def ga(problem, crossover, seed):
    """The genetic algorithm with ``crossover``, from ``seed``."""
    return gx.Ga(
        problem.genome,
        population_size=1000,
        select=gx.Tournament(4),
        crossover=crossover,
        mutation=gx.BitFlip(rate=1 / BITS),
        seed=seed,
    )


def blocks_of_ones(genome):
    """The blocks of ``genome`` that are all ones."""
    return sum(all(genome[block * K : (block + 1) * K]) for block in range(BLOCKS))


def median(values):
    """The median of ``values``."""
    values = sorted(values)
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


problem = gx.problems.binary.Trap(BLOCKS, K)
optimum = problem.optimum.value
print(f"Deceptive trap: {BLOCKS} blocks of {K} bits, maximum {optimum:.0f}")
print("a GA with two-point crossover, from seed 1")
print("generation  best  median  blocks of ones in the best")
# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(BITS, optimum)


def progress(progress):
    print(
        f"{progress.generation:>10}  {progress.best_fitness:>4.0f}  "
        f"{median(progress.scores):>6g}  {blocks_of_ones(progress.best_genome):>2}"
    )
    trace.record(progress)


result = ga(problem, gx.PointCrossover(2), 1).run(
    problem, target=optimum, generations=GENERATIONS, on_generation=progress
)
print(
    f"{result.best_fitness:.0f} after {result.generations} generations and "
    f"{result.evaluations} evaluations"
)

# two-point crossover from seeds 1 to 20, and uniform crossover as a contrast
reached = []
for seed in range(1, SEEDS + 1):
    result = ga(problem, gx.PointCrossover(2), seed).run(
        problem, target=optimum, generations=GENERATIONS
    )
    if result.stop_reason == "target":
        reached.append(result.generations)
print(
    f"\nseeds 1 to {SEEDS}, two-point crossover: {len(reached)} of {SEEDS} reach {optimum:.0f}, "
    f"after a median of {median(reached):.1f} generations"
)
uniform, bests = 0, []
for seed in range(1, SEEDS + 1):
    result = ga(problem, gx.UniformCrossover(), seed).run(
        problem, target=optimum, generations=GENERATIONS
    )
    uniform += result.stop_reason == "target"
    bests.append(blocks_of_ones(result.best_genome))
print(
    f"contrast, uniform crossover: {uniform} of {SEEDS} reach it in {GENERATIONS} generations; "
    f"a median of {median(bests):.1f} blocks of ones"
)

# hill climbing: one bit flipped at a time, kept if no worse
scores, blocks = [], []
for seed in range(1, SEEDS + 1):
    search = gx.LocalSearch(problem.genome, neighbor=gx.BitFlip(count=1), seed=seed)
    result = search.run(problem, target=optimum, evaluations=CLIMB)
    scores.append(result.best_fitness)
    blocks.append(blocks_of_ones(result.best_genome))
print(
    f"contrast, hill climbing ({CLIMB} evaluations): a median best of {median(scores):.1f}, "
    f"with {median(blocks):.1f} blocks of ones"
)
trace.write()
