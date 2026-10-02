"""NK landscape: maximize a landscape of Kauffman and Weinberger's NK model, N = 20 bits each
interacting with K = 4 others chosen at random, with iterated local search, and check it against
the optimum found by evaluating all 2^20 strings.

The landscape is drawn from seed 1 with genoxide's portable random numbers. Iterated local search
flips one bit at a time, keeping changes that are no worse, and after 100 steps without a better
best restarts from the best, changed by 5 random flips. A run from seed 1 prints its improvements;
then, on the landscapes of seeds 1 to 5, runs from seeds 1 to 20 count how often it reaches the
optimum, against a genetic algorithm as a contrast. The landscapes are genoxide's
problems.binary.NkLandscape, which run evaluates in Rust, and whose optimum searches every string.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/nk_landscape/main.py
"""

import genoxide as gx

from trace import Trace

N = 20
K = 4
LANDSCAPES = 5
SEEDS = 20
# the most evaluations of a run of iterated local search, and generations of the GA
BUDGET = 500_000
GENERATIONS = 200


def ils(landscape, seed):
    """Iterated local search from ``seed``."""
    return gx.LocalSearch(
        landscape.genome, neighbor=gx.BitFlip(count=1), restart=(100, 5), seed=seed
    )


def median(values):
    """The median of ``values``."""
    values = sorted(values)
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def text(bits):
    """A bit string as 0s and 1s."""
    return "".join("1" if bit else "0" for bit in bits)


landscape = gx.problems.binary.NkLandscape(N, K, "random", seed=1)
optimum = landscape.optimum
print(f"NK landscape: N = {N}, K = {K}, random neighbors, drawn from seed 1")
print(
    f"the optimum, over all 2^{N} strings: {optimum.value:.6f} at {text(optimum.solutions[0])}"
)
print("iterated local search from seed 1")
print("evaluations  best")
# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(N, optimum.value)
last = 0.0


def progress(progress):
    global last
    best = progress.best_fitness or 0.0
    if best > last:
        print(f"{progress.evaluations:>11}  {best:.6f}")
        last = best
    trace.record(progress)


result = ils(landscape, 1).run(
    landscape, target=optimum.value, evaluations=BUDGET, on_generation=progress
)
print(
    f"{result.best_fitness:.6f} after {result.evaluations} evaluations, at "
    f"{text(result.best_genome)}"
)

# iterated local search and, as a contrast, a GA on landscapes 1 to 5, from seeds 1 to 20
print("\nlandscape  optimum   ILS reaches  median evaluations  GA reaches")
for landscape_seed in range(1, LANDSCAPES + 1):
    landscape = gx.problems.binary.NkLandscape(N, K, "random", seed=landscape_seed)
    optimum = landscape.optimum.value
    evaluations = []
    ga_reached = 0
    for seed in range(1, SEEDS + 1):
        result = ils(landscape, seed).run(landscape, target=optimum, evaluations=BUDGET)
        if result.stop_reason == "target":
            evaluations.append(result.evaluations)
        ga = gx.Ga(
            landscape.genome,
            population_size=500,
            select=gx.Tournament(2),
            crossover=gx.PointCrossover(2),
            mutation=gx.BitFlip(rate=1 / N),
            seed=seed,
        )
        result = ga.run(landscape, target=optimum, generations=GENERATIONS)
        ga_reached += result.stop_reason == "target"
    print(
        f"{landscape_seed:>9}  {optimum:.6f}  {len(evaluations):>8}/{SEEDS}  "
        f"{median(evaluations):>18.1f}  {ga_reached:>7}/{SEEDS}"
    )
trace.write()
