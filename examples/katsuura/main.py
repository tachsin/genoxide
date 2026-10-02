"""Katsuura: minimize Katsuura's function, rugged everywhere, in 10 dimensions.

Runs CMA-ES with IPOP restarts (a population that doubles at each restart) from 10 seeds, with a
budget of 500,000 evaluations per run, and counts the runs that reach the minimum, 0 at the origin,
to within 1e-8. Then, as contrasts with the budget of 100,000 evaluations of the other functions'
pages: CMA-ES without restarts, differential evolution (SHADE), particle swarm optimization, which
reaches the minimum only by stopping at the bounds, and a real-coded genetic algorithm. The function
is genoxide's `problems::Katsuura`, which run evaluates in Rust. The runs evaluate in parallel, with
the same results on any number of threads.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of a run for the plot on the example's page,
with trace.py.

    python examples/katsuura/main.py
"""

import genoxide as gx

from trace import record_small

DIMENSIONS = 10
SEEDS = 10
# a run stops once its error to the minimum is at most this
ERROR = 1e-8
# the algorithms and their budgets of evaluations per run: the main method's, enough for every
# seed, then the contrasts', the 10,000 per dimension of the other functions' pages
ALGORITHMS = [
    ("CMA-ES with IPOP", 50_000 * DIMENSIONS),
    ("CMA-ES", 10_000 * DIMENSIONS),
    ("DE", 10_000 * DIMENSIONS),
    ("PSO", 10_000 * DIMENSIONS),
    ("GA", 10_000 * DIMENSIONS),
]


def build(name, genome, seed):
    """The algorithm called ``name``, on ``genome``, from ``seed``."""
    if name == "CMA-ES":
        # without restarts, the run ends once it has converged: sampling on around its point
        # wouldn't change its best
        return gx.Cmaes(genome, restarts="stop", objective="minimize", seed=seed)
    if name == "CMA-ES with IPOP":
        return gx.Cmaes(genome, restarts="ipop", objective="minimize", seed=seed)
    if name == "DE":
        return gx.De(genome, objective="minimize", seed=seed)
    if name == "PSO":
        return gx.Pso(genome, population_size=40, objective="minimize", seed=seed)
    return gx.Ga(
        genome,
        population_size=100,
        select=gx.Tournament(3),
        crossover=gx.SimulatedBinaryCrossover(15.0),
        mutation=gx.PolynomialMutation(20.0, rate=1 / DIMENSIONS),
        objective="minimize",
        seed=seed,
    )


def median(values):
    """The median of ``values``, None without any."""
    values = sorted(values)
    middle = len(values) // 2
    if not values:
        return None
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def error_text(error):
    """An error to two significant digits, as Rust writes it: 9.9e-9."""
    mantissa, exponent = f"{error:.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


problem = gx.problems.Katsuura(DIMENSIONS)
minimum = problem.optimum.value
print(f"Katsuura in {DIMENSIONS} dimensions, {SEEDS} seeds")
print("algorithm          budget  at min  evaluations  median error")
corners = 0
fewest = most = float("nan")
for name, budget in ALGORITHMS:
    # the evaluations of the runs that reach the minimum, and every run's best error
    evaluations, errors = [], []
    for seed in range(1, SEEDS + 1):
        result = build(name, problem.genome, seed).run(
            problem, target=minimum + ERROR, evaluations=budget, parallel=True
        )
        if result.stop_reason == "target":
            evaluations.append(float(result.evaluations))
        # rounding can put a solution a few ulps below the minimum
        errors.append(max(result.best_fitness - minimum, 0.0))
        # a corner of the box, where every gene is at a bound
        if name == "PSO" and all(abs(x) == 5.0 for x in result.best_genome):
            corners += 1
    reached = f"{len(evaluations)}/{SEEDS}"
    if name == "CMA-ES with IPOP":
        # the main method's fewest and most evaluations to the minimum
        fewest, most = min(evaluations), max(evaluations)
    middle = median(evaluations)
    evaluations_text = "-" if middle is None else f"{middle:.0f}"
    error = error_text(median(errors))
    print(f"{name:<16}  {budget:>7}  {reached:>6}  {evaluations_text:>11}  {error:>12}")
print("evaluations: the median of the runs that reach the minimum")
print(f"CMA-ES with IPOP: from {fewest:.0f} to {most:.0f} evaluations to the minimum")
print(f"PSO: {corners} of its {SEEDS} runs end on a corner of the box, every gene at a bound")

# with GENOXIDE_TRACE=<file>, a trace for the plot on the example's page, of a separate run in
# 2 dimensions: the plot is the function's contour
record_small()
