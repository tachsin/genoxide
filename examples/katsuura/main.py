"""Katsuura: minimize Katsuura's function, rugged everywhere, in 10 dimensions.

Runs CMA-ES without and with IPOP restarts (a population that doubles at each restart),
differential evolution (SHADE), particle swarm optimization and a real-coded genetic algorithm from
10 seeds each, and counts the runs that reach the minimum, 0 at the origin, to within 1e-8. The
function is genoxide's `problems::Katsuura`, which run evaluates in Rust.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of a run for the plot on the example's page,
with trace.py.

    python examples/katsuura/main.py
"""

import genoxide as gx

from trace import record_small

DIMENSIONS = 10
SEEDS = 10
BUDGET = 10_000 * DIMENSIONS
# a run stops once its error to the minimum is at most this
ERROR = 1e-8
ALGORITHMS = ["CMA-ES", "CMA-ES with IPOP", "DE", "PSO", "GA"]


def build(name, genome, seed):
    """The algorithm called ``name``, on ``genome``, from ``seed``."""
    if name == "CMA-ES":
        return gx.Cmaes(genome, objective="minimize", seed=seed)
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
print(f"Katsuura in {DIMENSIONS} dimensions, {SEEDS} seeds, {BUDGET} evaluations at most per run")
print("algorithm         at min  evaluations  median error")
for name in ALGORITHMS:
    # the evaluations of the runs that reach the minimum, and every run's best error
    evaluations, errors = [], []
    for seed in range(1, SEEDS + 1):
        result = build(name, problem.genome, seed).run(
            problem, target=minimum + ERROR, evaluations=BUDGET
        )
        if result.stop_reason == "target":
            evaluations.append(float(result.evaluations))
        # rounding can put a solution a few ulps below the minimum
        errors.append(max(result.best_fitness - minimum, 0.0))
    reached = f"{len(evaluations)}/{SEEDS}"
    middle = median(evaluations)
    evaluations_text = "-" if middle is None else f"{middle:.0f}"
    print(f"{name:<16}  {reached:>6}  {evaluations_text:>11}  {error_text(median(errors)):>12}")
print("evaluations: the median of the runs that reach the minimum")

# with GENOXIDE_TRACE=<file>, a trace for the plot on the example's page, of a separate run in
# 2 dimensions: the plot is the function's contour
record_small()
