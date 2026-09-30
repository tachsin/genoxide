"""Disc brake: minimize the mass of a multiple-disc brake and its stopping time, subject to five
constraints.

NSGA-II with a population of 100, simulated binary crossover and polynomial mutation at a rate of
1/4 per gene, for 250 generations. Prints how many solutions of the final front are feasible, the
range of each objective on it, and their hypervolume, as a share of that of genoxide's reference
front.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/disc_brake/main.py
"""

import genoxide as gx

from trace import REFERENCE, Trace, scaled

# the run's length
GENERATIONS = 250

problem = gx.problems.multi_engineering.DiscBrake()
nsga2 = gx.Nsga2(
    problem.genome,
    objectives=problem.objectives,
    population_size=100,
    crossover=gx.SimulatedBinaryCrossover(20),
    mutation=gx.PolynomialMutation(20, rate=0.25),
    seed=1,
)
# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(problem)
result = nsga2.run(problem, generations=GENERATIONS, on_generation=trace.on_generation)

feasible = result.front_violations == 0
front = result.front_objectives[feasible]
size = len(result.front_objectives)
count = "all feasible" if feasible.all() else f"{int(feasible.sum())} feasible"
print(f"NSGA-II, {GENERATIONS} generations: {size} solutions on the front, {count}")
low, high = front.min(axis=0), front.max(axis=0)
print(
    f"  mass from {low[0]:.4f} to {high[0]:.4f}, "
    f"stopping time from {low[1]:.4f} to {high[1]:.4f}"
)
found = scaled(problem, front)
volume = gx.indicators.hypervolume(found, [1.1, 1.1])
print(
    f"  hypervolume {volume:.4f}, {100 * volume / REFERENCE:.2f}% of the reference front's "
    f"{REFERENCE}"
)
trace.write()
