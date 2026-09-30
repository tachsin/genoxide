"""Car side impact, three objectives: minimize a car's weight, the pubic force on a passenger and
the mean velocity of the B-pillar and the front door in a side impact, subject to ten constraints.

NSGA-III with Jain and Deb's settings: the 153 reference directions of Das and Dennis's method with
16 divisions, a population of 156, for 500 generations. Prints how many solutions of the final
front are feasible, the range of each objective on it, and its hypervolume, as a share of that of
genoxide's reference front.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/car_side_impact_3obj/main.py
"""

import genoxide as gx

from trace import REFERENCE, Trace, scaled

# the run's length
GENERATIONS = 500

problem = gx.problems.multi_engineering.CarSideImpact()
# 16 divisions: the reference directions of Das and Dennis's method
nsga3 = gx.Nsga3(
    problem.genome,
    objectives=problem.objectives,
    reference_directions=gx.das_dennis(3, 16),
    population_size=156,
    crossover=gx.SimulatedBinaryCrossover(30),
    mutation=gx.PolynomialMutation(20, rate=1 / 7),
    seed=1,
)
# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(problem)
result = nsga3.run(problem, generations=GENERATIONS, on_generation=trace.on_generation)

feasible = result.front_violations == 0
front = result.front_objectives[feasible]
size = len(result.front_objectives)
count = "all feasible" if feasible.all() else f"{int(feasible.sum())} feasible"
print(f"NSGA-III, {GENERATIONS} generations: {size} solutions on the front, {count}")
low, high = front.min(axis=0), front.max(axis=0)
print(
    f"  weight from {low[0]:.3f} to {high[0]:.3f}, "
    f"pubic force from {low[1]:.4f} to {high[1]:.4f}, "
    f"mean velocity from {low[2]:.4f} to {high[2]:.4f}"
)
volume = gx.indicators.hypervolume(scaled(problem, front), [1.1, 1.1, 1.1])
print(
    f"  hypervolume {volume:.4f}, {100 * volume / REFERENCE:.2f}% of the reference front's "
    f"{REFERENCE}"
)
trace.write()
