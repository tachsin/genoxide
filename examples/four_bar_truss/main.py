"""Four-bar truss: minimize the volume of a truss of four bars and the displacement of its loaded
joint, whose front has three pieces.

NSGA-II with a population of 100, simulated binary crossover and polynomial mutation at a rate of
1/4 per gene, for 250 generations. Prints how many solutions of the final front are feasible, the
range of each objective on it, their IGD+ to the optimal front and their hypervolume.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/four_bar_truss/main.py
"""

import genoxide as gx

from trace import Trace, scaled, whole_front_hypervolume

# the run's length
GENERATIONS = 250

problem = gx.problems.multi_engineering.FourBarTruss()
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
    f"  volume from {low[0]:.2f} to {high[0]:.2f}, "
    f"displacement from {low[1]:.6f} to {high[1]:.6f}"
)
found = scaled(problem, front)
volume = gx.indicators.hypervolume(found, [1.1, 1.1])
optimal = scaled(problem, problem.optimal_front(2000))
distance = gx.indicators.igd_plus(found, optimal)
whole = whole_front_hypervolume(problem)
print(
    f"  IGD+ {distance:.5f}, hypervolume {volume:.4f}, {100 * volume / whole:.2f}% of the whole "
    f"front's {whole:.4f}"
)
trace.write()
