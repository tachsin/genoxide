"""Water resource planning: minimize five costs and losses of a storm drainage system, subject to
seven constraints.

SPEA2 with a population of 212, simulated binary crossover and polynomial mutation at a rate of 1/3
per gene, for 500 generations. Prints how many solutions of the final front are feasible, how far
they are from the optimal solutions in x₃, and their IGD+ to the optimal front.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/water_resource_planning/main.py
"""

import genoxide as gx

from trace import Trace, optimal_front, scaled

# the run's length
GENERATIONS = 500

problem = gx.problems.multi_engineering.WaterResourcePlanning()
spea2 = gx.Spea2(
    problem.genome,
    objectives=problem.objectives,
    population_size=212,
    crossover=gx.SimulatedBinaryCrossover(20),
    mutation=gx.PolynomialMutation(20, rate=1 / 3),
    seed=1,
)
# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(problem)
result = spea2.run(problem, generations=GENERATIONS, on_generation=trace.on_generation)

feasible = result.front_violations == 0
size = len(result.front_objectives)
count = "all feasible" if feasible.all() else f"{int(feasible.sum())} feasible"
print(f"SPEA2, {GENERATIONS} generations: {size} solutions on the front, {count}")
# the optimal solutions have x₃ = 0.01, its lower bound
highest = result.front_genomes[feasible, 2].max()
print(f"  x₃ at most {highest:.5f}, where the optimal solutions have 0.01")
front = scaled(problem, result.front_objectives[feasible])
reference = optimal_front(problem)
distance = gx.indicators.igd_plus(front, reference)
print(f"  IGD+ {distance:.5f} to {len(reference)} points of the optimal front")
trace.write()
