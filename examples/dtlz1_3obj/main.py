"""DTLZ1 with 3 objectives: minimize three conflicting objectives over 7 variables in [0, 1], whose
Pareto front is the plane f₁ + f₂ + f₃ = 0.5, behind 11⁵ − 1 local fronts.

NSGA-III with the 91 reference directions of Das and Dennis's method with 12 divisions, a
population of 92 and 400 generations, as in Deb and Jain (2014). Prints the size of the final
front, its hypervolume, its IGD+ to 1,035 points of the optimal front, and how far its farthest
solution is from the front. run evaluates the problem in Rust.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/dtlz1_3obj/main.py
"""

import genoxide as gx

from trace import Trace

VARIABLES = 7

# the reference point of the hypervolume: 1.1 times the nadir point (0.5, 0.5, 0.5)
REFERENCE = [0.55, 0.55, 0.55]

problem = gx.problems.Dtlz1(objectives=3, variables=VARIABLES)
nsga3 = gx.Nsga3(
    problem.genome,
    objectives=problem.objectives,
    reference_directions=gx.das_dennis(3, 12),
    population_size=92,
    crossover=gx.SimulatedBinaryCrossover(30),
    mutation=gx.PolynomialMutation(20, rate=1 / VARIABLES),
    seed=1,
)
# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(REFERENCE)
result = nsga3.run(problem, generations=400, on_generation=trace.on_generation)

# the hypervolume of the front; the whole front's is 0.55³ minus the corner that the plane cuts
# off, 0.5³ / 6
front = result.front_objectives
volume = gx.indicators.hypervolume(front, REFERENCE)
print(f"{len(front)} solutions on the front, hypervolume {volume:.4f} (the whole front: 0.1455)")
# IGD+ to 1,035 points spread evenly over the plane
distance = gx.indicators.igd_plus(front, problem.optimal_front(1000))
print(f"IGD+ to the optimal front {distance:.5f}")
# the objectives sum to (1 + g) / 2: g is 0 on the front
farthest = max(0.0, (2 * front.sum(axis=1) - 1).max())
print(f"the largest g on the front {farthest:.5f}")
trace.write()
