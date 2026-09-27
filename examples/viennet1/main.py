"""Viennet 1 (VNT1): minimize three objectives of two variables, the squared distances to three
points plus constants, whose Pareto front is a curved triangle.

NSGA-III with the 91 reference directions of Das and Dennis's method with 12 divisions, a
population of 92 and 50 generations. Prints the size of the final front, its hypervolume and its
IGD+ to 1,035 points of the optimal front. run evaluates the problem in Rust.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/viennet1/main.py
"""

import genoxide as gx

from trace import Trace

# the reference point of the hypervolume: the nadir point (4, 5, 4) plus a tenth of each
# objective's range on the front, whose ideal point is (0, 1, 2)
REFERENCE = [4.4, 5.4, 4.2]

problem = gx.problems.Viennet1()
nsga3 = gx.Nsga3(
    problem.genome,
    objectives=problem.objectives,
    reference_directions=gx.das_dennis(3, 12),
    population_size=92,
    crossover=gx.SimulatedBinaryCrossover(30),
    mutation=gx.PolynomialMutation(20, rate=0.5),
    seed=1,
)
# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(REFERENCE)
result = nsga3.run(problem, generations=50, on_generation=trace.on_generation)

# the hypervolume of the front, and its IGD+ to the images of 1,035 points spread evenly over the
# triangle of optimal solutions
front = result.front_objectives
volume = gx.indicators.hypervolume(front, REFERENCE)
distance = gx.indicators.igd_plus(front, problem.optimal_front(1000))
print(f"{len(front)} solutions on the front, hypervolume {volume:.4f} (the whole front: 33.52)")
print(f"IGD+ to the optimal front {distance:.4f}")
trace.write()
