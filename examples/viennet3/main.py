"""Viennet 3 (VNT3): minimize three objectives of two variables, two of which depend only on the
distance from the origin, so that the Pareto front is two curves.

NSGA-III with the 91 reference directions of Das and Dennis's method with 12 divisions, a
population of 92 and 50 generations. Prints the size of the final front, how many of its
solutions are on each of the two curves, and its hypervolume. run evaluates the problem in Rust.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/viennet3/main.py
"""

import numpy as np

import genoxide as gx

from trace import Trace

# the reference point of the hypervolume: the nadir point (8.1964, 17.0370, 0.1760) plus a tenth
# of each objective's range on the front, whose ideal point is (0, 15, −0.1)
REFERENCE = [9.0160, 17.2407, 0.2036]

problem = gx.problems.Viennet3()
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

# the optimal solutions are on two curves: one where x₁² + x₂² ≤ 1.5, near the origin, and one
# where x₁² + x₂² ≥ 4.19
inner = int(np.sum(np.sum(result.front_genomes**2, axis=1) < 3.0))
front = result.front_objectives
outer = len(front) - inner
print(f"{len(front)} solutions on the front: {inner} near the origin, {outer} farther out")
# the hypervolume of the front; the whole front's is about 5.3255
volume = gx.indicators.hypervolume(front, REFERENCE)
print(f"hypervolume {volume:.4f} (the whole front: 5.3255)")
trace.write()
