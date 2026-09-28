"""Viennet 2 (VNT2): minimize three convex quadratic objectives of two variables, whose Pareto
front is a curved triangle.

NSGA-III with the 91 reference directions of Das and Dennis's method with 12 divisions, a
population of 92 and 50 generations. Prints the size of the final front and its hypervolume. run
evaluates the problem in Rust.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/viennet2/main.py
"""

import numpy as np

import genoxide as gx

from trace import Trace

problem = gx.problems.Viennet2()
# the reference point of the hypervolume: the nadir point (4.2452, −16.4766, −12.0531) plus
# a tenth of each objective's range on the front, from the ideal point (3, −17, −13),
# rounded to 4 decimals: (4.3697, −16.4242, −11.9584)
ideal, nadir = problem.ideal_point, problem.nadir_point
REFERENCE = (np.round((nadir + (nadir - ideal) / 10) * 1e4) / 1e4).tolist()
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

# the hypervolume of the front; the whole front's is about 0.7744
front = result.front_objectives
volume = gx.indicators.hypervolume(front, REFERENCE)
print(f"{len(front)} solutions on the front, hypervolume {volume:.4f} (the whole front: 0.7744)")
trace.write()
