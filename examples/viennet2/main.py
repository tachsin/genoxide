"""Viennet 2 (VNT2): minimize three convex quadratic objectives of two variables, whose Pareto
front is a curved triangle.

NSGA-III with the 91 reference directions of Das and Dennis's method with 12 divisions, and
SMS-EMOA, each with a population of 92 for 50 generations. Prints the size of each final front and
its hypervolume. run
evaluates the problem in Rust.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of the SMS-EMOA run for the plot on the
example's page, with trace.py.

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
# the whole front's hypervolume, about 0.7744
WHOLE = 0.7744


def run(name, algorithm, on_generation=None):
    """Runs ``algorithm`` for 50 generations, and reports its front."""
    front = algorithm.run(problem, generations=50, on_generation=on_generation).front_objectives
    volume = gx.indicators.hypervolume(front, REFERENCE)
    print(
        f"{name:<8} {len(front)} solutions, hypervolume {volume:.4f}, "
        f"{volume / WHOLE * 100:.1f}% of the whole front's"
    )


# polynomial mutation at a rate of 0.5, one of the two genes per child on average
settings = dict(
    objectives=problem.objectives,
    population_size=92,
    crossover=gx.SimulatedBinaryCrossover(30),
    mutation=gx.PolynomialMutation(20, rate=0.5),
    seed=1,
)
# 91 directions and a population of 92, the multiple of 4 above
run("NSGA-III", gx.Nsga3(problem.genome, reference_directions=gx.das_dennis(3, 12), **settings))
# with GENOXIDE_TRACE=<file>, a trace of this run for the plot on the example's page
trace = Trace(REFERENCE)
run("SMS-EMOA", gx.SmsEmoa(problem.genome, **settings), trace.on_generation)

print(f"the whole front: hypervolume {WHOLE}")
trace.write()
