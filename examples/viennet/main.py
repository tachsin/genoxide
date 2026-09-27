"""Viennet's three problems: minimize three objectives of two variables, with NSGA-III.

VNT1, VNT2 and VNT3 from genoxide's problems; run evaluates them in Rust. Each with the 91
reference directions of Das and Dennis's method with 12 divisions, a population of 92 and 30
generations. Prints, per problem, the size of the final front and its normalized hypervolume: the
hypervolume with the objectives mapped to [0, 1] by the ideal and nadir points, and the reference
point (1.1, 1.1, 1.1). For VNT1, whose front is known, also the IGD+ to 1,035 points of it.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its runs for the plot on the example's
page, with trace.py.

    python examples/viennet/main.py
"""

import numpy as np

import genoxide as gx

from trace import Trace

GENERATIONS = 30


def normalized_hypervolume(front, scale):
    """The hypervolume of ``front`` with the objectives mapped to [0, 1] by ``scale``, the ideal
    and nadir points, and the reference point (1.1, 1.1, 1.1)."""
    ideal, nadir = scale
    return gx.indicators.hypervolume((front - ideal) / (nadir - ideal), [1.1, 1.1, 1.1])


def solve(problem, scale, trace):
    """Runs NSGA-III on the problem, and prints the size of its front, the front's normalized
    hypervolume and, if the optimal front is known, the IGD+ to it."""
    nsga3 = gx.Nsga3(
        problem.genome,
        objectives=problem.objectives,
        reference_directions=gx.das_dennis(3, 12),
        population_size=92,
        crossover=gx.SimulatedBinaryCrossover(30),
        mutation=gx.PolynomialMutation(20, rate=0.5),
        seed=1,
    )
    on_generation = trace.fronts(problem.name, scale)
    result = nsga3.run(problem, generations=GENERATIONS, on_generation=on_generation)

    front = result.front_objectives
    volume = normalized_hypervolume(front, scale)
    # the IGD+ to at least 1,000 points of the optimal front: for VNT1, the images of 1,035 points
    # spread evenly over the triangle of its optimal solutions
    optimal = problem.optimal_front(1000)
    distance = "-" if optimal is None else f"{gx.indicators.igd_plus(front, optimal):.4f}"
    print(f"{problem.name:<7}  {len(front):>9}  {volume:>11.4f}  {distance:>6}")


# with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
trace = Trace(normalized_hypervolume)
print("problem  solutions  hypervolume  IGD+")
# VNT1's ideal and nadir points are known: its objectives' minima, and their worst values on the
# front, at the minima of the other two
vnt1 = gx.problems.Viennet1()
solve(vnt1, (vnt1.ideal_point, vnt1.nadir_point), trace)
# VNT2's and VNT3's aren't in genoxide. VNT2's objectives are convex: the worst values on its
# front are at the minima of the other two, as for VNT1. VNT3's nadir point is an estimate, from
# the non-dominated points of a 2,001 × 2,001 grid of the variables.
scale = (np.array([3.0, -17.0, -13.0]), np.array([4.2452, -16.4766, -12.0531]))
solve(gx.problems.Viennet2(), scale, trace)
scale = (np.array([0.0, 15.0, -0.1]), np.array([8.1964, 17.0370, 0.1762]))
solve(gx.problems.Viennet3(), scale, trace)
trace.write()
