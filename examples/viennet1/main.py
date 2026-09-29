"""Viennet 1 (VNT1): minimize three objectives of two variables, the squared distances to three
points plus constants, whose Pareto front is a curved triangle.

NSGA-III twice, for 50 generations: with the 91 reference directions of Das and Dennis's method
with 12 divisions and a population of 92, and with 496 directions, 30 divisions, and as many
solutions. Prints each final front's size, hypervolume and IGD+ to 1,035 points of the optimal
front. run evaluates the problem in Rust.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of the second run for the plot on the
example's page, with trace.py.

    python examples/viennet1/main.py
"""

import genoxide as gx

from trace import Trace

# the reference point of the hypervolume: the nadir point (4, 5, 4) plus a tenth of each
# objective's range on the front, whose ideal point is (0, 1, 2)
REFERENCE = [4.4, 5.4, 4.2]

problem = gx.problems.Viennet1()
# the images of 1,035 points spread evenly over the triangle of optimal solutions, for IGD+
optimal = problem.optimal_front(1000)
# the front's range, from the ideal point (0, 1, 2) to the nadir point (4, 5, 4), to scale each
# objective to [0, 1] for the scaled IGD+
ideal, nadir = problem.ideal_point, problem.nadir_point


def run(name, divisions, population, on_generation=None):
    """Runs NSGA-III with the directions of Das and Dennis's method with ``divisions`` for 50
    generations, and reports its front."""
    nsga3 = gx.Nsga3(
        problem.genome,
        objectives=problem.objectives,
        reference_directions=gx.das_dennis(3, divisions),
        population_size=population,
        crossover=gx.SimulatedBinaryCrossover(30),
        mutation=gx.PolynomialMutation(20, rate=0.5),
        seed=1,
    )
    result = nsga3.run(problem, generations=50, on_generation=on_generation)

    front = result.front_objectives
    volume = gx.indicators.hypervolume(front, REFERENCE)
    distance = gx.indicators.igd_plus(front, optimal)
    scaled = gx.indicators.igd_plus(
        (front - ideal) / (nadir - ideal), (optimal - ideal) / (nadir - ideal)
    )
    print(
        f"{name:<14} {len(front)} solutions, hypervolume {volume:.4f}, IGD+ {distance:.4f} "
        f"(scaled {scaled:.4f})"
    )


# 91 directions and a population of 92, the multiple of 4 above
run("91 directions", 12, 92)

# with GENOXIDE_TRACE=<file>, a trace of this run for the plot on the example's page
trace = Trace(REFERENCE)
# 496 directions, and a solution for each
run("496 directions", 30, None, trace.on_generation)

# the whole front's hypervolume, from a 4,001 × 4,001 grid of the variables
print("the whole front: hypervolume 33.52")
trace.write()
