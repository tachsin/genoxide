"""Inverted DTLZ1: minimize three objectives over 7 variables, whose Pareto front is DTLZ1's
triangle turned upside down, with usual directions and inverted directions.

From genoxide's problems.InvertedDtlz1; run evaluates it in Rust. Runs usual directions and inverted
directions, and prints each final front's size, its hypervolume with the objectives divided by the
front's nadir point, as a share of that of a sample of the optimal front with as many points as
there are reference directions, and the median and largest distance g of its solutions from the
front; then the hypervolumes of the whole front and of the sample.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its runs for the plot on the example's
page, with trace.py.

    python examples/inverted_dtlz1/main.py
"""

import numpy as np

import genoxide as gx

from trace import Trace

# the reference point of the hypervolume, with the objectives divided by the nadir point
REFERENCE = [1.1, 1.1, 1.1]

problem = gx.problems.InvertedDtlz1()
ideal, nadir = problem.ideal_point, problem.nadir_point


def normalized(points):
    """The objectives divided by the front's nadir point (its ideal point is the origin)."""
    return (points - ideal) / (nadir - ideal)


# with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
trace = Trace(problem, normalized, REFERENCE)
# 91 reference directions: Das and Dennis's points with 12 divisions
directions = gx.das_dennis(3, 12)
# the same directions turned upside down, as the front is: (1 − d)/2
inverted = (1 - directions) / 2
# the sample: as many points of the optimal front, what a front of 91 solutions can be
sample = normalized(problem.optimal_front(len(directions)))


def run(name, algorithm, generations):
    """Runs ``algorithm`` for ``generations``, and prints its final front's size, its hypervolume,
    as a share of the sample's, and the median and largest distance g of its solutions from the
    front."""
    result = algorithm.run(problem, generations=generations, on_generation=trace.front(name))
    front = result.front_objectives
    volume = gx.indicators.hypervolume(normalized(front), REFERENCE)
    percent = 100 * volume / gx.indicators.hypervolume(sample, REFERENCE)
    # g, from f₁ + f₂ + f₃ = 1 + g: the median (the middle value, or the upper of the two middle
    # ones) and the largest
    P = normalized(front)
    g = np.sort(P.sum(axis=1) / 2 - 1)
    median, largest = g[len(g) // 2], g[-1]
    print(
        f"{name}, {generations} generations: {len(front)} solutions, "
        f"hypervolume {volume:.4f}, {percent:.1f}% of the sample's, "
        f"g {median:.5f} (median) to {largest:.5f}"
    )


# NSGA-III with Das and Dennis's 91 directions, Deb and Jain's settings
algorithm = gx.Nsga3(
    problem.genome,
    objectives=problem.objectives,
    reference_directions=directions,
    population_size=92,
    crossover=gx.SimulatedBinaryCrossover(30),
    mutation=gx.PolynomialMutation(20, rate=1 / 7),
    seed=1,
)
run("usual directions", algorithm, 2000)
# NSGA-III with the same directions turned upside down, (1 − d)/2
algorithm = gx.Nsga3(
    problem.genome,
    objectives=problem.objectives,
    reference_directions=inverted,
    population_size=92,
    crossover=gx.SimulatedBinaryCrossover(30),
    mutation=gx.PolynomialMutation(20, rate=1 / 7),
    seed=1,
)
run("inverted directions", algorithm, 2000)
whole = normalized(problem.optimal_front(3_000))
print(
    f"the whole front: hypervolume {gx.indicators.hypervolume(whole, REFERENCE):.4f}; the "
    f"sample of {len(sample)} of its points: "
    f"{gx.indicators.hypervolume(sample, REFERENCE):.4f}"
)
trace.write()
