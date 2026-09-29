"""Scaled DTLZ2: minimize three objectives over 12 variables, whose objectives range over 1, 10 and
100, with NSGA-III and MOEA/D.

From genoxide's problems.ScaledDtlz2; run evaluates it in Rust. Runs NSGA-III and MOEA/D, and prints
each final front's size, its hypervolume with the objectives divided by the front's nadir point, as
a share of that of a sample of the optimal front with as many points as there are reference
directions, and the median and largest distance g of its solutions from the front; then the
hypervolumes of the whole front and of the sample.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its runs for the plot on the example's
page, with trace.py.

    python examples/scaled_dtlz2/main.py
"""

import numpy as np

import genoxide as gx

from trace import Trace

# the reference point of the hypervolume, with the objectives divided by the nadir point
REFERENCE = [1.1, 1.1, 1.1]

problem = gx.problems.ScaledDtlz2()
ideal, nadir = problem.ideal_point, problem.nadir_point


def normalized(points):
    """The objectives divided by the front's nadir point (its ideal point is the origin)."""
    return (points - ideal) / (nadir - ideal)


# with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
trace = Trace(problem, normalized, REFERENCE)
# 91 reference directions: Das and Dennis's points with 12 divisions
directions = gx.das_dennis(3, 12)
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
    # g, from Σ (fᵢ / 10^(i − 1))² = (1 + g)²: the median (the middle value, or the upper of the two
    # middle ones) and the largest
    P = normalized(front)
    g = np.sort(np.sqrt((P * P).sum(axis=1)) - 1)
    median, largest = g[len(g) // 2], g[-1]
    print(
        f"{name}, {generations} generations: {len(front)} solutions, "
        f"hypervolume {volume:.4f}, {percent:.1f}% of the sample's, "
        f"g {median:.5f} (median) to {largest:.5f}"
    )


# NSGA-III with Deb and Jain's settings for 3 objectives
algorithm = gx.Nsga3(
    problem.genome,
    objectives=problem.objectives,
    reference_directions=directions,
    population_size=92,
    crossover=gx.SimulatedBinaryCrossover(30),
    mutation=gx.PolynomialMutation(20, rate=1 / 12),
    seed=1,
)
run("NSGA-III", algorithm, 500)
# MOEA/D with Tchebycheff decomposition and the same 91 directions as weights
algorithm = gx.Moead(
    problem.genome,
    objectives=problem.objectives,
    weights=directions,
    crossover=gx.SimulatedBinaryCrossover(20),
    mutation=gx.PolynomialMutation(20, rate=1 / 12),
    seed=1,
)
run("MOEA/D", algorithm, 500)
whole = normalized(problem.optimal_front(3_000))
print(
    f"the whole front: hypervolume {gx.indicators.hypervolume(whole, REFERENCE):.4f}; the "
    f"sample of {len(sample)} of its points: "
    f"{gx.indicators.hypervolume(sample, REFERENCE):.4f}"
)
trace.write()
