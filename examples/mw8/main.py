"""MW8: minimize three objectives over 15 variables, subject to one constraint, whose Pareto front
is the unit sphere in four bands, with η = 20 and η = 2.

From genoxide's problems.Mw8; run evaluates it in Rust. Runs η = 20 and η = 2, and prints each final
front's size, its hypervolume with the objectives divided by the front's nadir point, as a share of
that of a sample of the optimal front with as many points as there are reference directions, and the
median and largest distance g of its solutions from the front; then the hypervolumes of the whole
front and of the sample.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its runs for the plot on the example's
page, with trace.py.

    python examples/mw8/main.py
"""

import numpy as np

import genoxide as gx

from trace import Trace

# the reference point of the hypervolume, with the objectives divided by the nadir point
REFERENCE = [1.1, 1.1, 1.1]

problem = gx.problems.Mw8()
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
    # g₂, from f₁² + f₂² + f₃² = g₂²: the median (the middle value, or the upper of the two middle
    # ones) and the largest
    P = normalized(front)
    g = np.sort(np.sqrt((P * P).sum(axis=1)))
    median, largest = g[len(g) // 2], g[-1]
    print(
        f"{name}, {generations} generations: {len(front)} solutions, "
        f"hypervolume {volume:.4f}, {percent:.1f}% of the sample's, "
        f"g {median:.5f} (median) to {largest:.5f}"
    )


# NSGA-III with the paper's settings: polynomial mutation with η = 20, 600 generations
algorithm = gx.Nsga3(
    problem.genome,
    objectives=problem.objectives,
    reference_directions=directions,
    population_size=92,
    crossover=gx.SimulatedBinaryCrossover(20),
    mutation=gx.PolynomialMutation(20, rate=1 / 15),
    seed=1,
)
run("η = 20", algorithm, 600)
# NSGA-III with Deb and Jain's crossover (η = 30) and mutation with η = 2, for longer
algorithm = gx.Nsga3(
    problem.genome,
    objectives=problem.objectives,
    reference_directions=directions,
    population_size=92,
    crossover=gx.SimulatedBinaryCrossover(30),
    mutation=gx.PolynomialMutation(2, rate=1 / 15),
    seed=1,
)
run("η = 2", algorithm, 2000)
whole = normalized(problem.optimal_front(3_000))
print(
    f"the whole front: hypervolume {gx.indicators.hypervolume(whole, REFERENCE):.4f}; the "
    f"sample of {len(sample)} of its points: "
    f"{gx.indicators.hypervolume(sample, REFERENCE):.4f}"
)
trace.write()
