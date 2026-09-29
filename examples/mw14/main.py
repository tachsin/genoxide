"""MW14: minimize three objectives over 15 variables, subject to one constraint, whose Pareto front
is four disconnected patches, with NSGA-III and SMS-EMOA.

From genoxide's problems.Mw14; run evaluates it in Rust. Runs NSGA-III and SMS-EMOA, and prints each
final front's size, its hypervolume with the objectives divided by the front's nadir point, as a
share of that of a sample of the optimal front with as many points as there are reference
directions, and how many of its solutions lie in the gaps between the patches; then the hypervolumes
of the whole front and of the sample.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its runs for the plot on the example's
page, with trace.py.

    python examples/mw14/main.py
"""

import numpy as np

import genoxide as gx

from trace import Trace

# the reference point of the hypervolume, with the objectives divided by the nadir point
REFERENCE = [1.1, 1.1, 1.1]

# MW14's gaps: each of f₁ and f₂ on the front is in [0, A] or (B, 1.5]
A = 0.7313522974897325
B = 1.3296339087402259

problem = gx.problems.Mw14()
ideal, nadir = problem.ideal_point, problem.nadir_point


def normalized(points):
    """The objectives divided by the front's nadir point (its ideal point is the origin, but for
    f₃)."""
    return (points - ideal) / (nadir - ideal)


# with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
trace = Trace(problem, normalized, REFERENCE)
# 91 reference directions: Das and Dennis's points with 12 divisions
directions = gx.das_dennis(3, 12)
# the sample: as many points of the optimal front, what a front of 91 solutions can be
sample = normalized(problem.optimal_front(len(directions)))


def run(name, algorithm, generations):
    """Runs ``algorithm`` for ``generations``, and prints its final front's size, its hypervolume,
    as a share of the sample's, and how many of its solutions lie in the gaps between the
    patches."""
    result = algorithm.run(problem, generations=generations, on_generation=trace.front(name))
    front = result.front_objectives
    volume = gx.indicators.hypervolume(normalized(front), REFERENCE)
    percent = 100 * volume / gx.indicators.hypervolume(sample, REFERENCE)
    inside = lambda values: ((values > A) & (values <= B))
    gaps = int((inside(front[:, 0]) | inside(front[:, 1])).sum())
    print(
        f"{name}, {generations} generations: {len(front)} solutions, "
        f"hypervolume {volume:.4f}, {percent:.1f}% of the sample's, {gaps} in the gaps"
    )


# NSGA-III with the paper's operators: η = 20 for both
algorithm = gx.Nsga3(
    problem.genome,
    objectives=problem.objectives,
    reference_directions=directions,
    population_size=92,
    crossover=gx.SimulatedBinaryCrossover(20),
    mutation=gx.PolynomialMutation(20, rate=1 / 15),
    seed=1,
)
run("NSGA-III", algorithm, 3000)
# SMS-EMOA, which keeps the solutions that add the most hypervolume
algorithm = gx.SmsEmoa(
    problem.genome,
    objectives=problem.objectives,
    population_size=92,
    crossover=gx.SimulatedBinaryCrossover(20),
    mutation=gx.PolynomialMutation(20, rate=1 / 15),
    seed=1,
)
run("SMS-EMOA", algorithm, 2000)
whole = normalized(problem.optimal_front(3_000))
print(
    f"the whole front: hypervolume {gx.indicators.hypervolume(whole, REFERENCE):.4f}; the "
    f"sample of {len(sample)} of its points: "
    f"{gx.indicators.hypervolume(sample, REFERENCE):.4f}"
)
trace.write()
