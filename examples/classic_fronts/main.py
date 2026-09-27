"""Classic fronts: NSGA-II on four classic two-objective problems, Schaffer's first and second,
Fonseca and Fleming's, and Poloni's.

The problems come from genoxide's problems; run evaluates them in Rust. NSGA-II has the settings
of the NSGA-II paper, and the same 250 generations on each. Prints, per problem, the size of the
final front, its hypervolume with the objectives normalized by the problem's ideal and nadir
points, and, where the optimal front is known, its IGD+ to 500 points of it.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/classic_fronts/main.py
"""

import numpy as np

import genoxide as gx

from trace import Trace

GENERATIONS = 250


def normalized_hypervolume(front, ideal, nadir):
    """The hypervolume of ``front`` with each objective mapped to [0, 1] by the ideal and nadir
    points, and the reference point (1.1, 1.1)."""
    return gx.indicators.hypervolume((front - ideal) / (nadir - ideal), [1.1, 1.1])


def poloni_extremes(problem):
    """Poloni's ideal and nadir points, which genoxide doesn't give: the ends of its front are the
    minimum of f₁, 1 at (1, 2), and the minimum of f₂, 0 at (−3, −1)."""
    least_f1, least_f2 = problem([1.0, 2.0]), problem([-3.0, -1.0])
    return np.array([least_f1[0], least_f2[1]]), np.array([least_f2[0], least_f1[1]])


problems = {
    "Schaffer 1": gx.problems.Schaffer1(),
    "Schaffer 2": gx.problems.Schaffer2(),
    "Fonseca-Fleming": gx.problems.FonsecaFleming(3),
    "Poloni": gx.problems.Poloni(),
}

print(f"{'problem':<16}{'front':>6}{'hypervolume':>13}{'IGD+':>9}")
# with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
trace = Trace()
for name, problem in problems.items():
    # polynomial mutation at a rate of 1/n, one gene per child on average
    nsga2 = gx.Nsga2(
        problem.genome,
        objectives=problem.objectives,
        population_size=100,
        crossover=gx.SimulatedBinaryCrossover(15),
        mutation=gx.PolynomialMutation(20, rate=1 / problem.dimensions),
        seed=1,
    )
    if problem.ideal_point is not None and problem.nadir_point is not None:
        ideal, nadir = problem.ideal_point, problem.nadir_point
    else:
        ideal, nadir = poloni_extremes(problem)

    def volume(front, ideal=ideal, nadir=nadir):
        return normalized_hypervolume(front, ideal, nadir)

    record = trace.fronts(name, problem.optimal_front(100), volume)
    result = nsga2.run(problem, generations=GENERATIONS, on_generation=record)

    front = result.front_objectives
    # IGD+ to 500 points of the optimal front, where it's known
    optimal = problem.optimal_front(500)
    distance = "-" if optimal is None else f"{gx.indicators.igd_plus(front, optimal):.4f}"
    print(f"{name:<16}{len(front):>6}{volume(front):>13.4f}{distance:>9}")
trace.write()
