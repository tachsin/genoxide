"""Constrained two-objective fronts: NSGA-II with Deb's rules on SRN, TNK, OSY and CONSTR.

Four constrained problems of genoxide's problems, whose fitness is the two objectives and the
total constraint violation; run evaluates them in Rust. Prints, for each, the size of the final
front and how many of its solutions are feasible, the front's hypervolume and its IGD+ to 500
points of the optimal front, both with the objectives normalized by the problem's ideal and nadir
points.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its runs for the plot on the example's
page, with trace.py.

    python examples/constrained_fronts/main.py
"""

import genoxide as gx

from trace import Trace


def normalized(problem, front):
    """The objectives of ``front`` mapped to [0, 1] by the problem's ideal and nadir points."""
    ideal, nadir = problem.ideal_point, problem.nadir_point
    return (front - ideal) / (nadir - ideal)


def normalized_hypervolume(problem, front):
    """The hypervolume of ``front`` in normalized objectives, up to the reference point
    (1.1, 1.1)."""
    return gx.indicators.hypervolume(normalized(problem, front).reshape(-1, 2), [1.1, 1.1])


problems = [gx.problems.Srn(), gx.problems.Tnk(), gx.problems.Osy(), gx.problems.Constr()]
# with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
trace = Trace(normalized_hypervolume)
print("problem  front  feasible  hypervolume  (optimal)  IGD+")
for problem in problems:
    # the settings of the NSGA-II paper: a mutation rate of 1/n for n genes
    genes = problem.dimensions
    nsga2 = gx.Nsga2(
        problem.genome,
        objectives=problem.objectives,
        population_size=100,
        crossover=gx.SimulatedBinaryCrossover(20),
        mutation=gx.PolynomialMutation(20, rate=1 / genes),
        seed=1,
    )
    result = nsga2.run(problem, generations=200, on_generation=trace.panel(problem))

    feasible = result.front_objectives[result.front_violations == 0]
    # the hypervolume of the feasible front, beside that of 500 points of the optimal front, and
    # the IGD+ to them, all in normalized objectives
    optimal = problem.optimal_front(500)
    volume = normalized_hypervolume(problem, feasible)
    best = normalized_hypervolume(problem, optimal)
    distance = gx.indicators.igd_plus(normalized(problem, feasible), normalized(problem, optimal))
    print(
        f"{problem.name:<7}  {len(result.front_objectives):>5}  {len(feasible):>8}  "
        f"{volume:>11.4f}  {best:>9.4f}  {distance:.4f}"
    )
trace.write(problems)
