"""DTLZ6 with 3 objectives: minimize three conflicting objectives over 12 variables in [0, 1], whose
Pareto front is DTLZ5's curve, behind a distance function that is hard to bring to 0.

Three runs of 400 generations, with a population of 92 and polynomial mutation: NSGA-III and
NSGA-II with simulated binary crossover, as for DTLZ5, and NSGA-II with uniform crossover. Prints,
for each, the size of its front, its IGD+ to 1,000 points of the curve, its hypervolume, the
largest gap between its solutions along the curve and the median distance of its solutions from
the front. run evaluates the problem in Rust.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of the run with uniform crossover for the
plot on the example's page, with trace.py.

    python examples/dtlz6_3obj/main.py
"""

import numpy as np

import genoxide as gx

from trace import Trace

VARIABLES = 12

GENERATIONS = 400

# the reference point of the hypervolume: 1.1 times the nadir point (1/√2, 1/√2, 1)
REFERENCE = [1.1 * np.sqrt(0.5), 1.1 * np.sqrt(0.5), 1.1]

problem = gx.problems.Dtlz6(objectives=3, variables=VARIABLES)
# IGD+ to 1,000 points evenly spread along the curve
optimal = problem.optimal_front(1000)


def run(name, algorithm, on_generation=None):
    """Runs ``algorithm`` for 400 generations, and reports its front."""
    result = algorithm.run(problem, generations=GENERATIONS, on_generation=on_generation)
    front = result.front_objectives
    distance = gx.indicators.igd_plus(front, optimal)
    volume = gx.indicators.hypervolume(front, REFERENCE)
    # θ₁ in degrees, 0 at (1/√2, 1/√2, 0) and 90 at (0, 0, 1); the largest gap between
    # neighbors, or between an end of the curve and the nearest solution
    angles = np.degrees(np.arctan2(front[:, 2], np.sqrt(front[:, 0] ** 2 + front[:, 1] ** 2)))
    gap = np.diff(np.sort(np.concatenate([angles, [0.0, 90.0]]))).max()
    # the objectives lie on a sphere of radius 1 + g: g is 0 on the front
    g = np.maximum(np.sqrt((front**2).sum(axis=1)) - 1, 0.0)
    print(
        f"{name:<16} {len(front)} solutions, IGD+ {distance:.4f}, hypervolume {volume:.4f}, "
        f"largest gap {gap:.1f}°, median g {np.median(g):.4f}"
    )


# polynomial mutation at a rate of 1/12, one gene per child on average
settings = dict(
    objectives=problem.objectives,
    population_size=92,
    mutation=gx.PolynomialMutation(20, rate=1 / VARIABLES),
    seed=1,
)
sbx = gx.SimulatedBinaryCrossover(30)
directions = gx.das_dennis(3, 12)
nsga3 = gx.Nsga3(problem.genome, reference_directions=directions, crossover=sbx, **settings)
run("NSGA-III, SBX", nsga3)
run("NSGA-II, SBX", gx.Nsga2(problem.genome, crossover=sbx, **settings))

# uniform crossover takes each gene from either parent, unchanged
# with GENOXIDE_TRACE=<file>, a trace of this run for the plot on the example's page
trace = Trace(REFERENCE)
uniform = gx.Nsga2(problem.genome, crossover=gx.UniformCrossover(), **settings)
run("NSGA-II, uniform", uniform, trace.on_generation)

# the whole curve's hypervolume, 1.1³/2 − 1.1π/4 + 1/3
print("the whole front: hypervolume 0.1349")
trace.write()
