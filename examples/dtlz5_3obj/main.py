"""DTLZ5 with 3 objectives: minimize three conflicting objectives over 12 variables in [0, 1], whose
Pareto front is a curve, a quarter circle in the plane f₁ = f₂.

NSGA-III with the settings of the DTLZ2 example, and NSGA-II with the same population and
operators, each for 250 generations. Prints, for each, the size of its front, its IGD+ to 1,000
points of the curve, its hypervolume, the largest gap between its solutions along the curve and
how far its farthest solution is from the front. run evaluates the problem in Rust.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of the NSGA-II run for the plot on the
example's page, with trace.py.

    python examples/dtlz5_3obj/main.py
"""

import numpy as np

import genoxide as gx

from trace import Trace

VARIABLES = 12

# the reference point of the hypervolume: 1.1 times the nadir point (1/√2, 1/√2, 1)
REFERENCE = [1.1 * np.sqrt(0.5), 1.1 * np.sqrt(0.5), 1.1]

problem = gx.problems.Dtlz5(objectives=3, variables=VARIABLES)
# IGD+ to 1,000 points evenly spread along the curve
optimal = problem.optimal_front(1000)


def run(name, algorithm, on_generation=None):
    """Runs ``algorithm`` for 250 generations, and reports its front."""
    front = algorithm.run(problem, generations=250, on_generation=on_generation).front_objectives
    distance = gx.indicators.igd_plus(front, optimal)
    volume = gx.indicators.hypervolume(front, REFERENCE)
    # θ₁ in degrees, 0 at (1/√2, 1/√2, 0) and 90 at (0, 0, 1); the largest gap between
    # neighbors, or between an end of the curve and the nearest solution
    angles = np.degrees(np.arctan2(front[:, 2], np.sqrt(front[:, 0] ** 2 + front[:, 1] ** 2)))
    gap = np.diff(np.sort(np.concatenate([angles, [0.0, 90.0]]))).max()
    # the objectives lie on a sphere of radius 1 + g: g is 0 on the front
    farthest = max(0.0, (np.sqrt((front**2).sum(axis=1)) - 1).max())
    print(
        f"{name:<8} {len(front)} solutions, IGD+ {distance:.4f}, hypervolume {volume:.4f}, "
        f"largest gap {gap:.1f}°, largest g {farthest:.4f}"
    )


settings = dict(
    objectives=problem.objectives,
    population_size=92,
    crossover=gx.SimulatedBinaryCrossover(30),
    mutation=gx.PolynomialMutation(20, rate=1 / VARIABLES),
    seed=1,
)
run("NSGA-III", gx.Nsga3(problem.genome, reference_directions=gx.das_dennis(3, 12), **settings))

# with GENOXIDE_TRACE=<file>, a trace of the NSGA-II run for the plot on the example's page
trace = Trace(REFERENCE)
run("NSGA-II", gx.Nsga2(problem.genome, **settings), trace.on_generation)

# the whole curve's hypervolume, 1.1³/2 − 1.1π/4 + 1/3
print("the whole front: hypervolume 0.1349")
trace.write()
