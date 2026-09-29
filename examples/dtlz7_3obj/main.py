"""DTLZ7 with 3 objectives: minimize three conflicting objectives over 22 variables in [0, 1], whose
Pareto front is four disconnected regions.

NSGA-III with the settings of the DTLZ2 example, NSGA-II with the same population and operators,
and NSGA-III with 861 reference directions and as many solutions, each for 250 generations.
Prints, for each, how many solutions of its front lie in each region, its IGD+ to 1,024 points of
the front and its hypervolume. run evaluates the problem in Rust.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of the last run for the plot on the
example's page, with trace.py.

    python examples/dtlz7_3obj/main.py
"""

import numpy as np

import genoxide as gx

from trace import Trace

VARIABLES = 22

# f₁ and f₂ on the front are in [0, A] or (B, C], from genoxide's docs
A = 0.2514118360889171
B = 0.6316265307000612
C = 0.8594008566447239

# the reference point of the hypervolume: 1.1 times the nadir point (C, C, 6)
REFERENCE = [1.1 * C, 1.1 * C, 6.6]

problem = gx.problems.Dtlz7(objectives=3, variables=VARIABLES)
# IGD+ to a grid of 32 × 32 values of f₁ and f₂ over the regions
optimal = problem.optimal_front(1000)
# the front's range, from the ideal to the nadir point, to scale each objective to [0, 1] for the
# scaled IGD+
ideal, nadir = problem.ideal_point, problem.nadir_point


def run(name, algorithm, on_generation=None):
    """Runs ``algorithm`` for 250 generations, and reports its front."""
    front = algorithm.run(problem, generations=250, on_generation=on_generation).front_objectives
    # the solutions in each region: f₁ and f₂ low or high, split halfway between A and B
    high = front[:, :2] > (A + B) / 2
    low_low, low_high, high_low, high_high = (
        int(((high[:, 0] == first) & (high[:, 1] == second)).sum())
        for first in (False, True)
        for second in (False, True)
    )
    distance = gx.indicators.igd_plus(front, optimal)
    scaled = gx.indicators.igd_plus(
        (front - ideal) / (nadir - ideal), (optimal - ideal) / (nadir - ideal)
    )
    volume = gx.indicators.hypervolume(front, REFERENCE)
    print(
        f"{name:<24} {len(front)} solutions, {low_low} + {low_high} + {high_low} + {high_high} in "
        f"the four regions, IGD+ {distance:.4f} (scaled {scaled:.4f}), hypervolume {volume:.4f}"
    )


settings = dict(
    objectives=problem.objectives,
    crossover=gx.SimulatedBinaryCrossover(30),
    mutation=gx.PolynomialMutation(20, rate=1 / VARIABLES),
    seed=1,
)
# 91 directions and a population of 92, the multiple of 4 above
nsga3 = gx.Nsga3(
    problem.genome, reference_directions=gx.das_dennis(3, 12), population_size=92, **settings
)
run("NSGA-III, 91 directions", nsga3)
run("NSGA-II", gx.Nsga2(problem.genome, population_size=92, **settings))
# with GENOXIDE_TRACE=<file>, a trace of this run for the plot on the example's page
trace = Trace(REFERENCE)
# 861 directions, and a solution for each
nsga3 = gx.Nsga3(problem.genome, reference_directions=gx.das_dennis(3, 40), **settings)
run("NSGA-III, 861 directions", nsga3, trace.on_generation)

# the whole front's hypervolume, integrated numerically
print("the whole front: hypervolume 1.7392")
trace.write()
