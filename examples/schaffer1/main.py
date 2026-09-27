"""Schaffer 1: minimize x² and (x − 2)² over x in [−1000, 1000], with NSGA-II.

Schaffer's first problem, from genoxide's problems.Schaffer1; run evaluates it in Rust. Its front
is convex, and its optimal solutions, x in [0, 2], are a thousandth of the interval. Prints the
size of the final front, its IGD+ to 500 points of the optimal front, and its hypervolume.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/schaffer1/main.py
"""

import genoxide as gx

from trace import Trace

# the reference point of the hypervolume: 10% of the front's range beyond its worst point (4, 4)
REFERENCE = [4.4, 4.4]

problem = gx.problems.Schaffer1()
# one variable: polynomial mutation changes it in every child
nsga2 = gx.Nsga2(
    problem.genome,
    objectives=problem.objectives,
    population_size=100,
    crossover=gx.SimulatedBinaryCrossover(15),
    mutation=gx.PolynomialMutation(20, rate=1.0),
    seed=1,
)
# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(problem, REFERENCE)
result = nsga2.run(problem, generations=250, on_generation=trace.on_generation)

front = result.front_objectives
print(f"{len(front)} solutions on the front")

# IGD+ to the optimal front, x from 0 to 2
optimal = problem.optimal_front(500)
distance = gx.indicators.igd_plus(front, optimal)
print(f"IGD+ to the optimal front: {distance:.4f}")

# the whole front's hypervolume is 4.4² − 8/3, the box minus the area under f2 = (√f1 − 2)²
volume = gx.indicators.hypervolume(front, REFERENCE)
print(f"hypervolume {volume:.3f} (the whole front: 16.693)")
trace.write()
