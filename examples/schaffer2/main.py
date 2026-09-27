"""Schaffer 2: minimize a piecewise linear f₁ and (x − 5)² over x in [−5, 10], with NSGA-II.

Schaffer's second problem, from genoxide's problems.Schaffer2; run evaluates it in Rust. Its front
is in two pieces. Prints the size of the final front and how many of its solutions are on each
piece, its IGD+ to 500 points of the optimal front, and its hypervolume.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/schaffer2/main.py
"""

import genoxide as gx

from trace import Trace

# the reference point of the hypervolume: 10% of the front's range beyond its worst point (1, 16)
REFERENCE = [1.2, 17.6]

problem = gx.problems.Schaffer2()
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

# the first piece, x in [1, 2), has f1 in [−1, 0); the second, x in [4, 5], f1 in [0, 1]
front = result.front_objectives
first = int((front[:, 0] < 0).sum())
second = len(front) - first
print(f"{len(front)} solutions on the front: {first} on the first piece, {second} on the second")

# IGD+ to the optimal front, both pieces
optimal = problem.optimal_front(500)
distance = gx.indicators.igd_plus(front, optimal)
print(f"IGD+ to the optimal front: {distance:.4f}")

# the whole front's hypervolume is 2.2 × 17.6 − 37/3 − 1/3, the box minus the areas under its two
# pieces
volume = gx.indicators.hypervolume(front, REFERENCE)
print(f"hypervolume {volume:.3f} (the whole front: 26.053)")
trace.write()
