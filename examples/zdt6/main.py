"""ZDT6: minimize two conflicting objectives over 10 variables in [0, 1], with a concave Pareto
front and a search space that crowds solutions at one end of it, with NSGA-II.

Zitzler, Deb and Thiele's sixth problem, from genoxide's problems.Zdt6; run evaluates it in Rust.
Prints how unevenly x1 maps to f1, then the size and range of the final front, its IGD+ to 500
points of the optimal front, and its hypervolume.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/zdt6/main.py
"""

import numpy as np

import genoxide as gx

from trace import Trace

# the reference point of the hypervolume, beyond the front's worst point (1, 0.921)
REFERENCE = [1.1, 1.1]

# the optimal front's f1 runs from 0.2808 to 1; the middle of that range
MIDDLE = 0.6404

problem = gx.problems.Zdt6(10)

# x1 alone sets f1: at 1,001 evenly spaced values, with the other variables at 0, how many give
# f1 in the lower half of the front's range
grid = np.zeros((1001, 10))
grid[:, 0] = np.arange(1001) / 1000
lower = int((problem.evaluate(grid)[:, 0] < MIDDLE).sum())
print(f"{lower} of 1001 evenly spaced x1 give f1 below {MIDDLE}")

# polynomial mutation at a rate of 1/10, one gene per child on average
nsga2 = gx.Nsga2(
    problem.genome,
    objectives=problem.objectives,
    population_size=100,
    crossover=gx.SimulatedBinaryCrossover(15),
    mutation=gx.PolynomialMutation(20, rate=1 / 10),
    seed=1,
)
# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(problem, REFERENCE)
result = nsga2.run(problem, generations=250, on_generation=trace.on_generation)

front = result.front_objectives
below = int((front[:, 0] < MIDDLE).sum())
smallest = front[:, 0].min()
print(
    f"{len(front)} solutions on the front, {below} with f1 below {MIDDLE}, "
    f"the smallest {smallest:.4f}"
)

# IGD+ to the optimal front, f2 = 1 − f1², at 500 points evenly spaced in f1
optimal = problem.optimal_front(500)
distance = gx.indicators.igd_plus(front, optimal)
print(f"IGD+ to the optimal front: {distance:.4f}")

# the whole front's hypervolume, found from the curve's integral
volume = gx.indicators.hypervolume(front, REFERENCE)
print(f"hypervolume {volume:.4f} (the whole front: 0.5079)")
trace.write()
