"""Poloni: minimize two objectives over x1 and x2 in [−π, π], with NSGA-II.

Poloni's problem, from genoxide's problems.Poloni; run evaluates it in Rust. Its front is in two
pieces, and isn't known in closed form. Prints the size of the final front and how many of its
solutions are on each piece, and its hypervolume, against that of a fine grid over the box.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/poloni/main.py
"""

import numpy as np

import genoxide as gx

from trace import Trace

problem = gx.problems.Poloni()
# the reference point of the hypervolume: the nadir point, the front's worst point (16.77, 25),
# plus a tenth of the front's range from the ideal point (1, 0), rounded up to a tenth:
# (18.4, 27.5)
ideal, nadir = problem.ideal_point, problem.nadir_point
REFERENCE = (np.ceil((nadir + (nadir - ideal) / 10) * 10) / 10).tolist()
# polynomial mutation at a rate of 1/2, one gene per child on average
nsga2 = gx.Nsga2(
    problem.genome,
    objectives=problem.objectives,
    population_size=100,
    crossover=gx.SimulatedBinaryCrossover(15),
    mutation=gx.PolynomialMutation(20, rate=0.5),
    seed=1,
)
# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(problem, REFERENCE)
result = nsga2.run(problem, generations=250, on_generation=trace.on_generation)

# the first piece runs from (1, 25) down to f2 ≈ 20.9, the second from f2 ≈ 3.1 to 0
front = result.front_objectives
first = int((front[:, 1] > 12).sum())
second = len(front) - first
print(f"{len(front)} solutions on the front: {first} on the first piece, {second} on the second")

# the front isn't known: the non-dominated points of a 4001 × 4001 grid over the box give a lower
# bound on its hypervolume
volume = gx.indicators.hypervolume(front, REFERENCE)
print(f"hypervolume {volume:.2f} (a fine grid: 444.57)")
trace.write()
