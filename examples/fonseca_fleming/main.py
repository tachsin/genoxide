"""Fonseca-Fleming: minimize two objectives with a concave front over 3 variables in [−4, 4], with
NSGA-II.

Fonseca and Fleming's problem, from genoxide's problems.FonsecaFleming, with 3 variables as in the
NSGA-II paper; run evaluates it in Rust. Prints the size of the final front, its IGD+ to 500
points of the optimal front, and its hypervolume.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/fonseca_fleming/main.py
"""

import genoxide as gx

from trace import Trace

# the reference point of the hypervolume: beyond the front's worst point (0.9817, 0.9817), and
# every objective value, which is below 1
REFERENCE = [1.1, 1.1]

problem = gx.problems.FonsecaFleming(3)
# polynomial mutation at a rate of 1/3, one gene per child on average
nsga2 = gx.Nsga2(
    problem.genome,
    objectives=problem.objectives,
    population_size=100,
    crossover=gx.SimulatedBinaryCrossover(15),
    mutation=gx.PolynomialMutation(20, rate=1 / 3),
    seed=1,
)
# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(problem, REFERENCE)
result = nsga2.run(problem, generations=250, on_generation=trace.on_generation)

front = result.front_objectives
print(f"{len(front)} solutions on the front")

# IGD+ to the optimal front, x1 = x2 = x3 from −1/√3 to 1/√3
optimal = problem.optimal_front(500)
distance = gx.indicators.igd_plus(front, optimal)
print(f"IGD+ to the optimal front: {distance:.4f}")

# the whole front's hypervolume, the box minus the area under the curve, found numerically
volume = gx.indicators.hypervolume(front, REFERENCE)
print(f"hypervolume {volume:.4f} (the whole front: 0.5521)")
trace.write()
