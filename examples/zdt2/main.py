"""ZDT2: minimize two conflicting objectives over 30 variables in [0, 1], with a concave Pareto
front, with NSGA-II.

Zitzler, Deb and Thiele's second problem, from genoxide's problems.Zdt2; run evaluates it in Rust.
Prints the size of the final front, its IGD+ to 500 points of the optimal front, and its
hypervolume.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/zdt2/main.py
"""

import genoxide as gx

from trace import Trace

# the reference point of the hypervolume, beyond the front's worst point (1, 1)
REFERENCE = [1.1, 1.1]

problem = gx.problems.Zdt2(30)
# polynomial mutation at a rate of 1/30, one gene per child on average
nsga2 = gx.Nsga2(
    problem.genome,
    objectives=problem.objectives,
    population_size=100,
    crossover=gx.SimulatedBinaryCrossover(15),
    mutation=gx.PolynomialMutation(20, rate=1 / 30),
    seed=1,
)
# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(problem, REFERENCE)
result = nsga2.run(problem, generations=250, on_generation=trace.on_generation)

front = result.front_objectives
print(f"{len(front)} solutions on the front")

# IGD+ to the optimal front, f2 = 1 − f1², at 500 points evenly spaced in f1
optimal = problem.optimal_front(500)
distance = gx.indicators.igd_plus(front, optimal)
print(f"IGD+ to the optimal front: {distance:.4f}")

# the whole front's hypervolume: the box, 1.1 × 1.1, minus the area under the curve, 2/3
volume = gx.indicators.hypervolume(front, REFERENCE)
print(f"hypervolume {volume:.4f} (the whole front: 0.5433)")
trace.write()
