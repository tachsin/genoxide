"""CONSTR: minimize two objectives subject to two constraints, with NSGA-II and Deb's rules.

Deb's problem, from genoxide's problems.Constr, whose fitness is the two objectives and the
constraint violation; run evaluates it in Rust. Prints how many solutions of the final front are
feasible, how many lie along each of the two pieces of the optimal front, their IGD+ to 500 points
of the optimal front, and the front's hypervolume.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/constr/main.py
"""

import genoxide as gx

from trace import Trace

problem = gx.problems.Constr()
# the settings of the NSGA-II paper: a mutation rate of 1/n for n genes
nsga2 = gx.Nsga2(
    problem.genome,
    objectives=problem.objectives,
    population_size=100,
    crossover=gx.SimulatedBinaryCrossover(20),
    mutation=gx.PolynomialMutation(20, rate=0.5),
    seed=1,
)
# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(problem)
result = nsga2.run(problem, generations=250, on_generation=trace.on_generation)

front = result.front_objectives
feasible = int((result.front_violations == 0).sum())
print(f"{len(front)} solutions on the front, {feasible} feasible")

# the two pieces of the optimal front meet at f1 = x1 = 2/3
boundary = int((front[:, 0] < 2 / 3).sum())
print(f"along the boundary x2 = 6 - 9 x1: {boundary}, along x2 = 0: {len(front) - boundary}")

# IGD+ to the optimal front
optimal = problem.optimal_front(500)
distance = gx.indicators.igd_plus(front, optimal)
print(f"IGD+ to the optimal front: {distance:.6f}")

# the hypervolume with the reference point (1.1, 10); the whole front's is
# 19 · 5/18 − 7 ln(12/7) + 10/3 − ln(3/2) + 0.9 = 5.3327
volume = gx.indicators.hypervolume(front, [1.1, 10.0])
print(f"hypervolume {volume:.4f} (the whole front: 5.3327)")
trace.write()
