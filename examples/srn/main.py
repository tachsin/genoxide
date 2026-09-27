"""SRN: minimize two objectives subject to two constraints, with NSGA-II and Deb's rules.

Srinivas and Deb's problem, from genoxide's problems.Srn, whose fitness is the two objectives and
the constraint violation; run evaluates it in Rust. Prints how many solutions of the final front
are feasible, how many lie along each of the three pieces of the optimal front, their IGD+ to 500
points of the optimal front, and the front's hypervolume.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/srn/main.py
"""

import genoxide as gx

from trace import Trace

problem = gx.problems.Srn()
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

# the pieces of the optimal front meet at f1 = 24.5, at x = (-2.5, 2.5), and at f1 = 212.42, at
# x = (-2.5, √218.75) on the circle
f1 = front[:, 0]
line = (f1 < 24.5).sum()
middle = ((24.5 <= f1) & (f1 < 212.42)).sum()
circle = (f1 >= 212.42).sum()
print(
    f"along the line x1 = 3 x2 - 10: {line}, along x1 = -2.5: {middle}, "
    f"along the circle: {circle}"
)

# IGD+ to the optimal front, its three pieces
optimal = problem.optimal_front(500)
distance = gx.indicators.igd_plus(front, optimal)
print(f"IGD+ to the optimal front: {distance:.4f}")

# the hypervolume with the reference point (245, 25); the whole front's, from 2,000,000 of its
# points, is 35478.6
volume = gx.indicators.hypervolume(front, [245.0, 25.0])
print(f"hypervolume {volume:.1f} (the whole front: 35478.6)")
trace.write()
