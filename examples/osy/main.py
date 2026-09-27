"""OSY: minimize two objectives of six variables subject to six constraints, with NSGA-II and
Deb's rules.

Osyczka and Kundu's problem, from genoxide's problems.Osy, whose fitness is the two objectives
and the constraint violation; run evaluates it in Rust. Prints how many solutions of the final
front are feasible, how many lie along each of the five pieces of the optimal front, their IGD+ to
500 points of the optimal front, and the front's hypervolume.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/osy/main.py
"""

import math

import genoxide as gx

from trace import Trace

problem = gx.problems.Osy()
# the settings of the NSGA-II paper: a mutation rate of 1/n for n genes
nsga2 = gx.Nsga2(
    problem.genome,
    objectives=problem.objectives,
    population_size=100,
    crossover=gx.SimulatedBinaryCrossover(20),
    mutation=gx.PolynomialMutation(20, rate=1 / 6),
    seed=1,
)
# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(problem)
result = nsga2.run(problem, generations=250, on_generation=trace.on_generation)

front = result.front_objectives
feasible = int((result.front_violations == 0).sum())
print(f"{len(front)} solutions on the front, {feasible} feasible")

# the five pieces of the optimal front meet at f1 = -258, -242, -123.46 and -116
f1 = front[:, 0]
ends = [-math.inf, -258.0, -242.0, -123.46, -116.0, math.inf]
pieces = [str(((low <= f1) & (f1 < high)).sum()) for low, high in zip(ends, ends[1:])]
print(f"along the five pieces of the front, from f1 = -274 to -42: {', '.join(pieces)}")

# IGD+ to the optimal front
optimal = problem.optimal_front(500)
distance = gx.indicators.igd_plus(front, optimal)
print(f"IGD+ to the optimal front: {distance:.4f}")

# the hypervolume with the reference point (-20, 85); the whole front's, from 2,000,000 of its
# points, is 16546.1
volume = gx.indicators.hypervolume(front, [-20.0, 85.0])
print(f"hypervolume {volume:.1f} (the whole front: 16546.1)")
trace.write()
