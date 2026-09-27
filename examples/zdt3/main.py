"""ZDT3: minimize two conflicting objectives over 30 variables in [0, 1], with a Pareto front in
five disconnected pieces, with NSGA-II.

Zitzler, Deb and Thiele's third problem, from genoxide's problems.Zdt3; run evaluates it in Rust.
Prints the size of the final front and how many of its solutions are on each piece, its IGD+ to
500 points of the optimal front, and its hypervolume.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/zdt3/main.py
"""

import genoxide as gx

from trace import Trace

# the reference point of the hypervolume, beyond the front's worst point (0.852, 1): f2 is
# negative on much of the front, down to −0.773, but f1 and f2 are at most 1 there
REFERENCE = [1.1, 1.1]

# the five pieces of the optimal front, as ranges of f1: each ends at a local minimum of
# f2 = 1 − √f1 − f1 sin(10π f1), where the next piece's values drop below it
PIECES = [
    (0.0, 0.0830),
    (0.1822, 0.2578),
    (0.4093, 0.4539),
    (0.6184, 0.6525),
    (0.8233, 0.8518),
]

problem = gx.problems.Zdt3(30)
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

# the solutions on each piece: f1 within 0.001 of the piece's range
f1 = front[:, 0]
counts = [int(((f1 >= low - 0.001) & (f1 <= high + 0.001)).sum()) for low, high in PIECES]
print(f"on the five pieces: {', '.join(map(str, counts))}")

# IGD+ to the optimal front, 500 points spread over the pieces in proportion to their widths
optimal = problem.optimal_front(500)
distance = gx.indicators.igd_plus(front, optimal)
print(f"IGD+ to the optimal front: {distance:.4f}")

# the whole front's hypervolume, found numerically
volume = gx.indicators.hypervolume(front, REFERENCE)
print(f"hypervolume {volume:.4f} (the whole front: 1.3318)")
trace.write()
