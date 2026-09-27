"""TNK: minimize two objectives subject to two constraints, with NSGA-II and Deb's rules.

Tanaka's problem, from genoxide's problems.Tnk, whose fitness is the two objectives and the
constraint violation; run evaluates it in Rust. Prints how many solutions of the final front are
feasible, how many lie on each of the five pieces of the optimal front, their IGD+ to 500 points
of the optimal front, and the front's hypervolume.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/tnk/main.py
"""

import genoxide as gx

from trace import Trace

problem = gx.problems.Tnk()
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


def piece(f1, f2):
    """The piece of the optimal front near (f1, f2): the five pieces are symmetric in f1 and f2,
    the first ends at f1 = 0.1996, the second at f1 = 0.6147, and the middle one begins at
    f1 = 0.6202."""
    if f1 < 0.3:
        return 0
    if f1 < 0.6175:
        return 1
    if f2 < 0.3:
        return 4
    if f2 < 0.6175:
        return 3
    return 2


pieces = [0] * 5
for f1, f2 in front:
    pieces[piece(f1, f2)] += 1
a, b, c, d, e = pieces
print(f"on the five pieces of the front, from f1 = 0.04 to 1.04: {a}, {b}, {c}, {d}, {e}")

# IGD+ to the optimal front
optimal = problem.optimal_front(500)
distance = gx.indicators.igd_plus(front, optimal)
print(f"IGD+ to the optimal front: {distance:.6f}")

# the hypervolume with the reference point (1.2, 1.2); the whole front's, from 2,000,000 of its
# points, is 0.6551
volume = gx.indicators.hypervolume(front, [1.2, 1.2])
print(f"hypervolume {volume:.4f} (the whole front: 0.6551)")
trace.write()
