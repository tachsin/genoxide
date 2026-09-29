"""MW13: minimize two objectives over 15 variables subject to two constraints, with NSGA-II;
the optimal front is three pieces behind infeasible barriers.

Ma and Wang's MW13, from genoxide's problems.Mw13, whose fitness is the two objectives and the
constraint violation; run evaluates it in Rust. Runs NSGA-II twice: with the paper's settings
(polynomial mutation with η = 20, 600 generations), and with η = 2 for 5,000 generations. Prints
each final front's size, how many of its solutions are feasible, and its IGD+ to 500 points of the
optimal front and hypervolume, with the objectives normalized by the front's ideal and nadir
points; then the whole front's hypervolume.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its runs for the plot on the example's
page, with trace.py.

    python examples/mw13/main.py
"""

import numpy as np

import genoxide as gx

from trace import Trace

# the reference point of the hypervolume, with the objectives normalized by the front's ideal and
# nadir points: 1.1 times the nadir point
REFERENCE = [1.1, 1.1]

problem = gx.problems.Mw13()
ideal, nadir = problem.ideal_point, problem.nadir_point


def normalized(points):
    """The objectives normalized by the front's ideal and nadir points: the front spans [0, 1] in
    each."""
    return (points - ideal) / (nadir - ideal)


# with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
trace = Trace(problem, normalized, REFERENCE)


def run(name, eta, generations):
    """Runs NSGA-II with a population of 100, simulated binary crossover with η = 20 at
    genoxide's default rate of 0.9, and polynomial mutation with the distribution index ``eta``
    at a rate of 1/n per gene, for ``generations``; prints its final front's size, how many of it
    are feasible, and its IGD+ to 500 points of the optimal front and hypervolume, with normalized
    objectives."""
    nsga2 = gx.Nsga2(
        problem.genome,
        objectives=problem.objectives,
        population_size=100,
        crossover=gx.SimulatedBinaryCrossover(20),
        mutation=gx.PolynomialMutation(eta, rate=1 / problem.dimensions),
        seed=1,
    )
    result = nsga2.run(problem, generations=generations, on_generation=trace.fronts(name))
    front = result.front_objectives
    feasible = front[result.front_violations == 0]
    noun = "solution" if len(front) == 1 else "solutions"
    start = f"NSGA-II, {name}, {generations} generations: {len(front)} {noun}, "
    if len(feasible) == 0:
        print(start + f"none feasible, the least violation {result.front_violations.min():.4f}")
        return
    optimal = normalized(problem.optimal_front(500))
    distance = gx.indicators.igd_plus(normalized(feasible), optimal)
    volume = gx.indicators.hypervolume(normalized(feasible), REFERENCE)
    print(start + f"{len(feasible)} feasible, IGD+ {distance:.4f}, hypervolume {volume:.4f}")


# the paper's settings: polynomial mutation with η = 20, 600 generations
run("η = 20", 20, 600)
# polynomial mutation with η = 2, whose steps are larger, for 5,000 generations
run("η = 2", 2, 5_000)
# the hypervolume of the whole front, from 20,000 of its points
whole = normalized(problem.optimal_front(20_000))
print(f"the whole front: hypervolume {gx.indicators.hypervolume(whole, REFERENCE):.4f}")
trace.write()
