"""WFG9: minimize two objectives whose concave front lies behind multimodal, deceptive and
non-separable parameters, with NSGA-II and MOEA/D.

The Walking Fish Group's ninth problem, from genoxide's problems.Wfg9, with 2 objectives, 4
position and 20 distance parameters; run evaluates it in Rust. Runs each algorithm for 1,000
generations, and prints its front after 250 and after 1,000: the size, the IGD+ to 500 points of
the optimal front, the hypervolume, and how far the solutions are from the front.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its runs for the plot on the example's
page, with trace.py.

    python examples/wfg9/main.py
"""

import math

import genoxide as gx

from trace import Trace

# the reference point of the hypervolume: 1.1 times the front's nadir point (2, 4)
REFERENCE = [2.2, 4.4]

# the generation of the first report: 25,000 evaluations
FIRST = 250

# the generations of each run
GENERATIONS = 1_000

# 2 objectives, k = 4 position and l = 20 distance parameters
problem = gx.problems.Wfg9(2)
optimal = problem.optimal_front(500)
# with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
trace = Trace(REFERENCE)


def distance(f1, f2):
    """How far a point is from the front: the d with (f₁ − d, f₂ − d) on it, where
    ((f₁ − d) / 2)² + ((f₂ − d) / 4)² = 1, the smaller root of that quadratic. WFG adds the
    distance parameters' value, x_M, to both objectives: d is that value."""
    a = 0.25 + 0.0625
    b = f1 / 2 + f2 / 8
    c = f1 * f1 / 4 + f2 * f2 / 16 - 1
    return max((b - math.sqrt(b * b - 4 * a * c)) / (2 * a), 0.0)


def report(name, generations, front):
    """Prints the size of a front, its IGD+ to the optimal front, its hypervolume, and the
    distances of its points from the front."""
    igd = gx.indicators.igd_plus(front, optimal)
    volume = gx.indicators.hypervolume(front, REFERENCE)
    distances = [distance(f1, f2) for f1, f2 in front.tolist()]
    total = 0.0
    for d in distances:
        total += d
    print(
        f"{name:<8} after {generations} generations: {len(front)} solutions, IGD+ {igd:.4f}, "
        f"hypervolume {volume:.4f}"
    )
    print(
        f"         distance from the front {min(distances):.4f} to {max(distances):.4f}, "
        f"mean {total / len(distances):.4f}"
    )


def run(name, algorithm):
    """Runs ``algorithm`` for 1,000 generations, and reports its front after 250 and 1,000."""
    record = trace.fronts(name)
    first = []

    def on_generation(progress):
        if progress.generation == FIRST:
            first.append(progress.front_objectives)
        if record:
            record(progress)

    result = algorithm.run(problem, generations=GENERATIONS, on_generation=on_generation)
    report(name, FIRST, first[0])
    report(name, GENERATIONS, result.front_objectives)


# polynomial mutation at a rate of 1/24, one gene per child on average
settings = dict(
    objectives=problem.objectives,
    crossover=gx.SimulatedBinaryCrossover(15),
    mutation=gx.PolynomialMutation(20, rate=1 / 24),
    seed=1,
)
run("NSGA-II", gx.Nsga2(problem.genome, population_size=100, **settings))
# a subproblem per weight vector, 101 of them 0.01 apart, each scored by PBI
weights = gx.das_dennis(2, 100)
run("MOEA/D", gx.Moead(problem.genome, weights=weights, decomposition=gx.Pbi(5.0), **settings))

# the box up to the reference point, less the quarter ellipse under the front, of area 2π
whole = REFERENCE[0] * REFERENCE[1] - 2 * math.pi
print(f"the whole front: hypervolume {whole:.4f}")
trace.write()
