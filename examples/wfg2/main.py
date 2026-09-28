"""WFG2: minimize two objectives over 24 variables, with a convex front in six disconnected
regions and non-separable distance parameters, with NSGA-II and MOEA/D.

The second problem of Huband, Hingston, Barone and While's WFG toolkit, from genoxide's
problems.Wfg2, with 2 objectives and the recommended sizes, k = 4 and l = 20; run evaluates it in
Rust. Runs each algorithm for 1,000 generations, and prints its front after 250 and after 1,000:
the size, the solutions on each region of the optimal front, the IGD+ to 500 points of it, and
the hypervolume.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its runs for the plot on the example's
page, with trace.py.

    python examples/wfg2/main.py
"""

import genoxide as gx

from trace import Trace

# the reference point of the hypervolume: 1.1 times the front's worst point, (2, 4)
REFERENCE = [2.2, 4.4]

# the generation of the first report: the NSGA-II paper's budget
FIRST = 250

# the generation of the last report
LAST = 1_000

# the six regions of the optimal front, as ranges of f₁ = 2 (1 − cos(x₁π/2)), from the ranges of
# x₁ where 1 − x₁ cos²(5πx₁) is below all its values at smaller x₁
REGIONS = [
    (0.0, 0.0043),
    (0.0414, 0.1074),
    (0.3028, 0.3912),
    (0.7351, 0.8331),
    (1.2904, 1.3894),
    (1.9133, 2.0),
]

problem = gx.problems.Wfg2(objectives=2)
optimal = problem.optimal_front(500)
# with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
trace = Trace(REFERENCE, REGIONS)


def report(name, generations, front):
    """Prints the size of a front, its solutions on each region (f₁ within 0.01 of the region's
    range), its IGD+ to the optimal front and its hypervolume. A front has each solution once,
    though MOEA/D's subproblems can hold copies of one."""
    distance = gx.indicators.igd_plus(front, optimal)
    volume = gx.indicators.hypervolume(front, REFERENCE)
    counts = [
        int(((front[:, 0] >= low - 0.01) & (front[:, 0] <= high + 0.01)).sum())
        for low, high in REGIONS
    ]
    print(
        f"{name:<7} after {generations} generations: {len(front)} solutions "
        f"({', '.join(map(str, counts))}), IGD+ {distance:.4f}, hypervolume {volume:.4f}"
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

    result = algorithm.run(problem, generations=LAST, on_generation=on_generation)
    report(name, FIRST, first[0])
    report(name, LAST, result.front_objectives)


# polynomial mutation at a rate of 1/24, one gene per child on average
operators = dict(
    objectives=problem.objectives,
    crossover=gx.SimulatedBinaryCrossover(15),
    mutation=gx.PolynomialMutation(20, rate=1 / 24),
    seed=1,
)
run("NSGA-II", gx.Nsga2(problem.genome, population_size=100, **operators))
# 101 weight vectors, evenly spread, and the same operators
run("MOEA/D", gx.Moead(problem.genome, weights=gx.das_dennis(2, 100), **operators))

print("the whole front: hypervolume 6.1511")
trace.write()
