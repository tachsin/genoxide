"""WFG3: minimize two objectives over 24 variables, with a linear front and non-separable
distance parameters, with NSGA-II and SMS-EMOA.

The third problem of Huband, Hingston, Barone and While's WFG toolkit, from genoxide's
problems.Wfg3, with 2 objectives and the recommended sizes, k = 4 and l = 20; run evaluates it in
Rust. Runs each algorithm for 1,000 generations, and prints its front after 250 and after 1,000:
the size, the IGD+ to 500 points of the optimal front, the hypervolume, and the largest gap
between neighbors on the front.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its runs for the plot on the example's
page, with trace.py.

    python examples/wfg3/main.py
"""

import numpy as np

import genoxide as gx

from trace import Trace

# the reference point of the hypervolume: 1.1 times the front's worst point, (2, 4)
REFERENCE = [2.2, 4.4]

# the generation of the first report: the NSGA-II paper's budget
FIRST = 250

# the generation of the last report
LAST = 1_000

problem = gx.problems.Wfg3(objectives=2)
optimal = problem.optimal_front(500)
# with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
trace = Trace(REFERENCE)


def report(name, generations, front):
    """Prints the size of a front, its IGD+ to the optimal front, its hypervolume, and the
    largest distance between neighbors on it, sorted by f₁."""
    distance = gx.indicators.igd_plus(front, optimal)
    volume = gx.indicators.hypervolume(front, REFERENCE)
    ordered = front[np.lexsort((front[:, 1], front[:, 0]))]
    gap = max(np.hypot(*(ordered[1:] - ordered[:-1]).T), default=0.0)
    print(
        f"{name:<8} after {generations} generations: {len(front)} solutions, "
        f"IGD+ {distance:.4f}, hypervolume {volume:.4f}, largest gap {gap:.4f}"
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
settings = dict(
    objectives=problem.objectives,
    population_size=100,
    crossover=gx.SimulatedBinaryCrossover(15),
    mutation=gx.PolynomialMutation(20, rate=1 / 24),
    seed=1,
)
run("NSGA-II", gx.Nsga2(problem.genome, **settings))
run("SMS-EMOA", gx.SmsEmoa(problem.genome, **settings))

# the segment from (0, 4) to (2, 0): 2.2 × 4.4 less the triangle under it, 4
print("the whole front: hypervolume 5.6800; 100 evenly spaced points: gaps of 0.0452")
trace.write()
