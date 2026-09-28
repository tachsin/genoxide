"""WFG1: minimize two objectives over 24 variables, with a front of convex and mixed parts behind
a flat region and a strong polynomial bias, with NSGA-II and SMS-EMOA.

The first problem of Huband, Hingston, Barone and While's WFG toolkit, from genoxide's
problems.Wfg1, with 2 objectives and the recommended sizes, k = 4 and l = 20; run evaluates it in
Rust. Runs each algorithm for 2,500 generations, and prints its front after 250 and after 2,500:
the size, the IGD+ to 500 points of the optimal front, and the hypervolume. In double precision
no genome reaches the front, so it also prints the closest that a genome gets.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its runs for the plot on the example's
page, with trace.py.

    python examples/wfg1/main.py
"""

import numpy as np

import genoxide as gx

from trace import Trace

# the reference point of the hypervolume: 1.1 times the front's worst point, (2, 4)
REFERENCE = [2.2, 4.4]

# the generation of the first report: the NSGA-II paper's budget
FIRST = 250

# the generation of the last report
LAST = 2_500

problem = gx.problems.Wfg1(objectives=2)
optimal = problem.optimal_front(500)
# with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
trace = Trace(REFERENCE)


def report(name, generations, front):
    """Prints the size of a front, its IGD+ to the optimal front and its hypervolume."""
    distance = gx.indicators.igd_plus(front, optimal)
    volume = gx.indicators.hypervolume(front, REFERENCE)
    print(
        f"{name:<8} after {generations} generations: {len(front)} solutions, "
        f"IGD+ {distance:.4f}, hypervolume {volume:.4f}"
    )


def run(name, algorithm):
    """Runs ``algorithm`` for 2,500 generations, and reports its front after 250 and 2,500."""
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

# the closest a genome gets: the paper's optimal solutions, the distance parameters zᵢ at
# 0.35 × 2i, still have a distance x_M of about 0.069 in double precision; with the position
# parameters at their upper bounds, x₁ = 1, f₂ = x_M
z = np.array([2.0 * i if i <= 4 else 0.35 * 2.0 * i for i in range(1, 25)])
distance = problem(z)[1]
# every objective value of the front moved by x_M: the closest front a genome can have
closest = gx.indicators.igd_plus(optimal + distance, optimal)
volume = gx.indicators.hypervolume(problem.optimal_front(100) + distance, REFERENCE)
print(
    f"the closest front, {distance:.4f} away: IGD+ {closest:.4f}, "
    f"hypervolume {volume:.4f} with 100 points"
)
print("the whole front: hypervolume 6.7857")
trace.write()
