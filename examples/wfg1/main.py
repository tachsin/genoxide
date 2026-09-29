"""WFG1: minimize two objectives over 24 variables, with a front of convex and mixed parts behind
a flat region and a strong polynomial bias, with NSGA-II.

The first problem of Huband, Hingston, Barone and While's WFG toolkit, from genoxide's
problems.Wfg1, with 2 objectives and the recommended sizes, k = 4 and l = 20; run evaluates it in
Rust. In double precision no genome reaches the front, so the target is the closest front a genome
can have: the front moved by the smallest distance x_M that a genome gets. Runs NSGA-II with
simulated binary crossover at η = 15, the ZDT examples' setting, for 2,500 generations, and at
η = 2 for 20,000, and prints the fronts after 2,500 and after 20,000 generations: the size, the
IGD+ to 500 points of the optimal front and to the closest front, and the hypervolume.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its runs for the plot on the example's
page, with trace.py.

    python examples/wfg1/main.py
"""

import numpy as np

import genoxide as gx

from trace import Trace

# the reference point of the hypervolume: 1.1 times the front's worst point, (2, 4)
REFERENCE = [2.2, 4.4]

# the generation of the first report, ten times the NSGA-II paper's budget, and of the last
FIRST = 2_500
LAST = 20_000

# f₁ and f₂ divided by the front's range, 2 and 4
SCALE = np.array([2.0, 4.0])

problem = gx.problems.Wfg1(objectives=2)
optimal = problem.optimal_front(500)

# the closest a genome gets: the paper's optimal solutions, the distance parameters zᵢ at
# 0.35 × 2i, still have a distance x_M of about 0.069 in double precision; with the position
# parameters at their upper bounds, x₁ = 1, f₂ = x_M
z = np.array([2.0 * i if i <= 4 else 0.35 * 2.0 * i for i in range(1, 25)])
distance = problem(z)[1]
# every objective value of the front moved by x_M: the closest front a genome can have
closest = optimal + distance

# with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
trace = Trace(REFERENCE)


def report(name, generations, front):
    """Prints the size of a front; its IGD+ to 500 points of the optimal front, evenly spread
    along it; its IGD+ to the closest front, with the objectives scaled to the front's range; and
    its hypervolume."""
    to_optimal = gx.indicators.igd_plus(front, optimal)
    to_closest = gx.indicators.igd_plus(front / SCALE, closest / SCALE)
    volume = gx.indicators.hypervolume(front, REFERENCE)
    print(
        f"{name:<10} after {generations:>5} generations: {len(front)} solutions, "
        f"IGD+ {to_optimal:.4f} ({to_closest:.4f} to the closest front), "
        f"hypervolume {volume:.4f}"
    )


def run(name, algorithm, generations):
    """Runs ``algorithm`` for ``generations``, and reports its front after 2,500 and at the
    end."""
    record = trace.fronts(name)
    first = []

    def on_generation(progress):
        if progress.generation == FIRST:
            first.append(progress.front_objectives)
        if record:
            record(progress)

    result = algorithm.run(problem, generations=generations, on_generation=on_generation)
    if generations > FIRST:
        report(name, FIRST, first[0])
    report(name, generations, result.front_objectives)


# the ZDT examples' settings for 2,500 generations, then SBX with η = 2 for 20,000: children
# further from their parents
for name, eta, generations in [("SBX eta 15", 15, FIRST), ("SBX eta 2", 2, LAST)]:
    # polynomial mutation at a rate of 1/24, one gene per child on average
    nsga2 = gx.Nsga2(
        problem.genome,
        objectives=problem.objectives,
        population_size=100,
        crossover=gx.SimulatedBinaryCrossover(eta),
        mutation=gx.PolynomialMutation(20, rate=1 / 24),
        seed=1,
    )
    run(name, nsga2, generations)

to_optimal = gx.indicators.igd_plus(closest, optimal)
volume = gx.indicators.hypervolume(problem.optimal_front(100) + distance, REFERENCE)
print(
    f"the closest front, {distance:.4f} away: IGD+ {to_optimal:.4f}, "
    f"hypervolume {volume:.4f} with 100 points"
)
print("the whole front: hypervolume 6.7857")
trace.write()
