"""ZDT4: minimize two conflicting objectives over 10 variables, with the convex Pareto front of
ZDT1 behind 21⁹ local fronts, with NSGA-II and SMS-EMOA.

Zitzler, Deb and Thiele's fourth problem, from genoxide's problems.Zdt4; run evaluates it in Rust.
Runs each algorithm for 500 generations, and prints its front after 250 and after 500: the size,
the IGD+ to 500 points of the optimal front, and the hypervolume.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its runs for the plot on the example's
page, with trace.py.

    python examples/zdt4/main.py
"""

import genoxide as gx

from trace import Trace

# the reference point of the hypervolume, beyond the front's worst point (1, 1)
REFERENCE = [1.1, 1.1]

# the generation of the first report: the NSGA-II paper's budget
HALFWAY = 250

problem = gx.problems.Zdt4(10)
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
    """Runs ``algorithm`` for 500 generations, and reports its front after 250 and 500."""
    record = trace.fronts(name)
    halfway = []

    def on_generation(progress):
        if progress.generation == HALFWAY:
            halfway.append(progress.front_objectives)
        if record:
            record(progress)

    result = algorithm.run(problem, generations=2 * HALFWAY, on_generation=on_generation)
    report(name, HALFWAY, halfway[0])
    report(name, 2 * HALFWAY, result.front_objectives)


# polynomial mutation at a rate of 1/10, one gene per child on average
settings = dict(
    objectives=problem.objectives,
    population_size=100,
    crossover=gx.SimulatedBinaryCrossover(15),
    mutation=gx.PolynomialMutation(20, rate=1 / 10),
    seed=1,
)
run("NSGA-II", gx.Nsga2(problem.genome, **settings))
run("SMS-EMOA", gx.SmsEmoa(problem.genome, **settings))

print("the whole front: hypervolume 0.8767")
trace.write()
