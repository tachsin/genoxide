"""WFG5: minimize two conflicting objectives over 24 variables, with a concave Pareto front and
deceptive parameters, with NSGA-II and SMS-EMOA.

The Walking Fish Group's fifth problem, from genoxide's problems.Wfg5, with the recommended
sizes: 4 position and 20 distance parameters; run evaluates it in Rust. Runs each algorithm for
1,000 generations, and prints its front after 250 and after 1,000: the size, the IGD+ to 500
points of the optimal front and the hypervolume; then how far the front is from the optimal one,
and where the population's distance parameters are: at the bounds, where their deceptive shift
leads, or near 0.35, its optimum.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its runs for the plot on the example's
page, with trace.py.

    python examples/wfg5/main.py
"""

import math

import genoxide as gx

from trace import Trace

# the reference point of the hypervolume, 1.1 times the nadir point (2, 4)
REFERENCE = [2.2, 4.4]

# the generation of the first report: the NSGA-II paper's budget
FIRST = 250

# the length of the run
GENERATIONS = 1000

# the position parameters, k: the distance parameters follow them
POSITION = 4

problem = gx.problems.Wfg5(objectives=2)
optimal = problem.optimal_front(500)
# with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
trace = Trace(REFERENCE)


def distance_parameters(population):
    """The distance parameters of the population's genomes, normalized to [0, 1]: y = z / 2i for
    i from k + 1."""
    return [
        genome[i] / (2 * (i + 1)) for genome in population for i in range(POSITION, len(genome))
    ]


def distance(f):
    """The distance x_M of a point from the optimal front: its objectives are x_M + 2 sin(x₁π/2)
    and x_M + 4 cos(x₁π/2), so ((f₁ − x_M) / 2)² + ((f₂ − x_M) / 4)² = 1, of which x_M is the
    smaller root."""
    a, b, c = 5 / 16, f[0] / 2 + f[1] / 8, f[0] * f[0] / 4 + f[1] * f[1] / 16 - 1
    return (b - math.sqrt(max(b * b - 4 * a * c, 0.0))) / (2 * a)


def report(name, generations, front, parameters):
    """Prints the size of a front, its IGD+ to the optimal front and its hypervolume; then the
    least and the largest distance of its points from the optimal front, and how many of the
    population's distance parameters are within 0.001 of the bounds 0 and 1, and within 0.001 of
    0.35."""
    igd = gx.indicators.igd_plus(front, optimal)
    volume = gx.indicators.hypervolume(front, REFERENCE)
    print(
        f"{name} after {generations} generations: {len(front)} solutions, IGD+ {igd:.4f}, "
        f"hypervolume {volume:.4f}"
    )
    distances = [distance(f) for f in front]
    bounds = sum(1 for y in parameters if y < 0.001 or y > 0.999)
    optimum = sum(1 for y in parameters if abs(y - 0.35) < 0.001)
    print(
        f"  distance {min(distances):.5f} to {max(distances):.5f}; genes at the bounds {bounds}, "
        f"near 0.35 {optimum}, of {len(parameters)}"
    )


def run(name, algorithm):
    """Runs ``algorithm`` for 1,000 generations, and reports after 250 and after 1,000."""
    record = trace.fronts(name)
    reports = []

    def on_generation(progress):
        if progress.generation in (FIRST, GENERATIONS):
            front = progress.front_objectives.tolist()
            parameters = distance_parameters(progress.population.tolist())
            reports.append((progress.generation, front, parameters))
        if record:
            record(progress)

    algorithm.run(problem, generations=GENERATIONS, on_generation=on_generation)
    for generations, front, parameters in reports:
        report(name, generations, front, parameters)


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

print("the whole front: hypervolume 3.3968")
trace.write()
