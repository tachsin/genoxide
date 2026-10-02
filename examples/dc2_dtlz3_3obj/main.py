"""DC2-DTLZ3 with 3 objectives: minimize three objectives of DTLZ3 subject to two constraints on its
distance function, which leave almost nothing but the front feasible, with NSGA-III.

From genoxide's problems.Dc2Dtlz3; run evaluates it in Rust. Runs NSGA-III with the settings of the
C-TAEA paper's C-NSGA-III, a population of 92 for 2,000 generations, twice: with constraint
dominance, and on DTLZ3 without the constraints. Prints each final front's size, how many of its
solutions are feasible, their IGD+ to 2,000 points of the optimal front and their hypervolume, with
the objectives normalized by the front's ideal and nadir points, as a share of that of a sample of
the front with at least as many points as the population; then the hypervolumes of the whole front
and of the sample.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its runs for the plot on the example's
page, with trace.py.

    python examples/dc2_dtlz3_3obj/main.py
"""

import genoxide as gx

from trace import Trace

# the reference point of the hypervolume, with the objectives normalized by the front's ideal and
# nadir points
REFERENCE = [1.1, 1.1, 1.1]

problem = gx.problems.Dc2Dtlz3()
ideal, nadir = problem.ideal_point, problem.nadir_point
rate = 1 / problem.dimensions


def normalized(points):
    """The objectives normalized by the front's ideal and nadir points: the front spans [0, 1] in
    each."""
    return (points - ideal) / (nadir - ideal)


# with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
trace = Trace(problem, normalized, REFERENCE)
# the sample: at least as many points of the optimal front as the population has, about what a
# front of that many solutions can be
sample = normalized(problem.optimal_front(92))


def run(name, algorithm, fitness, generations):
    """Runs ``algorithm`` on ``fitness`` for ``generations``, and prints its final front's size,
    how many of its solutions the problem finds feasible, their IGD+ to 2,000 points of the
    optimal front and their hypervolume, with normalized objectives, as a share of the
    sample's."""
    result = algorithm.run(fitness, generations=generations, on_generation=trace.front(name))
    objectives, violations = problem.evaluate(result.front_genomes)
    feasible = objectives[violations == 0]
    noun = "solution" if len(objectives) == 1 else "solutions"
    start = f"{name}, {generations} generations: {len(objectives)} {noun}, "
    if len(feasible) == 0:
        print(start + f"none feasible, the least violation {violations.min():.4f}")
        return
    optimal = normalized(problem.optimal_front(2_000))
    distance = gx.indicators.igd_plus(normalized(feasible), optimal)
    volume = gx.indicators.hypervolume(normalized(feasible), REFERENCE)
    percent = 100 * volume / gx.indicators.hypervolume(sample, REFERENCE)
    print(
        start + f"{len(feasible)} feasible, IGD+ {distance:.4f}, hypervolume {volume:.4f}, "
        f"{percent:.1f}% of the sample's"
    )


# NSGA-III with the settings of the C-TAEA paper's C-NSGA-III, constraint dominance
algorithm = gx.Nsga3(
    problem.genome,
    objectives=problem.objectives,
    reference_directions=gx.das_dennis(3, 12),
    population_size=92,
    crossover=gx.SimulatedBinaryCrossover(30),
    crossover_rate=1.0,
    mutation=gx.PolynomialMutation(20, rate=rate),
    seed=1,
)
run("NSGA-III", algorithm, problem, 2_000)
# the same on DTLZ3, without the constraints, whose solutions the problem then scores
algorithm = gx.Nsga3(
    problem.genome,
    objectives=problem.objectives,
    reference_directions=gx.das_dennis(3, 12),
    population_size=92,
    crossover=gx.SimulatedBinaryCrossover(30),
    crossover_rate=1.0,
    mutation=gx.PolynomialMutation(20, rate=rate),
    seed=1,
)
run("NSGA-III, unconstrained", algorithm, gx.problems.Dtlz3(variables=problem.dimensions), 2_000)
whole = normalized(problem.optimal_front(3_000))
print(
    f"the whole front: hypervolume {gx.indicators.hypervolume(whole, REFERENCE):.4f}; "
    f"a sample of {len(sample)} of its points: "
    f"{gx.indicators.hypervolume(sample, REFERENCE):.4f}"
)
trace.write()
