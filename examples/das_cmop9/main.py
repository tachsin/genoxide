"""DAS-CMOP9: minimize three objectives over 30 variables subject to 7 constraints, whose front is
patches of a sphere, with linked variables, with NSGA-II and NSGA-III.

From genoxide's problems.DasCmop9; run evaluates it in Rust. Runs NSGA-II with the paper's settings
and NSGA-III with polynomial mutation with η = 5, a population of 300 for 1,000 generations each,
with the difficulty triplet of the paper's figure 6, (0.5, 0.5, 0.5). Prints each final front's
size, how many of its solutions are feasible, their IGD+ to 2,000 points of the optimal front and
their hypervolume, with the objectives normalized by the front's ideal and nadir points, as a share
of that of a sample of the front with at least as many points as the population; then the
hypervolumes of the whole front and of the sample.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its runs for the plot on the example's
page, with trace.py.

    python examples/das_cmop9/main.py
"""

import genoxide as gx

from trace import Trace

# the reference point of the hypervolume, with the objectives normalized by the front's ideal and
# nadir points
REFERENCE = [1.1, 1.1, 1.1]

problem = gx.problems.DasCmop9()
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
sample = normalized(problem.optimal_front(300))


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


# the paper's settings: NSGA-II, a population of 300 for 1,000 generations
algorithm = gx.Nsga2(
    problem.genome,
    objectives=problem.objectives,
    population_size=300,
    crossover=gx.SimulatedBinaryCrossover(20),
    mutation=gx.PolynomialMutation(20, rate=rate),
    seed=1,
)
run("NSGA-II, η = 20", algorithm, problem, 1_000)
# NSGA-III with 276 reference directions (22 divisions) and polynomial mutation with η = 5
algorithm = gx.Nsga3(
    problem.genome,
    objectives=problem.objectives,
    reference_directions=gx.das_dennis(3, 22),
    population_size=300,
    crossover=gx.SimulatedBinaryCrossover(20),
    crossover_rate=1.0,
    mutation=gx.PolynomialMutation(5, rate=rate),
    seed=1,
)
run("NSGA-III, η = 5", algorithm, problem, 1_000)
whole = normalized(problem.optimal_front(3_000))
print(
    f"the whole front: hypervolume {gx.indicators.hypervolume(whole, REFERENCE):.4f}; "
    f"a sample of {len(sample)} of its points: "
    f"{gx.indicators.hypervolume(sample, REFERENCE):.4f}"
)
trace.write()
