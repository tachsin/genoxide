"""DAS-CMOP3: minimize two objectives over 30 variables subject to 11 constraints, whose front is
disconnected, with MOEA/D-DE and NSGA-II.

Fan et al.'s DAS-CMOP3, from genoxide's problems.DasCmop3, with the difficulty triplet of the paper's
figure 6, (0, 0.5, 0.5); run evaluates it in Rust. Runs MOEA/D with differential evolution
(MOEA/D-DE) and, as a contrast, NSGA-II with the paper's settings, a population of 300 for 1,000
generations each (300,000 evaluations), with constraint dominance. Prints each final front's size,
how many of its solutions are feasible, and its IGD+ to 500 points of the optimal front and
hypervolume, with the objectives normalized by the front's ideal and nadir points, as a share of
that of a sample of the front with as many points as the population; then the hypervolumes of the
whole front and of the sample.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its runs for the plot on the example's
page, with trace.py.

    python examples/das_cmop3/main.py
"""

import genoxide as gx

from trace import Trace

# the reference point of the hypervolume, with the objectives normalized by the front's ideal and
# nadir points: 1.1 times the nadir point
REFERENCE = [1.1, 1.1]

# the paper's population and evaluations: 300 for 1,000 generations
POPULATION = 300
GENERATIONS = 1_000

problem = gx.problems.DasCmop3()
ideal, nadir = problem.ideal_point, problem.nadir_point
rate = 1 / problem.dimensions


def normalized(points):
    """The objectives normalized by the front's ideal and nadir points: the front spans [0, 1] in
    each."""
    return (points - ideal) / (nadir - ideal)


# with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
trace = Trace(problem, normalized, REFERENCE)
# the sample: as many points of the optimal front as the population has, about what a front of that
# many solutions can be
sample = normalized(problem.optimal_front(POPULATION))


def run(name, algorithm):
    """Runs ``algorithm`` for the paper's 1,000 generations, and prints its final front's size, how
    many of it are feasible, and its IGD+ to 500 points of the optimal front and hypervolume, with
    normalized objectives, as a share of the sample's."""
    result = algorithm.run(problem, generations=GENERATIONS, on_generation=trace.fronts(name))
    objectives, violations = problem.evaluate(result.front_genomes)
    feasible = objectives[violations == 0]
    noun = "solution" if len(objectives) == 1 else "solutions"
    start = f"{name}, {GENERATIONS} generations: {len(objectives)} {noun}, "
    if len(feasible) == 0:
        print(start + f"none feasible, the least violation {violations.min():.4f}")
        return
    optimal = normalized(problem.optimal_front(500))
    distance = gx.indicators.igd_plus(normalized(feasible), optimal)
    volume = gx.indicators.hypervolume(normalized(feasible), REFERENCE)
    percent = 100 * volume / gx.indicators.hypervolume(sample, REFERENCE)
    print(
        start + f"{len(feasible)} feasible, IGD+ {distance:.4f}, hypervolume {volume:.4f}, "
        f"{percent:.1f}% of the sample's"
    )


# MOEA/D-DE: a subproblem per weight vector, 30 neighbours (the paper's 0.1 N), parents from the
# neighbourhood with probability 0.2, differential evolution with F = 0.5 and CR = 1
moead = gx.Moead(
    problem.genome,
    objectives=problem.objectives,
    weights=gx.das_dennis(2, POPULATION - 1),
    neighbors=30,
    neighbor_mating=0.2,
    crossover=gx.DifferentialEvolutionCrossover(f=0.5, cr=1.0),
    mutation=gx.PolynomialMutation(20, rate=rate),
    seed=1,
)
run("MOEA/D-DE", moead)
# the contrast: NSGA-II with the paper's settings
nsga2 = gx.Nsga2(
    problem.genome,
    objectives=problem.objectives,
    population_size=POPULATION,
    crossover=gx.SimulatedBinaryCrossover(20),
    mutation=gx.PolynomialMutation(20, rate=rate),
    seed=1,
)
run("NSGA-II", nsga2)
# the hypervolume of the whole front, from 20,000 of its points
whole = normalized(problem.optimal_front(20_000))
print(
    f"the whole front: hypervolume {gx.indicators.hypervolume(whole, REFERENCE):.4f}; "
    f"a sample of {len(sample)} of its points: "
    f"{gx.indicators.hypervolume(sample, REFERENCE):.4f}"
)
trace.write()
