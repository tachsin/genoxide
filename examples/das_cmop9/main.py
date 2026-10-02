"""DAS-CMOP9: minimize three objectives over 30 variables subject to 7 constraints, whose front is
patches of a sphere, with linked variables, with MOEA/D-DE and NSGA-II.

From genoxide's problems.DasCmop9; run evaluates it in Rust. Runs MOEA/D with differential
evolution (MOEA/D-DE) and, as a contrast, NSGA-II with the paper's settings, a population of 300
for 1,000 generations each, with the difficulty triplet of the paper's figure 6, (0.5, 0.5, 0.5).
Prints each final front's size, how many of its solutions are feasible, their IGD+ to 2,000 points
of the optimal front and their hypervolume, with the objectives normalized by the front's ideal and
nadir points, as a share of that of a sample of the front with at least as many points as the
population, and the median and largest distance from them to a dense sample of the front. Then the
same indicators for the best point of the dense sample for each of MOEA/D's 300 weight vectors, and
for 666, and the hypervolumes of the whole front and of the sample.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its runs for the plot on the example's
page, with trace.py.

    python examples/das_cmop9/main.py
"""

import numpy as np

import genoxide as gx

from trace import Trace

# the reference point of the hypervolume, with the objectives normalized by the front's ideal and
# nadir points
REFERENCE = [1.1, 1.1, 1.1]

# the paper's population and evaluations: 300 for 1,000 generations; MOEA/D's 300 weight vectors
# are Das and Dennis's with 23 divisions
POPULATION = 300
GENERATIONS = 1_000
DIVISIONS = 23

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
sample = normalized(problem.optimal_front(POPULATION))
# a dense sample of the front, at least 40,000 points
dense = problem.optimal_front(40_000)


def indicators(points):
    """The IGD+ of ``points`` to 2,000 points of the optimal front, their hypervolume, and that as a
    percentage of the sample's, with normalized objectives."""
    found = normalized(points)
    distance = gx.indicators.igd_plus(found, normalized(problem.optimal_front(2_000)))
    volume = gx.indicators.hypervolume(found, REFERENCE)
    return distance, volume, 100 * volume / gx.indicators.hypervolume(sample, REFERENCE)


def distances(points):
    """The median and the largest distance from a point of ``points`` to the nearest point of the
    dense sample, with normalized objectives."""
    near = normalized(dense)
    found = [np.sqrt(((near - p) * (near - p)).sum(axis=1).min()) for p in normalized(points)]
    return np.median(found), max(found)


def per_weight(weights):
    """For each weight vector, the point of the dense sample that minimizes MOEA/D's Tchebycheff
    value maxⱼ wⱼ |fⱼ − zⱼ| around the ideal point z (of two equal, the one with the least
    Σ (fⱼ − zⱼ), then the first), each point once, sorted."""
    shifted = dense - ideal
    sums = (shifted[:, 0] + shifted[:, 1]) + shifted[:, 2]
    best = []
    for w in weights:
        values = (w * np.abs(shifted)).max(axis=1)
        tied = np.flatnonzero(values == values.min())
        best.append(dense[tied[np.argmin(sums[tied])]])
    return np.unique(np.array(best), axis=0)


def run(name, algorithm):
    """Runs ``algorithm`` for the paper's 1,000 generations, and prints its final front's size, how
    many of its solutions the problem finds feasible, their IGD+ to 2,000 points of the optimal
    front and their hypervolume, with normalized objectives, as a share of the sample's, and the
    median and largest distance from them to the dense sample of the front."""
    result = algorithm.run(problem, generations=GENERATIONS, on_generation=trace.front(name))
    objectives, violations = problem.evaluate(result.front_genomes)
    feasible = objectives[violations == 0]
    noun = "solution" if len(objectives) == 1 else "solutions"
    start = f"{name}, {GENERATIONS} generations: {len(objectives)} {noun}, "
    if len(feasible) == 0:
        print(start + f"none feasible, the least violation {violations.min():.4f}")
        return
    distance, volume, percent = indicators(feasible)
    median, largest = distances(feasible)
    print(
        start + f"{len(feasible)} feasible, IGD+ {distance:.4f}, hypervolume {volume:.4f}, "
        f"{percent:.1f}% of the sample's; from the front: {median:.4f} (median), "
        f"{largest:.4f} (farthest)"
    )


# MOEA/D-DE: a subproblem per weight vector, 30 neighbours (the paper's 0.1 N), parents from the
# neighbourhood with probability 0.2, differential evolution with F = 0.5 and CR = 1
moead = gx.Moead(
    problem.genome,
    objectives=problem.objectives,
    weights=gx.das_dennis(3, DIVISIONS),
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
# what a decomposition into Tchebycheff subproblems can reach: each weight vector's best point of
# the dense sample, with MOEA/D's 300 vectors and with 666 (35 divisions)
for divisions in (DIVISIONS, 35):
    weights = gx.das_dennis(3, divisions)
    best = per_weight(weights)
    distance, volume, percent = indicators(best)
    print(
        f"the best point of the front for each of {len(weights)} weight vectors: {len(best)} "
        f"points, IGD+ {distance:.4f}, hypervolume {volume:.4f}, {percent:.1f}% of the sample's"
    )
whole = normalized(problem.optimal_front(3_000))
print(
    f"the whole front: hypervolume {gx.indicators.hypervolume(whole, REFERENCE):.4f}; "
    f"a sample of {len(sample)} of its points: "
    f"{gx.indicators.hypervolume(sample, REFERENCE):.4f}"
)
trace.write()
