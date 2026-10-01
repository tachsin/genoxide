"""DAS-CMOP3: minimize two objectives over 30 variables subject to 11 constraints, whose front is disconnected,
with NSGA-II.

Fan et al.'s DAS-CMOP3, from genoxide's problems.DasCmop3, with the difficulty triplet of the paper's
figure 6, (0, 0.5, 0.5); run evaluates it in Rust. Runs NSGA-II twice with the paper's settings, a
population of 300 for 1,000 generations (300,000 evaluations): with polynomial mutation with
η = 20, as the paper's, and with η = 5. Prints each final front's size, how many of its solutions
are feasible, and its IGD+ to 500 points of the optimal front and hypervolume, with the objectives
normalized by the front's ideal and nadir points; then the whole front's hypervolume.

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


def normalized(points):
    """The objectives normalized by the front's ideal and nadir points: the front spans [0, 1] in
    each."""
    return (points - ideal) / (nadir - ideal)


# with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
trace = Trace(problem, normalized, REFERENCE)


def run(name, eta):
    """Runs NSGA-II with the paper's population, simulated binary crossover with η = 20 at
    genoxide's default rate of 0.9, and polynomial mutation with the distribution index ``eta``
    at a rate of 1/n per gene; prints its final front's size, how many of it are feasible, and its
    IGD+ to 500 points of the optimal front and hypervolume, with normalized objectives."""
    nsga2 = gx.Nsga2(
        problem.genome,
        objectives=problem.objectives,
        population_size=POPULATION,
        crossover=gx.SimulatedBinaryCrossover(20),
        mutation=gx.PolynomialMutation(eta, rate=1 / problem.dimensions),
        seed=1,
    )
    result = nsga2.run(problem, generations=GENERATIONS, on_generation=trace.fronts(name))
    objectives, violations = problem.evaluate(result.front_genomes)
    feasible = objectives[violations == 0]
    noun = "solution" if len(objectives) == 1 else "solutions"
    start = f"NSGA-II, {name}, {GENERATIONS} generations: {len(objectives)} {noun}, "
    if len(feasible) == 0:
        print(start + f"none feasible, the least violation {violations.min():.4f}")
        return
    optimal = normalized(problem.optimal_front(500))
    distance = gx.indicators.igd_plus(normalized(feasible), optimal)
    volume = gx.indicators.hypervolume(normalized(feasible), REFERENCE)
    print(start + f"{len(feasible)} feasible, IGD+ {distance:.4f}, hypervolume {volume:.4f}")


# the paper's settings: polynomial mutation with η = 20
run("η = 20", 20)
# polynomial mutation with η = 5, whose steps are larger
run("η = 5", 5)
# the hypervolume of the whole front, from 20,000 of its points
whole = normalized(problem.optimal_front(20_000))
print(f"the whole front: hypervolume {gx.indicators.hypervolume(whole, REFERENCE):.4f}")
trace.write()
