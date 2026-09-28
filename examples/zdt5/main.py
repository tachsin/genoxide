"""ZDT5: minimize two conflicting objectives over a string of 80 bits, whose Pareto front is 31
points behind deceptive fronts, with NSGA-II.

Zitzler, Deb and Thiele's fifth problem, from genoxide's problems.Zdt5; run evaluates it in Rust.
Runs NSGA-II three times for 250 generations: with uniform crossover and a population of 100, with
two-point crossover and a population of 100, and with two-point crossover and a population of
1000. Prints, for each, the g of its front (10 on the optimal front), how many of the 31 optimal
points it found, its IGD+ to them and its hypervolume.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its runs for the plot on the example's
page, with trace.py.

    python examples/zdt5/main.py
"""

import numpy as np

import genoxide as gx

from trace import Trace

# the reference point of the hypervolume: 1.1 times the nadir point (31, 10)
REFERENCE = [34.1, 11.0]

GENERATIONS = 250

problem = gx.problems.Zdt5()
optimal = problem.optimal_front(31)
# with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
trace = Trace(REFERENCE)


def run(name, algorithm):
    """Runs ``algorithm`` for 250 generations, and reports its front."""
    result = algorithm.run(problem, generations=GENERATIONS, on_generation=trace.fronts(name))
    front = result.front_objectives
    # f₁ f₂ = g, a whole number: 10 on the optimal front
    g = np.round(front[:, 0] * front[:, 1]).astype(int)
    low, high = g.min(), g.max()
    spread = f"= {low}" if low == high else f"{low} to {high}"
    # the distinct values of f₁ where g = 10
    found = len(np.unique(front[g == 10, 0]))
    distance = gx.indicators.igd_plus(front, optimal)
    volume = gx.indicators.hypervolume(front, REFERENCE)
    print(
        f"{name:<15}  g {spread:<8} {found:>2} of the 31 optimal points, IGD+ {distance:.4f}, "
        f"hypervolume {volume:.2f}"
    )


# bit-flip mutation at a rate of 1/80, one bit per child on average; uniform crossover takes each
# bit from either parent
settings = dict(objectives=problem.objectives, mutation=gx.BitFlip(rate=1 / 80), seed=1)
uniform = gx.Nsga2(problem.genome, population_size=100, crossover=gx.UniformCrossover(), **settings)
run("uniform, 100", uniform)
# two-point crossover exchanges a segment, and keeps most 5-bit substrings whole
for size in [100, 1000]:
    two_point = gx.Nsga2(
        problem.genome, population_size=size, crossover=gx.PointCrossover(2), **settings
    )
    run(f"two-point, {size}", two_point)

print("the whole front: g = 10, 31 points, hypervolume 323.15")
trace.write()
