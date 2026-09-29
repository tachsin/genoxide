"""Viennet 3 (VNT3): minimize three objectives of two variables, two of which depend only on the
distance from the origin, so that the Pareto front is two curves.

NSGA-III with the 91 reference directions of Das and Dennis's method with 12 divisions, and
SMS-EMOA, each with a population of 92 for 50 generations. Prints the size of each final front, how
many of its solutions are on each of the two curves, and its hypervolume. run evaluates the problem
in Rust.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of the SMS-EMOA run for the plot on the
example's page, with trace.py.

    python examples/viennet3/main.py
"""

import numpy as np

import genoxide as gx

from trace import Trace

problem = gx.problems.Viennet3()
# the reference point of the hypervolume: the nadir point (8.1964, 17.0370, 0.1760) plus
# a tenth of each objective's range on the front, from the ideal point (0, 15, −0.1),
# rounded to 4 decimals: (9.0160, 17.2407, 0.2036)
ideal, nadir = problem.ideal_point, problem.nadir_point
REFERENCE = (np.round((nadir + (nadir - ideal) / 10) * 1e4) / 1e4).tolist()
# the whole front's hypervolume, about 5.3255
WHOLE = 5.3255


def run(name, algorithm, on_generation=None):
    """Runs ``algorithm`` for 50 generations, and reports its front."""
    result = algorithm.run(problem, generations=50, on_generation=on_generation)
    # the optimal solutions are on two curves: one where x₁² + x₂² ≤ 1.5, near the origin, and one
    # where x₁² + x₂² ≥ 4.19
    inner = int((np.sum(result.front_genomes**2, axis=1) < 3.0).sum())
    front = result.front_objectives
    volume = gx.indicators.hypervolume(front, REFERENCE)
    print(
        f"{name:<8} {len(front)} solutions, {inner} near the origin and {len(front) - inner} "
        f"farther out, hypervolume {volume:.4f}, {volume / WHOLE * 100:.1f}% of the whole front's"
    )


# polynomial mutation at a rate of 0.5, one of the two genes per child on average
settings = dict(
    objectives=problem.objectives,
    population_size=92,
    crossover=gx.SimulatedBinaryCrossover(30),
    mutation=gx.PolynomialMutation(20, rate=0.5),
    seed=1,
)
# 91 directions and a population of 92, the multiple of 4 above
run("NSGA-III", gx.Nsga3(problem.genome, reference_directions=gx.das_dennis(3, 12), **settings))
# with GENOXIDE_TRACE=<file>, a trace of this run for the plot on the example's page
trace = Trace(REFERENCE)
run("SMS-EMOA", gx.SmsEmoa(problem.genome, **settings), trace.on_generation)

print(f"the whole front: hypervolume {WHOLE}")
trace.write()
