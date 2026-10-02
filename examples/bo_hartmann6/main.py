"""Bayesian optimization of Hartmann's 6-D function in batches: 4 points a round, chosen one after
the other with the Kriging believer and evaluated in parallel, to within 1e-4 of the global
minimum. Then, as a contrast, the same search one point a round: fewer evaluations, more rounds.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of both runs for the plot on the example's
page, with trace.py.

    python examples/bo_hartmann6/main.py
"""

import numpy as np

import genoxide as gx

import trace

# how close to the global minimum, and the evaluations each run may take at most
TOLERANCE = 1e-4
BUDGET = 200
SEED = 3


def scientific(value):
    """Two significant digits, e.g. 1.2e-7."""
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


problem = gx.problems.Hartmann6()
minimum = problem.optimum.value


def search(batch, show):
    """A search in batches of ``batch`` points evaluated in parallel, to within the tolerance of
    the minimum or the budget, printing a row per round if ``show``; its result, and its best
    value's distance above the minimum after each round."""
    rounds = []

    def on_generation(progress):
        rounds.append(progress.best_fitness - minimum)
        if show:
            print(
                f"{progress.generation:>5} {progress.evaluations:>12} "
                f"{progress.best_fitness:>13.6f} {scientific(progress.best_fitness - minimum):>10}"
            )

    bo = gx.Bo(problem.genome, batch=batch, objective="minimize", seed=SEED)
    result = bo.run(
        problem,
        target=minimum + TOLERANCE,
        evaluations=BUDGET,
        parallel=True,
        on_generation=on_generation,
    )
    return result, rounds


print(f"Hartmann's 6-D function in [0, 1]^6: global minimum {minimum:.6f}")
print("14 points of a Latin hypercube, then 4 points a round by log-EI and the Kriging believer")
print("round  evaluations          best     f - f*")
batched, rounds = search(4, True)
distance = float(np.linalg.norm(batched.best_genome - problem.optimum.solutions[0]))
print(
    f"4 points a round: within 1e-4 of the minimum after {batched.evaluations} evaluations in "
    f"{batched.generations} rounds, {scientific(distance)} from its point"
)
assert batched.best_fitness - minimum <= TOLERANCE

# the contrast: one point a round, the same seed
single, single_rounds = search(1, False)
reached = single.best_fitness - minimum <= TOLERANCE
print(
    f"1 point a round:  {'within 1e-4 of the minimum' if reached else 'not within the tolerance'}"
    f" after {single.evaluations} evaluations in {single.generations} rounds"
)
trace.write(rounds, single_rounds)
