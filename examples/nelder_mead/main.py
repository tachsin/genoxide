"""Nelder-Mead: minimize Rosenbrock's function in two dimensions from the classic start (-1.2, 1),
with the simplex method, to its minimum at (1, 1).

The simplex is a triangle that reflects, expands, contracts and shrinks down the curved valley of
the function. The run stops when the triangle has collapsed on the minimum. Then the same run
with speculative asks: the same triangles in fewer rounds of evaluations.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/nelder_mead/main.py
"""

import numpy as np

import genoxide as gx

from trace import Trace

# a row of the table every this many rounds
EVERY = 20
BOUNDS = [(-2.0, 2.0), (-1.0, 3.0)]


def scientific(value):
    """Two significant digits, e.g. 1.2e-7."""
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def nelder_mead(speculative):
    return gx.NelderMead(
        gx.Real(BOUNDS),
        initial_genome=[-1.2, 1.0],
        speculative=speculative,
        objective="minimize",
    )


def row(progress):
    """A row of the table: the round, the evaluations, the best value and the simplex size, the
    largest difference between a vertex and the best one in a gene, as a fraction of its range."""
    vertices = progress.population
    widths = np.array([high - low for low, high in BOUNDS])
    size = float(np.max(np.abs(vertices[1:] - vertices[0]) / widths))
    return (
        f"{progress.generation:>5}  {progress.evaluations:>11}  "
        f"{scientific(progress.best_fitness):>10}  {scientific(size):>12}"
    )


# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace()
rows = []


def on_generation(progress):
    trace.record(progress)
    rows.append(row(progress))


rosenbrock = gx.problems.Rosenbrock(2)
result = nelder_mead(False).run(rosenbrock, evaluations=10_000, on_generation=on_generation)

print("Rosenbrock's function in two dimensions, from (-1.2, 1) where f = 24.2")
print("Nelder-Mead with Gao and Han's coefficients, a first simplex of 0.1 of each range")
print("round  evaluations  best value  simplex size")
for round_, line in enumerate(rows):
    if round_ % EVERY == 0 or round_ == len(rows) - 1:
        print(line)
assert result.stop_reason == "converged"
best = result.best_genome
print(
    f"converged after {result.generations} rounds and {result.evaluations} evaluations, "
    f"at ({best[0]:.10f}, {best[1]:.10f}), f = {scientific(result.best_fitness)}"
)

speculative = nelder_mead(True).run(rosenbrock, evaluations=10_000)
same = "the same end" if np.array_equal(speculative.best_genome, best) else "another end"
print(
    f"speculative asks: {same} after {speculative.generations} rounds and "
    f"{speculative.evaluations} evaluations"
)
trace.write()
