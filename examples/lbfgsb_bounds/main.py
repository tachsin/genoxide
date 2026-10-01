"""L-BFGS-B with a bound that cuts the valley: minimize Rosenbrock's function in the box
[-2, 0.5] x [-1, 3], whose minimum (1, 1) lies outside it. The minimum in the box is on its edge,
at (0.5, 0.25), where f = 0.25, and L-BFGS-B lands on it exactly.

Then Nelder-Mead in the same box, for contrast: it approaches the bound without reaching it.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/lbfgsb_bounds/main.py
"""

import genoxide as gx

from trace import Trace

BOUNDS = [(-2.0, 0.5), (-1.0, 3.0)]
START = [-1.2, 1.0]


def scientific(value):
    """Two significant digits, e.g. 1.2e-7."""
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace()
rows = []
criteria = []


def control(lbfgsb, progress):
    x = progress.population[0]
    rows.append(
        f"{progress.generation:>5}  {progress.evaluations:>11}  {x[0]:>9.6f}  {x[1]:>9.6f}  "
        f"{progress.scores[0]:>9.6f}  {scientific(lbfgsb.projected_gradient):>18}"
    )
    criteria.append(lbfgsb.converged)


rosenbrock = gx.problems.Rosenbrock(2)
lbfgsb = gx.Lbfgsb(gx.Real(BOUNDS), initial_genome=START, objective="minimize")
result = lbfgsb.run(rosenbrock, evaluations=1_000, on_generation=trace.record, control=control)

print("Rosenbrock's function in [-2, 0.5] x [-1, 3], from (-1.2, 1); its minimum (1, 1) is outside")
print("L-BFGS-B, the current point after each round")
print("round  evaluations         x1         x2          f  projected gradient")
for row in rows:
    print(row)
assert result.stop_reason == "converged"
assert criteria[-1] == "projected_gradient"
x = [float(gene) for gene in result.best_genome]
# the gradient at the end: -400 x1 (x2 - x1^2) - 2 (1 - x1), and 200 (x2 - x1^2)
valley = x[1] - x[0] * x[0]
gradient = [-400.0 * x[0] * valley + 2.0 * (x[0] - 1.0), 200.0 * valley]
print(
    f"L-BFGS-B: converged at ({x[0]!r}, {x[1]!r}), f = {result.best_fitness!r}, "
    f"after {result.evaluations} evaluations"
)
print(
    f"the gradient there is ({gradient[0]!r}, {gradient[1]!r}): "
    "f falls only beyond the bound x1 = 0.5"
)

nelder_mead = gx.NelderMead(gx.Real(BOUNDS), initial_genome=START, objective="minimize")
contrast = nelder_mead.run(rosenbrock, evaluations=10_000)
y = [float(gene) for gene in contrast.best_genome]
print(
    f"Nelder-Mead, for contrast: ({y[0]!r}, {y[1]!r}), f = {contrast.best_fitness!r}, "
    f"after {contrast.evaluations} evaluations"
)
trace.write()
