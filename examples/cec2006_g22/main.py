"""CEC 2006 g22: the linear function x1 of 22 variables under 1 nonlinear inequality and 19
equality constraints, from the CEC 2006 special session on constrained optimization (Liang et al.,
2006). The best known value is 236.430975504001, with the equalities met within the report's
tolerance of 0.0001.

genoxide's ``G22`` gives the value of a solution and its constraint violation, which Deb's
feasibility rules compare: a feasible solution beats an infeasible one, and two infeasible ones
compare by violation. L-SHADE searches the 22 variables for the report's budget of 500,000
evaluations, and doesn't find a feasible solution. The example prints the least violation it
finds, with its solution and constraints. ``run`` evaluates the problem in Rust.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/cec2006_g22/main.py
"""

import math

import genoxide as gx
from genoxide.problems.cec2006 import EQUALITY_TOLERANCE

from trace import Trace

# the CEC 2006 report's budget of evaluations per run
BUDGET = 500_000
# a constraint within this of its boundary is active
ACTIVE = 1e-6


def significant(value, digits):
    """``digits`` significant digits, e.g. 29.9953 or -30665.5 for 6."""
    if value == 0:
        return "0"
    magnitude = math.floor(math.log10(abs(value)))
    return f"{value:.{max(digits - 1 - magnitude, 0)}f}"


def state(g):
    """A constraint g(x) <= 0: "active" on its boundary, else its value."""
    return "active" if abs(g) <= ACTIVE else significant(g, 4)


def scientific(value, decimals):
    """Scientific notation as Rust writes it, e.g. 1.2e-5 for 1 decimal."""
    mantissa, exponent = f"{value:.{decimals}e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def gene(value):
    """A gene: in scientific notation when it's that close to 0, e.g. 3.0e-12, else to 6
    significant digits."""
    return scientific(value, 1) if value != 0 and abs(value) < 1e-4 else significant(value, 6)


# the report's equality tolerance, EQUALITY_TOLERANCE
problem = gx.problems.cec2006.G22()
optimum = problem.optimum
# SHADE with a population that shrinks over the budget, from 18 · 22 = 396 to 4
l_shade = gx.De(problem.genome, l_shade=BUDGET, objective=problem.objective, seed=1)
# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(problem)
# the evaluations when the best is first feasible
first = {"feasible": None}


def on_generation(progress):
    _, violation = problem(progress.best_genome)
    if first["feasible"] is None and violation == 0.0:
        first["feasible"] = progress.evaluations
    trace.record(progress)


result = l_shade.run(problem, evaluations=BUDGET, on_generation=on_generation)

print("L-SHADE with Deb's feasibility rules on g22, seed 1")
if result.violation == 0.0:
    feasibility = "feasible"
else:
    feasibility = f"infeasible, violation {significant(result.violation, 4)}"
print(f"stopped after {result.evaluations} evaluations: {feasibility}")
if first["feasible"] is None:
    print("no feasible solution found")
else:
    print(f"first feasible after {first['feasible']} evaluations")
print(
    f"f(x) {significant(result.best_fitness, 6)}, f* {significant(optimum.value, 6)} (best known)"
)
x = result.best_genome.tolist()
print(", ".join(f"x{i} {gene(xi)}" for i, xi in enumerate(x, 1)))
# g1, then h1 to h19, each equality as its excess over the tolerance, 0 when it's met
constraints = problem.constraints(result.best_genome).tolist()
g, h = constraints[:1], constraints[1:]
states = [f"g{i} {state(gi)}" for i, gi in enumerate(g, 1)]
states += [f"h{i} {state(max(abs(hi) - EQUALITY_TOLERANCE, 0.0))}" for i, hi in enumerate(h, 1)]
print(", ".join(states))
trace.write()
