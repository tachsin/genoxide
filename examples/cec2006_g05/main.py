"""CEC 2006 g05: a cubic in 4 variables with 2 linear inequality constraints and 3 nonlinear
equality constraints, from the CEC 2006 special session on constrained optimization (Liang et
al., 2006). The best known value is 5126.4967140071, with the equalities met within the report's
tolerance of 0.0001.

genoxide's ``G05`` gives the value of a solution and its constraint violation, which Deb's
feasibility rules compare: a feasible solution beats an infeasible one. CMA-ES searches the 4
variables within the report's budget of 500,000 evaluations, and stops once the error f(x) − f*
is at most 1e-8. The example prints the best solution and its constraints. ``run`` evaluates the
problem in Rust.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/cec2006_g05/main.py
"""

import math

import genoxide as gx
from genoxide.problems.cec2006 import EQUALITY_TOLERANCE

from trace import Trace

# the CEC 2006 report's budget of evaluations per run
BUDGET = 500_000
# the run stops once its best is feasible with an error f(x) - f* at most this
ERROR = 1e-8
# a constraint within this of its boundary is active
ACTIVE = 1e-6


def significant(value, digits):
    """``digits`` significant digits, e.g. 29.9953 or -30665.5 for 6."""
    magnitude = math.floor(math.log10(abs(value)))
    return f"{value:.{max(digits - 1 - magnitude, 0)}f}"


def state(g):
    """A constraint g(x) <= 0: "active" on its boundary, else its value."""
    return "active" if abs(g) <= ACTIVE else significant(g, 4)


def scientific(value, decimals):
    """Scientific notation as Rust writes it, e.g. 1.2e-5 for 1 decimal."""
    mantissa, exponent = f"{value:.{decimals}e}".split("e")
    return f"{mantissa}e{int(exponent)}"


problem = gx.problems.cec2006.G05()
optimum = problem.optimum
f_star = optimum.value
cmaes = gx.Cmaes(problem.genome, objective=problem.objective, seed=1)
# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(problem)
result = cmaes.run(
    problem, target=f_star + ERROR, evaluations=BUDGET, on_generation=trace.on_generation
)

# the example prints what runs with other seeds agree on: no evaluations of a run that meets its
# target, and the solution to 4 significant digits
value = result.best_fitness
print("CMA-ES with Deb's feasibility rules on g05, seed 1")
if result.stop_reason == "target":
    stop, error = "stopped by the target", f"< {scientific(ERROR, 0)}"
else:
    stop, error = f"stopped after {result.evaluations} evaluations", scientific(value - f_star, 1)
feasibility = "feasible" if result.violation == 0.0 else "infeasible"
print(f"{stop}: f(x) - f* {error}, {feasibility}")
proven = "proven" if optimum.proven else "best known"
print(f"f(x) {significant(value, 6)}, f* {significant(f_star, 6)} ({proven})")
x = result.best_genome.tolist()
print(", ".join(f"x{i} {significant(xi, 4)}" for i, xi in enumerate(x, 1)))
# g1 and g2, then h3 to h5, each equality as its excess over the tolerance, 0 when it's met
g1, g2, *h = problem.constraints(result.best_genome).tolist()
constraints = [("g", g1), ("g", g2)] + [("h", max(abs(hj) - EQUALITY_TOLERANCE, 0.0)) for hj in h]
print(", ".join(f"{name}{i} {state(g)}" for i, (name, g) in enumerate(constraints, 1)))
trace.write()
