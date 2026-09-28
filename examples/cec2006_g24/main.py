"""CEC 2006 g24: the linear function −x1 − x2 of 2 variables under 2 quartic inequality
constraints, from the CEC 2006 special session on constrained optimization (Liang et al., 2006).
The minimum is −5.50801327159536, proven, with both constraints active.

genoxide's ``G24`` gives the value of a solution and its constraint violation, which Deb's
feasibility rules compare: a feasible solution beats an infeasible one. CMA-ES, restarted with a
growing population (IPOP) when it converges, searches the 2 variables within the report's budget
of 500,000 evaluations, and stops once the error f(x) − f* is at most 1e-8. The example prints the
best solution and its constraints. ``run`` evaluates the problem in Rust.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/cec2006_g24/main.py
"""

import math

import genoxide as gx
import numpy as np

from trace import Trace

# the CEC 2006 report's budget of evaluations per run
BUDGET = 500_000
# the run stops once its best is feasible with an error f(x) - f* at most this
ERROR = 1e-8
# the report counts a run as successful once its error is at most this
SUCCESS = 1e-4
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


def count(evaluations):
    """The evaluations, or "never"."""
    return "never" if evaluations is None else str(evaluations)


def restarts(sizes):
    """The restarts, from the population sizes of the runs."""
    joined = ", ".join(map(str, sizes))
    if len(sizes) == 1:
        return f"no restart, a population of {sizes[0]}"
    if len(sizes) == 2:
        return f"1 restart, populations {joined}"
    return f"{len(sizes) - 1} restarts, populations {joined}"


problem = gx.problems.cec2006.G24()
optimum = problem.optimum
f_star = optimum.value
cmaes = gx.Cmaes(problem.genome, objective=problem.objective, restarts="ipop", seed=1)
# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(problem)
# the evaluations when the best is first feasible, and when its error first meets the report's
# criterion of success
first = {"feasible": None, "success": None}
# the population sizes of the runs: IPOP doubles it at each restart
sizes = []


def on_generation(progress):
    _, violations = problem.evaluate(progress.best_genome[np.newaxis])
    if violations[0] == 0.0:
        error = progress.best_fitness - f_star
        if first["feasible"] is None:
            first["feasible"] = progress.evaluations
        if first["success"] is None and error <= SUCCESS:
            first["success"] = progress.evaluations
    size = len(progress.population)
    if not sizes or sizes[-1] != size:
        sizes.append(size)
    trace.record(progress)


result = cmaes.run(
    problem, target=f_star + ERROR, evaluations=BUDGET, on_generation=on_generation
)

value = result.best_fitness
print("CMA-ES with IPOP restarts and Deb's feasibility rules on g24, seed 1")
if result.stop_reason == "target":
    stop, error = "stopped by the target", f"< {scientific(ERROR, 0)}"
else:
    stop, error = "stopped", scientific(value - f_star, 1)
feasibility = "feasible" if result.violation == 0.0 else "infeasible"
print(f"{stop} after {result.evaluations} evaluations: f(x) - f* {error}, {feasibility}")
print(
    f"first feasible after {count(first['feasible'])} evaluations, f(x) - f* <= 1e-4 after "
    f"{count(first['success'])}"
)
print(restarts(sizes))
proven = "proven" if optimum.proven else "best known"
print(f"f(x) {significant(value, 6)}, f* {significant(f_star, 6)} ({proven})")
x = result.best_genome.tolist()
print(", ".join(f"x{i} {significant(xi, 6)}" for i, xi in enumerate(x, 1)))
constraints = problem.constraints(result.best_genome).tolist()
print(", ".join(f"g{i} {state(g)}" for i, g in enumerate(constraints, 1)))
trace.write()
