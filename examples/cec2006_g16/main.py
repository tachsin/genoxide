"""CEC 2006 g16: a nonlinear function of 5 variables, through a chain of 17 intermediate
quantities, under 38 inequality constraints, from the CEC 2006 special session on constrained
optimization (Liang et al., 2006). The best known value is −1.90515525853479.

genoxide's ``G16`` gives the value of a solution and its constraint violation, which Deb's
feasibility rules compare: a feasible solution beats an infeasible one. CMA-ES with IPOP restarts
searches the 5 variables within the report's budget of 500,000 evaluations, and stops once the
error f(x) − f* is at most 1e-8. The example prints the best solution and its constraints.
``run`` evaluates the problem in Rust.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/cec2006_g16/main.py
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
# a constraint within this of its boundary is active: the scales differ, g36 bounds y16, which is
# about 140,000 there, and the run stops with the active constraints within about 1e-5
ACTIVE = 1e-4


def significant(value, digits):
    """``digits`` significant digits, e.g. 29.9953 or -30665.5 for 6."""
    magnitude = math.floor(math.log10(abs(value)))
    return f"{value:.{max(digits - 1 - magnitude, 0)}f}"


def scientific(value, decimals):
    """Scientific notation as Rust writes it, e.g. 1.2e-5 for 1 decimal."""
    mantissa, exponent = f"{value:.{decimals}e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def count(evaluations):
    """The evaluations, or "never"."""
    return "never" if evaluations is None else str(evaluations)


problem = gx.problems.cec2006.G16()
optimum = problem.optimum
f_star = optimum.value
cmaes = gx.Cmaes(problem.genome, restarts="ipop", objective=problem.objective, seed=1)
# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(problem)
# the evaluations when the best is first feasible, and when its error first meets the report's
# criterion of success; the restarts, each of which doubles the population
first = {"feasible": None, "success": None, "restarts": 0, "size": 0}


def on_generation(progress):
    _, violations = problem.evaluate(progress.best_genome[np.newaxis])
    if violations[0] == 0.0:
        error = progress.best_fitness - f_star
        if first["feasible"] is None:
            first["feasible"] = progress.evaluations
        if first["success"] is None and error <= SUCCESS:
            first["success"] = progress.evaluations
    size = len(progress.scores)
    if first["size"] and size > first["size"]:
        first["restarts"] += 1
    first["size"] = size
    trace.record(progress)


result = cmaes.run(
    problem, target=f_star + ERROR, evaluations=BUDGET, on_generation=on_generation
)

value = result.best_fitness
print("IPOP-CMA-ES with Deb's feasibility rules on g16, seed 1")
if result.stop_reason == "target":
    stop, error = "stopped by the target", f"< {scientific(ERROR, 0)}"
else:
    stop, error = "stopped", scientific(value - f_star, 1)
feasibility = "feasible" if result.violation == 0.0 else "infeasible"
print(
    f"{stop} after {result.evaluations} evaluations and {first['restarts']} restarts: "
    f"f(x) - f* {error}, {feasibility}"
)
print(
    f"first feasible after {count(first['feasible'])} evaluations, f(x) - f* <= 1e-4 after "
    f"{count(first['success'])}"
)
proven = "proven" if optimum.proven else "best known"
print(f"f(x) {significant(value, 6)}, f* {significant(f_star, 6)} ({proven})")
x = result.best_genome.tolist()
print(", ".join(f"x{i} {significant(xi, 6)}" for i, xi in enumerate(x, 1)))
# g1 to g38: the active ones, and how many of the others are satisfied or violated
g = problem.constraints(result.best_genome).tolist()
active = [f"g{i}" for i, gi in enumerate(g, 1) if abs(gi) <= ACTIVE]
satisfied = sum(gi < -ACTIVE for gi in g)
violated = sum(gi > ACTIVE for gi in g)
print(f"active (|g| <= 1e-4): {', '.join(active)}; {satisfied} satisfied, {violated} violated")
trace.write()
