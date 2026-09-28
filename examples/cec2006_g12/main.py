"""CEC 2006 g12: a quadratic in 3 variables whose feasible region is 729 disjoint spheres, from the
CEC 2006 special session on constrained optimization (Liang et al., 2006). The minimum is −1 at
(5, 5, 5), the center of one of the spheres, proven.

genoxide's ``G12`` gives the value of a solution and its constraint violation, which Deb's
feasibility rules compare: a feasible solution beats an infeasible one. SHADE, genoxide's
differential evolution, searches the 3 variables within the report's budget of 500,000
evaluations, and stops once the error f(x) − f* is at most 1e-8. The example prints the best
solution, its constraint and the sphere that holds it. ``run`` evaluates the problem in Rust.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/cec2006_g12/main.py
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


problem = gx.problems.cec2006.G12()
optimum = problem.optimum
f_star = optimum.value
shade = gx.De(problem.genome, objective=problem.objective, seed=1)
# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(problem)
# the evaluations when the best is first feasible, and when its error first meets the report's
# criterion of success
first = {"feasible": None, "success": None}


def on_generation(progress):
    _, violations = problem.evaluate(progress.best_genome[np.newaxis])
    if violations[0] == 0.0:
        error = progress.best_fitness - f_star
        if first["feasible"] is None:
            first["feasible"] = progress.evaluations
        if first["success"] is None and error <= SUCCESS:
            first["success"] = progress.evaluations
    trace.record(progress)


result = shade.run(
    problem, target=f_star + ERROR, evaluations=BUDGET, on_generation=on_generation
)

value = result.best_fitness
print("SHADE with Deb's feasibility rules on g12, seed 1")
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
proven = "proven" if optimum.proven else "best known"
# both to 6 decimals, alike for f* = -1 and a value just above it
print(f"f(x) {value:.6f}, f* {f_star:.6f} ({proven})")
x = result.best_genome.tolist()
print(", ".join(f"x{i} {significant(xi, 6)}" for i, xi in enumerate(x, 1)))
# the nearest of the 729 centers (p, q, r), p, q, r in 1..=9, and g, the squared distance to it
# less 0.0625: at most 0 inside its sphere (x >= 0, rounded half up as Rust rounds it)
center = ", ".join(str(min(max(math.floor(xi + 0.5), 1), 9)) for xi in x)
(g,) = problem.constraints(result.best_genome).tolist()
print(f"nearest center ({center}), g {g:.6f} (<= 0 inside its sphere)")
trace.write()
