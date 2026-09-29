"""CEC 2006 g17: a piecewise linear function of 6 variables under 4 nonlinear equality
constraints, from the CEC 2006 special session on constrained optimization (Liang et al., 2006).
The best known value is 8853.5338748065, with the equalities met within the report's tolerance of
0.0001.

genoxide's ``G17`` gives the value of a solution and its constraint violation. SHADE, a
differential evolution, compares them with Deb's feasibility rules at an ε level (Takahama and
Sakai, 2006): a violation up to ε counts as none. ε starts at 300 and falls to 0 over the first
150,000 evaluations, and the population is scored again each time it falls. The run has the
report's budget of 500,000 evaluations, and stops once ε is 0 and the error f(x) − f* is at most
1e-8. The example prints the best solution, the pieces of the objective it's on, and its
constraints. The fitness function evaluates the problem in Rust, a generation at a time.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/cec2006_g17/main.py
"""

import math

import genoxide as gx
import numpy as np
from genoxide.problems.cec2006 import EQUALITY_TOLERANCE

from trace import Trace

# the CEC 2006 report's budget of evaluations per run
BUDGET = 500_000
# the run stops once its best is feasible with an error f(x) - f* at most this
ERROR = 1e-8
# the report counts a run as successful once its error is at most this
SUCCESS = 1e-4
# a constraint within this of its boundary is active
ACTIVE = 1e-6
# the ε level at the start: a violation up to it counts as none
EPSILON = 300.0
# the evaluations after which ε is 0
CONTROL = 150_000
# ε falls, and the population is scored again, every this many generations
EVERY = 10


def epsilon(evaluations):
    """The ε level after ``evaluations``: EPSILON (1 - evaluations / CONTROL)^5, then 0."""
    if evaluations >= CONTROL:
        return 0.0
    rest = 1.0 - evaluations / CONTROL
    return EPSILON * rest * rest * rest * rest * rest


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


problem = gx.problems.cec2006.G17()
optimum = problem.optimum
f_star = optimum.value
shade = gx.De(problem.genome, objective=problem.objective, seed=1)
# the ε level in use
level = {"epsilon": EPSILON}


def fitness(genomes):
    """The values of a generation, and their violations beyond ε."""
    values, violations = problem.evaluate(genomes)
    return values, np.maximum(violations - level["epsilon"], 0.0)


# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(problem)
# the evaluations when the best is first feasible, and when its error first meets the report's
# criterion of success
first = {"feasible": None, "success": None}


def on_generation(progress):
    # the best by the ε level, measured without it
    values, violations = problem.evaluate(progress.best_genome[np.newaxis])
    feasible = violations[0] == 0.0
    if feasible:
        error = values[0] - f_star
        if first["feasible"] is None:
            first["feasible"] = progress.evaluations
        if first["success"] is None and error <= SUCCESS:
            first["success"] = progress.evaluations
    trace.record(progress)
    # the run's target, once ε is 0: a feasible best within ERROR of f*
    return not (level["epsilon"] == 0.0 and feasible and values[0] <= f_star + ERROR)


def control(shade, progress):
    # every EVERY generations, and once it reaches 0, ε follows its schedule
    following = epsilon(progress.evaluations)
    due = progress.generation % EVERY == 0 or following == 0.0
    if progress.generation > 0 and due and following != level["epsilon"]:
        level["epsilon"] = following
        shade.reevaluate()


result = shade.run(
    fitness,
    batch=True,
    evaluations=BUDGET,
    on_generation=on_generation,
    control=control,
)

value = result.best_fitness
print("SHADE with Deb's feasibility rules at an epsilon level on g17, seed 1")
if result.stop_reason == "aborted":
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
print(f"f(x) {significant(value, 6)}, f* {significant(f_star, 6)} ({proven})")
x = result.best_genome.tolist()
print(", ".join(f"x{i} {significant(xi, 4)}" for i, xi in enumerate(x, 1)))
# the pieces of f1(x1) and f2(x2) that the solution is on
f1 = "30 x1" if x[0] < 300 else "31 x1"
f2 = "28 x2" if x[1] < 100 else "29 x2" if x[1] < 200 else "30 x2"
print(f"pieces: f1(x1) = {f1}, f2(x2) = {f2}")
# h1 to h4, each as its excess over the tolerance, 0 when it's met
h = problem.constraints(result.best_genome).tolist()
print(
    ", ".join(
        f"h{i} {state(max(abs(hi) - EQUALITY_TOLERANCE, 0.0))}" for i, hi in enumerate(h, 1)
    )
)
trace.write()
