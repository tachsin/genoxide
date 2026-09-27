"""CEC 2006 g01: a quadratic in 13 variables under 9 linear inequalities, whose minimum −15 lies on
the boundary of a feasible region that fills about 0.01 % of the box.

The fitness is the value and the constraint violation, which Deb's feasibility rules compare: a
feasible solution beats an infeasible one. SHADE, a differential evolution, has the CEC 2006
report's budget of 500,000 evaluations and stops once it's within 1e-8 (relative) of the minimum.
The example prints the best solution and the constraints active at it.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/cec2006_g01/main.py
"""

import genoxide as gx

from trace import Trace

# the CEC 2006 report's budget
BUDGET = 500_000
# a constraint g with |g| at most this is active: the solution lies on its boundary
ACTIVE = 1e-6

problem = gx.problems.cec2006.G01()
optimum = problem.optimum.value
# within 1e-8 of the minimum, relative to its size
target = optimum + 1e-8 * abs(optimum)
shade = gx.De(problem.genome, objective=problem.objective, seed=1)
# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(problem)
result = shade.run(problem, target=target, evaluations=BUDGET, on_generation=trace.on_generation)

x = result.best_genome.tolist()
constraints = problem.constraints(result.best_genome).tolist()
active = [f"g{i + 1}" for i, g in enumerate(constraints) if abs(g) <= ACTIVE]
print(f"CEC 2006 g01 with SHADE, seed 1: at most {BUDGET} evaluations")
print(
    f"f {result.best_fitness:.6f}, {'feasible' if result.violation == 0 else 'infeasible'}, "
    f"after {result.evaluations} evaluations (the minimum: {optimum:.6f}, proven)"
)
print("x " + " ".join(f"{xi:.4f}" for xi in x))
print(f"active constraints (|g| <= 1e-6): {' '.join(active)}")
trace.write()
