"""CEC 2006 g03: the largest product of 10 variables on the unit sphere, an equality constraint
that leaves only a thin shell of feasible solutions.

The fitness is the value and the constraint violation, which Deb's feasibility rules compare: a
feasible solution beats an infeasible one. The equality counts as met within the CEC 2006 report's
tolerance, 0.0001. CMA-ES has the report's budget of 500,000 evaluations and stops once it's
within 1e-8 (relative) of the minimum. The example prints the best solution and the equality's
value at it.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/cec2006_g03/main.py
"""

import genoxide as gx

from trace import Trace

# the CEC 2006 report's budget
BUDGET = 500_000

# the report's equality tolerance, EQUALITY_TOLERANCE
problem = gx.problems.cec2006.G03()
optimum = problem.optimum.value
# within 1e-8 of the minimum, relative to its size
target = optimum + 1e-8 * abs(optimum)
cmaes = gx.Cmaes(problem.genome, objective=problem.objective, seed=1)
# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(problem, gx.problems.cec2006.EQUALITY_TOLERANCE)
result = cmaes.run(problem, target=target, evaluations=BUDGET, on_generation=trace.on_generation)

x = result.best_genome.tolist()
(h,) = problem.constraints(result.best_genome).tolist()
print(f"CEC 2006 g03 with CMA-ES, seed 1: at most {BUDGET} evaluations")
print(
    f"f {result.best_fitness:.7f}, {'feasible' if result.violation == 0 else 'infeasible'}, "
    f"after {result.evaluations} evaluations (the minimum: {optimum:.7f}, proven)"
)
print("x " + " ".join(f"{xi:.4f}" for xi in x))
print(f"h1 = x1^2 + ... + x10^2 - 1: {h:.6f} (met when |h1| <= 0.0001)")
trace.write()
