"""CEC 2006 g02: a rugged function of 20 variables with many local optima, under a product and a
sum constraint. Its best known value, −0.803619, isn't proven optimal.

The fitness is the value and the constraint violation, which Deb's feasibility rules compare: a
feasible solution beats an infeasible one. SHADE, a differential evolution, with a population of
300 instead of its published 100, uses the CEC 2006 report's whole budget of 500,000 evaluations,
to see whether anything beats the best known value. The example prints the best solution, its gap
to the best known value and the constraints active at it.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/cec2006_g02/main.py
"""

import genoxide as gx

from trace import Trace

# the CEC 2006 report's budget
BUDGET = 500_000
# a constraint g with |g| at most this is active: the solution lies on its boundary
ACTIVE = 1e-6

problem = gx.problems.cec2006.G02()
best_known = problem.optimum.value
shade = gx.De(problem.genome, population_size=300, objective=problem.objective, seed=1)
# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(problem)
result = shade.run(problem, evaluations=BUDGET, on_generation=trace.on_generation)

value = result.best_fitness
# the gap to the best known value, relative to its size: negative if the run beats it
gap = (value - best_known) / abs(best_known)
x = result.best_genome.tolist()
constraints = problem.constraints(result.best_genome).tolist()
active = [f"g{i + 1}" for i, g in enumerate(constraints) if abs(g) <= ACTIVE]
print(f"CEC 2006 g02 with SHADE (population 300), seed 1: {BUDGET} evaluations")
print(
    f"f {value:.6f}, {'feasible' if result.violation == 0 else 'infeasible'} "
    f"(the best known: {best_known:.6f}, not proven)"
)
# the last digits of f differ between platforms: g02 calls the platform's cos
if 0 <= gap < 1e-8:
    print("relative gap to the best known: below 1e-8")
else:
    # as Rust writes it: 1.2e-4
    mantissa, exponent = f"{gap:.1e}".split("e")
    print(f"relative gap to the best known: {mantissa}e{int(exponent)}")
print("x1-x10  " + " ".join(f"{xi:.3f}" for xi in x[:10]))
print("x11-x20 " + " ".join(f"{xi:.3f}" for xi in x[10:]))
print(f"active constraints (|g| <= 1e-6): {' '.join(active)}")
trace.write()
