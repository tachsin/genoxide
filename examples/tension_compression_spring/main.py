"""The tension/compression spring (Belegundu, 1982; Arora, 1989): the lightest coil spring whose
deflection, shear stress, surge frequency and outer diameter stay within their limits. A
constrained continuous problem, whose best known weight is 0.01266523.

The variables are the wire diameter d, the mean coil diameter D and the number of active coils N.
genoxide's ``TensionCompressionSpring`` gives the weight and the violation of the four
constraints, which Deb's feasibility rules compare. SHADE, a differential evolution, searches the
genes, and the example prints the best design and the constraints at their limits.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/tension_compression_spring/main.py
"""

import genoxide as gx

from trace import Trace

# a constraint within this of 0 is at its limit: active
ACTIVE = 1e-6

problem = gx.problems.engineering.TensionCompressionSpring()
best_known = problem.optimum.value
de = gx.De(problem.genome, objective=problem.objective, seed=1)
# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(problem)
result = de.run(
    problem,
    target=best_known * (1.0 + 1e-10),
    evaluations=100_000,
    on_generation=trace.on_generation,
)

d, coil, n = result.best_genome.tolist()
constraints = problem.constraints(result.best_genome).tolist()
active = [f"g{i + 1}" for i, g in enumerate(constraints) if abs(g) <= ACTIVE]
print(
    f"weight {result.best_fitness:.7f} after {result.evaluations} evaluations "
    f"(the best known: {best_known})"
)
print(f"violation {result.violation:.6f}")
print(f"d {d:.6f}, D {coil:.6f}, N {n:.6f}")
print(f"active constraints: {', '.join(active)}")
trace.write()
