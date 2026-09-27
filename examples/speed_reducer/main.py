"""Golinski's speed reducer (1970, 1973): the lightest gearbox whose gear teeth and shafts stay
within their stress and deflection limits. A constrained mixed discrete-continuous problem, whose
best known weight is 2996.348165.

The variables are the face width of the gears, the module of their teeth, the number of teeth on
the pinion, an integer, and the lengths and diameters of the two shafts. genoxide's
``SpeedReducer`` rounds the number of teeth when it evaluates a genome, and its fitness is the
weight and the violation of the eleven constraints, which Deb's feasibility rules compare. SHADE,
a differential evolution, searches the genes, and the example prints the best design and the
constraints at their limits.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/speed_reducer/main.py
"""

import genoxide as gx

from trace import Trace

# a constraint within this of 0 is at its limit: active
ACTIVE = 1e-6

problem = gx.problems.engineering.SpeedReducer()
best_known = problem.optimum.value
de = gx.De(problem.genome, objective=problem.objective, seed=1)
# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(problem)
result = de.run(problem, evaluations=50_000, on_generation=trace.on_generation)

b, m, z, l1, l2, d1, d2 = problem.design(result.best_genome).tolist()
constraints = problem.constraints(result.best_genome).tolist()
active = [f"g{i + 1}" for i, g in enumerate(constraints) if abs(g) <= ACTIVE]
print(f"weight {result.best_fitness:.6f} (the best known: {best_known:.6f})")
print(f"violation {result.violation:.6f}")
print(f"face width {b:.6f}, module {m:.6f}, teeth {z:.0f}")
print(f"shaft 1: length {l1:.6f}, diameter {d1:.6f}")
print(f"shaft 2: length {l2:.6f}, diameter {d2:.6f}")
print(f"active constraints: {', '.join(active)}")
trace.write()
