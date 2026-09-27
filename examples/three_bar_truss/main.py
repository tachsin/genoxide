"""The three-bar truss (Nowacki, 1974): the least volume of a planar truss of three bars whose
stresses stay within the allowed stress under a load. A constrained continuous problem, whose
minimum volume, 100 (√2 + √6/2) ≈ 263.895843, is known in closed form.

The variables are the cross-sections of the two outer bars and of the middle one. genoxide's
``ThreeBarTruss`` gives the volume and the violation of the three stress constraints, which Deb's
feasibility rules compare. SHADE, a differential evolution, searches the genes until it's within
1e-10 of the minimum, relative to it, and the example prints the best design and the constraints
at their limits.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/three_bar_truss/main.py
"""

import genoxide as gx

from trace import Trace

# a constraint within this of 0 is at its limit: active
ACTIVE = 1e-6

problem = gx.problems.engineering.ThreeBarTruss()
minimum = problem.optimum.value
de = gx.De(problem.genome, objective=problem.objective, seed=1)
# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(problem)
result = de.run(
    problem,
    target=minimum * (1.0 + 1e-10),
    evaluations=50_000,
    on_generation=trace.on_generation,
)

a1, a2 = result.best_genome.tolist()
constraints = problem.constraints(result.best_genome).tolist()
active = [f"g{i + 1}" for i, g in enumerate(constraints) if abs(g) <= ACTIVE]
print(
    f"volume {result.best_fitness:.6f} after {result.evaluations} evaluations "
    f"(the minimum: {minimum:.6f})"
)
print(f"violation {result.violation:.6f}")
print(f"A1 {a1:.6f}, A2 {a2:.6f}")
print(f"active constraints: {', '.join(active)}")
trace.write()
