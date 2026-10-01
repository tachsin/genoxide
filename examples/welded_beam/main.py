"""Welded beam design: the cheapest beam welded to a support that carries 6000 lb at 14 inches,
subject to its weld's shear stress, its bending stress, its buckling load and its deflection. A
constrained continuous problem, in the two forms of the literature.

``WeldedBeam`` is the form with seven constraints (Rao, 1996, as restated by Coello Coello, 2000),
``WeldedBeamRagsdell`` the one with five (Ragsdell and Phillips, 1976, as restated by Deb, 2000).
Their fitness is the cost and the constraint violation, which Deb's feasibility rules compare.
SHADE, a differential evolution, solves each until it reaches the best known cost, and the example
prints the best design, the evaluations it took and the best known cost.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/welded_beam/main.py
"""

import genoxide as gx

from trace import Trace

forms = [gx.problems.engineering.WeldedBeam(), gx.problems.engineering.WeldedBeamRagsdell()]
# with GENOXIDE_TRACE=<file>, a trace of the first form's run for the plot on the example's page
trace = Trace(forms[0])
for problem in forms:
    best_known = problem.optimum.value
    de = gx.De(problem.genome, objective=problem.objective, seed=1)
    result = de.run(
        problem,
        target=best_known * (1.0 + 1e-10),
        evaluations=100_000,
        on_generation=trace.on_generation,
    )
    h, l, t, b = result.best_genome.tolist()
    print(
        f"{problem.name}: cost {result.best_fitness:.6f}, violation {result.violation:.6f}, "
        f"after {result.evaluations} evaluations (the best known: {best_known})"
    )
    print(f"  h {h:.6f}, l {l:.6f}, t {t:.6f}, b {b:.6f}")
trace.write()
