"""Pressure vessel design (Sandgren, 1990): the cheapest cylindrical vessel with hemispherical heads
that holds 1,296,000 cubic inches, a constrained mixed discrete-continuous problem. The minimum
cost is 6059.714335.

The variables are the thickness of the shell and of the heads, multiples of 0.0625 inch, and the
inner radius and the length of the shell. genoxide's ``PressureVessel`` rounds the first two genes
to whole plates, and its fitness is the cost and the violation of the four constraints, which
Deb's feasibility rules compare. SHADE, a differential evolution, searches the genes.

    python examples/pressure_vessel/main.py
"""

import genoxide as gx

problem = gx.problems.engineering.PressureVessel()
minimum = problem.optimum.value
de = gx.De(problem.genome, objective=problem.objective, seed=1)
result = de.run(problem, evaluations=50_000)

shell, head, radius, length = problem.design(result.best_genome).tolist()
print(
    f"cost {result.best_fitness:.6f} after {result.evaluations} evaluations "
    f"(the minimum: {minimum:.6f})"
)
print(f"violation {result.violation:.6f}")
print(f"shell {shell:.4f}, heads {head:.4f}, radius {radius:.6f}, length {length:.6f}")
