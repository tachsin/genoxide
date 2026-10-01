"""The cantilever beam (Fleury and Braibant, 1986): the lightest beam of five hollow square
segments that carries a load at its free end, subject to its deflection. A constrained continuous
problem with a proven minimum, 1.339956361.

The variables are the widths of the five segments, from the support to the free end. The weight
grows with their sum, and one constraint limits the deflection. The fitness is the weight and the
constraint violation, which Deb's feasibility rules compare. CMA-ES searches the widths, and the
example prints the best design next to the minimum.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/cantilever_beam/main.py
"""

import genoxide as gx

from trace import Trace

problem = gx.problems.engineering.CantileverBeam()
optimum = problem.optimum
cmaes = gx.Cmaes(problem.genome, objective=problem.objective, seed=1)
# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(problem)
result = cmaes.run(
    problem,
    target=optimum.value * (1.0 + 1e-10),
    evaluations=8_000,
    on_generation=trace.on_generation,
)

weight, minimum = result.best_fitness, optimum.value
print(f"weight {weight:.9f} after {result.evaluations} evaluations (the minimum: {minimum:.9f})")
print(f"violation {result.violation:.6f}")
# the widths, from the support to the free end, next to the minimum's
print("segment  width     the minimum's")
x = result.best_genome
for i, (width, exact) in enumerate(zip(x.tolist(), optimum.solutions[0])):
    print(f"{i + 1:>7}  {width:.6f}  {exact:.6f}")
# the constraint is 61/x1³ + 37/x2³ + 19/x3³ + 7/x4³ + 1/x5³ ≤ 1, as g = that sum − 1 ≤ 0
deflection = 1.0 + problem.constraints(x).tolist()[0]
print(f"61/x1^3 + 37/x2^3 + 19/x3^3 + 7/x4^3 + 1/x5^3 = {deflection:.6f} (at most 1)")
trace.write()
