"""The car side impact (Gu et al., 2001): the lightest car body whose side withstands the European
side-impact test, from the thicknesses of seven parts, subject to ten limits on the crash dummy's
injuries and the structure's velocities. A constrained continuous problem, whose best known weight
is 23.585658.

The constraints are response surfaces fitted to crash simulations. The fitness is the weight and
the constraint violation, which Deb's feasibility rules compare. SHADE, a differential evolution,
searches the thicknesses, and the example prints the best design and each response next to its
limit.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/car_side_impact/main.py
"""

import genoxide as gx

from trace import Trace

# L-SHADE's budget of evaluations: its population shrinks over it
BUDGET = 20_000

# the parts whose thicknesses are the genes, in their order
PARTS = [
    "B-pillar inner",
    "B-pillar reinforcement",
    "floor side inner",
    "cross members",
    "door beam",
    "door beltline reinforcement",
    "roof rail",
]

# the constraints, in their order: what each limits, the limit and its unit
LIMITS = [
    ("abdomen load", 1.0, "kN"),
    ("upper chest velocity", 0.32, "m/s"),
    ("middle chest velocity", 0.32, "m/s"),
    ("lower chest velocity", 0.32, "m/s"),
    ("upper rib deflection", 32.0, "mm"),
    ("middle rib deflection", 32.0, "mm"),
    ("lower rib deflection", 32.0, "mm"),
    ("pubic force", 4.0, "kN"),
    ("B-pillar velocity", 9.9, "mm/ms"),
    ("front door velocity", 15.7, "mm/ms"),
]


def scientific(value):
    """Two significant digits, as Rust writes them: 1.7e-9."""
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


problem = gx.problems.engineering.CarSideImpact()
best_known = problem.optimum.value
l_shade = gx.De(problem.genome, l_shade=BUDGET, objective=problem.objective, seed=1)
# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(problem)
result = l_shade.run(problem, evaluations=BUDGET, on_generation=trace.on_generation)

weight, evaluations = result.best_fitness, result.evaluations
print(f"weight {weight:.9f} after {evaluations} evaluations (the best known: {best_known:.9f})")
gap = scientific((weight - best_known) / best_known)
print(f"violation {result.violation:.6f}, a relative gap of {gap}")
x = result.best_genome
print(f"{'thickness (mm)':<29}{'best':>9}  range")
for part, xi, (low, high) in zip(PARTS, x.tolist(), problem.genome.bounds):
    print(f"{part:<29}{xi:>9.6f}  {low:g} to {high:g}")
# each constraint is the response minus its limit, at most 0
print(f"{'response':<29}{'best':>9}  limit")
for (name, limit, unit), g in zip(LIMITS, problem.constraints(x).tolist()):
    print(f"{name:<29}{g + limit:>9.4f}  {limit:g} {unit}")
trace.write()
