"""Engineering designs and CEC 2006: SHADE with Deb's feasibility rules on eleven constrained
problems, five engineering designs and the CEC 2006 problems g01 to g06.

Each problem's fitness is its value and its constraint violation, which Deb's rules compare: a
feasible solution beats an infeasible one. One seeded SHADE run per problem, with a budget per
problem, stops early within 1e-8 (relative) of the optimum or best known value f*. The table gives
the best value found, f*, the relative gap (f - f*) / |f*| and whether the best is feasible. The
problems come from genoxide's problems, which run evaluates in Rust.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/engineering_suite/main.py
"""

import math

import genoxide as gx
from genoxide.problems import cec2006, engineering

from trace import Trace

# the engineering designs' budget, and the CEC 2006 report's for its problems
DESIGN_BUDGET = 50_000
CEC_BUDGET = 500_000
# a run stops once its best is feasible and within this of f*, relative to |f*|
TOLERANCE = 1e-8


def significant(value):
    """6 significant digits, e.g. 0.0126652 or -30665.5."""
    digits = math.floor(math.log10(abs(value)))
    return f"{value:.{max(5 - digits, 0)}f}"


def scientific(value):
    """Two significant digits, e.g. 1.2e-7."""
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


suite = [
    (engineering.TensionCompressionSpring(), DESIGN_BUDGET),
    (engineering.SpeedReducer(), DESIGN_BUDGET),
    (engineering.ThreeBarTruss(), DESIGN_BUDGET),
    (engineering.CantileverBeam(), DESIGN_BUDGET),
    (engineering.CarSideImpact(), DESIGN_BUDGET),
    (cec2006.G01(), CEC_BUDGET),
    (cec2006.G02(), CEC_BUDGET),
    (cec2006.G03(), CEC_BUDGET),
    (cec2006.G04(), CEC_BUDGET),
    (cec2006.G05(), CEC_BUDGET),
    (cec2006.G06(), CEC_BUDGET),
]

print("SHADE with Deb's feasibility rules, one run per problem, seed 1")
row = "{:<25}{:>8}{:>12}{:>12}{:>12}{:>10}{:>10}"
print(row.format("problem", "budget", "best found", "f*", "f* is", "gap", "feasible"))
# with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
trace = Trace()
for problem, budget in suite:
    optimum = problem.optimum
    f_star = optimum.value
    target = f_star + TOLERANCE * abs(f_star)
    shade = gx.De(problem.genome, objective=problem.objective, seed=1)
    result = shade.run(
        problem,
        target=target,
        evaluations=budget,
        on_generation=trace.errors(f"{problem.name}/SHADE", problem, f_star),
    )
    value = result.best_fitness
    feasible = result.violation == 0.0
    # the relative gap; below the tolerance, the run stopped early
    if feasible and value <= target:
        gap = "< 1e-8"
    else:
        gap = scientific((value - f_star) / abs(f_star))
    print(
        row.format(
            problem.name,
            budget,
            significant(value),
            significant(f_star),
            "optimum" if optimum.proven else "best known",
            gap,
            "yes" if feasible else "no",
        )
    )
trace.write()
