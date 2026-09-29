"""CEC 2006 g22: the linear function x1 of 22 variables under 1 nonlinear inequality and 19
equality constraints, from the CEC 2006 special session on constrained optimization (Liang et al.,
2006). The best known value is 236.430975504001, with the equalities met within the report's
tolerance of 0.0001.

genoxide's ``G22`` gives the value of a solution and its constraint violation, which Deb's
feasibility rules compare: a feasible solution beats an infeasible one. L-SHADE on the 22
variables, for the report's budget of 500,000 evaluations, doesn't find a feasible solution. The
19 equalities can be solved in order, though, from x1, x8 and x9: SHADE on those three, with the
other 19 variables solved from them, finds the least value with every equality met, 236.370313,
below the report's best known. The example prints both runs, and the second's solution and
constraints. ``run`` evaluates the problem in Rust, and the fitness of the second run evaluates it
in Rust, a generation at a time.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of the second run for the plot on the
example's page, with trace.py.

    python examples/cec2006_g22/main.py
"""

import math

import genoxide as gx
import numpy as np
from genoxide.problems.cec2006 import EQUALITY_TOLERANCE

from trace import Trace

# the CEC 2006 report's budget of evaluations per run
BUDGET = 500_000
# the least value with every equality met exactly, at x8 = 130 and x9 = 170 (genoxide's docs)
EXACT = 236.370313314566
# the second run stops once its best is feasible and within this of EXACT
ERROR = 1e-8
# a constraint within this of its boundary is active
ACTIVE = 1e-6


def significant(value, digits):
    """``digits`` significant digits, e.g. 29.9953 or -30665.5 for 6."""
    if value == 0:
        return "0"
    magnitude = math.floor(math.log10(abs(value)))
    return f"{value:.{max(digits - 1 - magnitude, 0)}f}"


def state(g):
    """A constraint g(x) <= 0: "active" on its boundary, else its value."""
    return "active" if abs(g) <= ACTIVE else significant(g, 4)


def scientific(value, decimals):
    """Scientific notation as Rust writes it, e.g. 1.2e-5 for 1 decimal."""
    mantissa, exponent = f"{value:.{decimals}e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def feasibility(violation):
    """"feasible", or "infeasible" with the violation."""
    return "feasible" if violation == 0.0 else f"infeasible, violation {significant(violation, 4)}"


# the report's equality tolerance, EQUALITY_TOLERANCE
problem = gx.problems.cec2006.G22()
optimum = problem.optimum
low, high = np.array(problem.genome.bounds, dtype=np.float64).T


def solve(genes):
    """The 22 variables from x1, x8 and x9, the other 19 solved from the equalities in order, each
    kept within its bounds: a variable that its bounds cut leaves its equality unmet."""
    # x[i] is x_i, with x[0] unused; numpy's float64, which divides by 0 as Rust does
    x = np.zeros(23)

    def within(i, value):
        return min(max(value, low[i - 1]), high[i - 1])

    x[1], x[8], x[9] = genes
    # h1 to h6, h10 and h11 are linear
    x[5] = within(5, 100_000.0 * x[8] - 1e7)
    x[6] = within(6, 100_000.0 * x[9] - 100_000.0 * x[8])
    x[7] = within(7, 5e7 - 100_000.0 * x[9])
    x[10] = within(10, (3.3e7 - x[5]) / 100_000.0)
    x[11] = within(11, (4.4e7 - x[6]) / 100_000.0)
    x[12] = within(12, (6.6e7 - x[7]) / 100_000.0)
    x[16] = within(16, x[11] - x[8])
    x[17] = within(17, x[12] - x[9])
    # h12 to h16 give the logarithms: with x18 to x22 at 0, they're the logarithms themselves, as
    # genoxide computes them, to the bit what the Rust example gets
    h = problem.constraints(x[1:])[1:]
    for i in range(18, 23):
        x[i] = within(i, h[i - 7])
    # h17 to h19 give x13 to x15, and h7 to h9 x2 to x4
    x[13] = within(13, (x[8] + x[10] - 400.0) / (x[18] - x[19]))
    x[14] = within(14, (x[9] + x[11] - x[8] - 400.0) / (x[20] - x[21]))
    x[15] = within(15, (x[12] - x[9] - 100.0) / (x[22] - 4.60517))
    x[2] = within(2, x[5] / (120.0 * x[13]))
    x[3] = within(3, x[6] / (80.0 * x[14]))
    x[4] = within(4, x[7] / (40.0 * x[15]))
    return x[1:]


def fitness(genes):
    """The values and violations of a generation of x1, x8 and x9, with the other variables solved
    from them."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return problem.evaluate(np.array([solve(row) for row in genes]))


# the 22 variables, as the report poses the problem: SHADE with a population that shrinks over the
# budget, from 18 · 22 = 396 to 4
l_shade = gx.De(problem.genome, l_shade=BUDGET, objective=problem.objective, seed=1)
result = l_shade.run(problem, evaluations=BUDGET)
print("L-SHADE on the 22 variables with Deb's feasibility rules, seed 1")
print(f"stopped after {result.evaluations} evaluations: {feasibility(result.violation)}")

# x1, x8 and x9 within their bounds, and the other 19 variables solved from the equalities
free = gx.Real([problem.genome.bounds[0], problem.genome.bounds[7], problem.genome.bounds[8]])
shade = gx.De(free, objective="minimize", seed=1)
# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(problem, solve)
result = shade.run(
    fitness,
    batch=True,
    target=EXACT + ERROR,
    evaluations=BUDGET,
    on_generation=trace.on_generation,
)
value = result.best_fitness
x = solve(result.best_genome)
print("SHADE on x1, x8 and x9, the other 19 variables solved from the equalities, seed 1")
stop = "stopped by the target" if result.stop_reason == "target" else "stopped"
print(f"{stop} after {result.evaluations} evaluations: {feasibility(result.violation)}")
print(
    f"f(x) {value:.6f}, {scientific(value - EXACT, 1)} above {EXACT:.6f}, the least with every "
    "equality met"
)
print(
    f"f(x) - f* {value - optimum.value:.6f}, where f* {optimum.value:.6f} is the report's best "
    "known"
)
print(", ".join(f"x{i} {significant(xi, 6)}" for i, xi in enumerate(x.tolist(), 1)))
# g1, then h1 to h19, each equality as its excess over the tolerance, 0 when it's met
constraints = problem.constraints(x).tolist()
g, h = constraints[:1], constraints[1:]
states = [f"g{i} {state(gi)}" for i, gi in enumerate(g, 1)]
states += [f"h{i} {state(max(abs(hi) - EQUALITY_TOLERANCE, 0.0))}" for i, hi in enumerate(h, 1)]
print(", ".join(states))
trace.write()
