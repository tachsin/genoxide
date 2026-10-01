"""MMA, the method of moving asymptotes, on a million variables: minimize the sum of c_j / x_j
subject to the sum of x_j <= V, a problem whose minimum is known in closed form,
x_j = V sqrt(c_j) / sum(sqrt(c_k)).

Each iteration replaces the function and the constraint by convex, separable approximations
around the current point, and solves them through their dual, in the constraint's single
multiplier. The example prints the best value and the largest error of a variable as the run
goes, and at the end the multiplier against its exact value.

The sums run in order with ``np.cumsum``, as Rust's do, so the run is the Rust example's to the
bit. With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the
example's page, with trace.py.

    python examples/mma/main.py
"""

import numpy as np

import genoxide as gx

from trace import Trace

# the variables, and the volume: on average 1 per variable
N = 1_000_000
VOLUME = float(N)
# a row of the table every this many iterations
EVERY = 4


def total(values):
    """The sum of ``values`` one after the other, as Rust adds them."""
    return float(np.cumsum(values)[-1])


def scientific(value, digits=1):
    """``value`` in scientific notation as Rust writes it, e.g. 1.2e-7."""
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return f"{mantissa}e{int(exponent)}"


# the costs cycle through 1 to 9
c = 1.0 + (np.arange(N) % 9).astype(np.float64)
# the minimum: the Lagrange conditions c_j / x_j^2 = lambda and the volume give
# x_j = V sqrt(c_j) / sum(sqrt(c_k)), the value sum(sqrt(c_k))^2 / V and the multiplier
# lambda = sum(sqrt(c_k))^2 / V^2
roots = np.sqrt(c)
roots_total = total(roots)
exact = VOLUME * roots / roots_total
minimum = total(c / exact)
multiplier = (roots_total / VOLUME) * (roots_total / VOLUME)
ones = np.ones((1, N))


def volume(x):
    """The value, its gradient, the constraint sum(x) - V <= 0 and its gradient, all ones."""
    return total(c / x), -c / (x * x), np.array([total(x) - VOLUME]), ones


def largest_error(x):
    """The largest relative error of a variable."""
    return float(np.max(np.abs((x - exact) / exact)))


def row(progress):
    """A row of the table: the iteration, the best value, and the largest relative error of a
    variable of the current point."""
    value = scientific(progress.best_fitness, 10)
    current = progress.population[0]
    return f"{progress.generation:>9}  {value:<16}  {scientific(largest_error(current)):>27}"


# from x_j = 0.5, half the volume, in [0.01, 10]; the dual's sums in parallel, with the same
# results as one after the other
mma = gx.Mma(
    gx.Real((0.01, 10.0), length=N),
    initial_genome=np.full(N, 0.5),
    parallel_sums=True,
    objective="minimize",
)
# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(minimum)
rows = []
state = {}


def on_generation(progress):
    trace.record(progress)
    rows.append(row(progress))


def control(running, progress):
    # the state after each generation, the last one's at the end
    state["converged"] = running.converged
    state["iterations"] = running.iterations
    state["multiplier"] = running.multipliers[0]


result = mma.run(
    volume, constraints=1, evaluations=200, on_generation=on_generation, control=control
)

print(f"minimize the sum of c_j / x_j subject to the sum of x_j <= {N}")
print(f"{N} variables in [0.01, 10], c_j = 1 + (j mod 9), from x_j = 0.5")
print("iteration  best value        largest error of a variable")
for iteration, line in enumerate(rows):
    if iteration % EVERY == 0 or iteration == len(rows) - 1:
        print(line)
assert result.stop_reason == "converged"
criterion = "the KKT conditions" if state["converged"] == "kkt" else "the step"
print(
    f"converged by {criterion} after {state['iterations']} iterations and "
    f"{result.evaluations} evaluations"
)
print(f"value {scientific(result.best_fitness, 10)}, the minimum {scientific(minimum, 10)}")
x = result.best_genome
print(
    f"largest relative error of a variable {scientific(largest_error(x))}, the volume used "
    f"{total(x):.6f} (violation {scientific(result.violation)})"
)
print(
    f"multiplier {state['multiplier']:.10f}, the exact (sum of the roots of c_j)^2 / V^2 = "
    f"{multiplier:.10f}"
)
trace.write()
