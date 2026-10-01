"""A global method, then a local one: SHADE on Rastrigin's function in 10 dimensions for 40,000
evaluations, which finds the basin of the global minimum, then L-BFGS-B from SHADE's best, with the
analytic gradient, down to the minimum itself, f = 0, in a few evaluations.

For contrast, SHADE alone, run on to f <= 1e-12.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its runs for the plot on the example's
page, with trace.py.

    python examples/polish/main.py
"""

import genoxide as gx

import trace

N = 10
# SHADE's evaluations before the polish
GLOBAL = 40_000
# a row of SHADE's table every this many evaluations
EVERY = 5_000


def scientific(value):
    """Two significant digits, e.g. 1.2e-7."""
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def largest(x):
    """The largest distance of a gene from 0."""
    return max(abs(float(gene)) for gene in x)


def recorder(rows):
    """An on_generation that keeps the evaluations and the best value after each generation."""

    def record(progress):
        rows.append((progress.evaluations, progress.best_fitness))

    return record


problem = gx.problems.Rastrigin(N)


def shade():
    return gx.De(problem.genome, objective="minimize", seed=1)


# SHADE for the global search: the best value after each generation
global_rows = []
found = shade().run(problem, evaluations=GLOBAL, on_generation=recorder(global_rows))

# L-BFGS-B from SHADE's best, with its default tolerances
local_rows = []
criteria = []
lbfgsb = gx.Lbfgsb(problem.genome, initial_genome=found.best_genome, objective="minimize")
polished = lbfgsb.run(
    problem,
    evaluations=10_000,
    on_generation=recorder(local_rows),
    control=lambda algorithm, progress: criteria.append(algorithm.converged),
)

# SHADE alone, to f <= 1e-12, for contrast
alone_rows = []
without = shade().run(
    problem, target=1e-12, evaluations=1_000_000, on_generation=recorder(alone_rows)
)

print(f"Rastrigin's function in {N} dimensions, its minimum 0 at the origin")
print(f"SHADE, 100 individuals, seed 1, for {GLOBAL} evaluations")
print("evaluations  best value")
for evaluations, value in global_rows:
    if evaluations % EVERY == 0:
        print(f"{evaluations:>11}  {scientific(value):>10}")
print(
    f"SHADE's best: f = {scientific(found.best_fitness)}, every gene within "
    f"{scientific(largest(found.best_genome))} of 0: in the global minimum's basin"
)
print("L-BFGS-B from SHADE's best, with the analytic gradient")
print("round  evaluations  best value")
for round_, (evaluations, value) in enumerate(local_rows):
    print(f"{round_:>5}  {evaluations:>11}  {scientific(value):>10}")
assert polished.stop_reason == "converged"
assert polished.best_fitness == 0.0
how = (
    "converged: a projected gradient below 1e-5"
    if criteria[-1] == "projected_gradient"
    else "converged"
)
print(
    f"L-BFGS-B: f = {polished.best_fitness!r} after {polished.evaluations} evaluations, every "
    f"gene within {scientific(largest(polished.best_genome))} of 0 ({how})"
)
print(f"in all, {GLOBAL + polished.evaluations} evaluations to the minimum")
assert without.stop_reason == "target"
print(
    f"SHADE alone, for contrast: f = {scientific(without.best_fitness)} after "
    f"{without.evaluations} evaluations"
)

# with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
trace.write_runs(alone_rows, GLOBAL, local_rows)
