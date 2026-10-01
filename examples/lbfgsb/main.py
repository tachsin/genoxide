"""L-BFGS-B: minimize Rosenbrock's function in 100 dimensions from the classic start
(-1.2, 1, -1.2, 1, ...) to f <= 1e-10, once with its analytic gradient and once with forward
differences, to compare their cost.

Both runs take about the same steps: the gradient is the same to about 7 digits. The analytic
gradient costs one evaluation per trial point, forward differences 101.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its runs for the plot on the example's
page, with trace.py.

    python examples/lbfgsb/main.py
"""

import genoxide as gx

import trace

N = 100
# a row of the table every this many rounds
EVERY = 50


def scientific(value):
    """Two significant digits, e.g. 1.2e-7."""
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def run(gradients):
    """The rounds of a run to f <= 1e-10, as (evaluations, best value), its result and its
    iterations."""
    problem = gx.problems.Rosenbrock(N)
    lbfgsb = gx.Lbfgsb(
        problem.genome,
        initial_genome=[-1.2, 1.0] * (N // 2),
        gradients=gradients,
        gradient_tolerance=0.0,
        function_tolerance=0.0,
        objective="minimize",
    )
    rounds = []
    iterations = []

    def on_generation(progress):
        rounds.append((progress.evaluations, progress.best_fitness))

    def control(algorithm, progress):
        iterations.append(algorithm.iterations)

    result = lbfgsb.run(
        problem,
        target=1e-10,
        evaluations=200_000,
        on_generation=on_generation,
        control=control,
    )
    return rounds, result, iterations[-1]


value = gx.problems.Rosenbrock(N)([-1.2, 1.0] * (N // 2))
analytic, analytic_result, analytic_iterations = run("auto")
forward, forward_result, forward_iterations = run("forward")

print(
    f"Rosenbrock's function in {N} dimensions, from (-1.2, 1, -1.2, 1, ...) where f = {value:.0f}"
)
print("L-BFGS-B with 10 pairs to f <= 1e-10: the analytic gradient, and forward differences")
print("       analytic gradient        forward differences")
print("round  evaluations  best value  evaluations  best value")


def cell(rounds, round_):
    if round_ < len(rounds):
        evaluations, best = rounds[round_]
        return f"{evaluations:>11}  {scientific(best):>10}"
    return f"{'':>11}  {'':>10}"


for round_ in range(max(len(analytic), len(forward))):
    last = round_ + 1 in (len(analytic), len(forward))
    if round_ % EVERY == 0 or last:
        print(f"{round_:>5}  {cell(analytic, round_)}  {cell(forward, round_)}")
for name, result, iterations in [
    ("analytic gradient", analytic_result, analytic_iterations),
    ("forward differences", forward_result, forward_iterations),
]:
    assert result.stop_reason == "target"
    print(
        f"{name}: f = {scientific(result.best_fitness)} after {iterations} iterations and "
        f"{result.evaluations} evaluations"
    )
print(
    "forward differences took "
    f"{forward_result.evaluations / analytic_result.evaluations:.0f} times the evaluations"
)

# with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
trace.write_runs(analytic, forward)
