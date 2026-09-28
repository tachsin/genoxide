"""Easom: minimize Easom's function, a single narrow well in a nearly flat plane, with CMA-ES, from
30 seeds, without restarts and with IPOP restarts.

Away from its well, the function is within a hair of 0, and covered with tiny local minima. A run
of CMA-ES without restarts that doesn't find the well early converges into one of them; with IPOP
restarts, a run that has converged starts again from a random point with twice the population.
The table counts the runs that reach the minimum. The function, its bounds and its minimum come
from genoxide's problems.Easom, which run evaluates in Rust.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/easom/main.py
"""

import genoxide as gx

from trace import Trace

SEEDS = 30
BUDGET = 10_000
# a run stops once its error to the minimum is at most this
ERROR = 1e-8


def median(evaluations):
    """The median of the evaluations, rounded down; 0 without any."""
    evaluations = sorted(evaluations)
    middle = len(evaluations) // 2
    if not evaluations:
        return 0
    if len(evaluations) % 2:
        return evaluations[middle]
    return (evaluations[middle - 1] + evaluations[middle]) // 2


problem = gx.problems.Easom()
minimum = problem.optimum.value
target = minimum + ERROR
print(
    f"Easom: minimum {minimum:g} at (pi, pi), {SEEDS} seeds, {BUDGET} evaluations at most per run"
)
print("runs              at -1  elsewhere  evaluations: median  largest")
# with GENOXIDE_TRACE=<file>, a trace of the runs without restarts for the plot on the example's
# page
trace = Trace([problem.genome.bounds] * 2, problem.optimum)
for name, restarts in [("CMA-ES", None), ("CMA-ES with IPOP", "ipop")]:
    reached, elsewhere = 0, 0
    # the evaluations of the runs that reach the target
    evaluations = []
    for seed in range(1, SEEDS + 1):
        cmaes = gx.Cmaes(problem.genome, restarts=restarts, objective="minimize", seed=seed)
        on_generation = trace.on_generation if restarts is None else None
        result = cmaes.run(
            problem, target=target, evaluations=BUDGET, on_generation=on_generation
        )
        if result.stop_reason == "target":
            reached += 1
            evaluations.append(result.evaluations)
        else:
            elsewhere += 1
    largest = max(evaluations, default=0)
    print(
        f"{name:<16}  {reached:>5}  {elsewhere:>9}  {median(evaluations):>19}  {largest:>7}"
    )
print("evaluations: of the runs that reach the minimum, to within 1e-8")
trace.write()
