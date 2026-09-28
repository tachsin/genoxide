"""Hartmann 6-D: minimize Hartmann's function in 6 dimensions with CMA-ES, from 30 seeds, without
restarts and with IPOP restarts.

The function has two basins of nearly the same depth. A run of CMA-ES without restarts converges
into one of them and stays; with IPOP restarts, a run that has converged starts again from a
random point with twice the population. The table counts the runs that end in each minimum. The
function, its bounds and its best known minimum come from genoxide's problems.Hartmann6, which
run evaluates in Rust.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/hartmann6/main.py
"""

import genoxide as gx

from trace import Trace

SEEDS = 30
BUDGET = 10_000
# a run stops once its error to the best known minimum is at most this
ERROR = 1e-6
# the other local minimum, where a third of local searches end
OTHER_MINIMUM = -3.203161918396231


def median(evaluations):
    """The median of the evaluations, rounded down; 0 without any."""
    evaluations = sorted(evaluations)
    middle = len(evaluations) // 2
    if not evaluations:
        return 0
    if len(evaluations) % 2:
        return evaluations[middle]
    return (evaluations[middle - 1] + evaluations[middle]) // 2


problem = gx.problems.Hartmann6()
minimum = problem.optimum.value
target = minimum + ERROR
print(
    f"Hartmann 6-D: best known minimum {minimum:.5f}, {SEEDS} seeds, {BUDGET} evaluations at "
    "most per run"
)
columns = f"at {minimum:.5f}  at {OTHER_MINIMUM:.5f}  elsewhere"
print(f"{'runs':<16}  {columns}  evaluations: median  largest")
# the seed of the first run without restarts that ends at the other minimum: the trace records
# the run with IPOP restarts from that seed
trace_seed = None
for name, restarts in [("CMA-ES", None), ("CMA-ES with IPOP", "ipop")]:
    counts = {"global": 0, "other": 0, "elsewhere": 0}
    # the evaluations of the runs that reach the target
    evaluations = []
    for seed in range(1, SEEDS + 1):
        cmaes = gx.Cmaes(problem.genome, restarts=restarts, objective="minimize", seed=seed)
        trace = Trace(problem, recording=restarts == "ipop" and seed == trace_seed)
        result = cmaes.run(
            problem, target=target, evaluations=BUDGET, on_generation=trace.on_generation
        )
        trace.write()
        if result.stop_reason == "target":
            counts["global"] += 1
            evaluations.append(result.evaluations)
        elif abs(result.best_fitness - OTHER_MINIMUM) <= ERROR:
            counts["other"] += 1
            if trace_seed is None:
                trace_seed = seed
        else:
            counts["elsewhere"] += 1
    largest = max(evaluations, default=0)
    print(
        f"{name:<16}  {counts['global']:>11}  {counts['other']:>11}  {counts['elsewhere']:>9}  "
        f"{median(evaluations):>19}  {largest:>7}"
    )
print("evaluations: of the runs that reach the best known minimum")
