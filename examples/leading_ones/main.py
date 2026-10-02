"""LeadingOnes: maximize the number of ones before the first zero of a string of 100 bits, with
the (1+1) evolutionary algorithm.

The (1+1) EA keeps one string, flips each of its bits with probability 1/n and keeps the child if
it's no worse: genoxide's LocalSearch with BitFlip(rate=1/n) as its neighbor. A run from seed 1
prints the evaluations at which the leading ones reach 10, 20, … 100; then runs from seeds 1 to
100 give the evaluations to the optimum, which Droste, Jansen and Wegener (2002) proved to be
Θ(n²). The function is genoxide's problems.binary.LeadingOnes, which run evaluates in Rust.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/leading_ones/main.py
"""

import genoxide as gx

from trace import Trace

BITS = 100
SEEDS = 100
# the most evaluations of a run
BUDGET = 1_000_000


def one_plus_one(problem, seed):
    """The (1+1) evolutionary algorithm from ``seed``."""
    return gx.LocalSearch(problem.genome, neighbor=gx.BitFlip(rate=1 / BITS), seed=seed)


problem = gx.problems.binary.LeadingOnes(BITS)
optimum = problem.optimum.value
print(f"LeadingOnes of {BITS} bits, the (1+1) EA from seed 1")
print("leading ones  evaluations")
# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(BITS, optimum)
next_tenth = 10


def progress(progress):
    global next_tenth
    best = progress.best_fitness or 0.0
    # the evaluations at which the leading ones first reach each tenth of the string
    while best >= next_tenth:
        print(f"{next_tenth:>12}  {progress.evaluations:>11}")
        next_tenth += 10
    trace.record(progress)


result = one_plus_one(problem, 1).run(
    problem, target=optimum, evaluations=BUDGET, on_generation=progress
)
print(
    f"{result.best_fitness:.0f} leading ones after {result.evaluations} evaluations "
    f"(the optimum: {BITS})"
)

# the evaluations to the optimum from seeds 1 to 100
evaluations = []
for seed in range(1, SEEDS + 1):
    result = one_plus_one(problem, seed).run(problem, target=optimum, evaluations=BUDGET)
    if result.stop_reason == "target":
        evaluations.append(result.evaluations)
evaluations.sort()
count = len(evaluations)
mean = sum(evaluations) / count
if count % 2:
    median = float(evaluations[count // 2])
else:
    median = (evaluations[count // 2 - 1] + evaluations[count // 2]) / 2
print(f"\nseeds 1 to {SEEDS}: {count} reach the optimum, after {mean:.0f} evaluations on average")
print(f"(fewest {evaluations[0]}, median {median:.1f}, most {evaluations[-1]}); n² = {BITS * BITS}")
trace.write()
