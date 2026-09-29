"""Eggholder: minimize the eggholder function, deep local minima all over and the deepest on the
edge of the box, from 30 seeds each with particle swarms of a ring and a global topology and, for
contrast, with CMA-ES with IPOP restarts.

The runs that end close together are grouped, and the table gives each group's best point and
value, and how many runs of each algorithm end there. The function, its bounds and its best known
minimum come from genoxide's problems.Eggholder, which run evaluates in Rust.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/eggholder/main.py
"""

import genoxide as gx

from trace import Trace

SEEDS = 30
BUDGET = 50_000
PARTICLES = 80
# a run stops once its error to the best known minimum is at most this
ERROR = 1e-6
ALGORITHMS = ["PSO, ring", "PSO, global", "CMA-ES, IPOP"]

problem = gx.problems.Eggholder()
minimum = problem.optimum.value
target = minimum + ERROR
# per group of runs that ended close together: its best point and value, and its runs per
# algorithm
minima = []
# per algorithm, the runs that reach the target
reached = [0, 0, 0]
# with GENOXIDE_TRACE=<file>, a trace of the runs of the swarm with the global topology for the
# plot on the example's page
trace = Trace([problem.genome.bounds] * 2, problem.optimum)
for a in range(len(ALGORITHMS)):
    for seed in range(1, SEEDS + 1):
        if a == 2:
            algorithm = gx.Cmaes(problem.genome, restarts="ipop", objective="minimize", seed=seed)
        else:
            algorithm = gx.Pso(
                problem.genome,
                population_size=PARTICLES,
                ring=None if a == 1 else 1,
                objective="minimize",
                seed=seed,
            )
        on_generation = trace.on_generation if a == 1 else None
        result = algorithm.run(
            problem, target=target, evaluations=BUDGET, on_generation=on_generation
        )
        if result.stop_reason == "target":
            reached[a] += 1
        point = result.best_genome.tolist()
        value = result.best_fitness
        # the same minimum: within 2% of the bounds' width, 20.48, in both genes
        close = (
            group
            for group in minima
            if all(abs(point[i] - group["point"][i]) <= 20.48 for i in range(2))
        )
        group = next(close, None)
        if group is None:
            runs = [0, 0, 0]
            runs[a] = 1
            minima.append({"point": point, "value": value, "runs": runs})
        else:
            group["runs"][a] += 1
            if value < group["value"]:
                group["point"], group["value"] = point, value
minima.sort(key=lambda group: group["value"])

print(
    f"Eggholder: best known minimum {minimum:.4f} at (512, 404.2318), {SEEDS} seeds, {BUDGET} "
    "evaluations at most per run"
)
pso_ring, pso_global, cmaes = ALGORITHMS
print(f"runs ending near        best value  {pso_ring}  {pso_global}  {cmaes}")
for group in minima:
    x1, x2 = group["point"]
    a, b, c = group["runs"]
    at = f"({x1:.1f}, {x2:.1f})"
    print(f"{at:<16}  {group['value']:>16.4f}  {a:>9}  {b:>11}  {c:>12}")
a, b, c = reached
label = "error below 1e-6"
print(f"{label:<34}  {a:>9}  {b:>11}  {c:>12}")
trace.write()
