"""Schaffer F6: minimize Schaffer's F6, rings of local minima around the global one, from 30 seeds
each with SHADE, a differential evolution, and, for contrast, with a particle swarm and a genetic
algorithm.

The function depends only on the distance r from the origin, and its local minima are rings near
r = π, 2π, 3π, …; the table counts the runs whose best point ends on each ring. The function, its
bounds and its minimum come from genoxide's problems.SchafferF6, which run evaluates in Rust.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/schaffer_f6/main.py
"""

import math

import genoxide as gx

from trace import Trace

SEEDS = 30
BUDGET = 50_000
# a run stops once its error to the minimum is at most this
ERROR = 1e-6

problem = gx.problems.SchafferF6()
target = problem.optimum.value + ERROR
# by ring, 0 the minimum and k the ring near r = kπ: the best value among the runs that end near
# it, and the runs per algorithm
rings = {}
# per algorithm, the runs that reach the target
reached = [0, 0, 0]
# with GENOXIDE_TRACE=<file>, a trace of SHADE's runs for the plot on the example's page
trace = Trace(problem.optimum)
for a in range(3):
    for seed in range(1, SEEDS + 1):
        if a == 0:
            algorithm = gx.De(problem.genome, objective="minimize", seed=seed)
        elif a == 2:
            algorithm = gx.Ga(
                problem.genome,
                population_size=100,
                select=gx.Tournament(3),
                crossover=gx.SimulatedBinaryCrossover(15),
                mutation=gx.PolynomialMutation(20, rate=0.5),
                objective="minimize",
                seed=seed,
            )
        else:
            algorithm = gx.Pso(problem.genome, population_size=40, objective="minimize", seed=seed)
        on_generation = trace.on_generation if a == 0 else None
        result = algorithm.run(
            problem, target=target, evaluations=BUDGET, on_generation=on_generation
        )
        if result.stop_reason == "target":
            reached[a] += 1
        x1, x2 = result.best_genome.tolist()
        value = result.best_fitness
        ring = round(math.sqrt(x1 * x1 + x2 * x2) / math.pi)
        entry = rings.setdefault(ring, {"value": value, "runs": [0, 0, 0]})
        entry["value"] = min(entry["value"], value)
        entry["runs"][a] += 1

print(f"Schaffer F6: minimum 0 at the origin, {SEEDS} seeds, {BUDGET} evaluations at most per run")
print("runs ending near        best value  SHADE  PSO  GA")
for k in sorted(rings):
    near = "the minimum, r = 0" if k == 0 else f"the ring at r = {k * math.pi:.2f}"
    shade, swarm, ga = rings[k]["runs"]
    print(f"{near:<22}  {rings[k]['value']:>10.6f}  {shade:>5}  {swarm:>3}  {ga:>2}")
shade, swarm, ga = reached
print(f"{'error below 1e-6':<34}  {shade:>5}  {swarm:>3}  {ga:>2}")
trace.write()
