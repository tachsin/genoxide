"""Rocket injector: minimize two temperatures of a rocket injector and the length of its
combustion, three response surfaces.

SMS-EMOA with a population of 92, for 500 generations. Prints how many solutions of the final front
are feasible, the range of each objective on it, and its hypervolume, as a share of that of
genoxide's reference front.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/rocket_injector/main.py
"""

import genoxide as gx

from trace import REFERENCE, Trace, scaled

# the run's length
GENERATIONS = 500

problem = gx.problems.multi_engineering.RocketInjector()
sms_emoa = gx.SmsEmoa(
    problem.genome,
    objectives=problem.objectives,
    population_size=92,
    crossover=gx.SimulatedBinaryCrossover(20),
    mutation=gx.PolynomialMutation(20, rate=0.25),
    seed=1,
)
# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(problem)
result = sms_emoa.run(problem, generations=GENERATIONS, on_generation=trace.on_generation)

feasible = result.front_violations == 0
front = result.front_objectives[feasible]
size = len(result.front_objectives)
count = "all feasible" if feasible.all() else f"{int(feasible.sum())} feasible"
print(f"SMS-EMOA, {GENERATIONS} generations: {size} solutions on the front, {count}")
low, high = front.min(axis=0), front.max(axis=0)
print(
    f"  TF_max from {low[0]:.4f} to {high[0]:.4f}, "
    f"TT_max from {low[1]:.4f} to {high[1]:.4f}, "
    f"X_cc from {low[2]:.4f} to {high[2]:.4f}"
)
volume = gx.indicators.hypervolume(scaled(problem, front), [1.1, 1.1, 1.1])
print(
    f"  hypervolume {volume:.4f}, {100 * volume / REFERENCE:.2f}% of the reference front's "
    f"{REFERENCE}"
)
trace.write()
