"""Gear train design (Sandgren, 1990): the numbers of teeth of a compound gear train of four gears,
from 12 to 60 each, whose ratio is closest to 1/6.931. An integer problem.

The problem is genoxide's ``GearTrain``, on integer genes; its score is the squared error of the
ratio. A genetic algorithm with uniform crossover and a mutation that redraws each gene with
probability 0.25 searches the 49⁴ ≈ 5.8 million designs, until it reaches the minimum,
2.700857e-12, known by evaluating them all.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/gear_train/main.py
"""

import genoxide as gx

from trace import Trace

problem = gx.problems.engineering.GearTrain()
minimum = problem.optimum.value
ga = gx.Ga(
    problem.genome,
    objective=problem.objective,
    population_size=100,
    select=gx.Tournament(2),
    crossover=gx.UniformCrossover(),
    mutation=gx.UniformMutation(rate=0.25),
    seed=1,
)
# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(problem)
result = ga.run(problem, target=minimum, generations=20_000, on_generation=trace.on_generation)

teeth = result.best_genome.tolist()
ratio = teeth[0] * teeth[1] / (teeth[2] * teeth[3])
print(
    f"error {result.best_fitness:.6e} after {result.generations} generations "
    f"(the minimum: {minimum:.6e})"
)
print(f"teeth {tuple(teeth)}, ratio {ratio:.8f} (the target: {1 / 6.931:.8f})")
trace.write()
