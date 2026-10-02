"""OneMax: find the bit string with the most ones.

The "hello world" of genetic algorithms: a binary genome, tournament selection, uniform crossover
and bit-flip mutation, with the best count printed every 50 generations. The function is
genoxide's problems.binary.OneMax, which run evaluates in Rust.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/one_max/main.py
"""

import genoxide as gx

from trace import Trace

LEN = 500

problem = gx.problems.binary.OneMax(LEN)
ga = gx.Ga(
    problem.genome,
    population_size=100,
    select=gx.Tournament(3),
    crossover=gx.UniformCrossover(),
    mutation=gx.BitFlip(rate=1 / LEN),
    seed=42,
)


# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(LEN)


def progress(progress):
    if progress.generation % 50 == 0:
        print(f"{progress.generation:>10}  {progress.best_fitness:>4.0f}")
    trace.record(progress)


print("generation  best")
result = ga.run(
    problem, target=problem.optimum.value, generations=10_000, on_generation=progress
)
print(
    f"\n{result.best_fitness:.0f} ones after {result.generations} generations and "
    f"{result.evaluations} evaluations (the optimum: {LEN})"
)
trace.write()
