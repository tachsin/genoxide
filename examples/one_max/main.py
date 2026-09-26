"""OneMax: find the bit string with the most ones.

The "hello world" of genetic algorithms: a binary genome, tournament selection, uniform crossover
and bit-flip mutation, with the best count printed every 50 generations.

    python examples/one_max/main.py
"""

import genoxide as gx

LEN = 500

ga = gx.Ga(
    gx.Binary(LEN),
    population_size=100,
    select=gx.Tournament(3),
    crossover=gx.UniformCrossover(),
    mutation=gx.BitFlip(rate=1 / LEN),
    seed=42,
)


def progress(progress):
    if progress.generation % 50 == 0:
        print(f"{progress.generation:>10}  {progress.best_fitness:>4.0f}")


print("generation  best")
result = ga.run(lambda bits: bits.sum(), target=LEN, generations=10_000, on_generation=progress)
print(
    f"\n{result.best_fitness:.0f} ones after {result.generations} generations and "
    f"{result.evaluations} evaluations (the optimum: {LEN})"
)
